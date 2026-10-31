"""Pont asyncio pour l'UI GTK4 : boucle d\u00e9di\u00e9e sur un thread + file d'\u00e9v\u00e9nements.

Même architecture que desktop/controller.py (thread asyncio d\u00e9di\u00e9 + queue
thread-safe), adapt\u00e9 \u00e0 GTK : le thread GTK d\u00e9pose des coroutines via
run_coroutine_threadsafe et lit la file d'\u00e9v\u00e9nements avec GLib.idle_add.

R\u00e9utilise le c\u0153ur existant (config + LLMClient + McpToolProvider + Agent)
sans passer par GuiBackend, pour avoir tous les callbacks du tour (on_tool_result
avec statut/dur\u00e9e, que GuiBackend ne forward pas) :

  {"type": "reasoning",    "text": ...}
  {"type": "content",      "text": ...}
  {"type": "tool_call",    "name": ..., "args": {...}}
  {"type": "tool_result",  "name": ..., "args": {...}, "ok": bool,
                           "duration": float, "result": str}
  {"type": "done",         "reply": ..., "trace": {...}}
  {"type": "error",        "message": ...}

+ sessions persistantes (SessionStore), statut/d\u00e9marrage du llama-server
(gui.server), contexte r\u00e9el (GET /slots), r\u00e9glages UI persist\u00e9s dans
~/.config/jarvis/ui.json (th\u00e8me, ctx_size, gpu, tts).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import queue
import threading
from pathlib import Path

import httpx

from ..agent import Agent
from ..compact import compact_tool
from ..config import Config, load_config
from ..gpu import args_for_device, config_dir, env_for_device, load_gpu_choice, save_gpu_choice
from ..gui import server as server_mod
from ..gui.models import discover_models
from ..llm import LLMClient
from ..mcp import McpToolProvider
from ..sessions import SessionStore

log = logging.getLogger("jarvis.gtk.backend")

_LLM_PARAMS = set(inspect.signature(LLMClient.__init__).parameters) - {"self"}


def ui_settings_path() -> Path:
    return config_dir() / "ui.json"


def load_ui_settings() -> dict:
    try:
        return json.loads(ui_settings_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_ui_settings(patch: dict) -> dict:
    data = load_ui_settings()
    data.update(patch)
    ui_settings_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def state_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "jarvis"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def server_ctx_usage(base_url: str, timeout: float = 3.0) -> dict:
    """GET /props + /slots de llama-server : usage r\u00e9el du contexte.

    Retourne {"n_ctx": int|None, "used": int, "free": int}. Ne l\u00e8ve jamais.
    """
    out: dict = {"n_ctx": None, "used": 0, "free": 0}
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        async with httpx.AsyncClient(base_url=root, timeout=timeout) as http:
            props = await http.get("/props")
            if props.status_code == 200:
                out["n_ctx"] = (props.json().get("default_generation_settings") or {}).get("n_ctx")
            slots = await http.get("/slots")
            if slots.status_code == 200:
                data = slots.json()
                if isinstance(data, list) and data:
                    s = data[0]
                    n_ctx = s.get("n_ctx") or out["n_ctx"]
                    out["n_ctx"] = n_ctx
                    out["used"] = int(s.get("n_prompt_tokens") or 0)
                    out["free"] = max(0, int(n_ctx or 0) - out["used"])
    except Exception as exc:  # noqa: BLE001 — indicateur best-effort
        log.debug("ctx usage indisponible: %s", exc)
    return out


class GtkBackend:
    """C\u0153ur async de l'UI GTK : config + LLM + MCP + Agent + sessions."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.events: queue.Queue = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="jarvis-gtk-async",
                                        daemon=True)
        self.cfg: Config = load_config(config_path)
        self.llm = LLMClient(**{k: v for k, v in self.cfg.llm.__dict__.items()
                                if k in _LLM_PARAMS})
        self.tools = McpToolProvider(self.cfg.mcp_servers)
        self.agent = Agent(self.llm, self.tools, self.cfg.agent.system_prompt,
                           self.cfg.agent.max_iterations, stream=True,
                           internal_tools={"jarvis__compact": compact_tool})
        self.history: list[dict] = []
        self.busy = False
        self._lock = asyncio.Lock()
        self.sessions = SessionStore(state_dir())
        self.session_id: str | None = None
        self.models_llm = discover_models(use_defaults=True)["llm"]

    # ---------- cycle de vie ----------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> None:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._astart(), self.loop)

    async def _astart(self) -> None:
        try:
            if self.cfg.mcp_servers:
                await self.tools.start()
            # reprise de la derni\u00e8re session comme au CLI
            self.session_id = self.sessions.latest_id()
            if self.session_id:
                try:
                    self.history = self.sessions.load(self.session_id)
                except Exception as exc:  # noqa: BLE001
                    log.debug("session illisible: %s", exc)
                    self.session_id = None
            self.events.put({"type": "backend_ready",
                             "tools": self.tools.names(),
                             "session_id": self.session_id,
                             "history_len": len(self.history)})
        except Exception as exc:  # noqa: BLE001 — remonter à l'UI
            log.exception("démarrage backend GTK")
            self.events.put({"type": "backend_error", "message": str(exc)})

    def stop(self) -> None:
        async def _astop() -> None:
            if self.session_id and self.history:
                try:
                    self.sessions.save(self.session_id, self.history)
                except Exception as exc:  # noqa: BLE001
                    log.debug("sauvegarde session: %s", exc)
            try:
                await self.tools.stop()
            except RuntimeError as exc:
                log.debug("fermeture MCP dégradée: %s", exc)
            await self.llm.aclose()
        try:
            asyncio.run_coroutine_threadsafe(_astop(), self.loop).result(timeout=15)
        except Exception as exc:  # noqa: BLE001
            log.debug("arrêt backend: %s", exc)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)

    # ---------- appels depuis le thread GTK ----------

    def _run(self, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def _save_session(self) -> None:
        if self.session_id is None:
            self.session_id = self.sessions.new_id()
        try:
            self.sessions.save(self.session_id, self.history)
        except Exception as exc:  # noqa: BLE001
            log.debug("sauvegarde session: %s", exc)

    def stream_turn(self, text: str) -> None:
        self._run(self._astream_turn(text))

    async def _astream_turn(self, text: str) -> None:
        def sink(event: dict) -> None:
            self.events.put(event)

        if self.busy:
            sink({"type": "error", "message": "un tour est déjà en cours"})
            return
        async with self._lock:
            self.busy = True
        try:
            reply, self.history, trace = await self.agent.turn(
                self.history, text,
                on_tool_call=lambda n, a: sink({"type": "tool_call", "name": n, "args": a}),
                on_tool_result=lambda n, a, ok, dur, res: sink({
                    "type": "tool_result", "name": n, "args": a, "ok": ok,
                    "duration": dur, "result": res}),
                on_reasoning=lambda f: sink({"type": "reasoning", "text": f}),
                on_content=lambda f: sink({"type": "content", "text": f}),
            )
            sink({"type": "done", "reply": reply,
                  "trace": {"tool_calls": trace.tool_calls,
                            "iterations": trace.iterations,
                            "error": trace.error}})
            self._save_session()
        except Exception as exc:  # noqa: BLE001 — remonter à l'UI
            log.exception("tour en erreur")
            sink({"type": "error", "message": str(exc)})
        finally:
            self.busy = False

    def new_session(self) -> None:
        self._run(self._anew())

    async def _anew(self) -> None:
        self.history.clear()
        self.session_id = self.sessions.new_id()
        self._save_session()
        self.events.put({"type": "session_new", "session_id": self.session_id,
                         "history_len": 0})

    def load_session(self, sid: str) -> None:
        self._run(self._aload(sid))

    async def _aload(self, sid: str) -> None:
        try:
            history = self.sessions.load(sid)
        except Exception as exc:  # noqa: BLE001
            self.events.put({"type": "error", "message": f"session illisible: {exc}"})
            return
        self.session_id = sid
        self.history = history
        self.events.put({"type": "session_loaded", "session_id": sid,
                         "history_len": len(history)})

    def list_sessions(self) -> None:
        self._run(self._alist_sessions())

    async def _alist_sessions(self) -> None:
        self.events.put({"type": "sessions_list", "sessions": self.sessions.list(limit=50)})

    # ---------- serveur LLM ----------

    def refresh_status(self) -> None:
        self._run(self._astatus())

    async def _astatus(self) -> None:
        st = await server_mod.get_status(self.cfg.llm.base_url)
        usage = await server_ctx_usage(self.cfg.llm.base_url)
        self.events.put({"type": "server_status", "state": st.state, "host": st.host,
                         "port": st.port, "models": st.models, "detail": st.detail,
                         "ctx": usage})

    def start_server(self, model_path: str, ctx_size: int | None = None) -> None:
        self._run(self._astart_server(model_path, ctx_size))

    async def _astart_server(self, model_path: str, ctx_size: int | None) -> None:
        extra = ["--ctx-size", str(ctx_size)] if ctx_size else None
        st = await server_mod.start_server(model_path, self.cfg.llm.base_url,
                                           extra_args=extra)
        self.events.put({"type": "server_started", "state": st.state, "host": st.host,
                         "port": st.port, "models": st.models, "detail": st.detail})
        await self._astatus()

    def stop_server(self) -> None:
        self._run(self._astop_server())

    async def _astop_server(self) -> None:
        st = await server_mod.stop_server(self.cfg.llm.base_url)
        self.events.put({"type": "server_stopped", "state": st.state, "host": st.host,
                         "port": st.port, "models": st.models, "detail": st.detail})

    # ---------- voix ----------

    def stt_record_and_transcribe(self, seconds: float = 6.0) -> None:
        self._run(self._arecord_stt(seconds))

    async def _arecord_stt(self, seconds: float) -> None:
        from ..voice import VoiceError, record_wav, transcribe_file
        vcfg = self.cfg.voice
        try:
            self.events.put({"type": "stt_recording", "seconds": seconds})
            path = await record_wav(seconds, vcfg)
            text = await transcribe_file(vcfg, path, language="fr")
            self.events.put({"type": "stt_done", "text": text})
        except VoiceError as exc:
            self.events.put({"type": "voice_error", "message": str(exc)})

    def tts_speak(self, text: str) -> None:
        self._run(self._atts(text))

    async def _atts(self, text: str) -> None:
        import tempfile
        from pathlib import Path as _P
        from ..voice import VoiceError, play_wav, synthesize
        vcfg = self.cfg.voice
        try:
            wav = await synthesize(vcfg, text)
            out = _P(tempfile.gettempdir()) / f"jarvis-gtk-tts-{os.getpid()}.wav"
            out.write_bytes(wav)
            self.events.put({"type": "tts_ready", "path": str(out)})
            await play_wav(out, vcfg)
        except VoiceError as exc:
            self.events.put({"type": "voice_error", "message": str(exc)})

    # ---------- réglages ----------

    def apply_gpu(self, device: str) -> dict:
        """Persiste le choix GPU (gpu.json, appliqué aux serveurs locaux au
        prochain démarrage) et dans ui.json. Retourne le choix effectif."""
        env = env_for_device(device)
        data = save_gpu_choice(device, env)
        save_ui_settings({"gpu": device})
        return {"device": data.get("device", device), "env": env,
                "args": args_for_device(device)}

    def gpu_choice(self) -> dict:
        return load_gpu_choice()
