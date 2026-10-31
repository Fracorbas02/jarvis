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
from ..models import discover_models
from .. import server as server_mod
from ..llm import LLMClient
from ..mcp import McpToolProvider
from ..sessions import SessionStore

log = logging.getLogger("jarvis.gtk.backend")

APP_ID = "io.github.Fracorbas02.Jarvis"

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
    path = ui_settings_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)  # écriture atomique
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


def _wav_rms(path) -> float | None:
    """Niveau RMS d'un WAV 16-bit mono (diagnostic : 0 = silence, micro muet)."""
    import math
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                return None
            frames = w.readframes(w.getnframes())
        if not frames:
            return 0.0
        n = len(frames) // 2
        total = 0
        for i in range(0, min(len(frames), 16000 * 2 * 10), 2):
            total += int.from_bytes(frames[i:i + 2], "little", signed=True) ** 2
        return math.sqrt(total / max(1, min(n, 16000 * 10)))
    except Exception:  # noqa: BLE001
        return None


class GtkBackend:
    """C\u0153ur async de l'UI GTK : config + LLM + MCP + Agent + sessions."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.events: queue.Queue = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="jarvis-async",
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
        self._apply_persisted_tts_api()

    def _apply_persisted_tts_api(self) -> None:
        """Réapplique les réglages TTS API persistants (ui.json) sur la config :
        enabled / model / voice_id survivent à l'arrêt de l'app."""
        from dataclasses import replace
        s = load_ui_settings()
        tts_api = self.cfg.tts_api
        if "tts_api_enabled" in s:
            tts_api = replace(tts_api, enabled=bool(s["tts_api_enabled"]))
        if s.get("tts_api_model"):
            tts_api = replace(tts_api, model=str(s["tts_api_model"]))
        if "tts_api_voice" in s:
            tts_api = replace(tts_api, voice_id=str(s["tts_api_voice"]))
        self.cfg.tts_api = tts_api

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

    def stt_record_and_transcribe(self, seconds: float = 6.0,
                                  device: str | None = None) -> None:
        self._run(self._arecord_stt(seconds, device))

    def _record_cfg(self, device: str | None):
        """VoiceConfig cloné avec un record_cmd pointant sur le micro choisi.

        - device None/"default" -> record_cmd d'origine (arecord, fallback ffmpeg) ;
        - device commençant par "pulse:" -> source PipeWire explicite, enregistrée
          via ffmpeg -f pulse -i <source> (arecord absent sur les systèmes
          100 % PipeWire/PulseAudio sans alsa-utils) ;
        - sinon device ALSA -> arecord -D <device>."""
        from dataclasses import replace
        if device and device != "default":
            if device.startswith("pulse:"):
                source = device[len("pulse:"):]
                return replace(self.cfg.voice, record_cmd=[
                    "ffmpeg", "-loglevel", "error", "-f", "pulse", "-i", source,
                    "-ar", "16000", "-ac", "1", "-f", "s16le", "-"])
            return replace(self.cfg.voice, record_cmd=[
                "arecord", "-q", "-D", device,
                "-f", "S16_LE", "-r", "16000", "-c", "1", "-"])
        return self.cfg.voice

    async def _arecord_stt(self, seconds: float, device: str | None) -> None:
        from ..voice import VoiceError, record_wav, transcribe_file
        vcfg = self._record_cfg(device)
        try:
            self.events.put({"type": "stt_recording", "seconds": seconds})
            path = await record_wav(seconds, vcfg)
            text = await transcribe_file(self.cfg.voice, path, language="fr")
            self.events.put({"type": "stt_done", "text": text})
        except VoiceError as exc:
            self.events.put({"type": "voice_error", "message": str(exc)})

    def refresh_stt_status(self) -> None:
        """Statut du serveur STT -> événement stt_server_status."""
        self._run(self._astt_status())

    def _stt_status_event(self, st) -> dict:
        from ..voice import _find_mmproj, resolve_model_path
        mmproj = None
        if (self.cfg.voice.stt.protocol or "whisper") == "voxtral":
            try:
                mmproj = _find_mmproj(resolve_model_path(self.cfg.voice.stt.model))
            except Exception:  # noqa: BLE001
                mmproj = None
            if mmproj is None and "--mmproj" in (self.cfg.voice.stt.extra_args or []):
                i = self.cfg.voice.stt.extra_args.index("--mmproj")
                if i + 1 < len(self.cfg.voice.stt.extra_args):
                    mmproj = self.cfg.voice.stt.extra_args[i + 1]
        return {"type": "stt_server_status", "state": st.state, "host": st.host,
                "port": st.port, "detail": st.detail, "healthy": st.healthy,
                "protocol": self.cfg.voice.stt.protocol,
                "model": self.cfg.voice.stt.model, "binary": self.cfg.voice.stt.binary,
                "mmproj": str(mmproj) if mmproj else ""}

    async def _astt_status(self) -> None:
        from ..voice import voice_server_status
        st = await voice_server_status("stt", self.cfg.voice.stt)
        self.events.put(self._stt_status_event(st))

    def start_stt_server(self) -> None:
        self._run(self._astt_start())

    async def _astt_start(self) -> None:
        from ..voice import start_voice_server
        st = await start_voice_server("stt", self.cfg.voice.stt)
        self.events.put(self._stt_status_event(st))

    def stop_stt_server(self) -> None:
        self._run(self._astt_stop())

    async def _astt_stop(self) -> None:
        from ..voice import stop_voice_server
        st = await stop_voice_server("stt", self.cfg.voice.stt)
        self.events.put(self._stt_status_event(st))

    def stt_test_transcribe(self, seconds: float = 6.0,
                            device: str | None = None) -> None:
        """Phase de test STT : enregistre puis transcrit, SANS lancer de tour
        LLM. Événements : stt_test_recording -> stt_test_done / voice_error."""
        self._run(self._astt_test(seconds, device))

    async def _astt_test(self, seconds: float, device: str | None) -> None:
        import time
        from ..voice import VoiceError, ensure_voice_server, record_wav, STTClient
        vcfg = self._record_cfg(device)
        try:
            st = await ensure_voice_server("stt", self.cfg.voice.stt)
            if st.state == "offline":
                raise VoiceError(f"serveur STT indisponible : {st.detail}")
            self.events.put({"type": "stt_test_recording", "seconds": seconds})
            path = await record_wav(seconds, vcfg)
            audio_rms = _wav_rms(path)
            t0 = time.monotonic()
            client = STTClient(self.cfg.voice.stt.base_url,
                               protocol=self.cfg.voice.stt.protocol)
            try:
                text = await client.transcribe(path, language="fr")
            finally:
                await client.aclose()
            self.events.put({"type": "stt_test_done", "text": text,
                             "seconds": seconds,
                             "elapsed": time.monotonic() - t0,
                             "protocol": self.cfg.voice.stt.protocol,
                             "server": f"{st.host}:{st.port}",
                             "audio_rms": audio_rms,
                             "wav_path": str(path),
                             "wav_bytes": path.stat().st_size if path.is_file() else 0})
        except VoiceError as exc:
            self.events.put({"type": "voice_error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 — JAMAIS laisser la coroutine mourir en silence
            log.exception("stt_test en erreur")
            self.events.put({"type": "voice_error",
                             "message": f"erreur interne STT : {type(exc).__name__}: {exc}"})

    def test_mic(self, device: str | None = None) -> None:
        """Enregistrement de test (0.5 s) pour v\u00e9rifier l'autorisation micro."""
        self._run(self._atest_mic(device))

    async def _atest_mic(self, device: str | None) -> None:
        from ..voice import VoiceError, record_wav
        vcfg = self._record_cfg(device)
        try:
            await record_wav(0.5, vcfg)
            self.events.put({"type": "mic_test", "ok": True, "message": ""})
        except VoiceError as exc:
            self.events.put({"type": "mic_test", "ok": False, "message": str(exc)})

    def check_mic_permission(self) -> None:
        """Interroge le portail XDG (PermissionStore, table 'microphone') pour
        savoir si NOTRE app_id a l'autorisation micro : événement
        {"type": "mic_permission", "state": "yes"|"no"|"unknown"}.
        Best-effort via gdbus (pas de dep D-Bus) : 'unknown' si indisponible."""
        self._run(self._acheck_perm())

    async def _acheck_perm(self) -> None:
        import re
        import subprocess
        state = "unknown"
        try:
            out = subprocess.run(
                ["gdbus", "call", "--session",
                 "--dest", "org.freedesktop.portal.PermissionStore",
                 "--object-path", "/org/freedesktop/portal/permission_store",
                 "--method", "org.freedesktop.portal.PermissionStore.Lookup",
                 "microphone", "microphone"],
                capture_output=True, text=True, timeout=5).stdout
            # (<{'autre.app': <['yes']>, 'io.github.Fracorbas02.Jarvis': <['no']>},)
            # -> on ancre sur NOTRE app-id, pas sur le 1er 'yes'/'no' venu
            m = re.search(rf"['\"]{re.escape(APP_ID)}['\"]:\s*<\[['\"](yes|no)['\"]", out)
            if m:
                state = m.group(1)
        except Exception as exc:  # noqa: BLE001 — portail absent (pas GNOME/Flatpak)
            log.debug("portail permission indisponible: %s", exc)
        self.events.put({"type": "mic_permission", "state": state})

    def tts_speak(self, text: str) -> None:
        self._run(self._atts(text))

    async def _atts(self, text: str) -> None:
        """Lit `text` à voix haute.

        Source : API Mistral si [voice.tts_api] enabled ET clé enregistrée,
        sinon serveur TTS local. Vrai fallback : si l'API est choisie mais
        échoue (réseau, 401, 403, 404…), on retombe sur le TTS local au lieu
        de ne rien lire."""
        import tempfile
        from pathlib import Path as _P
        from ..voice import play_wav
        vcfg = self.cfg.voice
        tts_api = getattr(self.cfg, "tts_api", None)
        audio: bytes | None = None
        ext = "wav"
        api_error: str | None = None
        if tts_api is not None and tts_api.enabled:
            try:
                audio, ext = await self._atts_api(text)
            except Exception as exc:  # noqa: BLE001 — MistralTTSError + réseau
                api_error = str(exc)
                log.warning("TTS API Mistral en échec, fallback local : %s", exc)
        if audio is None:
            # pas d'API choisie, pas de clé, ou API en échec -> TTS local
            from ..voice import synthesize
            try:
                audio = await synthesize(vcfg, text)
                ext = "wav"
            except Exception as exc:  # noqa: BLE001 — VoiceError
                detail = f" (API: {api_error})" if api_error else ""
                self.events.put({"type": "voice_error",
                                 "message": f"TTS local en échec aussi : {exc}{detail}"})
                return
            if api_error:
                self.events.put({"type": "tts_fallback",
                                 "message": f"API Mistral en échec ({api_error}) "
                                            f"— lu par le TTS local"})
        try:
            out = _P(tempfile.gettempdir()) / f"jarvis-tts-{os.getpid()}.{ext}"
            out.write_bytes(audio)
            self.events.put({"type": "tts_ready", "path": str(out)})
            await play_wav(out, vcfg)
        except Exception as exc:  # noqa: BLE001
            self.events.put({"type": "voice_error", "message": str(exc)})

    async def _atts_api(self, text: str) -> tuple[bytes, str]:
        """TTS via l'API Mistral. Retourne (octets audio, extension)."""
        from ..mistral_tts import (MistralTTSClient, MistralTTSError,
                                    load_api_key)
        tts_api = self.cfg.tts_api
        key = load_api_key()
        if not key:
            raise MistralTTSError(
                "clé API Mistral absente — Réglages > Voix API Mistral, "
                "ou export MISTRAL_API_KEY")
        client = MistralTTSClient(key, base_url=tts_api.base_url,
                                  model=tts_api.model)
        try:
            audio = await client.synthesize(
                text, voice_id=tts_api.voice_id or None,
                response_format=tts_api.response_format or "mp3")
            return audio, (tts_api.response_format or "mp3")
        finally:
            await client.aclose()

    # ---------- réglages voix API Mistral ----------

    def set_tts_api(self, enabled: bool | None = None, model: str | None = None,
                    voice_id: str | None = None) -> None:
        """Met à jour les réglages TTS API persistants (ui.json) et les applique
        à la config en mémoire. La clé reste dans mistral.json (séparée)."""
        from dataclasses import replace
        patch: dict = {}
        tts_api = self.cfg.tts_api
        if enabled is not None:
            tts_api = replace(tts_api, enabled=enabled)
            patch["tts_api_enabled"] = enabled
        if model:
            tts_api = replace(tts_api, model=model)
            patch["tts_api_model"] = model
        if voice_id is not None:
            tts_api = replace(tts_api, voice_id=voice_id)
            patch["tts_api_voice"] = voice_id
        self.cfg.tts_api = tts_api
        if patch:
            save_ui_settings(patch)

    def save_mistral_api_key(self, api_key: str) -> bool:
        """Enregistre la clé API localement (mistral.json, 0600). Vide = supprime."""
        from ..mistral_tts import delete_api_key, save_api_key
        key = api_key.strip()
        if not key:
            delete_api_key()
            return False
        save_api_key(key)
        return True

    def mistral_api_key_status(self) -> dict:
        from ..mistral_tts import has_api_key, load_api_key
        return {"has_key": has_api_key(),
                "source": "env" if os.environ.get("MISTRAL_API_KEY") else "file",
                "key_len": len(load_api_key())}

    def list_mistral_voices(self) -> None:
        """GET /v1/audio/voices -> événement mistral_voices (preset + custom)."""
        self._run(self._alist_mistral_voices())

    async def _alist_mistral_voices(self) -> None:
        from ..mistral_tts import (MistralTTSClient, MistralTTSError,
                                    load_api_key)
        key = load_api_key()
        if not key:
            self.events.put({"type": "mistral_voices", "voices": [],
                             "error": "clé API absente (Réglages > Voix API)"})
            return
        tts_api = self.cfg.tts_api
        client = MistralTTSClient(key, base_url=tts_api.base_url, model=tts_api.model,
                                  timeout=20.0)
        try:
            voices = await client.list_voices()
            self.events.put({"type": "mistral_voices",
                             "voices": [v.__dict__ for v in voices], "error": ""})
        except MistralTTSError as exc:
            self.events.put({"type": "mistral_voices", "voices": [],
                             "error": str(exc)})
        finally:
            await client.aclose()

    def test_mistral_tts(self, text: str = "Bonjour, ceci est un test de la voix Mistral.") -> None:
        """Synthétise un extrait via l'API et le joue (teste clé + voix + réseau)."""
        self._run(self._atest_mistral_tts(text))

    async def _atest_mistral_tts(self, text: str) -> None:
        import tempfile
        from pathlib import Path as _P
        from ..voice import play_wav
        try:
            audio, ext = await self._atts_api(text)
            out = _P(tempfile.gettempdir()) / f"jarvis-tts-test.{ext}"
            out.write_bytes(audio)
            await play_wav(out, self.cfg.voice)
            self.events.put({"type": "mistral_tts_test", "ok": True,
                             "message": f"{len(audio) // 1024} Ko ({ext})"})
        except Exception as exc:  # noqa: BLE001
            self.events.put({"type": "mistral_tts_test", "ok": False,
                             "message": str(exc)})

    # ---------- réglages ----------

    def list_input_devices(self) -> None:
        """Liste les cartes son d'entr\u00e9e (arecord -L) -> \u00e9v\u00e9nement mic_devices."""
        self._run(self._alist_mic())

    @staticmethod
    def _pactl_sources() -> list[tuple[str, str]]:
        """Sources d'entrée PipeWire/PulseAudio via `pactl list short sources`
        (pour les systèmes sans alsa-utils/arecord) : [(pulse:<source>, label)].
        On exclut les .monitor (ce sont des sorties)."""
        import shutil
        import subprocess
        if not shutil.which("pactl"):
            return []
        out = subprocess.run(["pactl", "list", "short", "sources"],
                             capture_output=True, text=True, timeout=10).stdout
        devs: list[tuple[str, str]] = []
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name, desc = parts[1], parts[2]
            if name.endswith(".monitor"):
                continue
            label = desc if desc and desc != name else name
            devs.append((f"pulse:{name}", f"{label}  ({name})"))
        return devs

    async def _alist_mic(self) -> None:
        import re
        import shutil
        import subprocess
        devices: list[tuple[str, str]] = [("default", "Micro par d\u00e9faut du syst\u00e8me")]
        error = ""
        if shutil.which("arecord"):
            try:
                out = subprocess.run(["arecord", "-L"], capture_output=True, text=True,
                                     timeout=10).stdout
                cur = None
                for line in out.splitlines():
                    m = re.match(r"^(\S+)\s*$", line)
                    if m and not line.startswith((" ", "\t")):
                        cur = m.group(1)
                        devices.append((cur, cur))
                    elif cur and line.startswith((" ", "\t")):
                        desc = line.strip()
                        # premi\u00e8re ligne descriptive = nom de la carte (ex. "HDA Intel PCH, ...")
                        if devices[-1][1] == cur and desc and not desc.startswith(
                                ("Discard", "Direct", "Hardware", "Default", "Sysdefault")):
                            devices[-1] = (cur, f"{cur} \u2014 {desc}")
            except Exception as exc:  # noqa: BLE001
                error = f"arecord -L a \u00e9chou\u00e9 : {exc}"
        else:
            error = "arecord indisponible (alsa-utils non install\u00e9)"
        # sources PipeWire en compl\u00e9ment (utile sans arecord, et pour choisir
        # explicitement une source : pactl list short sources)
        try:
            devices += self._pactl_sources()
        except Exception as exc:  # noqa: BLE001
            error = (error + " ; " if error else "") + f"pactl indisponible : {exc}"
        if len(devices) == 1 and error:
            error += " \u2014 aucun micro d\u00e9tect\u00e9 (installe alsa-utils ou pipewire/pulseaudio)"
        self.events.put({"type": "mic_devices", "devices": devices, "error": error})

    def apply_gpu(self, device: str) -> dict:
        """Persiste le choix GPU (gpu.json, appliqué aux serveurs locaux au
        prochain démarrage) et dans ui.json. Retourne le choix effectif.

        Si le device choisi n'existe plus dans `llama-server --list-devices`
        (GPU débranché), on persiste le 1er device ggml disponible à la place."""
        from ..gpu import device_is_available, list_llama_devices
        if not device_is_available(device):
            devs = list_llama_devices()
            if devs:
                import logging
                logging.getLogger("jarvis.gtk.backend").warning(
                    "GPU %s indisponible, choix forcé sur %s", device, devs[0][0])
                device = devs[0][0]
        env = env_for_device(device)
        data = save_gpu_choice(device, env)
        save_ui_settings({"gpu": device})
        return {"device": data.get("device", device), "env": env,
                "args": args_for_device(device)}

    def gpu_choice(self) -> dict:
        return load_gpu_choice()
