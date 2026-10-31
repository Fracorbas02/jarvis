"""Pont thread Tk <-> boucle asyncio du backend (LLM + MCP).

Toute la logique async (GuiBackend, statut llama-server) tourne dans une
boucle asyncio dédiée sur un thread daemon. Le thread Tk ne touche jamais
aux objets async : il dépose des coroutines via run_coroutine_threadsafe et
reçoit les événements via une queue.Queue (thread-safe), dépouillée avec
root.after().

Événements émis vers le thread Tk (dicts) :
  {"type": "backend_ready", "tools": [...]}
  {"type": "backend_error", "message": ...}
  {"type": "server_status", "state": ..., "host": ..., "port": ..., "models": [...]}
  {"type": "server_started", ...idem...} / {"type": "server_stopped", ...}
  + les événements du puits de stream_turn (reasoning/content/tool_call/done/error)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import tempfile
import threading
from pathlib import Path

from ..gui import server as server_mod
from ..gui.backend import GuiBackend
from ..gui.models import discover_models
from ..voice import (VoiceError, play_wav, record_wav, start_voice_server,
                     stop_voice_server, synthesize, transcribe_bytes,
                     voice_server_status)

log = logging.getLogger("jarvis.desktop.controller")


def default_repo_root() -> Path:
    # src/jarvis/desktop/controller.py -> racine du dépôt
    return Path(__file__).resolve().parents[3]


class DesktopController:
    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.events: queue.Queue = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="jarvis-async",
                                        daemon=True)
        self.backend: GuiBackend | None = None
        self.models_by_mode = discover_models()
        self.mode = "llm"
        self.selected: dict[str, object] = {}

    # ---------- cycle de vie ----------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> None:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._astart(), self.loop)

    async def _astart(self) -> None:
        try:
            self.backend = GuiBackend(self.config_path)
            await self.backend.start()
            self.events.put({"type": "backend_ready",
                             "tools": self.backend.tool_names()})
        except Exception as exc:  # noqa: BLE001 — remonter à l'UI
            log.exception("démarrage backend desktop")
            self.events.put({"type": "backend_error", "message": str(exc)})

    def stop(self) -> None:
        async def _astop() -> None:
            if self.backend is not None:
                await self.backend.stop()
        try:
            asyncio.run_coroutine_threadsafe(_astop(), self.loop).result(timeout=15)
        except Exception as exc:  # noqa: BLE001 — arrêt best-effort
            log.debug("arrêt backend: %s", exc)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)

    # ---------- appels depuis le thread Tk ----------

    def _run(self, coro) -> None:
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stream_turn(self, text: str) -> None:
        if self.backend is None:
            return

        def sink(event: dict) -> None:
            # appelé dans le thread de la boucle : la queue est thread-safe
            self.events.put(event)

        self._run(self.backend.stream_turn(text, sink))

    def reset(self) -> None:
        if self.backend is not None:
            self._run(self._areset())

    async def _areset(self) -> None:
        self.backend.reset()

    def refresh_status(self) -> None:
        if self.backend is None:
            return
        self._run(self._astatus())

    async def _astatus(self) -> None:
        st = await server_mod.get_status(self.backend.cfg.llm.base_url)
        self.events.put({"type": "server_status", "state": st.state, "host": st.host,
                         "port": st.port, "models": st.models, "detail": st.detail})

    def start_server(self, model_path: str) -> None:
        if self.backend is None:
            return
        self._run(self._astart_server(model_path))

    async def _astart_server(self, model_path: str) -> None:
        st = await server_mod.start_server(model_path, self.backend.cfg.llm.base_url)
        self.events.put({"type": "server_started", "state": st.state, "host": st.host,
                         "port": st.port, "models": st.models, "detail": st.detail})

    def stop_server(self) -> None:
        if self.backend is None:
            return
        self._run(self._astop_server())

    async def _astop_server(self) -> None:
        st = await server_mod.stop_server(self.backend.cfg.llm.base_url)
        self.events.put({"type": "server_stopped", "state": st.state, "host": st.host,
                         "port": st.port, "models": st.models, "detail": st.detail})

    # ---------- voix (STT / TTS) ----------

    def refresh_voice_status(self, mode: str) -> None:
        if self.backend is None:
            return
        self._run(self._avoice_status(mode))

    async def _avoice_status(self, mode: str) -> None:
        vcfg = self.backend.cfg.voice
        scfg = vcfg.stt if mode == "stt" else vcfg.tts
        st = await voice_server_status(mode, scfg)
        self.events.put({"type": "voice_status", "mode": mode, "state": st.state,
                         "host": st.host, "port": st.port, "detail": st.detail})

    def start_voice_server(self, mode: str, model_path: str | None = None) -> None:
        if self.backend is None:
            return
        self._run(self._astart_voice(mode, model_path))

    async def _astart_voice(self, mode: str, model_path: str | None) -> None:
        vcfg = self.backend.cfg.voice
        scfg = vcfg.stt if mode == "stt" else vcfg.tts
        st = await start_voice_server(mode, scfg, model_path=model_path)
        self.events.put({"type": "voice_status", "mode": mode, "state": st.state,
                         "host": st.host, "port": st.port, "detail": st.detail})

    def stop_voice_server(self, mode: str) -> None:
        if self.backend is None:
            return
        self._run(self._astop_voice(mode))

    async def _astop_voice(self, mode: str) -> None:
        vcfg = self.backend.cfg.voice
        scfg = vcfg.stt if mode == "stt" else vcfg.tts
        st = await stop_voice_server(mode, scfg)
        self.events.put({"type": "voice_status", "mode": mode, "state": st.state,
                         "host": st.host, "port": st.port, "detail": st.detail})

    def tts_synthesize(self, mode: str, text: str, voice: str | None) -> None:
        if self.backend is None:
            return
        self._run(self._atts(mode, text, voice))

    async def _atts(self, mode: str, text: str, voice: str | None) -> None:
        vcfg = self.backend.cfg.voice
        try:
            wav = await synthesize(vcfg, text, voice=voice)
            out = Path(tempfile.gettempdir()) / f"jarvis-tts-{os.getpid()}.wav"
            out.write_bytes(wav)
            self.events.put({"type": "tts_done", "path": str(out),
                             "bytes": len(wav)})
        except VoiceError as exc:
            self.events.put({"type": "tts_error", "message": str(exc)})

    def stt_transcribe_bytes(self, mode: str, data: bytes, filename: str) -> None:
        if self.backend is None:
            return
        self._run(self._astt_bytes(mode, data, filename))

    async def _astt_bytes(self, mode: str, data: bytes, filename: str) -> None:
        vcfg = self.backend.cfg.voice
        try:
            text = await transcribe_bytes(vcfg, data, filename=filename)
            self.events.put({"type": "stt_done", "text": text})
        except VoiceError as exc:
            self.events.put({"type": "stt_error", "message": str(exc)})

    def stt_transcribe_file(self, path: str) -> None:
        if self.backend is None:
            return
        self._run(self._astt_file(path))

    async def _astt_file(self, path: str) -> None:
        from ..voice import transcribe_file
        vcfg = self.backend.cfg.voice
        try:
            text = await transcribe_file(vcfg, path, language="fr")
            self.events.put({"type": "stt_done", "text": text})
        except VoiceError as exc:
            self.events.put({"type": "stt_error", "message": str(exc)})

    def stt_record(self, seconds: float = 5.0) -> None:
        if self.backend is None:
            return
        self._run(self._arecord(seconds))

    async def _arecord(self, seconds: float) -> None:
        vcfg = self.backend.cfg.voice
        try:
            path = await record_wav(seconds, vcfg)
            self.events.put({"type": "stt_recorded", "path": str(path)})
            await self._astt_file(str(path))
        except VoiceError as exc:
            self.events.put({"type": "stt_error", "message": str(exc)})

    def play_wav_file(self, path: str) -> None:
        if self.backend is None:
            return
        self._run(self._aplay(path))

    async def _aplay(self, path: str) -> None:
        vcfg = self.backend.cfg.voice
        try:
            player = await play_wav(path, vcfg)
            self.events.put({"type": "wav_played", "player": player, "path": path})
        except VoiceError as exc:
            self.events.put({"type": "wav_error", "message": str(exc)})
