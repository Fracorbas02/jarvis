"""Panneaux des modes voix (TTS / STT) de la GUI web.

Fonctionnels : chaque panneau pilote son serveur voix local (statut, start/
stop) et offre l'interaction :
- TTS : choix de la voix, saisie du texte, synthèse via llama-server
  (/v1/audio/speech, fallback /tts), lecture dans le navigateur (ui.audio)
  et téléchargement du WAV ;
- STT : upload d'un fichier audio (wav/mp3/webm…) ou enregistrement micro
  directement dans le navigateur (getUserMedia côté client, POST vers
  /api/voice/record), puis transcription via whisper-server.

Le démarrage d'un serveur voix charge un modèle (dizaines de secondes à
minutes) : il ne bloque JAMAIS la boucle NiceGUI — le subprocess est lancé
via ui.run.io_bound et un timer rafraîchit le statut jusqu'au /health. Les
erreurs (binaire absent, modèle introuvable, processus mort) remontent en
console (log), en notification et dans le chip de statut.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from nicegui import app, ui
from starlette.requests import Request

from ..config import VoiceConfig
from ..voice import (VoiceError, list_voices, start_voice_server,
                     stop_voice_server, synthesize, transcribe_bytes,
                     voice_server_status)
from .models import MODES
from .state import GuiState

log = logging.getLogger("jarvis.gui.voice")

_REC_JS = """
async function jarvisRecordAndTranscribe(token, outId, btnId) {
  const out = document.getElementById(outId);
  const btn = document.getElementById(btnId);
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    out.textContent = "getUserMedia non disponible (HTTPS ou localhost requis)";
    return;
  }
  if (window.__jarvisRec && window.__jarvisRec.active) {
    window.__jarvisRec.stop();
    btn.disabled = false;
    return;
  }
  btn.disabled = true;
  out.textContent = "enregistrement… (reclique sur le bouton pour arrêter)";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const rec = new MediaRecorder(stream);
    const chunks = [];
    rec.ondataavailable = e => chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      btn.disabled = false;
      out.textContent = "transcription en cours…";
      const blob = new Blob(chunks, {type: rec.mimeType || "audio/webm"});
      const fd = new FormData();
      fd.append("file", blob, "record.webm");
      try {
        const resp = await fetch("/api/voice/record", {method: "POST", body: fd});
        const data = await resp.json();
        out.textContent = data.ok ? ("« " + data.text + " »") : ("erreur : " + data.error);
      } catch (e) {
        out.textContent = "erreur réseau : " + e;
      }
    };
    window.__jarvisRec = {active: true, stop: () => { rec.stop(); window.__jarvisRec.active = false; }};
    rec.start();
  } catch (e) {
    btn.disabled = false;
    out.textContent = "micro refusé : " + e;
  }
}
"""


def voice_cfg(state: GuiState) -> VoiceConfig:
    if state.backend is not None:
        return state.backend.cfg.voice
    return VoiceConfig()


def register_voice_api(state: GuiState) -> None:
    """Route API appelée par le JS d'enregistrement micro du panneau STT."""
    ui.add_body_html(f"<script>{_REC_JS}</script>", shared=True)

    @app.post("/api/voice/record")
    async def record(request: Request) -> dict:
        form = await request.form()
        upload = form.get("file")
        data = await upload.read() if upload is not None else b""
        name = getattr(upload, "filename", "record.webm") or "record.webm"
        if not data:
            return {"ok": False, "error": "fichier vide"}
        try:
            text = await transcribe_bytes(voice_cfg(state), data, filename=name)
        except VoiceError as exc:
            log.warning("record STT en erreur: %s", exc)
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "text": text}


_STATUS_STYLE = {
    "online": ("● en ligne", "positive"),
    "managed": ("● démarrage…", "info"),
    "offline": ("● hors ligne", "negative"),
}


def build_voice_panel(state: GuiState, mode: str) -> ui.element:
    voice_select: ui.select | None = None  # rempli uniquement en mode tts
    meta = MODES[mode]
    models = state.models_by_mode.get(mode, [])
    selected = state.selected.get(mode)
    vcfg = voice_cfg(state)
    server_cfg = vcfg.stt if mode == "stt" else vcfg.tts

    async def refresh_voices() -> None:
        if voice_select is None:
            return
        try:
            voices = await list_voices(vcfg)
        except Exception as exc:  # noqa: BLE001 — statut best-effort
            log.debug("list_voices: %s", exc)
            voices = []
        if voices:
            opts = {v: v for v in voices}
            voice_select.options = opts
            if voice_select.value not in opts:
                voice_select.value = voices[0]
        else:
            voice_select.options = {}
            voice_select.value = None

    with ui.column().classes("w-full h-full items-center justify-center gap-4 overflow-y-auto py-6") as root:
        ui.icon(meta["icon"], size="56px").classes("text-[#4cc2ff] opacity-80")
        ui.label(meta["label"]).classes("text-2xl font-bold")
        ui.label(meta["description"]).classes("text-gray-400 text-sm")

        # --- serveur voix : statut + start/stop ---
        with ui.card().classes("w-full max-w-xl bg-[#14171d] rounded-2xl p-4"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Serveur").classes("text-sm font-semibold grow")
                status_chip = ui.chip("● …", color="grey", text_color="white").props("dense")
                addr_label = ui.label("").classes("text-xs text-gray-500")
                btn_start = ui.button("Démarrer", icon="play_arrow")
                btn_start.props("dense unelevated color=primary no-caps").classes("rounded-lg text-xs")
                btn_stop = ui.button("Arrêter", icon="stop")
                btn_stop.props("dense unelevated color=negative no-caps").classes("rounded-lg text-xs")

            async def refresh_voice_status() -> None:
                st = await voice_server_status(mode, server_cfg)
                text, color = _STATUS_STYLE.get(st.state, ("● ?", "grey"))
                status_chip.text = text
                status_chip.props(f"color={color}")
                # l'erreur éventuelle (binaire absent, modèle introuvable, proc mort)
                # est visible directement dans le chip, pas enfouie
                if st.detail and st.state in ("offline", "managed"):
                    status_chip.text = f"{text} — {st.detail}"
                addr_label.text = f"{st.host}:{st.port}"
                btn_start.visible = st.state == "offline"
                btn_stop.visible = st.state == "managed"

            def on_start() -> None:
                model = state.selected.get(mode)
                path = model.path if model is not None else None
                if model is None:
                    ui.notify(f"Choisis d'abord un modèle {mode.upper()} dans la sidebar",
                              type="warning")
                    log.warning("start %s: aucun modèle sélectionné", mode)
                    return
                ui.notify(f"Démarrage du serveur {mode.upper()} avec {model.name}… "
                          f"(chargement du modèle, voir ~/.cache/jarvis/voice-{mode}.log)",
                          type="info")
                log.info("start serveur %s: modèle=%s cfg=%s", mode, path, server_cfg)
                btn_start.disable()

                async def _do_start() -> None:
                    st = await start_voice_server(mode, server_cfg, model_path=path)
                    log.info("start serveur %s -> %s (%s)", mode, st.state, st.detail)
                    if st.state == "offline":
                        ui.notify(f"Échec démarrage {mode.upper()} : {st.detail}",
                                  type="negative", timeout=15)
                    elif st.state == "managed":
                        ui.notify(f"Serveur {mode.upper()} en cours de chargement…",
                                  type="info")
                    else:
                        ui.notify(f"Serveur {mode.upper()} prêt", type="positive")
                    await refresh_voice_status()
                    btn_start.enable()
                    if mode == "tts":
                        await refresh_voices()

                # le chargement du modèle prend des dizaines de secondes :
                # on lance la coroutine via un timer (tâche serveur dédiée),
                # jamais en await direct dans le handler (ça figerait le client)
                ui.timer(0.1, _do_start, once=True)

            def on_stop() -> None:
                ui.timer(0.1, _stop_and_refresh, once=True)

            async def _stop_and_refresh() -> None:
                st = await stop_voice_server(mode, server_cfg)
                log.info("stop serveur %s -> %s", mode, st.state)
                await refresh_voice_status()
                ui.notify(f"Serveur {mode.upper()} arrêté", type="info")

            btn_start.on_click(on_start)
            btn_stop.on_click(on_stop)
            # statut : une fois vite, puis en boucle (le chargement peut durer longtemps)
            ui.timer(0.3, refresh_voice_status, once=True)
            ui.timer(3.0, refresh_voice_status)

        # --- modèles détectés pour ce mode ---
        with ui.card().classes("w-full max-w-xl bg-[#14171d] rounded-2xl p-4"):
            ui.label(f"Modèles détectés ({len(models)})").classes("text-sm font-semibold mb-2")
            if not models:
                hint = ("Dépose un modèle whisper (.gguf) dans ~/models/stt/ (ou $JARVIS_MODEL_DIRS) : "
                        "il apparaîtra ici." if mode == "stt" else
                        "Dépose un modèle de synthèse vocale (.gguf, ex. voxtral-4b-tts) dans "
                        "~/models/tts/ : il apparaîtra ici.")
                ui.label(f"Aucun modèle {mode.upper()} détecté.").classes("text-sm text-gray-300")
                ui.label(hint).classes("text-xs text-gray-500 mt-1")
            else:
                with ui.column().classes("w-full gap-1"):
                    for m in models:
                        is_sel = selected is not None and m.path == selected.path
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.icon("check_circle" if is_sel else "radio_button_unchecked",
                                    size="18px").classes(
                                "text-[#4cc2ff]" if is_sel else "text-gray-600")
                            ui.label(m.name).classes("text-sm grow")
                            ui.label(m.size_human).classes("text-xs text-gray-500 mono")
                        ui.label(str(m.path)).classes("text-xs text-gray-600 pl-6 -mt-1")

        # --- zone d'interaction ---
        with ui.card().classes("w-full max-w-xl bg-[#14171d] rounded-2xl p-4"):
            ui.label("Interaction").classes("text-sm font-semibold mb-2")
            if mode == "tts":
                voice_select = _build_tts_interaction(state, vcfg)
                # voix exposées par le serveur TTS (vide tant qu'il est hors ligne)
                ui.timer(1.0, refresh_voices, once=True)
                ui.timer(10.0, refresh_voices)
            else:
                _build_stt_interaction(state, vcfg)

    return root


def _build_tts_interaction(state: GuiState, vcfg: VoiceConfig) -> ui.select:
    voice_select = ui.select(options={}, label="Voix").classes("w-full").props("dense outlined rounded")
    text_input = ui.input(placeholder="Texte à synthétiser…").classes("w-full").props(
        "outlined dense rounded type=textarea").style("min-height: 80px")
    btn_synth = ui.button("Synthétiser", icon="volume_up").props(
        "unelevated color=primary no-caps mt-2").classes("rounded-lg")
    out_box = ui.column().classes("w-full mt-3 gap-2")

    async def on_synthesize() -> None:
        text = (text_input.value or "").strip()
        if not text:
            ui.notify("Saisis un texte", type="warning")
            return
        btn_synth.disable()
        out_box.clear()
        with out_box:
            ui.spinner(size="sm")
            ui.label("synthèse en cours…").classes("text-xs text-gray-400")
        try:
            wav = await synthesize(vcfg, text, voice=voice_select.value or None)
        except VoiceError as exc:
            log.warning("synthèse TTS en erreur: %s", exc)
            ui.notify(str(exc), type="negative", timeout=15)
            out_box.clear()
            btn_synth.enable()
            return
        out_box.clear()
        with out_box:
            tmp = Path(tempfile.gettempdir()) / f"jarvis-tts-{uuid.uuid4().hex[:8]}.wav"
            tmp.write_bytes(wav)
            url = app.add_media_file(local_file=str(tmp))
            ui.audio(url).classes("w-full")
            ui.button("Télécharger le WAV", icon="download",
                      on_click=lambda: ui.download(wav, "jarvis-tts.wav")).props(
                "flat dense no-caps").classes("rounded-lg text-xs")
        btn_synth.enable()

    btn_synth.on_click(on_synthesize)
    return voice_select


def _build_stt_interaction(state: GuiState, vcfg: VoiceConfig) -> None:
    token = uuid.uuid4().hex

    async def on_upload(e) -> None:
        data = await e.file.read()
        out_label.text = "transcription en cours…"
        try:
            text = await transcribe_bytes(vcfg, data, filename=e.file.name)
        except VoiceError as exc:
            log.warning("upload STT en erreur: %s", exc)
            out_label.text = f"erreur : {exc}"
            ui.notify(str(exc), type="negative", timeout=15)
            return
        out_label.text = f"« {text} »" if text else "(aucune parole détectée)"

    out_label = ui.label("").classes("text-sm text-gray-200 mt-2")
    with ui.row().classes("w-full gap-2 items-center"):
        upload = ui.upload(label="Transcrire un fichier audio…", auto_upload=True,
                           on_upload=on_upload).props("outlined dense no-caps").classes(
            "rounded-lg text-xs flex-1")
        upload.props('accept="audio/*,.wav,.mp3,.webm,.ogg,.m4a"')
        rec_btn = ui.button("🎤 Parler", icon="mic").props(
            "unelevated color=primary no-caps round").tooltip(
            "Enregistre depuis le micro du navigateur (getUserMedia) et transcrit")
        rec_btn.props(f'id="jarvis-rec-{token}"')
        rec_btn.on_click(lambda: ui.run_javascript(
            f"jarvisRecordAndTranscribe('{token}', 'jarvis-rec-out-{token}', 'jarvis-rec-{token}')"))

    with ui.row().classes("w-full items-center gap-2"):
        ui.html(f'<div id="jarvis-rec-out-{token}" class="text-sm text-gray-200"></div>')
