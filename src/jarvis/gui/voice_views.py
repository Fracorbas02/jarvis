"""Panneaux des modes voix (TTS / STT).

La phase voix n'est pas encore câblée côté backend : ces panneaux listent les
modèles détectés pour le mode, mettent en évidence le modèle choisi dans la
sidebar, et proposent la zone d'interaction (désactivée pour l'instant) qui
sera branchée quand le micro/TTS local arrivera.
"""

from __future__ import annotations

from nicegui import ui

from .models import MODES
from .state import GuiState

_EMPTY_HINT = {
    "tts": "Dépose un modèle de synthèse vocale (piper, kokoro, xtts, coqui…) "
           "en .gguf dans ~/models (ou $JARVIS_MODEL_DIRS) : il apparaîtra ici.",
    "stt": "Dépose un modèle de reconnaissance vocale (whisper, faster-whisper…) "
           "en .gguf dans ~/models (ou $JARVIS_MODEL_DIRS) : il apparaîtra ici.",
}


def build_voice_panel(state: GuiState, mode: str) -> ui.element:
    meta = MODES[mode]
    models = state.models_by_mode.get(mode, [])
    selected = state.selected.get(mode)

    with ui.column().classes("w-full h-full items-center justify-center gap-4 overflow-y-auto py-6") as root:
        ui.icon(meta["icon"], size="56px").classes("text-[#4cc2ff] opacity-80")
        ui.label(meta["label"]).classes("text-2xl font-bold")
        ui.label(meta["description"]).classes("text-gray-400 text-sm")

        # --- modèles détectés pour ce mode ---
        with ui.card().classes("w-full max-w-xl bg-[#14171d] rounded-2xl p-4"):
            ui.label(f"Modèles détectés ({len(models)})").classes("text-sm font-semibold mb-2")
            if not models:
                ui.label(f"Aucun modèle {mode.upper()} détecté.").classes("text-sm text-gray-300")
                ui.label(_EMPTY_HINT[mode]).classes("text-xs text-gray-500 mt-1")
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

        # --- zone d'interaction (phase voix à venir) ---
        with ui.card().classes("w-full max-w-xl bg-[#14171d] rounded-2xl p-4"):
            ui.label("Interaction").classes("text-sm font-semibold mb-2")
            if mode == "tts":
                ui.input(placeholder="Texte à synthétiser…").classes("w-full").props(
                    "outlined dense rounded").disable()
                ui.button("Synthétiser", icon="volume_up").props(
                    "unelevated color=primary no-caps mt-2").classes("rounded-lg").disable()
            else:
                ui.button("Parler", icon="mic").props(
                    "unelevated color=primary no-caps round").classes("mt-1").disable()
            ui.label("Phase voix : le micro permanent (VAD → STT) et le TTS local seront "
                     "branchés ici. Les modèles détectés ci-dessus sont prêts.").classes(
                "text-xs text-gray-500 mt-2")
    return root
