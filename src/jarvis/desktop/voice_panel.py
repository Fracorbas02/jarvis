"""Panneaux des modes voix (TTS / STT) de l'application desktop.

Phase voix pas encore câblée côté backend : le panneau liste les modèles
détectés pour le mode (sélection surlignée) et affiche la zone
d'interaction désactivée qui sera branchée avec le micro/TTS local.
"""

from __future__ import annotations

import customtkinter as ctk

from ..gui.models import MODES
from .controller import DesktopController
from .theme import ACCENT, BG, PANEL_2, TEXT_DIM

_EMPTY_HINT = {
    "tts": "Dépose un modèle de synthèse vocale (piper, kokoro, xtts, coqui…) "
           "en .gguf dans ~/models (ou $JARVIS_MODEL_DIRS) : il apparaîtra ici.",
    "stt": "Dépose un modèle de reconnaissance vocale (whisper, faster-whisper…) "
           "en .gguf dans ~/models (ou $JARVIS_MODEL_DIRS) : il apparaîtra ici.",
}


class VoicePanel(ctk.CTkFrame):
    def __init__(self, master, ctrl: DesktopController, mode: str):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.ctrl = ctrl
        self.mode = mode
        meta = MODES[mode]

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=meta["label"], font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=ACCENT).grid(row=0, column=0, padx=20, pady=(16, 2),
                                             sticky="w")
        ctk.CTkLabel(self, text=meta["description"], font=ctk.CTkFont(size=12),
                     text_color=TEXT_DIM).grid(row=1, column=0, padx=20, sticky="w")

        box = ctk.CTkFrame(self, fg_color=PANEL_2, corner_radius=12)
        box.grid(row=2, column=0, padx=20, pady=16, sticky="nsew")
        box.grid_columnconfigure(0, weight=1)

        models = ctrl.models_by_mode.get(mode, [])
        selected = ctrl.selected.get(mode)
        ctk.CTkLabel(box, text=f"Modèles détectés ({len(models)})",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(12, 6), sticky="w")
        r = 1
        if not models:
            ctk.CTkLabel(box, text=f"Aucun modèle {mode.upper()} détecté.",
                         font=ctk.CTkFont(size=12)).grid(row=r, column=0, padx=14,
                                                          sticky="w")
            r += 1
            ctk.CTkLabel(box, text=_EMPTY_HINT[mode], font=ctk.CTkFont(size=11),
                         text_color=TEXT_DIM, wraplength=520,
                         justify="left").grid(row=r, column=0, padx=14, pady=(2, 12),
                                             sticky="w")
        else:
            for m in models:
                is_sel = selected is not None and str(m.path) == str(selected)
                row = ctk.CTkFrame(box, fg_color="transparent")
                row.grid(row=r, column=0, padx=14, pady=2, sticky="ew")
                ctk.CTkLabel(row, text="●" if is_sel else "○",
                             text_color=ACCENT if is_sel else TEXT_DIM,
                             font=ctk.CTkFont(size=12)).pack(side="left")
                ctk.CTkLabel(row, text=m.name, font=ctk.CTkFont(size=12)).pack(
                    side="left", padx=(6, 0))
                ctk.CTkLabel(row, text=m.size_human, font=ctk.CTkFont(size=11),
                             text_color=TEXT_DIM).pack(side="right")
                r += 1
            ctk.CTkLabel(box, text="sélection : voir le combo MODÈLE de la sidebar",
                         font=ctk.CTkFont(size=10), text_color=TEXT_DIM).grid(
                row=r, column=0, padx=14, pady=(4, 12), sticky="w")
            r += 1

        # zone d'interaction (phase voix à venir)
        r += 1
        ctk.CTkLabel(box, text="Interaction", font=ctk.CTkFont(size=13, weight="bold")
                     ).grid(row=r, column=0, padx=14, pady=(6, 4), sticky="w")
        r += 1
        if mode == "tts":
            ctk.CTkEntry(box, placeholder_text="Texte à synthétiser…",
                         state="disabled", height=32).grid(row=r, column=0, padx=14,
                                                          pady=2, sticky="ew")
            r += 1
            ctk.CTkButton(box, text="Synthétiser", state="disabled",
                          fg_color=ACCENT, text_color="#06121a").grid(
                row=r, column=0, padx=14, pady=(6, 4), sticky="w")
        else:
            ctk.CTkButton(box, text="● Parler", state="disabled",
                          fg_color=ACCENT, text_color="#06121a").grid(
                row=r, column=0, padx=14, pady=(6, 4), sticky="w")
        r += 1
        ctk.CTkLabel(box, text="Phase voix : le micro permanent (VAD → STT) et le TTS "
                               "local seront branchés ici.",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM,
                     wraplength=520, justify="left").grid(row=r, column=0, padx=14,
                                                          pady=(2, 14), sticky="w")
