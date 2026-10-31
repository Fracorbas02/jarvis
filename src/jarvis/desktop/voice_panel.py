"""Panneaux des modes voix (TTS / STT) de l'application desktop.

Fonctionnels : chaque panneau pilote son serveur voix local (statut, démarrer/
arrêter) et offre l'interaction — TTS : saisie + voix + synthèse + lecture ;
STT : enregistrement micro (arecord/ffmpeg) ou fichier .wav, transcription.
Les appels passent par le DesktopController (boucle asyncio dédiée) et les
résultats remontent via la queue d'événements.
"""

from __future__ import annotations

import customtkinter as ctk
from tkinter import filedialog

from ..gui.models import MODES
from .controller import DesktopController
from .theme import ACCENT, BG, PANEL_2, TEXT_DIM

_VOICE_STATUS_TEXT = {"online": "● en ligne", "managed": "● géré par l'app",
                      "offline": "● hors ligne"}
_VOICE_STATUS_COLOR = {"online": "#3fb950", "managed": "#4cc2ff", "offline": "#f85149"}

_EMPTY_HINT = {
    "tts": "Dépose un modèle de synthèse vocale (voxtral-4b-tts…) en .gguf dans "
           "~/models/tts/ : il apparaîtra ici.",
    "stt": "Dépose un modèle whisper en .gguf dans ~/models/stt/ : il apparaîtra ici.",
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
        self.box = box

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

        # --- serveur voix : statut + démarrer/arrêter ---
        r += 1
        ctk.CTkLabel(box, text="Serveur voix", font=ctk.CTkFont(size=13, weight="bold")
                     ).grid(row=r, column=0, padx=14, pady=(6, 4), sticky="w")
        r += 1
        srow = ctk.CTkFrame(box, fg_color="transparent")
        srow.grid(row=r, column=0, padx=14, pady=2, sticky="ew")
        self.status_dot = ctk.CTkLabel(srow, text="●", font=ctk.CTkFont(size=12),
                                       text_color="#888888")
        self.status_dot.pack(side="left")
        self.status_label = ctk.CTkLabel(srow, text="…", font=ctk.CTkFont(size=11))
        self.status_label.pack(side="left", padx=(4, 8))
        self.btn_vstart = ctk.CTkButton(srow, text="Démarrer ▶", width=90, height=26,
                                        fg_color=ACCENT, text_color="#06121a",
                                        command=self._start_server)
        self.btn_vstart.pack(side="left", padx=(0, 4))
        self.btn_vstop = ctk.CTkButton(srow, text="Arrêter ■", width=80, height=26,
                                       fg_color="#f85149", text_color="#1a0505",
                                       command=self._stop_server)
        self.addr_label = ctk.CTkLabel(box, text="", font=ctk.CTkFont(size=10),
                                       text_color=TEXT_DIM)
        self.addr_label.grid(row=r + 1, column=0, padx=14, sticky="w")
        r += 2

        # --- interaction ---
        r += 1
        ctk.CTkLabel(box, text="Interaction", font=ctk.CTkFont(size=13, weight="bold")
                     ).grid(row=r, column=0, padx=14, pady=(6, 4), sticky="w")
        r += 1
        self.result_label = ctk.CTkLabel(box, text="", font=ctk.CTkFont(size=12),
                                         text_color="#e6edf3", wraplength=520,
                                         justify="left")
        if mode == "tts":
            self.entry = ctk.CTkEntry(box, placeholder_text="Texte à synthétiser…",
                                      height=32)
            self.entry.grid(row=r, column=0, padx=14, pady=2, sticky="ew")
            self.entry.bind("<Return>", lambda _e: self._synthesize())
            r += 1
            brow = ctk.CTkFrame(box, fg_color="transparent")
            brow.grid(row=r, column=0, padx=14, pady=(6, 4), sticky="w")
            self.btn_synth = ctk.CTkButton(brow, text="Synthétiser", height=30,
                                           fg_color=ACCENT, text_color="#06121a",
                                           command=self._synthesize)
            self.btn_synth.pack(side="left", padx=(0, 8))
            self.btn_play = ctk.CTkButton(brow, text="▶ Lire le WAV", height=30,
                                          fg_color=PANEL_2, hover_color="#2a2e37",
                                          command=self._play_last)
            self.btn_play.pack(side="left")
            self.btn_play.configure(state="disabled")
            self._last_wav: str | None = None
            self.result_label.grid(row=r + 1, column=0, padx=14, pady=(4, 14),
                                   sticky="w")
        else:
            brow = ctk.CTkFrame(box, fg_color="transparent")
            brow.grid(row=r, column=0, padx=14, pady=(6, 4), sticky="w")
            ctk.CTkButton(brow, text="● Enregistrer 5 s (micro)", height=30,
                          fg_color=ACCENT, text_color="#06121a",
                          command=lambda: self.ctrl.stt_record(5.0)).pack(
                side="left", padx=(0, 8))
            ctk.CTkButton(brow, text="… Ouvrir un WAV", height=30,
                          fg_color=PANEL_2, hover_color="#2a2e37",
                          command=self._pick_file).pack(side="left")
            self.result_label.grid(row=r + 1, column=0, padx=14, pady=(4, 14),
                                   sticky="w")

    # ---------- serveur voix ----------

    def _model_path(self) -> str | None:
        sel = self.ctrl.selected.get(self.mode)
        return str(sel) if sel else None

    def _start_server(self) -> None:
        self.ctrl.start_voice_server(self.mode, self._model_path())

    def _stop_server(self) -> None:
        self.ctrl.stop_voice_server(self.mode)

    def refresh_status(self) -> None:
        self.ctrl.refresh_voice_status(self.mode)

    def set_voice_status(self, state: str, host: str, port: int, detail: str) -> None:
        self.status_dot.configure(text_color=_VOICE_STATUS_COLOR.get(state, "#888888"))
        self.status_label.configure(text=_VOICE_STATUS_TEXT.get(state, state))
        txt = f"{host}:{port}"
        if detail and state == "offline":
            txt += f" — {detail}"
        self.addr_label.configure(text=txt)
        if state == "offline":
            self.btn_vstart.pack(side="left", padx=(0, 4))
        else:
            self.btn_vstart.pack_forget()
        if state == "managed":
            self.btn_vstop.pack(side="left")
        else:
            self.btn_vstop.pack_forget()

    # ---------- interaction ----------

    def _synthesize(self) -> None:
        if self.mode != "tts":
            return
        text = (self.entry.get() or "").strip()
        if not text:
            self.result_label.configure(text="Saisis un texte d'abord.")
            return
        self.result_label.configure(text="synthèse en cours…")
        self.btn_synth.configure(state="disabled")
        self.ctrl.tts_synthesize(self.mode, text, voice=None)

    def on_tts_done(self, path: str) -> None:
        self._last_wav = path
        self.result_label.configure(text=f"WAV prêt : {path}")
        self.btn_synth.configure(state="normal")
        self.btn_play.configure(state="normal")

    def on_tts_error(self, message: str) -> None:
        self.result_label.configure(text=f"erreur TTS : {message}")
        self.btn_synth.configure(state="normal")

    def _play_last(self) -> None:
        if self._last_wav:
            self.ctrl.play_wav_file(self._last_wav)

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Fichier audio à transcrire",
            filetypes=[("Audio", "*.wav *.mp3 *.webm *.ogg *.m4a"), ("Tous", "*.*")])
        if path:
            self.result_label.configure(text=f"transcription de {path}…")
            self.ctrl.stt_transcribe_file(path)

    def on_stt_recorded(self, path: str) -> None:
        if path:
            self.result_label.configure(text=f"WAV enregistré : {path} — transcription…")

    def on_stt_done(self, text: str) -> None:
        self.result_label.configure(text=f"« {text} »" if text else "(aucune parole détectée)")

    def on_stt_error(self, message: str) -> None:
        self.result_label.configure(text=f"erreur STT : {message}")
