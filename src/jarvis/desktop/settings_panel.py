"""Panneau Réglages de l’application desktop : choix du GPU, infos.

Le GPU choisi est persisté dans ~/.config/jarvis/gpu.json et appliqué
aux serveurs voix (STT/TTS) au prochain démarrage (voir jarvis.gpu).
"""

from __future__ import annotations

import customtkinter as ctk

from ..gpu import detect_gpus, env_for_device, load_gpu_choice, save_gpu_choice
from .theme import ACCENT, BG, PANEL_2, TEXT_DIM


class SettingsPanel(ctk.CTkFrame):
    def __init__(self, master, ctrl):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.ctrl = ctrl
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Réglages", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=ACCENT).grid(row=0, column=0, padx=16, pady=(12, 4),
                                             sticky="w")

        box = ctk.CTkFrame(self, fg_color=PANEL_2, corner_radius=12)
        box.grid(row=1, column=0, padx=20, pady=16, sticky="nsew")
        box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(box, text="GPU utilisé par les serveurs locaux (STT/TTS/LLM)",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(12, 6), sticky="w")

        self.gpus = detect_gpus()
        labels = [f"{g.name}  ({g.device})" for g in self.gpus]
        choice = load_gpu_choice()
        current = choice.get("device", "")
        initial = next((l for l in labels if l.endswith(f"({current})")), labels[0])
        self.gpu_combo = ctk.CTkComboBox(box, values=labels, height=30,
                                         fg_color=BG, button_color=ACCENT)
        self.gpu_combo.set(initial)
        self.gpu_combo.grid(row=1, column=0, padx=14, pady=4, sticky="ew")

        self.gpu_hint = ctk.CTkLabel(box, text="", font=ctk.CTkFont(size=11),
                                     text_color=TEXT_DIM, wraplength=520,
                                     justify="left")
        self.gpu_hint.grid(row=2, column=0, padx=14, pady=(2, 8), sticky="w")
        self._update_hint()

        ctk.CTkButton(box, text="Enregistrer", width=110, height=30,
                      fg_color=ACCENT, text_color="#06121a",
                      command=self._save).grid(row=3, column=0, padx=14,
                                               pady=(0, 12), sticky="w")

    def _device_of(self, label: str) -> str:
        return label.rsplit("(", 1)[1].rstrip(")")

    def _update_hint(self) -> None:
        dev = self._device_of(self.gpu_combo.get())
        env = env_for_device(dev)
        txt = f"device = {dev}"
        if env:
            txt += "   env = " + ", ".join(f"{k}={v}" for k, v in env.items())
        txt += "\nEnregistré dans ~/.config/jarvis/gpu.json"
        self.gpu_hint.configure(text=txt)

    def _save(self) -> None:
        dev = self._device_of(self.gpu_combo.get())
        save_gpu_choice(dev, env_for_device(dev))
        self._update_hint()
