"""Panneau Réglages de l’application desktop : choix du GPU, infos.

Le GPU choisi est persisté dans ~/.config/jarvis/gpu.json et appliqué par
variables d env (CUDA_VISIBLE_DEVICES & co) au llama-server principal et aux
serveurs voix (STT/TTS) au prochain démarrage (voir jarvis.gpu).
"""

from __future__ import annotations

import customtkinter as ctk

from ..gpu import (args_for_device, detect_gpus, env_for_device,
                   load_gpu_choice, save_gpu_choice)
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
        self.dev_by_label = {f"{g.name}  ({g.device})": g.device for g in self.gpus}
        labels = list(self.dev_by_label)
        choice = load_gpu_choice()
        current = choice.get("device", "")
        initial = next((l for l, g in zip(labels, self.gpus) if g.device == current),
                       labels[0])
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
        # dict label -> device (robuste même si le nom du GPU contient "(")
        return self.dev_by_label.get(label, label)

    def _update_hint(self) -> None:
        dev = self._device_of(self.gpu_combo.get())
        env = env_for_device(dev)
        args = args_for_device(dev)
        saved = load_gpu_choice().get("device", "")
        txt = f"device = {dev}"
        if env:
            txt += "   env = " + ", ".join(f"{k}={v}" for k, v in env.items())
        if args:
            txt += "   args = " + " ".join(args)
        if saved:
            txt += (f"\nEnregistré : {saved} dans ~/.config/jarvis/gpu.json "
                    "(appliqué au prochain démarrage des serveurs locaux)")
        else:
            txt += ("\nAUCUN choix enregistré (pas de gpu.json) : le GPU par défaut "
                    "sera utilisé. Sélectionne un GPU puis clique sur Enregistrer.")
        self.gpu_hint.configure(text=txt)

    def _save(self) -> None:
        dev = self._device_of(self.gpu_combo.get())
        save_gpu_choice(dev, env_for_device(dev))
        self._update_hint()
