"""Barre latérale de l'application desktop : mode, modèle, serveur, outils."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from ..gui.models import MODES, MODE_ORDER
from .controller import DesktopController
from .theme import ACCENT, PANEL, PANEL_2, TEXT_DIM

MODE_VALUES = [MODES[m]["label"] for m in MODE_ORDER]
MODE_BY_LABEL = {MODES[m]["label"]: m for m in MODE_ORDER}

_STATUS_TEXT = {"online": "● en ligne", "managed": "● géré par l'app",
                "offline": "● hors ligne"}
_STATUS_COLOR = {"online": "#3fb950", "managed": "#4cc2ff", "offline": "#f85149"}


class Sidebar(ctk.CTkFrame):
    def __init__(self, master, ctrl: DesktopController,
                 on_mode_change: Callable[[str], None],
                 on_reset: Callable[[], None]):
        super().__init__(master, fg_color=PANEL, corner_radius=0)
        self.ctrl = ctrl
        self.on_mode_change = on_mode_change
        self.on_reset = on_reset
        self._model_paths: dict[str, str] = {}  # label combo -> chemin

        # largeur contrôlée par le volet (PanedWindow) parent
        self.grid_columnconfigure(0, weight=1)

        r = 0
        # titre
        ctk.CTkLabel(self, text="JARVIS", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=ACCENT).grid(row=r, column=0, padx=16, pady=(16, 0),
                                             sticky="w")
        r += 1
        ctk.CTkLabel(self, text="assistant local", font=ctk.CTkFont(size=11),
                     text_color=TEXT_DIM).grid(row=r, column=0, padx=16, pady=(0, 8),
                                               sticky="w")
        r += 1
        self._sep(r); r += 1

        # mode
        ctk.CTkLabel(self, text="MODE", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_DIM).grid(row=r, column=0, padx=16, sticky="w")
        r += 1
        self.mode_seg = ctk.CTkSegmentedButton(
            self, values=MODE_VALUES, height=30,
            selected_color=ACCENT, unselected_color=PANEL_2,
            command=self._on_mode)
        self.mode_seg.grid(row=r, column=0, padx=12, pady=(4, 8), sticky="ew")
        self.mode_seg.set(MODES[self.ctrl.mode]["label"])
        r += 1
        self._sep(r); r += 1

        # modèle
        ctk.CTkLabel(self, text="MODÈLE", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_DIM).grid(row=r, column=0, padx=16, sticky="w")
        r += 1
        self.model_combo = ctk.CTkComboBox(self, values=[], height=28,
                                           fg_color=PANEL_2, button_color=ACCENT,
                                           command=self._on_model)
        self.model_combo.grid(row=r, column=0, padx=12, pady=(4, 2), sticky="ew")
        r += 1
        self.model_hint = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=9),
                                       text_color=TEXT_DIM, wraplength=230,
                                       justify="left")
        self.model_hint.grid(row=r, column=0, padx=16, sticky="w")
        r += 1
        self.refresh_models()
        self._sep(r); r += 1

        # serveur
        ctk.CTkLabel(self, text="SERVEUR LLM", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_DIM).grid(row=r, column=0, padx=16, sticky="w")
        r += 1
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=r, column=0, padx=12, pady=(4, 2), sticky="ew")
        self.status_dot = ctk.CTkLabel(row, text="●", font=ctk.CTkFont(size=12),
                                       text_color="#888888")
        self.status_dot.pack(side="left")
        self.status_label = ctk.CTkLabel(row, text="…", font=ctk.CTkFont(size=11))
        self.status_label.pack(side="left", padx=(4, 0))
        r += 1
        self.addr_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=10),
                                       text_color=TEXT_DIM)
        self.addr_label.grid(row=r, column=0, padx=16, sticky="w")
        r += 1
        self.served_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=10),
                                         text_color=TEXT_DIM, wraplength=230,
                                         justify="left")
        self.served_label.grid(row=r, column=0, padx=16, pady=(0, 4), sticky="w")
        r += 1
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=r, column=0, padx=12, pady=(0, 8), sticky="ew")
        self.btn_refresh = ctk.CTkButton(btns, text="↻", width=32, height=26,
                                         fg_color=PANEL_2, hover_color=ACCENT,
                                         command=self.ctrl.refresh_status)
        self.btn_refresh.pack(side="left", padx=(0, 4))
        self.btn_start = ctk.CTkButton(btns, text="Démarrer ▶", height=26,
                                       fg_color=ACCENT, text_color="#06121a",
                                       command=self._start_server)
        self.btn_start.pack(side="left", padx=(0, 4))
        self.btn_stop = ctk.CTkButton(btns, text="Arrêter ■", height=26,
                                      fg_color="#f85149", text_color="#1a0505",
                                      command=self.ctrl.stop_server)
        self.btn_stop.pack(side="left")
        r += 1
        self._sep(r); r += 1

        # outils
        self.tools_label = ctk.CTkLabel(self, text="OUTILS MCP",
                                        font=ctk.CTkFont(size=10, weight="bold"),
                                        text_color=TEXT_DIM)
        self.tools_label.grid(row=r, column=0, padx=16, sticky="w")
        r += 1
        self.tools_box = ctk.CTkScrollableFrame(self, height=120, fg_color=PANEL_2)
        self.tools_box.grid(row=r, column=0, padx=12, pady=(4, 8), sticky="ew")
        r += 1

        # reset en bas
        self.btn_reset = ctk.CTkButton(self, text="Réinitialiser la conversation",
                                       height=28, fg_color=PANEL_2,
                                       hover_color="#2a2e37",
                                       font=ctk.CTkFont(size=11),
                                       command=self._reset)
        self.btn_reset.grid(row=r, column=0, padx=12, pady=(8, 12), sticky="ew")

    def _sep(self, r: int) -> None:
        ctk.CTkFrame(self, height=1, fg_color="#232833").grid(
            row=r, column=0, padx=12, pady=6, sticky="ew")

    # ---------- interactions ----------

    def _on_mode(self, label: str) -> None:
        self.ctrl.mode = MODE_BY_LABEL[label]
        self.refresh_models()
        self.on_mode_change(self.ctrl.mode)

    def refresh_models(self) -> None:
        models = self.ctrl.models_by_mode.get(self.ctrl.mode, [])
        self._model_paths = {f"{m.name}  ·  {m.size_human}": str(m.path)
                             for m in models}
        self.model_combo.configure(values=list(self._model_paths))
        if self._model_paths:
            first = next(iter(self._model_paths))
            self.model_combo.set(first)
            self.ctrl.selected[self.ctrl.mode] = self._model_paths[first]
            self.model_hint.configure(text=self._model_paths[first])
        else:
            self.model_combo.set("")
            self.model_hint.configure(
                text="aucun modèle détecté pour ce mode (voir ~/models)")

    def _on_model(self, label: str) -> None:
        path = self._model_paths.get(label)
        if path:
            self.ctrl.selected[self.ctrl.mode] = path
            self.model_hint.configure(text=path)

    def _start_server(self) -> None:
        model = self.ctrl.selected.get("llm")
        if model:
            self.ctrl.start_server(str(model))

    def _reset(self) -> None:
        self.ctrl.reset()
        self.on_reset()

    # ---------- mises à jour pilotées par les événements ----------

    def set_tools(self, names: list[str]) -> None:
        for w in self.tools_box.winfo_children():
            w.destroy()
        self.tools_label.configure(text=f"OUTILS MCP ({len(names)})")
        if not names:
            ctk.CTkLabel(self.tools_box, text="aucun outil",
                         text_color=TEXT_DIM).pack(anchor="w", padx=4, pady=2)
        for name in names:
            ctk.CTkLabel(self.tools_box, text=f"• {name}",
                         font=ctk.CTkFont(size=11),
                         text_color="#c9d1d9").pack(anchor="w", padx=4, pady=1)

    def set_server_status(self, state: str, host: str, port: int,
                          models: list[str], detail: str = "") -> None:
        self.status_dot.configure(text="●", text_color=_STATUS_COLOR.get(state, "#888888"))
        self.status_label.configure(text=_STATUS_TEXT.get(state, state))
        self.addr_label.configure(text=f"{host}:{port}")
        served = models[0] if models else ""
        if models:
            self.served_label.configure(text=f"modèle servi : {served.rsplit('/', 1)[-1]}")
        elif detail:
            # ex. "llama-server mort (code 1), voir ~/.cache/jarvis/…" —
            # sinon l'utilisateur clique sur Démarrer et ne voit « rien »
            self.served_label.configure(text=detail[:140], text_color="#f85149")
        else:
            self.served_label.configure(text="aucun modèle servi", text_color=TEXT_DIM)
        offline = state == "offline"
        managed = state == "managed"
        if offline and self.ctrl.mode == "llm":
            self.btn_start.pack(side="left", padx=(0, 4))
        else:
            self.btn_start.pack_forget()
        if managed:
            self.btn_stop.pack(side="left")
        else:
            self.btn_stop.pack_forget()
