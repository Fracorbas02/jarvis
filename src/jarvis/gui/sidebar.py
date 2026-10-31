"""Barre latérale de la GUI : mode, modèle, serveur llama-server, outils MCP."""

from __future__ import annotations

from typing import Callable

from nicegui import ui

from . import server as server_mod
from .models import MODES, MODE_ORDER, ModelInfo
from .state import GuiState

_STATUS_STYLE = {
    "online": ("● en ligne", "positive", "serveur externe détecté"),
    "managed": ("● géré par la GUI", "info", "llama-server lancé depuis l'interface"),
    "offline": ("● hors ligne", "negative", "aucun serveur ne répond"),
}


class Sidebar:
    def __init__(self, state: GuiState, on_mode_change: Callable[[str], None]):
        self.state = state
        self.on_mode_change = on_mode_change
        self.mode_buttons: dict[str, ui.button] = {}
        self.model_select: ui.select | None = None
        self.model_hint: ui.label | None = None
        self.status_chip: ui.chip | None = None
        self.served_label: ui.label | None = None
        self.addr_label: ui.label | None = None
        self.btn_refresh: ui.button | None = None
        self.btn_start: ui.button | None = None
        self.btn_stop: ui.button | None = None
        self.tools_expansion: ui.expansion | None = None
        self.tools_box: ui.column | None = None
        self._build()

    # ---------- construction ----------

    def _build(self) -> None:
        with ui.column().classes("w-full gap-3"):
            # en-tête
            with ui.row().classes("items-center gap-2 px-1"):
                ui.html('<div class="jarvis-dot"></div>')
                with ui.column().classes("gap-0"):
                    ui.label("JARVIS").classes("text-xl font-bold tracking-widest")
                    ui.label("assistant local").classes("text-xs text-gray-500 -mt-1")

            ui.separator()

            # sélecteur de mode (segmentation)
            ui.label("MODE").classes("text-xs text-gray-500 tracking-wider px-1")
            with ui.row().classes("w-full gap-1"):
                for mode in MODE_ORDER:
                    meta = MODES[mode]
                    btn = ui.button(meta["label"], icon=meta["icon"], on_click=lambda m=mode: self._set_mode(m))
                    btn.props("flat dense no-caps").classes("flex-1 rounded-lg text-xs")
                    self.mode_buttons[mode] = btn
            self._refresh_mode_buttons()

            ui.separator()

            # sélecteur de modèle
            ui.label("MODÈLE").classes("text-xs text-gray-500 tracking-wider px-1")
            self.model_select = ui.select(
                options={}, label="Modèle installé",
                on_change=self._on_model_change,
            ).classes("w-full").props("dense outlined rounded")
            self.model_hint = ui.label("").classes("text-xs text-gray-500 px-1 min-h-4")
            self._refresh_model_select()

            ui.separator()

            # panneau serveur
            ui.label("SERVEUR LLM").classes("text-xs text-gray-500 tracking-wider px-1")
            with ui.row().classes("items-center gap-2 px-1"):
                self.status_chip = ui.chip("● …", color="grey", text_color="white")
                self.addr_label = ui.label("").classes("text-xs text-gray-500")
            self.served_label = ui.label("").classes("text-xs text-gray-400 px-1 break-all")
            with ui.row().classes("items-center gap-1 px-1"):
                self.btn_refresh = ui.button(icon="refresh", on_click=self.refresh_status)
                self.btn_refresh.props("flat dense round size=sm").tooltip("Rafraîchir le statut")
                self.btn_start = ui.button("Démarrer", icon="play_arrow", on_click=self.start_server)
                self.btn_start.props("dense unelevated color=primary no-caps").classes("rounded-lg text-xs")
                self.btn_stop = ui.button("Arrêter", icon="stop", on_click=self.stop_server)
                self.btn_stop.props("dense unelevated color=negative no-caps").classes("rounded-lg text-xs")

            ui.separator()

            # outils MCP
            self.tools_expansion = ui.expansion("Outils MCP", icon="build").classes("w-full text-sm")
            with self.tools_expansion:
                self.tools_box = ui.column().classes("w-full gap-1")
            self.refresh_tools()

            # pied de barre
            ui.space()
            ui.button("Réinitialiser la conversation", icon="restart_alt",
                      on_click=self._reset).props("flat no-caps").classes(
                          "w-full justify-start rounded-lg text-xs text-gray-400")

    # ---------- mode ----------

    def _set_mode(self, mode: str) -> None:
        self.state.mode = mode
        self._refresh_mode_buttons()
        self._refresh_model_select()
        self.on_mode_change(mode)

    def _refresh_mode_buttons(self) -> None:
        for mode, btn in self.mode_buttons.items():
            if mode == self.state.mode:
                btn.props("unelevated color=primary")
            else:
                btn.props("flat color=grey")

    # ---------- modèle ----------

    def _refresh_model_select(self) -> None:
        assert self.model_select is not None
        models = self.state.models_by_mode.get(self.state.mode, [])
        options = {str(m.path): f"{m.name}  ·  {m.size_human}" for m in models}
        current = self.state.selected.get(self.state.mode)
        self.model_select.options = options
        if current and str(current.path) in options:
            self.model_select.value = str(current.path)
        elif options:
            first = next(iter(options))
            self.model_select.value = first
            self.state.selected[self.state.mode] = self._model_by_path(first)
        else:
            self.model_select.value = None
        self._refresh_model_hint()

    def _model_by_path(self, path: str) -> ModelInfo | None:
        for m in self.state.models_by_mode.get(self.state.mode, []):
            if str(m.path) == path:
                return m
        return None

    def _on_model_change(self) -> None:
        assert self.model_select is not None
        if self.model_select.value:
            self.state.selected[self.state.mode] = self._model_by_path(self.model_select.value)
        self._refresh_model_hint()
        self.state.notify_model_change()

    def _refresh_model_hint(self) -> None:
        assert self.model_hint is not None
        m = self.state.selected.get(self.state.mode)
        if m is None:
            n = len(self.state.models_by_mode.get(self.state.mode, []))
            self.model_hint.text = "aucun modèle détecté pour ce mode" if n == 0 else ""
        else:
            self.model_hint.text = str(m.path)

    # ---------- serveur ----------

    async def refresh_status(self) -> None:
        if self.state.backend is None:
            return
        self.state.server_status = await server_mod.get_status(
            self.state.backend.cfg.llm.base_url)
        self._paint_status()

    def _paint_status(self) -> None:
        st = self.state.server_status
        if st is None or self.status_chip is None:
            return
        text, color, tooltip = _STATUS_STYLE.get(st.state, ("● ?", "grey", ""))
        self.status_chip.text = text
        self.status_chip.props(f"color={color}")
        self.status_chip.tooltip(tooltip)
        assert self.addr_label is not None and self.served_label is not None
        self.addr_label.text = f"{st.host}:{st.port}"
        served = st.models[0] if st.models else ""
        name = served.rsplit("/", 1)[-1] if served else "—"
        self.served_label.text = f"modèle servi : {name}" if st.models else "aucun modèle servi"
        # visibilité des boutons selon l'état
        managed = st.state == "managed"
        offline = st.state == "offline"
        assert self.btn_start is not None and self.btn_stop is not None
        self.btn_start.visible = offline and self.state.mode == "llm"
        self.btn_stop.visible = managed

    async def start_server(self) -> None:
        if self.state.backend is None:
            return
        model = self.state.selected.get("llm")
        if model is None:
            ui.notify("Choisis d'abord un modèle LLM", type="warning")
            return
        ui.notify(f"Démarrage de llama-server avec {model.name}… "
                  f"(chargement du modèle, voir ~/.cache/jarvis/llama-server-gui.log)",
                  type="info")
        self.btn_start.disable() if self.btn_start else None

        async def _do_start() -> None:
            st = await server_mod.start_server(model.path,
                                               self.state.backend.cfg.llm.base_url)
            self.state.server_status = st
            self._paint_status()
            if st.models:
                ui.notify("Serveur prêt", type="positive")
            elif st.state == "offline":
                ui.notify(f"Échec démarrage : {st.detail}", type="negative", timeout=15)
            else:
                ui.notify(st.detail or "démarrage en cours", type="warning")
            if self.btn_start:
                self.btn_start.enable()

        # le chargement du modèle prend des dizaines de secondes : on lance la
        # coroutine via un timer (tâche serveur dédiée), jamais en await direct
        # dans le handler (ça figerait le client)
        ui.timer(0.1, _do_start, once=True)

    async def stop_server(self) -> None:
        if self.state.backend is None:
            return
        st = await server_mod.stop_server(self.state.backend.cfg.llm.base_url)
        self.state.server_status = st
        self._paint_status()
        ui.notify("Serveur arrêté", type="info")

    # ---------- outils ----------

    def refresh_tools(self) -> None:
        assert self.tools_box is not None and self.tools_expansion is not None
        self.tools_box.clear()
        names = self.state.backend.tool_names() if self.state.backend else []
        self.tools_expansion.text = f"Outils MCP ({len(names)})"
        with self.tools_box:
            if not names:
                ui.label("aucun outil (MCP non configuré)").classes("text-xs text-gray-500")
            for name in names:
                ui.chip(name, icon="extension").props("dense outline").classes("text-xs")

    # ---------- reset ----------

    def _reset(self) -> None:
        if self.state.backend is not None:
            self.state.backend.reset()
        self.state.notify_reset()
        ui.notify("Conversation réinitialisée", type="info")


def build_sidebar(state: GuiState, on_mode_change: Callable[[str], None]) -> Sidebar:
    """Construit la sidebar dans le conteneur courant."""
    return Sidebar(state, on_mode_change)
