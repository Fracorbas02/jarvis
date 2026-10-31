"""Interface graphique locale de Jarvis — point d'entrée `uv run jarvis-gui`.

Layout : barre latérale (mode / modèle / serveur / outils) + zone principale
qui change selon le mode choisi (chat LLM, TTS, STT).
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from nicegui import app, ui

from .backend import GuiBackend
from .chat_view import ChatView
from .models import MODES, discover_models
from .sidebar import build_sidebar
from .state import GuiState
from .voice_views import build_voice_panel

log = logging.getLogger("jarvis.gui")

HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  body { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
  .q-field__control, .q-btn, .q-chip, .q-menu { border-radius: 10px !important; }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2b303b; border-radius: 8px; }
  ::-webkit-scrollbar-thumb:hover { background: #3a4150; }
  .jarvis-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background: #4cc2ff; box-shadow: 0 0 14px 3px rgba(76,194,255,.65);
    animation: jarvis-pulse 2.2s ease-in-out infinite;
  }
  @keyframes jarvis-pulse { 0%,100% { opacity: 1; } 50% { opacity: .45; } }
  .mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
</style>
"""


def build_page(state: GuiState) -> None:
    ui.add_head_html(HEAD_HTML)
    ui.dark_mode(True)
    ui.colors(primary="#4cc2ff", secondary="#7c6cff", accent="#ffb454",
              dark="#16181d", dark_page="#0d0f13")

    with ui.splitter(value=300).classes("w-full h-screen") as splitter:
        splitter.props("vertical=false")
        with splitter.before:
            with ui.column().classes("w-full h-full gap-3 p-4 overflow-y-auto bg-[#121419]"):
                sidebar = build_sidebar(state, on_mode_change=lambda m: paint_main(m))
                state.sidebar = sidebar  # type: ignore[attr-defined]
        with splitter.after:
            with ui.column().classes("w-full h-full p-6 overflow-hidden"):
                main_box = ui.column().classes("w-full h-full")
                state.main_box = main_box  # type: ignore[attr-defined]

    # Panneaux de chaque mode, construits une fois puis shown/hidden.
    panels: dict[str, ui.element] = {}
    with main_box:
        panels["llm"] = ChatView(state).root
        for mode in ("tts", "stt"):
            panels[mode] = build_voice_panel(state, mode)

    def paint_main(mode: str) -> None:
        for m, panel in panels.items():
            panel.visible = (m == mode)

    paint_main(state.mode)

    # Le backend MCP démarre en on_startup : attendre qu'il soit prêt pour
    # remplir outils + statut, puis rafraîchir le statut périodiquement.
    async def _ready_check() -> None:
        if state.backend is not None and state.backend.started:
            sidebar.refresh_tools()
            await sidebar.refresh_status()
            ready_timer.cancel()

    ready_timer = ui.timer(0.5, _ready_check)
    ui.timer(5.0, lambda: sidebar.refresh_status())


async def _startup(state: GuiState) -> None:
    try:
        state.backend = GuiBackend()
        await state.backend.start()
        state.models_by_mode = discover_models()
        log.info("backend GUI démarré (%d outils)", len(state.backend.tool_names()))
    except Exception as exc:  # noqa: BLE001 — la GUI doit s'ouvrir même sans config
        log.exception("démarrage backend GUI")
        state.backend_error = str(exc)


async def _shutdown(state: GuiState) -> None:
    if state.backend is not None:
        await state.backend.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis-gui",
                                     description="Interface graphique locale de Jarvis")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    state = GuiState()
    state.models_by_mode = discover_models()

    app.on_startup(lambda: _startup(state))
    app.on_shutdown(lambda: _shutdown(state))

    @ui.page("/")
    def index() -> None:
        if state.backend_error:
            ui.dark_mode(True)
            with ui.column().classes("w-full h-screen items-center justify-center gap-2"):
                ui.label("Jarvis — erreur de configuration").classes("text-2xl font-bold")
                ui.label(state.backend_error).classes("text-negative")
            return
        build_page(state)

    ui.run(
        title="Jarvis",
        host=args.host,
        port=args.port,
        dark=True,
        reload=False,
        show=args.open_browser,
    )


if __name__ == "__main__":
    main()
