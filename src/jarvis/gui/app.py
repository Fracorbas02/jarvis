"""Point d'entrée de l'interface graphique : `uv run jarvis-gui`."""

from __future__ import annotations

import argparse

from nicegui import ui


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis-gui", description="Interface graphique locale de Jarvis")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    @ui.page("/")
    def index() -> None:
        ui.dark_mode(True)
        ui.colors(primary="#6ea8fe", dark="#141518", dark_page="#0d0e11")
        with ui.column().classes("w-full h-screen items-center justify-center"):
            ui.label("Jarvis").classes("text-4xl font-bold")
            ui.label("interface graphique — scaffold").classes("text-gray-400")

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
