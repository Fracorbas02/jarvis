"""Couche d'affichage terminal (rich) du REPL Jarvis.

La réponse du modèle est rendue en Markdown dans un panneau, en streaming :
chaque fragment reçu met à jour un Live qui re-rend le Markdown au fur et à
mesure (throttle ~15 Hz pour ne pas reparser à chaque token).
"""

from __future__ import annotations

import time

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

JARVIS_STYLE = "green"
DIM = "grey50"


class Display:
    """Affiche le flux d'un tour : raisonnement (grisé) puis réponse (Markdown)."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._reasoning_started = False
        self._content_parts: list[str] = []
        self._live: Live | None = None
        self._last_render = 0.0
        self.content_started = False

    # -- raisonnement (provisoire : grisé inline) --------------------------
    def reasoning(self, chunk: str) -> None:
        if not self._reasoning_started:
            self._reasoning_started = True
            self.console.print("↯ réflexion… ", style=DIM, end="")
        self.console.print(chunk, style=DIM, end="")

    # -- réponse ------------------------------------------------------------
    def content(self, chunk: str) -> None:
        if self._reasoning_started:
            self.console.print()  # termine la ligne du raisonnement
            self._reasoning_started = False
        self.content_started = True
        self._content_parts.append(chunk)
        now = time.monotonic()
        if self._live is None:
            self._live = Live(self._render_content(), console=self.console,
                              transient=False, refresh_per_second=15)
            self._live.start()
            self._last_render = now
        elif now - self._last_render > 0.07:
            self._live.update(self._render_content())
            self._last_render = now

    def print_content(self, text: str) -> None:
        """Mode non-stream : affiche la réponse d'un bloc."""
        self.content_started = True
        self.console.print(self._panel(text))

    def _render_content(self):
        return self._panel("".join(self._content_parts))

    def _panel(self, text: str) -> Panel:
        return Panel(Markdown(text), title=f"[bold {JARVIS_STYLE}]jarvis[/]",
                     border_style=JARVIS_STYLE, box=box.ROUNDED, padding=(0, 1))

    # -- fin de tour ---------------------------------------------------------
    def end_turn(self) -> None:
        """Fige le rendu final et remet à zéro l'état de streaming."""
        if self._live is not None:
            self._live.update(self._render_content())
            self._live.stop()
            self._live = None
        elif self._reasoning_started:
            self.console.print()
        self._reasoning_started = False
        self._content_parts.clear()
