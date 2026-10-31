"""Couche d'affichage terminal (rich) du REPL Jarvis.

- Raisonnement du modèle : panneau grisé en direct (Live) pendant le thinking,
  puis replié à la fin du stream en une ligne résumé (durée, tokens) ; le texte
  complet reste consultable via /thinking (Display.last_reasoning).
- Réponse : rendue en Markdown dans un panneau, en streaming (throttle ~15 Hz
  pour ne pas reparser à chaque token).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

JARVIS_STYLE = "green"
DIM = "grey50"
RENDER_THROTTLE = 0.07  # secondes entre deux re-rendus du live


@dataclass
class Reasoning:
    text: str
    duration: float
    tokens: int


class Display:
    """Affiche le flux d'un tour : raisonnement (repliable) puis réponse (Markdown)."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._reasoning_parts: list[str] = []
        self._reasoning_live: Live | None = None
        self._reasoning_start = 0.0
        self._reasoning_last_render = 0.0
        self._content_parts: list[str] = []
        self._content_live: Live | None = None
        self._content_last_render = 0.0
        self.content_started = False
        # persistants entre les tours (consultables via /thinking)
        self.last_reasoning: Reasoning | None = None

    # -- raisonnement -------------------------------------------------------
    def reasoning(self, chunk: str) -> None:
        self._reasoning_parts.append(chunk)
        now = time.monotonic()
        if self._reasoning_live is None:
            self._reasoning_start = now
            self._reasoning_live = Live(self._render_reasoning(), console=self.console,
                                         transient=True, refresh_per_second=15)
            self._reasoning_live.start()
            self._reasoning_last_render = now
        elif now - self._reasoning_last_render > RENDER_THROTTLE:
            self._reasoning_live.update(self._render_reasoning())
            self._reasoning_last_render = now

    def _render_reasoning(self) -> Panel:
        text = Text("".join(self._reasoning_parts), style=DIM)
        return Panel(text, title=f"[{DIM}]↯ réflexion[/]", border_style=DIM,
                     box=box.ROUNDED, padding=(0, 1))

    def _close_reasoning(self) -> None:
        """Replie le panneau de raisonnement en une ligne résumé."""
        if self._reasoning_live is None:
            return
        text = "".join(self._reasoning_parts)
        duration = time.monotonic() - self._reasoning_start
        tokens = len(text.split())
        self._reasoning_live.stop()  # transient=True : le panneau est effacé
        self._reasoning_live = None
        self._reasoning_parts = []
        self.last_reasoning = Reasoning(text=text, duration=duration, tokens=tokens)
        self.console.print(
            f"[{DIM}]↯ réflexion · {duration:.1f}s · ~{tokens} tok "
            f"(/thinking pour revoir)[/]"
        )

    # -- réponse -------------------------------------------------------------
    def content(self, chunk: str) -> None:
        if self._reasoning_live is not None:
            self._close_reasoning()
        self.content_started = True
        self._content_parts.append(chunk)
        now = time.monotonic()
        if self._content_live is None:
            self._content_live = Live(self._render_content(), console=self.console,
                                      transient=False, refresh_per_second=15)
            self._content_live.start()
            self._content_last_render = now
        elif now - self._content_last_render > RENDER_THROTTLE:
            self._content_live.update(self._render_content())
            self._content_last_render = now

    def print_content(self, text: str) -> None:
        """Mode non-stream : affiche la réponse d'un bloc."""
        self.content_started = True
        self.console.print(self._panel(text))

    def _render_content(self) -> Panel:
        return self._panel("".join(self._content_parts))

    def _panel(self, text: str) -> Panel:
        return Panel(Markdown(text), title=f"[bold {JARVIS_STYLE}]jarvis[/]",
                     border_style=JARVIS_STYLE, box=box.ROUNDED, padding=(0, 1))

    # -- frontières de stream / tour ------------------------------------------
    def end_llm_stream(self) -> None:
        """Appelé à chaque fin de réponse LLM (stream terminé)."""
        self._close_reasoning()
        if self._content_live is not None:
            self._content_live.update(self._render_content())
            self._content_live.stop()  # transient=False : le rendu reste affiché
            self._content_live = None

    def end_turn(self) -> None:
        """Fige le rendu final et remet à zéro l'état de streaming du tour."""
        self.end_llm_stream()
        self._content_parts = []
        self.content_started = False

    # -- consultation ----------------------------------------------------------
    def show_reasoning(self) -> None:
        """Réaffiche le dernier raisonnement (/thinking)."""
        r = self.last_reasoning
        if r is None:
            self.console.print(f"[{DIM}]pas encore de raisonnement[/]")
            return
        panel = Panel(Text(r.text, style=DIM),
                      title=f"[{DIM}]↯ réflexion · {r.duration:.1f}s · ~{r.tokens} tok[/]",
                      border_style=DIM, box=box.ROUNDED, padding=(0, 1))
        if len(r.text.splitlines()) > 30:
            with self.console.pager(styles=True):
                self.console.print(panel)
        else:
            self.console.print(panel)
