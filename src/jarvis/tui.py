"""Couche d'affichage terminal (rich) du REPL Jarvis.

- Raisonnement du modèle : panneau grisé en direct (Live) pendant le thinking,
  puis replié à la fin du stream en une ligne résumé (durée, tokens) ; le texte
  complet reste consultable via /thinking (Display.last_reasoning).
- Réponse : rendue en Markdown dans un panneau, en streaming (throttle ~15 Hz
  pour ne pas reparser à chaque token).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

JARVIS_STYLE = "green"
DIM = "grey50"
RENDER_THROTTLE = 0.07  # secondes entre deux re-rendus du live
MAX_LINE = 110  # troncature des lignes d'appel d'outils à l'affichage


@dataclass
class Reasoning:
    text: str
    duration: float
    tokens: int


@dataclass
class ToolResult:
    name: str
    args: dict
    ok: bool
    duration: float
    result: str


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
        # persistants entre les tours (consultables via /thinking, /tool N)
        self.last_reasoning: Reasoning | None = None
        self.tool_results: list[ToolResult] = []
        self.last_tool_results: list[ToolResult] = []
        self._tool_live: Live | None = None
        self._tool_line = ""

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
            self.console.print()  # aère avant le panneau
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

    # -- outils ----------------------------------------------------------------
    def tool_call(self, name: str, args: dict) -> None:
        """Début d'exécution d'un outil : spinner sur la ligne d'appel."""
        raw = f"→ {name} {json.dumps(args, ensure_ascii=False)}"
        self._tool_line = raw if len(raw) <= MAX_LINE else raw[: MAX_LINE - 1] + "…"
        self._tool_live = Live(Spinner("dots", text=Text(self._tool_line, style=DIM)),
                               console=self.console, transient=True,
                               refresh_per_second=15)
        self._tool_live.start()

    def tool_result(self, name: str, args: dict, ok: bool,
                    duration: float, result: str) -> None:
        """Fin d'exécution : statut ✓/✗ + durée, aperçu du résultat, stockage."""
        if self._tool_live is not None:
            self._tool_live.stop()
            self._tool_live = None
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        raw_args = json.dumps(args, ensure_ascii=False)
        shown = raw_args if len(raw_args) <= 60 else raw_args[:57] + "…"
        self.console.print(f"{mark} {name}({shown})  [{DIM}]{duration:.2f}s[/]")
        n = len(self.tool_results) + 1
        self.tool_results.append(ToolResult(name, args, ok, duration, result))
        if result:
            lines = result.splitlines()
            preview = "\n".join(lines[:3])
            if len(lines) > 3 or len(result) > 200:
                preview += f"\n… ({len(result)} car. au total — /tool {n})"
            self.console.print(Text("  └─ " + preview.replace("\n", "\n  "), style=DIM))

    def show_tool(self, n: int) -> None:
        """Réaffiche le résultat complet du n-ième outil du dernier tour."""
        if not 1 <= n <= len(self.last_tool_results):
            self.console.print(
                f"[{DIM}]pas d'outil n°{n} sur le dernier tour "
                f"({len(self.last_tool_results)} appel(s))[/]"
            )
            return
        t = self.last_tool_results[n - 1]
        mark = "✓" if t.ok else "✗"
        style = DIM if t.ok else "red"
        panel = Panel(
            Text(t.result),
            title=f"[{style}]{mark} {t.name} · {t.duration:.2f}s[/]",
            border_style=style, box=box.ROUNDED, padding=(0, 1),
        )
        if len(t.result.splitlines()) > 30:
            with self.console.pager(styles=True):
                self.console.print(panel)
        else:
            self.console.print(panel)

    # -- frontières de stream / tour ------------------------------------------
    def end_llm_stream(self) -> None:
        """Appelé à chaque fin de réponse LLM (stream terminé)."""
        self._close_reasoning()
        if self._content_live is not None:
            self._content_live.update(self._render_content())
            self._content_live.stop()  # transient=False : le rendu reste affiché
            self._content_live = None

    def end_turn(self) -> None:
        """Fige le rendu final et archive les résultats d'outils du tour."""
        self.end_llm_stream()
        self._content_parts = []
        self.content_started = False
        self.last_tool_results = self.tool_results
        self.tool_results = []

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
