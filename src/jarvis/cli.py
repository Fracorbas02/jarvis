"""REPL Jarvis : affichage terminal riche (voir tui.py).

Le thinking du modèle et la réponse sont streamés en direct ; la réponse est
rendue en Markdown dans un panneau (tui.Display).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import tomllib
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .config import load_config
from .llm import LLMClient
from .mcp import McpToolProvider
from .tui import Display

HELP = """\
[bold]commandes[/]
  /help        cette aide
  /tools       liste les outils MCP avec leur description
  /thinking    réaffiche le raisonnement du dernier tour
  /tool N      résultat complet du N-ième outil du dernier tour
  /reset       efface l'historique de conversation
  /verbose     bascule les logs console en DEBUG (fichier: ~/.cache/jarvis/jarvis.log)
  /quit        quitte"""


def _make_console() -> Console:
    """Console rich : couleurs forcées en vrai terminal (sauf NO_COLOR).

    Nettoie os.environ (lu par défaut par rich) : COLUMNS/LINES figeraient la
    largeur, et TERM=dumb ferait retomber rich sur 80x25 au lieu de lire la
    taille réelle du tty. La taille des panneaux live est de toute façon
    relue par ioctl à chaque rendu (tui._LiveRegion.tty_width).
    """
    if os.environ.get("NO_COLOR"):
        return Console(no_color=True)
    os.environ.pop("COLUMNS", None)
    os.environ.pop("LINES", None)
    if os.environ.get("TERM", "") == "dumb":
        os.environ["TERM"] = "xterm-256color"
    if sys.stdout.isatty():
        return Console(force_terminal=True, color_system="truecolor")
    return Console()


def _banner(cfg, tools: McpToolProvider) -> Panel:
    n = len(tools.names())
    servers = ", ".join(sorted(cfg.mcp_servers)) or "aucun"
    body = Text.from_markup(
        f"[bold]assistant local[/]\n"
        f"[dim]modèle[/]  {cfg.llm.model} @ {cfg.llm.base_url}\n"
        f"[dim]outils[/]  {n} chargés ({servers})\n"
        f"[dim]/help[/] pour les commandes"
    )
    return Panel(body, title="[bold cyan]JARVIS[/]", border_style="cyan",
                 box=box.ROUNDED, padding=(0, 1))


def _setup_logging(verbose: bool = False) -> None:
    """Console en WARNING par défaut (lisible), DEBUG complet dans un fichier.

    Les logs httpx/jarvis restent disponibles pour le debug post-mortem dans
    ~/.cache/jarvis/jarvis.log (ou $JARVIS_LOG_DIR/jarvis.log).
    """
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)
    log_dir = Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "jarvis.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
    except OSError:
        pass  # répertoire non writable : la console suffit, on ne crash pas


def _set_console_log_level(level: int) -> None:
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)


async def amain() -> int:
    _setup_logging()
    try:
        cfg = load_config()
    except FileNotFoundError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2
    except tomllib.TOMLDecodeError as exc:
        print(f"config: TOML invalide: {exc}", file=sys.stderr)
        return 2

    llm = LLMClient(**cfg.llm.__dict__)
    tools = McpToolProvider(cfg.mcp_servers)
    if cfg.mcp_servers:
        await tools.start()
    agent = Agent(llm, tools, cfg.agent.system_prompt, cfg.agent.max_iterations,
                  stream=True)

    console = _make_console()
    display = Display(console)
    display.set_header(_banner(cfg, tools))
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGWINCH, display.on_resize)
    except (NotImplementedError, AttributeError):
        pass  # Windows / pas de signaux : le repaint suivra au prochain token
    history: list[dict] = []
    verbose = False
    try:
        while True:
            try:
                user = console.input("\n[bold cyan][toi] › [/]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user == "/quit":
                break
            if user == "/help":
                console.print(HELP)
                continue
            if user == "/reset":
                history.clear()
                display.reset_blocks()  # sinon les blocs reviennent au prochain resize
                console.print("[dim]historique réinitialisé[/]")
                continue
            if user == "/tools":
                table = Table(box=box.SIMPLE_HEAD, title="Outils MCP",
                              title_style="bold", header_style="bold")
                table.add_column("Outil")
                table.add_column("Description")
                for s in tools.schemas():
                    fn = s["function"]
                    desc = (fn.get("description") or "").strip().splitlines()
                    table.add_row(fn["name"], desc[0] if desc else "")
                console.print(table)
                continue
            if user == "/verbose":
                verbose = not verbose
                _set_console_log_level(logging.DEBUG if verbose else logging.WARNING)
                console.print(f"[dim]niveau de log console: "
                              f"{'DEBUG' if verbose else 'WARNING'} "
                              f"(log complet: ~/.cache/jarvis/jarvis.log)[/]")
                continue
            if user == "/thinking":
                display.show_reasoning()
                continue
            if user.startswith("/tool"):
                parts = user.split()
                if len(parts) == 2 and parts[1].isdigit():
                    display.show_tool(int(parts[1]))
                else:
                    console.print(f"[dim]usage: /tool N "
                                  f"(1..{max(len(display.last_tool_results), 1)})[/]")
                continue

            snapshot = list(history)  # agent.turn mute history : on restaure si ça plante
            try:
                reply, history, trace = await agent.turn(
                    history, user,
                    on_tool_call=display.tool_call,
                    on_tool_result=display.tool_result,
                    on_reasoning=display.reasoning,
                    on_content=display.content,
                    on_llm_end=display.end_llm_stream,
                )
                if reply and not display.content_started:
                    display.print_content(reply)
                display.end_turn()
            except KeyboardInterrupt:
                history = snapshot
                display.end_turn()
                console.print("[yellow]interrompu (Ctrl+C)[/]")
                continue
            except Exception as exc:  # noqa: BLE001 — le REPL ne doit pas mourir
                history = snapshot
                display.end_turn()
                console.print(f"[bold red]erreur:[/] {exc}")
                continue
            if trace.error:
                console.print(f"[bold yellow]warn:[/] {trace.error}")
    finally:
        await tools.stop()
        await llm.aclose()
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
