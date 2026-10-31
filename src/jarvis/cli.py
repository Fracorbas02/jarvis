"""REPL Jarvis : affichage terminal riche (voir tui.py).

Le thinking du modèle et la réponse sont streamés en direct ; la réponse est
rendue en Markdown dans un panneau (tui.Display).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from rich.console import Console

from .agent import Agent
from .config import load_config
from .llm import LLMClient
from .mcp import McpToolProvider
from .tui import Display

BANNER = """Jarvis (pilote) — /help pour les commandes"""


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
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "jarvis.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)


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

    llm = LLMClient(**cfg.llm.__dict__)
    tools = McpToolProvider(cfg.mcp_servers)
    if cfg.mcp_servers:
        await tools.start()
    agent = Agent(llm, tools, cfg.agent.system_prompt, cfg.agent.max_iterations,
                  stream=True)

    console = Console()
    console.print(BANNER)
    display = Display(console)
    history: list[dict] = []
    verbose = False
    try:
        while True:
            try:
                user = input("\n[toi] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user == "/quit":
                break
            if user == "/reset":
                history.clear()
                print("historique réinitialisé")
                continue
            if user == "/tools":
                for name in tools.names():
                    print(f"  {name}")
                continue
            if user == "/verbose":
                verbose = not verbose
                _set_console_log_level(logging.DEBUG if verbose else logging.WARNING)
                print(f"niveau de log console: {'DEBUG' if verbose else 'WARNING'} "
                      f"(log complet: ~/.cache/jarvis/jarvis.log)")
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

            try:
                reply, history, trace = await agent.turn(
                    history, user,
                    on_tool_call=display.tool_call,
                    on_tool_result=display.tool_result,
                    on_reasoning=display.reasoning,
                    on_content=display.content,
                    on_llm_end=display.end_llm_stream,
                )
                display.end_turn()
                if reply and not display.content_started:
                    display.print_content(reply)
            except Exception as exc:  # noqa: BLE001 — le REPL ne doit pas mourir
                display.end_turn()
                console.print(f"[bold red]erreur:[/] {exc}")
                continue
            if trace.error:
                print(f"warn: {trace.error}", file=sys.stderr)
    finally:
        await tools.stop()
        await llm.aclose()
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
