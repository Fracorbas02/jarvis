"""REPL minimal avec affichage streaming du raisonnement.

Le thinking du modèle s'affiche en grisé pendant qu'il se produit (comme le
résumé de pensée de Le Chat, sauf que c'est la vraie pensée, en direct), puis
la réponse, puis l'exécution des outils avec leur statut.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from .agent import Agent
from .config import load_config
from .llm import LLMClient
from .mcp import McpToolProvider

DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"

BANNER = """Jarvis (pilote) — /tools /reset /quit"""


class StreamDisplay:
    """Affiche le flux de raisonnement en grisé puis la réponse, en direct."""

    def __init__(self):
        self._reasoning_started = False
        self._content_started = False

    def reasoning(self, text: str) -> None:
        if not self._reasoning_started:
            self._reasoning_started = True
            print(f"{DIM}↯ réflexion… {RESET}", end="", flush=True)
        print(f"{DIM}{text}{RESET}", end="", flush=True)

    def content(self, text: str) -> None:
        if not self._content_started:
            self._content_started = True
            print("\n\n[jarvis] ", end="", flush=True)
        print(text, end="", flush=True)

    def end(self) -> None:
        if self._reasoning_started or self._content_started:
            print()


def _tool_status(name: str, args: dict) -> None:
    print(f"\n{DIM}→ {name} {json.dumps(args, ensure_ascii=False)}{RESET}", flush=True)


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

    print(BANNER)
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

            display = StreamDisplay()
            try:
                reply, history, trace = await agent.turn(
                    history, user,
                    on_tool_call=_tool_status,
                    on_reasoning=display.reasoning,
                    on_content=display.content,
                )
                display.end()
                if not reply and not display._content_started:
                    print(f"[jarvis] {reply}")
            except Exception as exc:  # noqa: BLE001 — le REPL ne doit pas mourir
                display.end()
                print(f"erreur: {exc}", file=sys.stderr)
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
