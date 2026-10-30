"""REPL minimal. TUI (Textual) viendra par-dessus agent.py sans le modifier."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from .agent import Agent
from .config import load_config
from .llm import LLMClient
from .mcp import McpToolProvider

BANNER = """Jarvis (pilote) — /tools /reset /quit"""


def _print_tool(name: str, args: dict) -> None:
    print(f"  -> {name} {json.dumps(args, ensure_ascii=False)}", flush=True)


async def amain() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config()
    except FileNotFoundError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2

    llm = LLMClient(**cfg.llm.__dict__)
    tools = McpToolProvider(cfg.mcp_servers)
    if cfg.mcp_servers:
        await tools.start()
    agent = Agent(llm, tools, cfg.agent.system_prompt, cfg.agent.max_iterations)

    print(BANNER)
    history: list[dict] = []
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

            print("[jarvis] …", flush=True)
            try:
                reply, history, trace = await agent.turn(history, user,
                                                          on_tool_call=_print_tool)
            except Exception as exc:  # noqa: BLE001 — le REPL ne doit pas mourir
                print(f"erreur: {exc}", file=sys.stderr)
                continue
            print(f"\n[jarvis]\n{reply}")
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
