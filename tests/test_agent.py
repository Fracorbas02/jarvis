"""Test end-to-end de la boucle agent : mock LLM + MCP server echo réel (sous-processus).

Valide, sans modèle ni réseau :
- la transmission des schémas MCP au LLM (format tools),
- le déroulé tool_call -> exécution via MCP -> injection role=tool -> réponse finale.

Usage: uv run python tests/test_agent.py  (depuis la racine du repo)
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from jarvis.agent import Agent  # noqa: E402
from jarvis.llm import LLMClient  # noqa: E402
from jarvis.mcp import McpToolProvider  # noqa: E402
from mock_llm import start_mock  # noqa: E402

ECHO_SERVER = ROOT / "mcp_servers" / "echo.py"


async def run_case(name: str, user_text: str, expect_in_reply: str,
                   expect_tool: bool) -> bool:
    srv, thread = start_mock(18080)
    try:
        llm = LLMClient("http://127.0.0.1:18080/v1", model="mock")
        tools = McpToolProvider({"echo": [sys.executable, str(ECHO_SERVER)]})
        await tools.start()

        schemas = tools.schemas()
        assert schemas, "aucun outil exposé par le MCP server echo"
        assert schemas[0]["function"]["name"] == "echo__ping"

        agent = Agent(llm, tools, "system de test", max_iterations=4)
        calls: list[str] = []
        reply, _history, trace = await agent.turn(
            [], user_text, on_tool_call=lambda n, a: calls.append(n))

        ok = expect_in_reply in reply
        if expect_tool:
            ok = ok and calls == ["echo__ping"] and trace.iterations >= 2
        else:
            ok = ok and not calls
        status = "OK " if ok else "FAIL"
        print(f"[{status}] {name}: reply={reply!r} tools={calls}")
        await tools.stop()
        await llm.aclose()
        return ok
    finally:
        srv.shutdown()
        thread.join(timeout=5)


async def run_case_failing_tool() -> bool:
    """Un outil qui lève une exception ne doit pas tuer le tour : le résultat
    d'outil contient ERREUR: et le modèle peut conclure normalement."""
    srv, thread = start_mock(18080)
    try:
        llm = LLMClient("http://127.0.0.1:18080/v1", model="mock")
        tools = McpToolProvider({"echo": [sys.executable, str(ECHO_SERVER)]})
        await tools.start()

        async def boom(name, arguments):
            raise RuntimeError("serveur MCP mort")

        tools.call = boom  # type: ignore[method-assign]
        agent = Agent(llm, tools, "system de test", max_iterations=4)
        results: list[tuple] = []
        reply, _history, trace = await agent.turn(
            [], "fais un ping vers 127.0.0.1",
            on_tool_result=lambda n, a, ok, d, r: results.append((n, ok, r)))
        ok = ("ERREUR" in reply and results and results[0][1] is False
              and "RuntimeError" in results[0][2] and trace.iterations >= 2)
        status = "OK " if ok else "FAIL"
        print(f"[{status}] outil en erreur: reply={reply!r} results={results}")
        await tools.stop()
        await llm.aclose()
        return ok
    finally:
        srv.shutdown()
        thread.join(timeout=5)


async def main() -> int:
    results = [
        await run_case("tour avec outil", "fais un ping vers 127.0.0.1",
                       "pong vers 127.0.0.1", expect_tool=True),
        await run_case("tour sans outil", "dis simplement bonjour",
                       "réponse directe", expect_tool=False),
        await run_case_failing_tool(),
    ]
    print(f"\n{sum(results)}/{len(results)} tests OK")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
