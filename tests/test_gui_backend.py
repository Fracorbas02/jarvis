"""Test du pont GUI (GuiBackend) : mock LLM + MCP echo, flux d'événements streaming.

Valide sans modèle réel que la GUI recevra bien la séquence :
reasoning -> tool_call -> reasoning -> content -> done.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from jarvis.config import Config, LLMConfig, AgentConfig  # noqa: E402
from jarvis.gui.backend import GuiBackend  # noqa: E402
from mock_llm import start_mock  # noqa: E402

ECHO_SERVER = ROOT / "mcp_servers" / "echo.py"


async def main() -> int:
    srv, thread = start_mock(18081)
    try:
        backend = GuiBackend.__new__(GuiBackend)  # contourne load_config
        cfg = Config(
            llm=LLMConfig(base_url="http://127.0.0.1:18081/v1", model="mock"),
            agent=AgentConfig(system_prompt="test"),
            mcp_servers={"echo": [sys.executable, str(ECHO_SERVER)]},
        )
        backend.cfg = cfg
        from jarvis.llm import LLMClient
        from jarvis.mcp import McpToolProvider
        from jarvis.agent import Agent
        backend.llm = LLMClient(**cfg.llm.__dict__)
        backend.tools = McpToolProvider(cfg.mcp_servers)
        backend.agent = Agent(backend.llm, backend.tools, cfg.agent.system_prompt,
                              cfg.agent.max_iterations, stream=True)
        backend.history = []
        backend.busy = False
        backend._lock = asyncio.Lock()
        await backend.start()

        assert backend.tool_names() == ["echo__ping"]

        events: list[dict] = []
        await backend.stream_turn("fais un ping vers 127.0.0.1", events.append)

        types = [e["type"] for e in events]
        assert "reasoning" in types, types
        assert "tool_call" in types, types
        assert events[types.index("tool_call")]["name"] == "echo__ping"
        assert "content" in types, types
        done = [e for e in events if e["type"] == "done"]
        assert done and "pong vers 127.0.0.1" in done[0]["reply"], events
        assert not backend.busy
        # l'historique a bien été alimenté (system + user + assistant + tool + assistant)
        assert any(m.get("role") == "tool" for m in backend.history)

        backend.reset()
        assert backend.history == []

        # second tour : réutilise l'historique sans erreur
        events.clear()
        await backend.stream_turn("bonjour", events.append)
        assert [e for e in events if e["type"] == "done"]

        await backend.stop()
        print("OK  séquence:", " -> ".join(types))
        return 0
    finally:
        srv.shutdown()
        thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
