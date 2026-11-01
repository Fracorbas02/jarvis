"""Usage réel des tokens remonté par le serveur (stream_options.include_usage).

- LLMClient capture le champ `usage` en streaming (chunk final) et en non-stream ;
- Agent.turn l'accumule dans TurnTrace (prompt/completion + temps de génération
  seul, hors exécution des outils) ;
- l'événement "done" du GtkBackend expose alors gen_tokens/gen_tps RÉELS
  (tokens_real=True) avec prompt_tokens — et retombe sur l'estimation
  ~4 car/tok si le serveur ne renvoie pas d'usage.

Lancement : uv run pytest tests/test_llm_usage.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from mock_llm import USAGE, start_mock  # noqa: E402
from jarvis.llm import LLMClient  # noqa: E402
from jarvis.agent import Agent  # noqa: E402


def test_llm_captures_usage_stream_and_json() -> None:
    async def run() -> None:
        srv, thread = start_mock(18095)
        try:
            llm = LLMClient("http://127.0.0.1:18095/v1", model="mock")
            msg = await llm.chat([{"role": "user", "content": "hello"}], stream=False)
            assert msg.get("usage", {}).get("completion_tokens") == USAGE["completion_tokens"]
            assert msg.get("usage", {}).get("prompt_tokens") == USAGE["prompt_tokens"]
            chunks: list[str] = []
            msg = await llm.chat([{"role": "user", "content": "hello"}], stream=True,
                                 on_content=lambda c: chunks.append(c))
            assert msg.get("usage", {}).get("completion_tokens") == USAGE["completion_tokens"]
            assert "".join(chunks), "contenu streamé attendu"
            await llm.aclose()
        finally:
            srv.shutdown()
            thread.join(timeout=5)
    asyncio.run(run())


def test_agent_trace_accumulates_usage_and_gen_seconds() -> None:
    async def run() -> None:
        srv, thread = start_mock(18096)
        try:
            llm = LLMClient("http://127.0.0.1:18096/v1", model="mock")
            agent = Agent(llm, None, "system de test", max_iterations=2, stream=True)
            reply, _history, trace = await agent.turn([], "hello")
            assert reply
            assert trace.completion_tokens == USAGE["completion_tokens"], trace
            assert trace.prompt_tokens == USAGE["prompt_tokens"], trace
            assert trace.gen_seconds > 0, trace
            await llm.aclose()
        finally:
            srv.shutdown()
            thread.join(timeout=5)
    asyncio.run(run())


def test_backend_done_event_has_real_tokens() -> None:
    async def run() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="jarvis-usage-test-"))
        os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
        os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")
        srv, thread = start_mock(18097)
        try:
            import queue as _q
            from jarvis.gtk import backend as gtk_backend
            from jarvis.gtk.backend import GtkBackend
            from jarvis.config import AgentConfig, Config, LLMConfig
            from jarvis.mcp import McpToolProvider
            from jarvis.sessions import SessionStore

            backend = GtkBackend.__new__(GtkBackend)  # contourne le constructeur (thread)
            backend.events = _q.Queue()
            backend.config_path = None
            backend.busy = False
            backend._lock = asyncio.Lock()
            cfg = Config(llm=LLMConfig(base_url="http://127.0.0.1:18097/v1", model="mock"),
                         agent=AgentConfig(system_prompt="test"), mcp_servers={})
            backend.cfg = cfg
            backend.llm = LLMClient("http://127.0.0.1:18097/v1", model="mock")
            backend.tools = McpToolProvider({})
            backend.agent = Agent(backend.llm, backend.tools, "test", stream=True)
            backend.history = []
            backend.sessions = SessionStore(gtk_backend.state_dir())
            backend.session_id = None

            await backend._astream_turn("hello")
            events = []
            while not backend.events.empty():
                events.append(backend.events.get_nowait())
            done = [e for e in events if e["type"] == "done"]
            assert done, events
            d = done[0]
            assert d["tokens_real"] is True
            assert d["gen_tokens"] == USAGE["completion_tokens"], d
            assert d["prompt_tokens"] == USAGE["prompt_tokens"], d
            assert d["gen_tps"] and d["gen_tps"] > 0, d
            assert d["gen_seconds"] and d["gen_seconds"] > 0, d
            assert d["trace"]["completion_tokens"] == USAGE["completion_tokens"], d
        finally:
            srv.shutdown()
            thread.join(timeout=5)
    asyncio.run(run())
