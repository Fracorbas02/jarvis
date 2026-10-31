"""Test du pont GTK (jarvis.gtk.backend) sans GTK ni LLM réel.

Valide, avec le mock LLM + MCP echo :
- la séquence d'événements streaming (reasoning -> tool_call -> content -> done)
  via la file d'événements du GtkBackend ;
- la persistance de session (SessionStore) après un tour ;
- les réglages UI (ui.json : thème, ctx_size, gpu) ;
- server_ctx_usage qui ne lève jamais (serveur absent -> n_ctx None) ;
- apply_gpu qui écrit gpu.json + ui.json.

Lancement : uv run python tests/test_gtk_backend.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from jarvis.gtk import backend as gtk_backend  # noqa: E402
from jarvis.gtk.backend import GtkBackend, server_ctx_usage  # noqa: E402
from mock_llm import start_mock  # noqa: E402

ECHO_SERVER = ROOT / "mcp_servers" / "echo.py"


def _drain(backend: GtkBackend, timeout: float = 10.0) -> list[dict]:
    """Lit la file d'événements jusqu'au 'done'/'error' (bloquant, thread test)."""
    import queue as _q
    import time as _t
    events: list[dict] = []
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        try:
            ev = backend.events.get(timeout=0.2)
        except _q.Empty:
            continue
        events.append(ev)
        if ev["type"] in ("done", "error"):
            break
    return events


async def amain() -> int:
    # environnement isolé : XDG_CONFIG_HOME / XDG_CACHE_HOME temporaires
    tmp = Path(tempfile.mkdtemp(prefix="jarvis-app-test-"))
    os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
    os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")

    srv, thread = start_mock(18091)
    try:
        backend = GtkBackend.__new__(GtkBackend)  # contourne le constructeur (thread)
        import queue as _q
        backend.events = _q.Queue()
        backend.config_path = None
        backend.busy = False
        backend._lock = asyncio.Lock()
        from jarvis.config import AgentConfig, Config, LLMConfig
        from jarvis.llm import LLMClient
        from jarvis.mcp import McpToolProvider
        from jarvis.agent import Agent
        from jarvis.compact import compact_tool
        from jarvis.sessions import SessionStore

        cfg = Config(
            llm=LLMConfig(base_url="http://127.0.0.1:18091/v1", model="mock"),
            agent=AgentConfig(system_prompt="test"),
            mcp_servers={"echo": [sys.executable, str(ECHO_SERVER)]},
        )
        backend.cfg = cfg
        backend.llm = LLMClient(**{k: v for k, v in cfg.llm.__dict__.items()
                                   if k in ("base_url", "model", "api_key", "temperature",
                                            "max_tokens", "timeout", "chat_template_kwargs")})
        backend.tools = McpToolProvider(cfg.mcp_servers)
        backend.agent = Agent(backend.llm, backend.tools, cfg.agent.system_prompt,
                              cfg.agent.max_iterations, stream=True,
                              internal_tools={"jarvis__compact": compact_tool})
        backend.history = []
        backend.sessions = SessionStore(gtk_backend.state_dir())
        backend.session_id = None
        backend.models_llm = []

        await backend.tools.start()
        assert backend.tools.names() == ["echo__ping"]

        await backend._astream_turn("fais un ping vers 127.0.0.1")
        events = []
        while not backend.events.empty():
            events.append(backend.events.get_nowait())
        types = [e["type"] for e in events]
        assert "reasoning" in types, types
        assert "tool_call" in types and \
            events[types.index("tool_call")]["name"] == "echo__ping", types
        assert "tool_result" in types, types
        tr = events[types.index("tool_result")]
        assert tr["ok"] and tr["duration"] >= 0 and "pong" in tr["result"], tr
        assert "content" in types, types
        done = [e for e in events if e["type"] == "done"]
        assert done and "pong vers 127.0.0.1" in done[0]["reply"], events
        assert any(m.get("role") == "tool" for m in backend.history)
        assert not backend.busy

        # persistance de session après le tour (fait par _astream_turn)
        assert backend.session_id is not None
        sessions = backend.sessions.list(limit=5)
        assert sessions, "session non persistée"
        saved = backend.sessions.load(backend.session_id)
        assert any(m.get("role") == "tool" for m in saved)

        # nouvelle session -> historique vidé + nouvelle session persistée
        await backend._anew()
        assert backend.history == []
        ev = backend.events.get_nowait()
        assert ev["type"] == "session_new" and ev["history_len"] == 0, ev

        # réglages UI : thème + ctx_size + gpu
        gtk_backend.save_ui_settings({"theme": "dark", "ctx_size": 32768})
        s = gtk_backend.load_ui_settings()
        assert s["theme"] == "dark" and s["ctx_size"] == 32768, s
        result = backend.apply_gpu("cpu")
        assert result["device"] == "cpu"
        gpu_json = json.loads((Path(os.environ["XDG_CONFIG_HOME"]) / "jarvis" / "gpu.json")
                              .read_text(encoding="utf-8"))
        assert gpu_json["device"] == "cpu", gpu_json
        assert gtk_backend.load_ui_settings()["gpu"] == "cpu"

        # parsing de la réponse du portail PermissionStore.Lookup, scopé APP_ID
        import re
        APP_ID = gtk_backend.APP_ID
        def parse_lookup(out: str) -> str:
            m = re.search(rf"['\"]{re.escape(APP_ID)}['\"]:\s*<\[['\"](yes|no)['\"]", out)
            return m.group(1) if m else "unknown"
        assert parse_lookup("(<{'io.github.Fracorbas02.Jarvis': <['yes']>},)") == "yes"
        assert parse_lookup("(<{'io.github.Fracorbas02.Jarvis': <['no']>},)") == "no"
        assert parse_lookup("(<{},>)") == "unknown"
        # réplique exacte du rapport de revue : une autre app ne doit pas l'emporter
        assert parse_lookup(
            "(<{'autre.app': <['yes']>, 'io.github.Fracorbas02.Jarvis': <['no']>},)") == "no"
        assert parse_lookup("(<{'autre.app': <['yes']>},)") == "unknown"
        assert parse_lookup("(<{'io.github.Fracorbas02.JarvisX': <['yes']>},)") == "unknown"
        assert parse_lookup('(<{"io.github.Fracorbas02.Jarvis": <["yes"]>},)') == "yes"

        # flag de 1er lancement : le dialogue d'autorisation ne se montre qu'une fois
        gtk_backend.save_ui_settings({"mic_permission_asked": True})
        assert gtk_backend.load_ui_settings()["mic_permission_asked"] is True

        # server_ctx_usage ne lève jamais, même serveur absent
        usage = await server_ctx_usage("http://127.0.0.1:1/v1")
        assert usage["n_ctx"] is None and usage["used"] == 0, usage
        usage = await server_ctx_usage("http://127.0.0.1:18091/v1")
        assert usage["n_ctx"] is None  # le mock ne sert pas /props ni /slots

        await backend.tools.stop()
        await backend.llm.aclose()
        print("OK  gtk backend:", " -> ".join(types))
        return 0
    finally:
        srv.shutdown()
        thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
