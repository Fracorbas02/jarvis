"""Régression gestion des conversations (sessions) côté backend GTK.

- une session vide n'est plus persistée (plus d'entrée "(vide)" dans la
  sidebar après Ctrl+N ou suppression de la conversation courante) ;
- la liste émise à la sidebar filtre les sessions vides (count=0) ;
- load/new/delete sont sérialisés avec un tour en cours (lock + busy) :
  un chargement refusé pendant un tour émet "error" et ne touche pas
  l'historique — évite qu'un prompt parti pendant le chargement soit
  enregistré dans la mauvaise conversation.

Lancement : uv run pytest tests/test_gtk_sessions.py -v
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jarvis.config import AgentConfig, Config, LLMConfig  # noqa: E402
from jarvis.gtk.backend import GtkBackend  # noqa: E402
from jarvis.llm import LLMClient  # noqa: E402
from jarvis.mcp import McpToolProvider  # noqa: E402
from jarvis.agent import Agent  # noqa: E402
from jarvis.sessions import SessionStore  # noqa: E402


def _make_backend() -> GtkBackend:
    import queue as _q
    backend = GtkBackend.__new__(GtkBackend)  # contourne le constructeur (thread)
    backend.events = _q.Queue()
    backend.config_path = None
    backend.busy = False
    backend._lock = asyncio.Lock()
    cfg = Config(llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="mock"),
                 agent=AgentConfig(system_prompt="test"), mcp_servers={})
    backend.cfg = cfg
    backend.llm = LLMClient("http://127.0.0.1:1/v1", model="mock")
    backend.tools = McpToolProvider({})
    backend.agent = Agent(backend.llm, backend.tools, "test", stream=True)
    backend.history = []
    backend.sessions = SessionStore(Path(tempfile.mkdtemp(prefix="jarvis-sess-")))
    backend.session_id = None
    return backend


def _drain(backend: GtkBackend) -> list[dict]:
    out = []
    while not backend.events.empty():
        out.append(backend.events.get_nowait())
    return out


def test_empty_session_not_saved_no_vide_entry() -> None:
    async def run() -> None:
        b = _make_backend()
        await b._anew()
        evs = _drain(b)
        assert evs[0]["type"] == "session_new"
        assert b.session_id is not None
        # aucune session persistée tant qu'il n'y a pas de message
        assert b.sessions.list() == []
        assert not b.sessions._path(b.session_id).exists()
        # une vraie session avec contenu est persistée, et listée
        b.history = [{"role": "user", "content": "hello"}]
        b._save_session()
        listed = b.sessions.list()
        assert len(listed) == 1 and listed[0]["count"] == 1
    asyncio.run(run())


def test_listed_sessions_filters_empty() -> None:
    b = _make_backend()
    b.sessions.save("empty-1", [])  # vieux fichier "(vide)"
    b.sessions.save("full-1", [{"role": "user", "content": "hi"}])
    listed = b._listed_sessions()
    assert [s["id"] for s in listed] == ["full-1"], listed


def test_delete_current_removes_file_without_vide_remains() -> None:
    async def run() -> None:
        b = _make_backend()
        b.session_id = "cur-1"
        b.history = [{"role": "user", "content": "hello"}]
        b._save_session()
        old = b.session_id
        await b._adelete(old)
        evs = _drain(b)
        types = [e["type"] for e in evs]
        assert types == ["session_new", "session_deleted", "sessions_list"], types
        assert evs[1]["ok"] is True
        assert not b.sessions._path(old).exists()
        assert b.session_id != old and b.history == []
        # pas de nouvelle entrée "(vide)" : ni fichier, ni liste
        assert b.sessions.list() == []
        assert evs[2]["sessions"] == []
        # supprimer une session inconnue -> ok=False, pas de reset
        await b._adelete("nope-123")
        evs = _drain(b)
        assert evs[0]["type"] == "session_deleted" and evs[0]["ok"] is False
    asyncio.run(run())


def test_load_refused_during_turn() -> None:
    async def run() -> None:
        b = _make_backend()
        b.session_id = "s1"
        b.history = [{"role": "user", "content": "old"}]
        b.sessions.save("s1", b.history)
        b.sessions.save("s2", [{"role": "user", "content": "other"}])
        b.busy = True  # un tour est en cours
        await b._aload("s2")
        evs = _drain(b)
        assert evs[0]["type"] == "error", evs
        assert b.session_id == "s1" and b.history[0]["content"] == "old"
        # une fois le tour fini, le chargement passe
        b.busy = False
        await b._aload("s2")
        evs = _drain(b)
        assert evs[0]["type"] == "session_loaded" and evs[0]["session_id"] == "s2"
        assert b.history[0]["content"] == "other"
    asyncio.run(run())


def test_new_refused_during_turn() -> None:
    async def run() -> None:
        b = _make_backend()
        b.history = [{"role": "user", "content": "old"}]
        b.busy = True
        await b._anew()
        evs = _drain(b)
        assert evs[0]["type"] == "error", evs
        assert b.history == [{"role": "user", "content": "old"}]
    asyncio.run(run())
