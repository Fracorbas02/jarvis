"""Test d'intégration du chat GUI contre le mock LLM (streaming complet).

Le mock llama-server tourne dans le processus de test ; la page servie est
tests/gui_main.py (main_file) et lit JARVIS_CONFIG au démarrage du fixture
user : on le force donc via un fixture autouse AVANT le fixture user.

Usage: uv run pytest tests/test_gui_chat.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from nicegui import ui
from nicegui.testing import User

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from mock_llm import start_mock  # noqa: E402

CHATMOCK_CONFIG = ROOT / "tests" / "config.chatmock.toml"
MOCK_PORT = 18082


@pytest.fixture(autouse=True)
def _chatmock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_CONFIG", str(CHATMOCK_CONFIG))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _wait_ready(user: User, marker: str, tries: int = 40) -> None:
    for _ in range(tries):
        await asyncio.sleep(0.5)
        try:
            await asyncio.wait_for(user.should_see(marker), timeout=1)
            return
        except Exception:
            pass
    await user.should_see(marker)


async def _wait_markdown(user: User, text: str, tries: int = 40) -> None:
    """Attend qu'un ui.markdown contienne le texte (prop content, pas should_see)."""
    for _ in range(tries):
        await asyncio.sleep(0.5)
        for el in user.find(ui.markdown).elements:
            if text in (el.content or ""):
                return
    raise AssertionError(f"markdown contenant {text!r} introuvable")


@pytest.mark.anyio
async def test_chat_streaming_with_mock_llm(user: User):
    srv, thread = start_mock(MOCK_PORT)
    try:
        await user.open("/")
        await _wait_ready(user, "Outils MCP (1)")
        await user.should_see("Discute avec Jarvis")
        # taper un message et envoyer avec Entrée
        user.find(kind=ui.input).type("fais un ping vers 127.0.0.1")
        user.find(kind=ui.input).trigger("keydown.enter")
        # le raisonnement du mock s'affiche dans l'expansion "Réflexion"
        await _wait_ready(user, "Réflexion")
        await _wait_markdown(user, "j'appelle echo__ping")
        # l'appel d'outil apparaît en chip
        await _wait_ready(user, "echo__ping")
        # la réponse finale streamée
        await _wait_markdown(user, "Résultat du ping")
    finally:
        srv.shutdown()
        thread.join(timeout=5)
