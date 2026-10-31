"""Tests d'intégration de la page GUI via le plugin pytest de NiceGUI (User, sans navigateur).

La page servie est tests/gui_main.py (défini comme main_file dans pyproject).

Usage: uv run pytest tests/test_gui_page.py -v
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
# config allégée (un seul MCP server) pour des tests GUI rapides
os.environ["JARVIS_CONFIG"] = str(ROOT / "tests" / "config.test.toml")

from jarvis.gui.backend import GuiBackend  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _tool_count() -> int:
    backend = GuiBackend()
    await backend.start()
    try:
        return len(backend.tool_names())
    finally:
        await backend.stop()


async def _wait_ready(user: User, marker: str, tries: int = 20) -> None:
    for _ in range(tries):
        await asyncio.sleep(0.5)
        try:
            await asyncio.wait_for(user.should_see(marker), timeout=1)
            return
        except Exception:
            pass
    await user.should_see(marker)


async def _wait_markdown(user: User, text: str, tries: int = 40) -> None:
    """Attend qu'un élément ui.markdown contienne le texte (rendu client-side,
    donc invisible de should_see : on inspecte la prop content)."""
    for _ in range(tries):
        await asyncio.sleep(0.5)
        for el in user.find(ui.markdown).elements:
            if text in (el.content or ""):
                return
    raise AssertionError(f"markdown contenant {text!r} introuvable")


@pytest.mark.anyio
async def test_page_renders(user: User):
    await user.open("/")
    # titre + modes
    await user.should_see("JARVIS")
    await user.should_see("Chat LLM")
    await user.should_see("TTS")
    await user.should_see("STT")
    # panneau serveur : le serveur de dev tourne sur 9931
    await user.should_see("127.0.0.1:9931")
    await user.should_see("SERVEUR LLM")
    # outils MCP chargés depuis config.toml (le backend démarre en on_startup)
    n = await _tool_count()
    await _wait_ready(user, f"Outils MCP ({n})")
    # zone principale : contenu du mode courant (chat LLM)
    await user.should_see("Discute avec Jarvis")


@pytest.mark.anyio
async def test_tools_expansion_and_mode_switch(user: User):
    await user.open("/")
    n = await _tool_count()
    await _wait_ready(user, f"Outils MCP ({n})")
    # ouvrir l'expansion -> les outils apparaissent
    user.find(f"Outils MCP ({n})").click()
    await user.should_see("echo__ping")
    # switch de mode via les boutons segmentés
    user.find(kind=ui.button, content="TTS").click()
    await user.should_see("Synthèse vocale locale")
    user.find(kind=ui.button, content="STT").click()
    await user.should_see("Reconnaissance vocale locale")
    user.find(kind=ui.button, content="Chat LLM").click()
    await user.should_see("Discute avec Jarvis")
