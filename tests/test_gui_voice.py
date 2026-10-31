"""Tests GUI des panneaux voix (TTS/STT) via le plugin pytest de NiceGUI (User).

Vérifie que le bouton « Démarrer » produit un retour visible (notification +
statut), y compris en cas d'échec (binaire absent) — le bug rapporté était
un clic sans aucun feedback ni erreur.
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
os.environ["JARVIS_CONFIG"] = str(ROOT / "tests" / "config.test.toml")

# un faux modèle TTS/STT pour que le sélecteur ait une sélection (sinon le
# bouton Démarrer répond « choisis d'abord un modèle » — comportement voulu,
# mais on veut ici tester le chemin « lancement » avec feedback d'erreur)
import tempfile
_FAKE_MODELS = Path(tempfile.mkdtemp(prefix="jarvis-test-models-"))
(_FAKE_MODELS / "tts").mkdir()
(_FAKE_MODELS / "tts" / "voxtral-4b-tts-q4_k.gguf").write_bytes(b"x" * 64)
(_FAKE_MODELS / "stt").mkdir()
(_FAKE_MODELS / "stt" / "Q4_K_M.gguf").write_bytes(b"x" * 64)
os.environ["JARVIS_MODEL_DIRS"] = str(_FAKE_MODELS)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _wait(user: User, marker: str, tries: int = 20) -> None:
    for _ in range(tries):
        await asyncio.sleep(0.3)
        try:
            await asyncio.wait_for(user.should_see(marker), timeout=1)
            return
        except Exception:
            pass
    await user.should_see(marker)


@pytest.mark.anyio
async def test_voice_panels_render_and_start_button_gives_feedback(user: User):
    await user.open("/")
    await user.should_see("JARVIS")

    # panneau TTS : le bouton Démarrer existe et le serveur est hors ligne
    user.find(kind=ui.button, content="TTS").click()
    await user.should_see("Serveur")
    await _wait(user, "hors ligne")
    await user.should_see("Démarrer")

    # le modèle TTS détecté (faux GGUF) doit être listé et sélectionné
    await _wait(user, "voxtral-4b-tts-q4_k.gguf")

    # clic sur Démarrer : llama-server est absent du sandbox -> on doit voir
    # un feedback explicite (notification + chip avec l'erreur), JAMAIS un silence
    user.find(kind=ui.button, content="Démarrer").click()
    await _wait(user, "Démarrage du serveur TTS", tries=10)
    # puis le statut final avec le détail de l'erreur dans le chip
    await _wait(user, "binaire introuvable", tries=40)

    # panneau STT idem
    user.find(kind=ui.button, content="STT").click()
    await user.should_see("Serveur")
    await user.should_see("Démarrer")
