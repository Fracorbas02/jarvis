"""Main file pour les tests d'intégration NiceGUI (fixture User).

Importé dans un sous-processus par le plugin de test : doit appeler ui.run()
au niveau module (sans main guard).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JARVIS_CONFIG", str(ROOT / "tests" / "config.test.toml"))

from nicegui import app, ui  # noqa: E402

from jarvis.gui.app import build_page  # noqa: E402
from jarvis.gui.backend import GuiBackend  # noqa: E402
from jarvis.gui.models import discover_models  # noqa: E402
from jarvis.gui.state import GuiState  # noqa: E402

state = GuiState()
state.models_by_mode = discover_models()


async def _startup() -> None:
    state.backend = GuiBackend()
    await state.backend.start()


async def _shutdown() -> None:
    if state.backend is not None:
        await state.backend.stop()


app.on_startup(_startup)
app.on_shutdown(_shutdown)


@ui.page("/")
def index() -> None:
    build_page(state)


ui.run(title="Jarvis (test)", host="127.0.0.1", port=8321, dark=True,
       reload=False, show=False)
