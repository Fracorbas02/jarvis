"""Tests du module de statut llama-server (contre le vrai serveur s'il tourne)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jarvis.gui import server


def test_host_port():
    assert server.host_port("http://127.0.0.1:9931/v1") == ("127.0.0.1", 9931)
    assert server.host_port("http://localhost:8080/v1") == ("localhost", 8080)
    assert server.host_port("127.0.0.1:1234/v1") == ("127.0.0.1", 1234)


def test_status_offline():
    st = asyncio.run(server.get_status("http://127.0.0.1:59999/v1", timeout=1.0))
    assert st.state == "offline"
    assert st.port == 59999


def test_status_live_server():
    # le serveur de dev tourne sur 9931 avec le Qwen3.6 chargé
    st = asyncio.run(server.get_status("http://127.0.0.1:9931/v1"))
    assert st.state in ("online", "managed")
    assert st.models, "le serveur live doit servir au moins un modèle"
    assert any("Qwen" in m for m in st.models), st.models


if __name__ == "__main__":
    test_host_port()
    test_status_offline()
    test_status_live_server()
    print("OK")
