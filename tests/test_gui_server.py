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


def test_start_server_passe_env_gpu(monkeypatch, tmp_path):
    """Le choix GPU global (onglet Réglages) doit être passé en env au Popen
    du llama-server principal lancé par l'app (jamais de --device brut)."""
    import jarvis.gpu as gpu
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    gpu.save_gpu_choice("cuda:1", gpu.env_for_device("cuda:1"))
    captured = {}

    class FakeProc:
        def poll(self):
            return None

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["env"] = kw.get("env")
        return FakeProc()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_log_path", tmp_path / "llama-server-gui.log")
    try:
        st = asyncio.run(server.start_server("/tmp/m.gguf", "http://127.0.0.1:59998/v1",
                                             ready_timeout=0.05))
    finally:
        server._proc = None
    assert st.state == "managed"
    assert "--device" not in captured["argv"]
    assert captured["env"] is not None
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_start_server_passe_device_vulkan(monkeypatch, tmp_path):
    """Choix vulkan (nom ggml "Vulkan1") -> --device Vulkan1 dans l'argv
    (llama.cpp refuse un index brut "0" : "invalid device"), pas d'env."""
    import jarvis.gpu as gpu
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    gpu.save_gpu_choice("Vulkan1", {})
    captured = {}

    class FakeProc:
        def poll(self):
            return None

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["env"] = kw.get("env")
        return FakeProc()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_log_path", tmp_path / "llama-server-gui.log")
    try:
        st = asyncio.run(server.start_server("/tmp/m.gguf", "http://127.0.0.1:59998/v1",
                                             ready_timeout=0.05))
    finally:
        server._proc = None
    assert st.state == "managed"
    assert captured["argv"][captured["argv"].index("--device") + 1] == "Vulkan1"
    assert captured["env"] is None  # aucun env pour vulkan


if __name__ == "__main__":
    test_host_port()
    test_status_offline()
    test_status_live_server()
    print("OK")
