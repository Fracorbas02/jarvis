"""Tests du lanceur llama-server (jarvis.server) : le choix GPU global
(gpu.json) doit être appliqué en --device <nom ggml> / env au Popen."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jarvis import server  # noqa: E402


def test_start_server_applique_device_gpu(monkeypatch, tmp_path):
    """gpu.json = Vulkan0 -> --device Vulkan0 dans l'argv (jamais d'index brut
    ni de 'vulkan:0'), et fallback si le device enregistré n'existe plus."""
    import jarvis.gpu as gpu
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        gpu, "_run",
        lambda cmd: ("  Vulkan0: AMD Radeon 890M (48701 MiB, 46000 MiB free)\n"
                     if cmd[:2] == ["llama-server", "--list-devices"] else ""))
    gpu.save_gpu_choice("Vulkan0", {})
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
    assert captured["argv"][captured["argv"].index("--device") + 1] == "Vulkan0"

    # device enregistré absent des devices actuels (GPU débranché) -> fallback
    gpu.save_gpu_choice("Vulkan9", {})
    try:
        asyncio.run(server.start_server("/tmp/m.gguf", "http://127.0.0.1:59998/v1",
                                        ready_timeout=0.05))
    finally:
        server._proc = None
    assert captured["argv"][captured["argv"].index("--device") + 1] == "Vulkan0"
    print("OK  start_server applique le choix GPU (avec fallback)")
