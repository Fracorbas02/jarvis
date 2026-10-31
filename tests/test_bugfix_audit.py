"""Tests de r\u00e9gression pour l'audit (bugs trouv\u00e9s et corrig\u00e9s).

Couvre en particulier la ligne AMD (CPU/GPU) et la robustesse multi-config :
- jarvis.gpu : device_index, env_for_device, parsing lspci/rocminfo sans crash ;
- jarvis.voice : --device avec index (pas 'hip:0'), record_wav qui s'arr\u00eatre
  proprement apr\u00e8s N s, host_port, transcribe_bytes (uuid + nettoyage) ;
- jarvis.agent : arguments JSON invalides -> pas de NameError ;
- jarvis.gui.server : parsing /v1/models au format OpenAI {"data": [...]} ;
- mcp_servers/sysinfo : _amdgpu_sysfs sur un faux /sys, gpu_status/temperatures
  ne l\u00e8vent jamais ;
- mcp_servers/nettools : la regex du certificat TLS compile et matche.

Usage: uv run pytest tests/test_bugfix_audit.py -v
"""

from __future__ import annotations

import asyncio
import re
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jarvis.gpu as gpu
from jarvis.agent import Agent
from jarvis.config import VoiceConfig, VoiceServerConfig
from jarvis.voice import _argv_for, host_port, record_wav, transcribe_bytes


# ---------- gpu.py ----------

def test_device_index():
    # signature main: str (jamais None) ; "Vulkan1" -> "1"
    assert gpu.device_index("hip:1") == "1"
    assert gpu.device_index("cuda:2") == "2"
    assert gpu.device_index("Vulkan1") == "1"
    assert gpu.device_index("vulkan:1") == "1"




def test_env_for_device_amd():
    env = gpu.env_for_device("hip:1")
    assert env == {"HIP_VISIBLE_DEVICES": "1", "ROCR_VISIBLE_DEVICES": "1"}
    assert gpu.env_for_device("cuda:0") == {"CUDA_VISIBLE_DEVICES": "0"}
    assert gpu.env_for_device("vulkan:1") == {}
    assert gpu.env_for_device("Vulkan1") == {}   # vulkan -> --device, pas env
    assert gpu.env_for_device("CUDA1") == {"CUDA_VISIBLE_DEVICES": "1"}
    # cpu masque tous les GPU (chaîne vide = aucun device visible)
    assert gpu.env_for_device("cpu") == {"CUDA_VISIBLE_DEVICES": "",
                                         "HIP_VISIBLE_DEVICES": "",
                                         "ROCR_VISIBLE_DEVICES": ""}




def test_args_for_device_nom_ggml():
    """args_for_device('Vulkan1') -> ['--device', 'Vulkan1'] (jamais un index)."""
    assert gpu.args_for_device("Vulkan1") == ["--device", "Vulkan1"]
    assert gpu.args_for_device("CUDA0") == []      # cuda -> env seule
    assert gpu.args_for_device("hip:0") == []      # hip -> env seule


def test_detect_gpus_lspci_sans_deux_points(monkeypatch):
    """lspci peut renvoyer des lignes VGA/3D sans ': ' apr\u00e8s l'adresse PCI :
    detect_gpus ne doit pas lever d'IndexError."""
    monkeypatch.setattr(gpu, "_run", lambda cmd, timeout=10.0: (
        "00:00.0 Host bridge: Intel\n"
        "01:00.0 VGA\n"
        "02:00.0 3D controller: AMD Radeon 890M\n" if cmd == ["lspci"] else ""
    ))
    gpus = gpu.detect_gpus()
    assert gpus[-1].backend == "cpu"
    amd = [g for g in gpus if "890M" in g.name]
    assert amd and amd[0].backend == "hip" and amd[0].device == "hip:0"


def test_detect_gpus_rocminfo_multi_cartes(monkeypatch):
    """rocminfo : un bloc par carte (Marketing Name), pas un gfx global."""
    rocminfo = (
        "Device #0\n  Marketing Name: AMD Radeon RX 7900 XTX\n  gfx1100\nAgent 1\nAgent 2\n"
        "Device #1\n  Marketing Name: AMD Radeon 890M\n  gfx1151\n"
    )
    monkeypatch.setattr(gpu, "_run", lambda cmd, timeout=10.0: rocminfo if cmd == ["rocminfo"] else "")
    gpus = [g for g in gpu.detect_gpus() if g.backend == "hip"]
    assert len(gpus) == 2
    assert "7900 XTX" in gpus[0].name and "890M" in gpus[1].name
    assert gpus[0].device == "hip:0" and gpus[1].device == "hip:1"


def test_detect_gpus_fallback_cpu(monkeypatch):
    monkeypatch.setattr(gpu, "_run", lambda cmd, timeout=10.0: "")
    gpus = gpu.detect_gpus()
    assert len(gpus) == 1 and gpus[0].device == "cpu"


# ---------- voice.py ----------

def test_argv_for_device_nom_ggml_pas_prefixe():
    """--device recoit le NOM ggml tel quel ('Vulkan1'), ou l'index pour
    l'ancien format ('cuda:1' -> '1') — jamais 'cuda:1'/'hip:0' brut."""
    vcfg = VoiceServerConfig(binary="llama-server",
                             base_url="http://127.0.0.1:9932", device="Vulkan1")
    argv = _argv_for("tts", vcfg, Path("/tmp/m.gguf"))
    assert argv[argv.index("--device") + 1] == "Vulkan1"
    vcfg2 = VoiceServerConfig(binary="whisper-server",
                              base_url="http://127.0.0.1:8178", device="cuda:1")
    argv2 = _argv_for("stt", vcfg2, Path("/tmp/m.gguf"))
    assert argv2[argv2.index("--device") + 1] == "1"




def test_host_port_defaut():
    assert host_port("http://127.0.0.1") == ("127.0.0.1", 80)
    assert host_port("127.0.0.1:8178") == ("127.0.0.1", 8178)


def test_record_wav_s_arrete_apres_n_secondes(tmp_path):
    """Un 'micro' infini (cat) doit \u00eatre stopp\u00e9 apr\u00e8s `seconds`, pas apr\u00e8s
    le timeout (seconds+10) : le WAV produit est valide et le test est rapide."""
    import time
    cfg = VoiceConfig()
    # flux infini de PCM (0x0001 répété) sur stdout, comme arecord
    cfg.record_cmd = ["sh", "-c", "while true; do printf '\\x00\\x01'; done", "-"]
    out = tmp_path / "rec.wav"

    async def go():
        t0 = time.monotonic()
        p = await record_wav(1.0, cfg, out_path=out)
        return p, time.monotonic() - t0

    p, dt = asyncio.run(go())
    assert p == out and out.is_file()
    assert dt < 8.0, f"record_wav a attendu le timeout au lieu de couper \u00e0 1s ({dt:.1f}s)"
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)


def test_transcribe_bytes_nettoie_tmp(tmp_path, monkeypatch):
    """L'upload .wav ne doit pas n\u00e9cessiter ffmpeg et le tmp est supprim\u00e9."""
    import mock_voice
    srv, _t = mock_voice.start_mock_voice(0, "stt")
    port = srv.server_address[1]
    try:
        vcfg = VoiceConfig()
        vcfg.stt.base_url = f"http://127.0.0.1:{port}"
        import jarvis.voice as v
        wav = v.pcm_to_wav_bytes(b"\x00\x00" * 16000)
        before = set(Path("/tmp").glob("jarvis-stt-*"))

        async def go():
            return await transcribe_bytes(vcfg, wav, filename="upload.wav")

        text = asyncio.run(go())
        assert "octets audio" in text
        after = set(Path("/tmp").glob("jarvis-stt-*"))
        assert after == before, f"fichiers temporaires laiss\u00e9s : {after - before}"
    finally:
        srv.shutdown()
        srv.server_close()


# ---------- agent.py ----------

def test_agent_args_json_invalides_pas_de_nameerror():
    """Un tool_call avec arguments invalides ne doit pas lever NameError
    (args doit exister m\u00eame dans le branche except)."""

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tools=None, stream=False,
                       on_reasoning=None, on_content=None):
            self.calls += 1
            if self.calls == 1:
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "x__y", "arguments": "{invalide"}}]}
            return {"role": "assistant", "content": "ok apr\u00e8s erreur"}

    class FakeTools:
        def schemas(self):
            return []

        async def call(self, name, args):
            return "unused"

    agent = Agent(FakeLLM(), FakeTools(), "sys", max_iterations=3, stream=False)
    reply, history, trace = asyncio.run(agent.turn([], "test"))
    assert reply == "ok apr\u00e8s erreur"
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("ERREUR: arguments JSON invalides")
    assert len(trace.tool_calls) == 1  # pas de crash sur json.dumps(args)


# ---------- gui/server.py ----------

def test_get_status_parse_openai_data():
    """GET /v1/models renvoie {"object":"list","data":[...]} : la GUI doit lire
    data (et pas models) pour conna\u00eetre le mod\u00e8le servi."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json as _json
    import threading

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                body = _json.dumps({"object": "list",
                                    "data": [{"id": "qwen3.6-35b-a3b", "object": "model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        from jarvis.gui import server as gui_server
        gui_server._proc = None
        st = asyncio.run(gui_server.get_status(f"http://127.0.0.1:{port}/v1"))
        assert st.state == "online"
        assert st.models == ["qwen3.6-35b-a3b"], st.models
    finally:
        srv.shutdown()
        srv.server_close()


# ---------- sysinfo.py ----------

def test_amdgpu_sysfs_faux_sys(tmp_path):
    """_amdgpu_sysfs lit un faux /sys/class/drm avec une carte amdgpu."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sysinfo_srv", Path(__file__).parent.parent / "mcp_servers" / "sysinfo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    drm = tmp_path / "drm"
    card = drm / "card0"
    dev = card / "device"
    (dev / "hwmon" / "hwmon1").mkdir(parents=True)
    (dev / "product_name").write_text("AMD Radeon 890M\n")
    (dev / "gpu_busy_percent").write_text("7\n")
    (dev / "mem_info_vram_total").write_text(str(16 * 1024**3))
    (dev / "mem_info_vram_used").write_text(str(2 * 1024**3))
    (dev / "hwmon" / "hwmon1" / "temp1_input").write_text("45000\n")

    out = mod._amdgpu_sysfs(drm)
    assert "AMD Radeon 890M" in out
    assert "busy 7%" in out
    assert "2048/16384 Mo" in out
    assert "temp 45" in out


def test_sysinfo_tools_ne_levent_pas():
    """gpu_status / temperatures r\u00e9pondent toujours (donn\u00e9es ou ERREUR propre)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sysinfo_srv2", Path(__file__).parent.parent / "mcp_servers" / "sysinfo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for res in (mod.gpu_status(), mod.temperatures(), mod.load(), mod.processes(limit=3)):
        assert isinstance(res, str) and res.strip(), res


# ---------- nettools.py ----------

def test_tls_cert_regex_compile():
    """La regex d'extraction du certificat PEM doit compiler et matcher."""
    pat = re.compile(r"-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----", re.S)
    cert = ("-----BEGIN CERTIFICATE-----\nMIIBszCCAVmgAwIBAgIU\n-----END CERTIFICATE-----\n"
            "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n")
    m = pat.search(cert)
    assert m and "MIIBsz" in m.group(1)


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v) and "monkeypatch" not in inspect.signature(v).parameters
           and "tmp_path" not in inspect.signature(v).parameters]
    for fn in fns:
        fn()
        print("OK ", fn.__name__)
    print("all OK")
