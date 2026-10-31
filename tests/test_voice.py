"""Tests de la voix locale (STT/TTS) contre les mocks de tests/mock_voice.py."""

import asyncio
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mock_voice
from jarvis.config import VoiceConfig, VoiceServerConfig
from jarvis.voice import (STTClient, TTSClient, host_port, pcm_to_wav_bytes,
                         resolve_model_path, voice_server_status, write_wav)

_srvs: list = []


def _start_mock(port: int, mode: str, tts_fallback: bool = False) -> None:
    srv, _t = mock_voice.start_mock_voice(port, mode, tts_fallback=tts_fallback)
    _srvs.append(srv)


def _stop_mocks() -> None:
    for srv in _srvs:
        srv.shutdown()
        srv.server_close()
    _srvs.clear()


def _run(coro):
    return asyncio.run(coro)


class _Mock:
    """context manager : démarre un mock voix sur un port libre."""

    def __init__(self, mode: str, tts_fallback: bool = False):
        self.mode = mode
        self.tts_fallback = tts_fallback
        self.port = 0

    def __enter__(self):
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        _start_mock(self.port, self.mode, self.tts_fallback)
        return self

    def __exit__(self, *exc):
        _stop_mocks()
        return False


def test_host_port_and_model_path():
    assert host_port("http://127.0.0.1:8178") == ("127.0.0.1", 8178)
    assert host_port("127.0.0.1:8178") == ("127.0.0.1", 8178)
    p = resolve_model_path("stt/Q4_K_M.gguf")
    assert p == Path.home() / "models" / "stt" / "Q4_K_M.gguf"


def test_wav_helpers(tmp_path):
    pcm = b"\x01\x02" * 800
    data = pcm_to_wav_bytes(pcm)
    assert data[:4] == b"RIFF"
    out = write_wav(tmp_path / "x.wav", pcm)
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.readframes(10**6) == pcm


def test_stt_transcribe():
    with _Mock("stt") as m:
        async def go():
            client = STTClient(f"http://127.0.0.1:{m.port}")
            wav = pcm_to_wav_bytes(b"\x00\x00" * 16000)
            tmp = Path("/tmp/jarvis-test-stt.wav")
            tmp.write_bytes(wav)
            text = await client.transcribe(tmp, language="fr")
            await client.aclose()
            return text
        text = _run(go())
        assert "transcription de" in text and "octets audio" in text


def test_tts_synthesize_speech_endpoint():
    with _Mock("tts") as m:
        async def go():
            client = TTSClient(f"http://127.0.0.1:{m.port}/v1")
            wav = await client.synthesize("Bonjour Jarvis", voice="casual_female")
            voices = await client.voices()
            await client.aclose()
            return wav, voices
        wav, voices = _run(go())
        assert wav[:4] == b"RIFF"
        assert voices == ["casual_female", "casual_male"]


def test_tts_fallback_tts_route():
    with _Mock("tts", tts_fallback=True) as m:
        async def go():
            client = TTSClient(f"http://127.0.0.1:{m.port}/v1")
            wav = await client.synthesize("test fallback")
            await client.aclose()
            return wav
        wav = _run(go())
        assert wav[:4] == b"RIFF"


def test_voice_server_status_offline():
    vcfg = VoiceServerConfig(binary="whisper-server",
                             base_url="http://127.0.0.1:59998")
    st = _run(voice_server_status("stt", vcfg, timeout=0.5))
    assert st.state == "offline" and st.port == 59998


def test_voice_server_status_online():
    with _Mock("stt") as m:
        vcfg = VoiceServerConfig(binary="whisper-server",
                                 base_url=f"http://127.0.0.1:{m.port}")
        st = _run(voice_server_status("stt", vcfg))
        assert st.state == "online"


def test_voice_config_defaults():
    cfg = VoiceConfig()
    assert cfg.stt.binary == "whisper-server"
    assert cfg.tts.binary == "llama-server"
    assert cfg.record_cmd and cfg.play_cmd


if __name__ == "__main__":
    test_host_port_and_model_path()
    test_wav_helpers(Path("/tmp"))
    test_stt_transcribe()
    test_tts_synthesize_speech_endpoint()
    test_tts_fallback_tts_route()
    test_voice_server_status_offline()
    test_voice_server_status_online()
    test_voice_config_defaults()
    print("OK")

def test_gpu_choice_persistence(tmp_path, monkeypatch) -> None:
    """save/load du choix GPU en JSON sous XDG_CONFIG_HOME."""
    import jarvis.gpu as gpu
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # devices ggml fictifs (2 GPU) pour choice_args/choice_env
    list_devices = ("  Vulkan0: AMD Radeon RX 9070 XT (RADV GFX1201) (16304 MiB, 16115 MiB free)\n"
                    "  Vulkan1: AMD Radeon 890M Graphics (RADV STRIX1) (48701 MiB, 45878 MiB free)\n")
    monkeypatch.setattr(gpu, "_run",
                        lambda cmd: list_devices if cmd[:2] == ["llama-server", "--list-devices"] else "")
    assert gpu.load_gpu_choice() == {}
    assert gpu.choice_env() == {}
    gpu.save_gpu_choice("hip:1", gpu.env_for_device("hip:1"))
    choice = gpu.load_gpu_choice()
    assert choice["device"] == "hip:1"
    assert choice["env"] == {"HIP_VISIBLE_DEVICES": "1", "ROCR_VISIBLE_DEVICES": "1"}
    assert gpu.choice_env() == {"HIP_VISIBLE_DEVICES": "1", "ROCR_VISIBLE_DEVICES": "1"}
    assert gpu.env_for_device("cuda:0") == {"CUDA_VISIBLE_DEVICES": "0"}
    # "cpu" masque tous les GPU (chaîne vide = aucun device visible)
    assert gpu.env_for_device("cpu") == {"CUDA_VISIBLE_DEVICES": "",
                                          "HIP_VISIBLE_DEVICES": "",
                                          "ROCR_VISIBLE_DEVICES": ""}
    assert gpu.env_for_device("vulkan:0") == {}  # pas d'env pour vulkan…
    assert gpu.env_for_device("Vulkan0") == {}
    assert gpu.env_for_device("CUDA1") == {"CUDA_VISIBLE_DEVICES": "1"}
    # nom ggml passé tel quel en --device (llama.cpp refuse un index brut)
    assert gpu.args_for_device("Vulkan1") == ["--device", "Vulkan1"]
    assert gpu.args_for_device("cuda:1") == []   # cuda/hip : l'env suffit
    assert gpu.args_for_device("CUDA1") == []
    assert gpu.args_for_device("cpu") == []
    gpu.save_gpu_choice("Vulkan1", {})
    assert gpu.choice_args() == ["--device", "Vulkan1"]
    assert gpu.choice_env() == {}
    gpu.save_gpu_choice("cuda:1", gpu.env_for_device("cuda:1"))
    assert gpu.choice_args() == []
    assert gpu.choice_env() == {"CUDA_VISIBLE_DEVICES": "1"}
    # index numérique extrait du device
    assert gpu.device_index("cuda:1") == "1"
    assert gpu.device_index("Vulkan0") == "0"
    assert gpu.device_index("cpu") == "cpu"
    print("OK  gpu choice persistence")


def test_choice_args_fallback_gpu_debranche(monkeypatch, tmp_path) -> None:
    """gpu.json = Vulkan1 mais seul Vulkan0 existe (GPU débranché) -> fallback
    --device Vulkan0, au lieu d'un --device invalide qui empêcherait
    llama-server de démarrer (« invalid device: Vulkan1 »)."""
    import jarvis.gpu as gpu
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    only_apu = "  Vulkan0: AMD Radeon 890M Graphics (RADV STRIX1) (48701 MiB, 46971 MiB free)\n"
    monkeypatch.setattr(gpu, "_run",
                        lambda cmd: only_apu if cmd[:2] == ["llama-server", "--list-devices"] else "")
    gpu.save_gpu_choice("Vulkan1", {})
    assert gpu.choice_args() == ["--device", "Vulkan0"]
    gpu.save_gpu_choice("Vulkan0", {})
    assert gpu.choice_args() == ["--device", "Vulkan0"]  # valide : pas de fallback
    print("OK  choice_args fallback gpu débranché")


def test_choice_env_hors_plage(monkeypatch, tmp_path) -> None:
    """cuda:1 enregistré mais un seul device CUDA existe -> env ignoré
    (CUDA_VISIBLE_DEVICES=1 masquerait tous les GPU)."""
    import jarvis.gpu as gpu
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    one_cuda = "  CUDA0: NVIDIA RTX 4090 (24576 MiB, 24000 MiB free)\n"
    monkeypatch.setattr(gpu, "_run",
                        lambda cmd: one_cuda if cmd[:2] == ["llama-server", "--list-devices"] else "")
    gpu.save_gpu_choice("cuda:1", gpu.env_for_device("cuda:1"))
    assert gpu.choice_env() == {}
    gpu.save_gpu_choice("cuda:0", gpu.env_for_device("cuda:0"))
    assert gpu.choice_env() == {"CUDA_VISIBLE_DEVICES": "0"}
    print("OK  choice_env hors plage")


def test_args_for_device_vulkan_legacy_index(monkeypatch) -> None:
    """Ancien choix "vulkan:N" (index vulkaninfo) -> nom ggml par correspondance
    de nom : l'ordre vulkaninfo diffère de celui de ggml (chez l'utilisateur,
    GPU0=890M iGPU dans vulkaninfo mais Vulkan1 dans llama-server)."""
    import jarvis.gpu as gpu
    list_devices = ("  Vulkan0: AMD Radeon RX 9070 XT (RADV GFX1201) (16304 MiB, 16115 MiB free)\n"
                    "  Vulkan1: AMD Radeon 890M Graphics (RADV STRIX1) (48701 MiB, 45878 MiB free)\n")
    vkinfo = ("GPU0:\n\tdeviceName = AMD Radeon 890M Graphics (RADV STRIX1)\n"
              "GPU1:\n\tdeviceName = AMD Radeon RX 9070 XT (RADV GFX1201)\n")

    def fake_run(cmd):
        if cmd[:2] == ["llama-server", "--list-devices"]:
            return list_devices
        if cmd[:1] == ["vulkaninfo"]:
            return vkinfo
        return ""

    monkeypatch.setattr(gpu, "_run", fake_run)
    assert gpu.args_for_device("vulkan:0") == ["--device", "Vulkan1"]  # 890M
    assert gpu.args_for_device("vulkan:1") == ["--device", "Vulkan0"]  # 9070 XT
    print("OK  args_for_device vulkan legacy")


def test_record_wav_stdout_pcm() -> None:
    """record_cmd produisant du PCM s16le sur stdout ('-' = sortie, conservé :
    ffmpeg exige une sortie explicite sinon « At least one output file must
    be specified ») -> WAV écrit avec un signal non nul."""
    import jarvis.voice as voice
    cfg = VoiceConfig(record_cmd=["ffmpeg", "-loglevel", "error", "-f", "lavfi",
                                  "-i", "sine=frequency=440:duration=30",
                                  "-ar", "16000", "-ac", "1", "-f", "s16le", "-"])
    p = _run(voice.record_wav(1.0, cfg))
    assert p.is_file() and p.stat().st_size > 44 + 1000
    with wave.open(str(p)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        frames = w.readframes(w.getnframes())
    peak = max(abs(int.from_bytes(frames[i:i + 2], "little", signed=True))
               for i in range(0, min(len(frames), 4000), 2))
    assert peak > 100, peak
    print("OK  record_wav stdout pcm")


def test_record_wav_strip_riff() -> None:
    """Si le binaire envoie un WAV (header RIFF) sur stdout, on strippe les
    44 octets d'en-tête avant d'écrire notre propre header WAV."""
    import jarvis.voice as voice
    cfg = VoiceConfig(record_cmd=["ffmpeg", "-loglevel", "error", "-f", "lavfi",
                                  "-i", "sine=frequency=440:duration=30",
                                  "-ar", "16000", "-ac", "1", "-f", "wav", "-"])
    p = _run(voice.record_wav(1.0, cfg))
    with wave.open(str(p)) as w:  # wave lit NOTRE header -> le RIFF a été strip
        assert w.getframerate() == 16000
        frames = w.readframes(4000)
    peak = max(abs(int.from_bytes(frames[i:i + 2], "little", signed=True))
               for i in range(0, len(frames), 2))
    assert peak > 100, peak
    print("OK  record_wav strip RIFF")


def test_detect_gpus_list_devices(monkeypatch) -> None:
    """Source primaire = llama-server --list-devices : les devices proposés
    ont directement le nom ggml utilisable en --device."""
    import jarvis.gpu as gpu
    list_devices = ("  Vulkan0: AMD Radeon RX 9070 XT (RADV GFX1201) (16304 MiB, 16115 MiB free)\n"
                    "  Vulkan1: AMD Radeon 890M Graphics (RADV STRIX1) (48701 MiB, 45878 MiB free)\n")
    monkeypatch.setattr(gpu, "_run",
                        lambda cmd: list_devices if cmd[:2] == ["llama-server", "--list-devices"] else "")
    gpus = gpu.detect_gpus()
    assert [g.device for g in gpus] == ["Vulkan0", "Vulkan1", "cpu"]
    assert all(g.backend == "vulkan" for g in gpus[:2])
    assert "9070" in gpus[0].name and "890M" in gpus[1].name
    print("OK  detect_gpus list-devices")


def test_argv_for_device_index() -> None:
    """Un device explicite "cuda:1" doit devenir --device 1 (index llama.cpp),
    jamais --device cuda:1 (invalide). gpu_args (choix global vulkan) est
    passé tel quel en tête d'extra."""
    from jarvis.voice import _argv_for
    vcfg = VoiceServerConfig(binary="whisper-server",
                             base_url="http://127.0.0.1:8178",
                             device="cuda:1")
    argv = _argv_for("stt", vcfg, Path("/tmp/m.gguf"))
    assert argv[argv.index("--device") + 1] == "1"
    vcfg2 = VoiceServerConfig(binary="whisper-server",
                              base_url="http://127.0.0.1:8178")
    assert "--device" not in _argv_for("stt", vcfg2, Path("/tmp/m.gguf"))
    argv2 = _argv_for("stt", vcfg2, Path("/tmp/m.gguf"), ["--device", "Vulkan1"])
    assert argv2[argv2.index("--device") + 1] == "Vulkan1"
    # device explicite au format nom ggml -> passé tel quel (pas l'index "1")
    vcfg3 = VoiceServerConfig(binary="llama-server",
                              base_url="http://127.0.0.1:9932",
                              device="Vulkan1")
    argv3 = _argv_for("tts", vcfg3, Path("/tmp/m.gguf"))
    assert argv3[argv3.index("--device") + 1] == "Vulkan1"
    print("OK  argv_for device index")


def test_detect_gpus_vulkan_types(monkeypatch) -> None:
    """vulkaninfo --summary parsé par bloc GPU : deviceName + deviceType
    (integrated/discrete) pour distinguer iGPU et dGPU."""
    import jarvis.gpu as gpu
    summary = (
        "Devices:\n========\nGPU0:\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n"
        "\tdeviceName         = AMD Radeon 890M Graphics (RADV STRIX1)\n"
        "GPU1:\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
        "\tdeviceName         = AMD Radeon RX 9070 XT (RADV GFX1201)\n"
    )
    monkeypatch.setattr(gpu, "_run",
                        lambda cmd: summary if cmd[:1] == ["vulkaninfo"] else "")
    gpus = gpu.detect_gpus()
    vulkans = [g for g in gpus if g.backend == "vulkan"]
    assert [g.device for g in vulkans] == ["vulkan:0", "vulkan:1"]
    assert "890M" in vulkans[0].name and "integrated" in vulkans[0].name
    assert "9070" in vulkans[1].name and "discrete" in vulkans[1].name
    assert gpus[-1].backend == "cpu"
    print("OK  detect_gpus vulkan")

