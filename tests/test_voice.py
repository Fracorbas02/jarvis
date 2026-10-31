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


def test_start_voice_server_erreurs():
    """Les 3 échecs de start_voice_server doivent retourner un statut offline
    avec un detail explicite (jamais d'exception, jamais de blocage)."""
    from jarvis.voice import start_voice_server
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    # 1. binaire introuvable
    vcfg = VoiceServerConfig(binary="binaire-qui-nexiste-pas-xyz",
                             base_url=base, model="~/models/stt/x.gguf")
    st = _run(start_voice_server("stt", vcfg))
    assert st.state == "offline" and "binaire introuvable" in st.detail

    # 2. modèle introuvable (binaire = /bin/true qui existe)
    vcfg = VoiceServerConfig(binary="/bin/true", base_url=base,
                             model="/chemin/inexistant/gguf.gguf")
    st = _run(start_voice_server("stt", vcfg))
    assert st.state == "offline" and "modèle introuvable" in st.detail

    # 3. processus qui meurt immédiatement (/bin/false), ready_timeout court
    vcfg = VoiceServerConfig(binary="/bin/false", base_url=base,
                             model=str(Path(__file__).resolve()), ready_timeout=5.0)
    st = _run(start_voice_server("stt", vcfg))
    assert st.state == "offline" and "mort (code" in st.detail, st


def test_start_voice_server_deja_en_ligne():
    """Si /health répond déjà, start_voice_server ne relance rien."""
    from jarvis.voice import start_voice_server
    with _Mock("stt") as m:
        vcfg = VoiceServerConfig(binary="/bin/false",
                                 base_url=f"http://127.0.0.1:{m.port}",
                                 model=str(Path(__file__).resolve()))
        st = _run(start_voice_server("stt", vcfg))
        assert st.state == "online"


if __name__ == "__main__":
    test_host_port_and_model_path()
    test_wav_helpers(Path("/tmp"))
    test_stt_transcribe()
    test_tts_synthesize_speech_endpoint()
    test_tts_fallback_tts_route()
    test_voice_server_status_offline()
    test_voice_server_status_online()
    test_voice_config_defaults()
    test_start_voice_server_erreurs()
    test_start_voice_server_deja_en_ligne()
    print("OK")
