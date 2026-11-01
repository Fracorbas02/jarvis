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


def test_stt_transcribe_voxtral():
    """Protocole voxtral : POST /v1/audio/transcriptions (multipart OpenAI)."""
    with _Mock("voxtral") as m:
        async def go():
            client = STTClient(f"http://127.0.0.1:{m.port}", protocol="voxtral",
                               model="voxtral-mini-4b-realtime")
            wav = pcm_to_wav_bytes(b"\x00\x00" * 16000)
            tmp = Path("/tmp/jarvis-test-voxtral.wav")
            tmp.write_bytes(wav)
            text = await client.transcribe(tmp, language="fr")
            await client.aclose()
            return text
        text = _run(go())
        assert "transcription de" in text and "octets audio" in text


def test_stt_fallback_501_goes_websocket():
    """Serveur voxtral sans route batch (501 "does not support chat/completions")
    : le client ne tente PAS le chat multimodal (inactif) et passe au
    websocket /v1/realtime. Ici le mock HTTP renvoie 404 sur le WS ->
    l'erreur nomme bien /v1/realtime."""
    with _Mock("voxtral-501") as m:
        async def go():
            client = STTClient(f"http://127.0.0.1:{m.port}", protocol="voxtral")
            wav = pcm_to_wav_bytes(b"\x00\x00" * 8000)
            tmp = Path("/tmp/jarvis-test-501.wav")
            tmp.write_bytes(wav)
            text = await client.transcribe(tmp, language="fr")
            await client.aclose()
            return text
        try:
            _run(go())
            raise AssertionError("devait échouer")
        except Exception as exc:
            assert "/v1/realtime" in str(exc), exc
            assert "chat inactif" in str(exc), exc


def test_stt_fallback_404_whisper_to_voxtral():
    """Un serveur whisper (protocole voxtral configuré) répond 404 sur
    /v1/audio/transcriptions : le client retombe sur /inference."""
    with _Mock("stt") as m:
        async def go():
            client = STTClient(f"http://127.0.0.1:{m.port}", protocol="voxtral")
            wav = pcm_to_wav_bytes(b"\x00\x00" * 8000)
            tmp = Path("/tmp/jarvis-test-fallback.wav")
            tmp.write_bytes(wav)
            text = await client.transcribe(tmp)
            await client.aclose()
            return text
        text = _run(go())
        assert "transcription de" in text


def test_stt_fallback_404_voxtral_to_whisper():
    """Un serveur voxtral (protocole whisper configuré) répond 404 sur
    /inference : le client retombe sur /v1/audio/transcriptions."""
    with _Mock("voxtral") as m:
        async def go():
            client = STTClient(f"http://127.0.0.1:{m.port}", protocol="whisper")
            wav = pcm_to_wav_bytes(b"\x00\x00" * 8000)
            tmp = Path("/tmp/jarvis-test-fallback2.wav")
            tmp.write_bytes(wav)
            text = await client.transcribe(tmp)
            await client.aclose()
            return text
        text = _run(go())
        assert "transcription de" in text


def test_args_for_device_fallback_quand_gpu_absent(monkeypatch) -> None:
    """Le device enregistré (Vulkan1) n'existe plus (2e GPU HS) : args_for_device
    retombe sur le 1er device ggml dispo au lieu de renvoyer un --device invalide."""
    import jarvis.gpu as gpu
    monkeypatch.setattr(gpu, "list_llama_devices",
                        lambda: [("Vulkan0", "AMD Radeon 890M Graphics (RADV STRIX1) (48701 MiB)")])
    assert gpu.args_for_device("Vulkan1") == ["--device", "Vulkan0"]
    # device toujours présent -> inchangé
    monkeypatch.setattr(gpu, "list_llama_devices",
                        lambda: [("Vulkan0", "dGPU"), ("Vulkan1", "iGPU 890M")])
    assert gpu.args_for_device("Vulkan1") == ["--device", "Vulkan1"]
    # llama-server absent (list_llama_devices vide) -> on tente tel quel
    monkeypatch.setattr(gpu, "list_llama_devices", lambda: [])
    assert gpu.args_for_device("Vulkan1") == ["--device", "Vulkan1"]
    print("OK  args_for_device fallback GPU absent")


def test_argv_for_stt_voxtral():
    """protocol=voxtral -> llama-server avec -ngl 99 --jinja (sans -fa),
    comme un serveur audio llama.cpp."""
    from jarvis.voice import _argv_for
    vcfg = VoiceServerConfig(binary="llama-server",
                             base_url="http://127.0.0.1:8178",
                             model="~/models/stt/Q4_K_M.gguf",
                             protocol="voxtral")
    argv = _argv_for("stt", vcfg, Path("/tmp/m.gguf"))
    assert argv[0] == "llama-server"
    assert "-ngl" in argv and "99" in argv
    assert "--jinja" in argv
    assert "-fa" not in argv
    assert argv[argv.index("-m") + 1] == "/tmp/m.gguf"
    # protocole whisper (défaut) : whisper-server inchangé, pas de --jinja
    vcfg2 = VoiceServerConfig(binary="whisper-server",
                              base_url="http://127.0.0.1:8178",
                              model="~/models/stt/Q4_K_M.gguf")
    argv2 = _argv_for("stt", vcfg2, Path("/tmp/m.gguf"))
    assert argv2[0] == "whisper-server"
    assert "--jinja" not in argv2 and "-ngl" not in argv2


def test_find_mmproj_and_argv_voxtral(tmp_path):
    """_find_mmproj trouve le mmproj du même dossier (BF16 préféré) et
    _argv_for l'ajoute en --mmproj pour le protocole voxtral."""
    from jarvis.voice import _argv_for, _find_mmproj
    model = tmp_path / "Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf"
    model.write_bytes(b"GGUF")
    (tmp_path / "mmproj-Voxtral-Mini-4B-Realtime-2602-Q8_0.gguf").write_bytes(b"MM")
    mm = tmp_path / "mmproj-Voxtral-Mini-4B-Realtime-2602-BF16.gguf"
    mm.write_bytes(b"MM")
    assert _find_mmproj(model) == mm  # BF16 préféré au Q8_0
    vcfg = VoiceServerConfig(binary="llama-server",
                             base_url="http://127.0.0.1:8178",
                             model=str(model), protocol="voxtral")
    argv = _argv_for("stt", vcfg, model)
    assert argv[argv.index("--mmproj") + 1] == str(mm)
    # extra_args avec --mmproj explicite : on ne le double pas
    vcfg2 = VoiceServerConfig(binary="llama-server",
                              base_url="http://127.0.0.1:8178",
                              model=str(model), protocol="voxtral",
                              extra_args=["--mmproj", "/chemin/explicite.gguf"])
    argv2 = _argv_for("stt", vcfg2, model)
    assert argv2.count("--mmproj") == 1
    assert argv2[argv2.index("--mmproj") + 1] == "/chemin/explicite.gguf"
    # pas de mmproj dans le dossier -> pas de --mmproj
    alone = tmp_path / "sous" / "Q4_K_M.gguf"
    alone.parent.mkdir()
    alone.write_bytes(b"GGUF")
    assert _find_mmproj(alone) is None
    argv3 = _argv_for("stt", vcfg, alone)
    assert "--mmproj" not in argv3


def test_voice_config_protocol_default():
    cfg = VoiceConfig()
    assert cfg.stt.protocol == "whisper"
    assert VoiceServerConfig().protocol == "whisper"


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


def test_load_config_protocol(tmp_path):
    """[voice.stt] protocol = "voxtral" est bien chargé depuis le TOML."""
    from jarvis.config import load_config
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[voice.stt]\n"
        'binary = "llama-server"\n'
        'base_url = "http://127.0.0.1:8178"\n'
        'model = "~/models/stt/Q4_K_M.gguf"\n'
        'protocol = "voxtral"\n',
        encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.voice.stt.protocol == "voxtral"
    assert cfg.voice.stt.binary == "llama-server"
    assert cfg.voice.tts.protocol == "whisper"  # défaut inchangé


def test_load_error_detail(tmp_path):
    """_load_error_detail extrait l'erreur de chargement du log (ex. binaire
    llama-server trop vieux pour l'architecture voxtral_realtime)."""
    from jarvis.voice import _load_error_detail
    log = tmp_path / "voice-stt.log"
    log.write_text(
        "0.00.160.343 I srv  llama_server: initializing ...\n"
        "0.00.219.665 E llama_model_load: error loading model: "
        "unknown model architecture: 'voxtral_realtime'\n"
        "0.00.272.635 E srv  llama_server: exiting due to model loading error\n",
        encoding="utf-8")
    out = _load_error_detail(log)
    assert "unknown model architecture: 'voxtral_realtime'" in out, out
    # log sans erreur de chargement -> ""
    log2 = tmp_path / "ok.log"
    log2.write_text("0.00.1 I srv  all good\n", encoding="utf-8")
    assert _load_error_detail(log2) == ""
    assert _load_error_detail(tmp_path / "inexistant.log") == ""


def test_start_voice_server_reports_load_error(tmp_path, monkeypatch):
    """Un binaire qui meurt au chargement du modèle -> statut offline avec
    l'erreur extraite du log (et non juste « mort (code N) »)."""
    import asyncio
    from jarvis.voice import start_voice_server
    fake = tmp_path / "fake-server"
    fake.write_text(
        "#!/bin/sh\n"
        "echo \"0.00.219 E llama_model_load: error loading model: "
        "unknown model architecture: 'voxtral_realtime'\" >&2\n"
        "exit 1\n",
        encoding="utf-8")
    fake.chmod(0o755)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setenv("JARVIS_LOG_DIR", str(tmp_path))
    vcfg = VoiceServerConfig(binary=str(fake), base_url="http://127.0.0.1:59997",
                             model=str(model), ready_timeout=5.0)
    st = asyncio.run(start_voice_server("stt", vcfg))
    assert st.state == "offline", st
    assert "mort (code 1)" in st.detail, st.detail
    assert "unknown model architecture: 'voxtral_realtime'" in st.detail, st.detail


def test_realtime_ws_transcribe():
    """RealtimeSTTClient contre le mock websocket /v1/realtime (tests/ws_mock.py) :
    handshake 101, session.update, append PCM, commit -> transcription reçue."""
    import asyncio
    from jarvis.voice import RealtimeSTTClient
    from ws_mock import start_ws_realtime
    port, _ready, stop = start_ws_realtime(text="bonjour jarvis")
    try:
        async def go():
            client = RealtimeSTTClient(f"http://127.0.0.1:{port}", timeout=30.0)
            wav = pcm_to_wav_bytes(b"\x01\x02" * 16000)  # 1 s de PCM 16 kHz
            tmp = Path("/tmp/jarvis-test-ws.wav")
            tmp.write_bytes(wav)
            return await client.transcribe_wav(tmp, language="fr")
        text = asyncio.run(go())
        assert "bonjour jarvis" in text, text
    finally:
        stop()


def test_record_wav_ffmpeg_t_avant_sortie() -> None:
    """record_wav place -t <seconds> AVANT la sortie '-' de ffmpeg (sinon -t
    en fin d'argv est un param de sortie ignoré -> process tué -> WAV 0 s)."""
    import jarvis.voice as voice
    captured = {}
    class FakeProc:
        returncode = 0
        async def communicate(self):
            return b"\x00\x00" * 100, b""
    async def fake_exec(*a, **k):
        captured["argv"] = list(a)
        return FakeProc()
    import shutil as _shutil, tempfile
    orig_exec, orig_which = voice.asyncio.create_subprocess_exec, voice.shutil.which
    voice.asyncio.create_subprocess_exec = fake_exec
    voice.shutil.which = lambda c: "/usr/bin/" + c
    try:
        cfg = VoiceConfig(record_cmd=["ffmpeg", "-loglevel", "error", "-f", "pulse",
                                      "-i", "default", "-ar", "16000", "-ac", "1",
                                      "-f", "s16le", "-"])
        out = Path(tempfile.mkdtemp()) / "o.wav"
        _run(voice.record_wav(6.0, cfg, out_path=out))
        argv = captured["argv"]
        assert argv[-1] == "-", argv
        i_out = argv.index("-")
        assert argv[i_out - 2:i_out] == ["-t", "6.0"], argv
    finally:
        voice.asyncio.create_subprocess_exec = orig_exec
        voice.shutil.which = orig_which


def test_record_wav_vad_fin_de_parole() -> None:
    """record_wav_vad s'arrête après `silence_seconds` de blocs sous le seuil
    RMS (fin de parole), pas au bout de max_seconds."""
    import asyncio
    import jarvis.voice as voice
    # 10 blocs de 100 ms de "parole" (RMS haut) puis silence infini
    loud = (b"\x00\x40" * 1600)          # 100 ms, amplitude 16384 -> RMS ~16384
    silent = b"\x00\x00" * 1600           # 100 ms de silence
    chunks = [loud] * 10 + [silent] * 200

    class FakeStdout:
        def __init__(self):
            self.i = 0
        async def read(self, n):
            await asyncio.sleep(0)  # laisse la boucle tourner
            if self.i < len(chunks):
                c = chunks[self.i]
                self.i += 1
                return c
            await asyncio.sleep(3600)  # bloque -> sera tué par le finally
            return b""

    class FakeProc:
        def __init__(self):
            self.stdout = FakeStdout()
            self.returncode = None
        def terminate(self):
            self.returncode = -15
        async def wait(self):
            return self.returncode
        async def communicate(self):
            return b"", b""

    captured = {}
    async def fake_exec(*a, **k):
        captured["argv"] = list(a)
        return FakeProc()
    orig_exec, orig_which = voice.asyncio.create_subprocess_exec, voice.shutil.which
    voice.asyncio.create_subprocess_exec = fake_exec
    voice.shutil.which = lambda c: "/usr/bin/" + c
    try:
        cfg = VoiceConfig(record_cmd=["ffmpeg", "-loglevel", "error", "-f", "pulse",
                                      "-i", "default", "-ar", "16000", "-ac", "1",
                                      "-f", "s16le", "-"])
        import tempfile
        out = Path(tempfile.mkdtemp()) / "vad.wav"
        p = _run(voice.record_wav_vad(cfg, out_path=out, max_seconds=60.0,
                                      silence_seconds=0.25, silence_rms=800.0,
                                      min_seconds=0.5))
        import wave
        with wave.open(str(p)) as w:
            secs = w.getnframes() / w.getframerate()
        # 1.0 s de parole + ~0.3 s de silence -> ~1.3 s, PAS 60 s
        assert 1.0 <= secs <= 2.0, secs
        # -t plafond présent avant la sortie "-"
        argv = captured["argv"]
        assert argv[argv.index("-") - 2] == "-t", argv
    finally:
        voice.asyncio.create_subprocess_exec = orig_exec
        voice.shutil.which = orig_which


def test_wav_rms_via_pcm() -> None:
    """RMS d'un WAV écrit par write_wav (via _pcm_rms sur les frames)."""
    import tempfile
    from jarvis.voice import _pcm_rms, write_wav
    tmp = Path(tempfile.mkdtemp())
    silence = write_wav(tmp / "s.wav", b"\x00\x00" * 16000)
    import wave
    with wave.open(str(silence)) as w:
        assert _pcm_rms(w.readframes(w.getnframes())) == 0.0
    sig = write_wav(tmp / "g.wav", b"\x00\x40" * 16000)
    with wave.open(str(sig)) as w:
        assert _pcm_rms(w.readframes(w.getnframes())) > 10000


def test_record_wav_vad_seuil_adaptatif() -> None:
    """Seuil adaptatif + garde heard_speech : avec floor=1000 (seuil 2500),
    un bloc fort (16384 >= 2*2500) est de la parole, un bloc faible (512) est
    du silence -> le VAD s'arrête. Sans parole entendue, il ne coupe pas."""
    import asyncio
    import jarvis.voice as voice
    loud = b"\x00\x40" * 1600      # RMS ~16384 (parole forte)
    mid = b"\x00\x02" * 1600       # RMS ~512 (silence sous seuil 2500)
    chunks = [loud] * 10 + [mid] * 30

    class FakeStdout:
        def __init__(self): self.i = 0
        async def read(self, n):
            await asyncio.sleep(0)
            if self.i < len(chunks):
                c = chunks[self.i]; self.i += 1; return c
            return b""
    class FakeProc:
        def __init__(self): self.stdout = FakeStdout(); self.returncode = 0
        def terminate(self): self.returncode = -15
        async def wait(self): return self.returncode
        async def communicate(self): return b"", b""
    async def fake_exec(*a, **k): return FakeProc()
    orig_exec, orig_which = voice.asyncio.create_subprocess_exec, voice.shutil.which
    voice.asyncio.create_subprocess_exec = fake_exec
    voice.shutil.which = lambda c: "/usr/bin/" + c
    try:
        import tempfile, wave
        from jarvis.config import VoiceConfig
        cfg = VoiceConfig(record_cmd=["ffmpeg", "-loglevel", "error", "-f", "pulse",
                                      "-i", "default", "-ar", "16000", "-ac", "1",
                                      "-f", "s16le", "-"])
        out = Path(tempfile.mkdtemp()) / "vad2.wav"
        p = _run(voice.record_wav_vad(cfg, out_path=out, max_seconds=60.0,
                                      silence_seconds=0.25, silence_rms=300.0,
                                      min_seconds=0.5, noise_floor=1000.0))
        with wave.open(str(p)) as w:
            secs = w.getnframes() / w.getframerate()
        assert 1.0 <= secs <= 2.0, secs  # 1 s parole + 0.25 s silence -> coupe
    finally:
        voice.asyncio.create_subprocess_exec = orig_exec
        voice.shutil.which = orig_which


def test_cleanup_recordings_garde_le_dernier(tmp_path, monkeypatch) -> None:
    """cleanup_recordings(keep=1) supprime les vieux rec-*.wav, garde le plus récent."""
    import os, time
    from jarvis.voice import cleanup_recordings
    monkeypatch.setenv("JARVIS_LOG_DIR", str(tmp_path))
    files = []
    for i in range(5):
        f = tmp_path / f"rec-1-{i}.wav"
        f.write_bytes(b"RIFF" + bytes([i]))
        os.utime(f, (1000 + i, 1000 + i))  # mtime croissant
        files.append(f)
    (tmp_path / "voice-stt.log").write_text("log")  # pas un rec-*.wav -> intact
    n = cleanup_recordings(keep=1)
    assert n == 4, n
    assert files[-1].is_file()          # le plus récent est gardé
    assert not files[0].exists()
    assert (tmp_path / "voice-stt.log").is_file()
    assert cleanup_recordings(keep=1) == 0  # idempotent


def test_record_wav_vad_seuil_relatif_ambiant() -> None:
    """Seuil relatif : avec une ambiance soutenue (tous les blocs ~2000), le
    seuil monte (~4000) ; un bloc de silence (~100) est détecté -> coupe.
    Et inversement : ambiance forte, parole forte (6000) -> heard, puis silence."""
    import asyncio
    import jarvis.voice as voice
    amb = b"\x00\x08" * 1600        # RMS ~2048 (ambiance musique)
    speech = b"\x00\x18" * 1600     # RMS ~6144 (parole)
    silence = b"\x00\x00" * 1600    # RMS 0
    # 10 blocs ambiance, 10 blocs parole, puis silence
    chunks = [amb] * 10 + [speech] * 10 + [silence] * 40

    class FakeStdout:
        def __init__(self): self.i = 0
        async def read(self, n):
            await asyncio.sleep(0)
            if self.i < len(chunks):
                c = chunks[self.i]; self.i += 1; return c
            return b""
    class FakeProc:
        def __init__(self): self.stdout = FakeStdout(); self.returncode = 0
        def terminate(self): self.returncode = -15
        async def wait(self): return self.returncode
        async def communicate(self): return b"", b""
    async def fake_exec(*a, **k): return FakeProc()
    orig_exec, orig_which = voice.asyncio.create_subprocess_exec, voice.shutil.which
    voice.asyncio.create_subprocess_exec = fake_exec
    voice.shutil.which = lambda c: "/usr/bin/" + c
    try:
        import tempfile, wave
        from jarvis.config import VoiceConfig
        cfg = VoiceConfig(record_cmd=["ffmpeg", "-loglevel", "error", "-f", "pulse",
                                      "-i", "default", "-ar", "16000", "-ac", "1",
                                      "-f", "s16le", "-"])
        out = Path(tempfile.mkdtemp()) / "vad3.wav"
        p = _run(voice.record_wav_vad(cfg, out_path=out, max_seconds=60.0,
                                      silence_seconds=0.25, silence_rms=800.0,
                                      min_seconds=0.5, noise_factor=2.0))
        with wave.open(str(p)) as w:
            secs = w.getnframes() / w.getframerate()
        # 1 s ambiance + 1 s parole + ~0.3 s silence -> ~2.3 s (coupe bien malgré l'ambiance)
        assert 1.8 <= secs <= 3.0, secs
    finally:
        voice.asyncio.create_subprocess_exec = orig_exec
        voice.shutil.which = orig_which


def test_pcm_rms():
    from jarvis.voice import _pcm_rms
    assert _pcm_rms(b"") == 0.0
    assert _pcm_rms(b"\x00\x00" * 100) == 0.0
    assert _pcm_rms(b"\x00\x40" * 100) > 10000


def test_voice_config_defaults():
    cfg = VoiceConfig()
    assert cfg.stt.binary == "whisper-server"
    assert cfg.tts.binary == "llama-server"
    assert cfg.record_cmd and cfg.play_cmd


if __name__ == "__main__":
    test_host_port_and_model_path()
    test_wav_helpers(Path("/tmp"))
    test_stt_transcribe()
    test_stt_transcribe_voxtral()
    test_stt_fallback_501_goes_websocket()
    test_stt_fallback_404_whisper_to_voxtral()
    test_stt_fallback_404_voxtral_to_whisper()
    test_argv_for_stt_voxtral()
    test_voice_config_protocol_default()
    test_tts_synthesize_speech_endpoint()
    test_tts_fallback_tts_route()
    test_voice_server_status_offline()
    test_voice_server_status_online()
    test_voice_config_defaults()
    import tempfile
    test_load_config_protocol(Path(tempfile.mkdtemp()))
    test_load_error_detail(Path(tempfile.mkdtemp()))
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


def test_record_wav_riff_stdout_kept_intact() -> None:
    """Si le binaire envoie un WAV complet (RIFF) sur stdout, on garde le
    fichier tel quel (header inclus) : pas de strip de 44 octets à l'aveugle
    qui produirait un WAV illisible de 0 s."""
    import jarvis.voice as voice
    wav_bytes = pcm_to_wav_bytes(b"\x01\x02" * 16000)
    class FakeProc:
        returncode = -15
        async def communicate(self):
            return wav_bytes, b""
    async def fake_exec(*a, **k):
        return FakeProc()
    import shutil as _shutil
    orig_exec = voice.asyncio.create_subprocess_exec
    orig_which = voice.shutil.which
    voice.asyncio.create_subprocess_exec = fake_exec
    voice.shutil.which = lambda c: "/usr/bin/" + c  # pretend arecord exists
    try:
        cfg = VoiceConfig(record_cmd=["arecord", "-q", "-"])
        import tempfile, os
        out = Path(tempfile.mkdtemp()) / "out.wav"
        p = _run(voice.record_wav(0.2, cfg, out_path=out))
        assert p.read_bytes() == wav_bytes  # WAV complet conservé
        import wave
        with wave.open(str(p)) as w:
            assert w.getnframes() == 16000  # lisible, pas tronqué
    finally:
        voice.asyncio.create_subprocess_exec = orig_exec
        voice.shutil.which = orig_which



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

