"""Test du flux de test STT de l'UI GTK (page « Test STT »).

Valide, sans GTK ni modèle réel (mock voix voxtral + record_wav simulé) :
- GtkBackend._astt_test émet stt_test_recording puis stt_test_done avec le
  texte transcrit, le protocole et le serveur ;
- _astt_status émet stt_server_status avec le protocole de la config ;
- le record_wav est remplacé par un WAV factice (pas de micro en CI).

Lancement : uv run python tests/test_gtk_stt.py
"""

from __future__ import annotations

import asyncio
import os
import queue as _q
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import mock_voice  # noqa: E402
from jarvis.config import Config, LLMConfig, VoiceConfig, VoiceServerConfig  # noqa: E402
from jarvis.gtk.backend import GtkBackend  # noqa: E402
from jarvis.voice import pcm_to_wav_bytes  # noqa: E402


def test_gtk_stt_test_flow():
    return asyncio.run(_amain())


async def _amain() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="jarvis-stt-test-"))
    os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
    os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")

    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv, thread = mock_voice.start_mock_voice(port, "voxtral")
    try:
        backend = GtkBackend.__new__(GtkBackend)
        backend.events = _q.Queue()
        backend.busy = False
        backend._lock = asyncio.Lock()
        cfg = Config(llm=LLMConfig(base_url=f"http://127.0.0.1:{port}/v1", model="mock"))
        cfg.voice = VoiceConfig(stt=VoiceServerConfig(
            binary="llama-server",
            base_url=f"http://127.0.0.1:{port}",
            model="~/models/stt/Q4_K_M.gguf",
            protocol="voxtral",
        ))
        backend.cfg = cfg
        backend.session_id = None
        backend.history = []

        # record_wav simulé : écrit un WAV factice au lieu d'enregistrer le micro
        import jarvis.voice as voice_mod
        wav_file = tmp / "fake-rec.wav"
        wav_file.write_bytes(pcm_to_wav_bytes(b"\x00\x00" * 16000))

        async def fake_record_wav(seconds, vcfg, out_path=None):
            return wav_file

        orig = voice_mod.record_wav
        voice_mod.record_wav = fake_record_wav
        try:
            await backend._astt_test(3.0, None)
        finally:
            voice_mod.record_wav = orig

        events = []
        while not backend.events.empty():
            events.append(backend.events.get_nowait())
        types = [e["type"] for e in events]
        assert types == ["stt_test_recording", "stt_test_done"], types
        done = events[-1]
        assert "transcription de" in done["text"], done
        assert done["protocol"] == "voxtral", done
        assert done["seconds"] == 3.0 and done["elapsed"] >= 0, done
        assert done["server"] == f"127.0.0.1:{port}", done

        # statut du serveur STT -> événement stt_server_status avec le protocole
        await backend._astt_status()
        ev = backend.events.get_nowait()
        assert ev["type"] == "stt_server_status", ev
        assert ev["state"] == "online" and ev["healthy"], ev
        assert ev["protocol"] == "voxtral", ev
        assert ev["binary"] == "llama-server" and ev["model"].endswith("Q4_K_M.gguf"), ev

        # serveur coupé -> offline, détail renseigné
        srv.shutdown()
        srv.server_close()
        await backend._astt_status()
        ev = backend.events.get_nowait()
        assert ev["type"] == "stt_server_status" and ev["state"] == "offline", ev

        print("OK  gtk stt test flow:", " -> ".join(types))
        return 0
    finally:
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:  # noqa: BLE001
            pass
        thread.join(timeout=5)


def test_pactl_sources_parsing(monkeypatch):
    """_pactl_sources parse `pactl list short sources` (tab-séparé), exclut
    les .monitor, et produit des ids "pulse:<source>"."""
    import subprocess
    from jarvis.gtk.backend import GtkBackend
    fake_out = (
        "59\talsa_output.pci-0000_38_00.1.HiFi__HDMI4__sink.monitor\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n"
        "68\talsa_input.usb-Jieli_Technology.mono-fallback\tPipeWire\ts16le 1ch 48000Hz\tSUSPENDED\n"
        "789\talsa_input.pci-0000_d4_00.6.HiFi__Mic1__source\tPipeWire\ts32le 2ch 48000Hz\tRUNNING\n"
    )
    monkeypatch.setattr("shutil.which", lambda c: "/usr/bin/pactl" if c == "pactl" else None)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=fake_out, stderr=""))
    devs = GtkBackend._pactl_sources()
    ids = [d for d, _ in devs]
    assert "pulse:alsa_input.pci-0000_d4_00.6.HiFi__Mic1__source" in ids
    assert "pulse:alsa_input.usb-Jieli_Technology.mono-fallback" in ids
    assert not any("monitor" in d for d in ids), ids
    labels = dict(devs)
    assert "Mic1" in labels["pulse:alsa_input.pci-0000_d4_00.6.HiFi__Mic1__source"]


def test_record_cfg_pulse_source(monkeypatch):
    """Un micro choisi au format pulse:<source> produit un record_cmd
    ffmpeg -f pulse -i <source> ; si la source n'existe plus (bus PCI renommé),
    on retombe sur default. Un device ALSA classique reste arecord -D."""
    from jarvis.config import Config
    from jarvis.gtk.backend import GtkBackend
    b = GtkBackend.__new__(GtkBackend)
    b.cfg = Config()
    monkeypatch.setattr(GtkBackend, "_pactl_sources",
                        staticmethod(lambda: [("pulse:alsa_input.pci-0000_69_00.6.analog-stereo", "x")]))
    # source encore présente -> gardée telle quelle
    vcfg0 = b._record_cfg("pulse:alsa_input.pci-0000_69_00.6.analog-stereo")
    assert "alsa_input.pci-0000_69_00.6.analog-stereo" in vcfg0.record_cmd
    # ancienne source (pci-0000_d4_00.6) disparue -> fallback default
    vcfg = b._record_cfg("pulse:alsa_input.pci-0000_d4_00.6.HiFi__Mic1__source")
    assert vcfg.record_cmd[vcfg.record_cmd.index("-i") + 1] == "default", vcfg.record_cmd
    assert vcfg.record_cmd[:5] == ["ffmpeg", "-loglevel", "error", "-f", "pulse"]
    assert vcfg.record_cmd[-1] == "-"
    vcfg2 = b._record_cfg("hw:CARD=0,DEV=0")
    assert vcfg2.record_cmd[0] == "arecord" and "-D" in vcfg2.record_cmd
    vcfg3 = b._record_cfg("default")
    assert vcfg3.record_cmd == b.cfg.voice.record_cmd


def test_wav_rms():
    """_wav_rms : silence -> ~0, signal -> RMS > 0."""
    import wave
    from jarvis.gtk.backend import _wav_rms
    from jarvis.voice import write_wav
    tmp = Path(tempfile.mkdtemp())
    silence = write_wav(tmp / "silence.wav", b"\x00\x00" * 16000)
    assert _wav_rms(silence) == 0.0
    sig = write_wav(tmp / "sig.wav", (b"\x00\x40" * 16000))  # amplitude 16384
    rms = _wav_rms(sig)
    assert rms is not None and rms > 10000, rms
    assert _wav_rms(tmp / "inexistant.wav") is None


if __name__ == "__main__":
    sys.exit(asyncio.run(_amain()))
