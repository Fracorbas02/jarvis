"""Régression: le terminal ne doit JAMAIS rester en mode brut après jarvis.

Bug trouvé: le key binding "enter" de cli._make_session appelait
Buffer.submit(), qui n'existe plus dans prompt_toolkit 3.x. L'AttributeError
levée à chaque Entrée cassait le chemin de sortie propre de l'application :
le raw_mode de prompt_toolkit n'était pas quitté proprement et le tty
restait sans ECHO/ICANON après fermeture de l'app (le shell n'affichait
plus rien).

Couvre :
- le binding "enter" soumet bien via validate_and_handle (test unitaire) ;
- test PTY de bout en bout : prompt dans un thread (comme cli._prompt),
  soumission, sortie -> les termios du terminal sont restaurés ;
- ttyguard : snapshot/restore des termios, handlers SIGTERM/SIGINT ;
- tous les sous-processus (serveurs voix/LLM, record, play, outils système)
  reçoivent stdin=DEVNULL pour ne jamais hériter/interférer avec le tty.

Usage: uv run pytest tests/test_tty_guard.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import termios
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # noqa: E402

from jarvis import ttyguard  # noqa: E402
from jarvis.cli import _make_session  # noqa: E402


# ---------- binding "enter" ----------

def test_enter_binding_soumet_sans_erreur():
    """Le handler 'enter' doit soumettre le buffer via validate_and_handle
    (Buffer.submit n'existe pas dans prompt_toolkit 3.x -> AttributeError)."""
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.keys import Keys
    assert not hasattr(Buffer, "submit"), "prompt_toolkit a réintroduit submit ?"
    assert hasattr(Buffer, "validate_and_handle")

    session = _make_session(None)
    enter = [b for b in session.app.key_bindings.bindings
             if b.keys == (Keys.Enter,)]
    assert enter, "aucun binding 'enter' trouvé"
    handler = enter[-1].handler  # le nôtre, enregistré après ceux par défaut

    buf = Buffer()
    accepted: list[str] = []
    buf.accept_handler = lambda b: accepted.append(b.text) or True
    buf.text = "/quit"
    handler(SimpleNamespace(current_buffer=buf))
    assert accepted == ["/quit"]

    # continuation "\\" en fin de ligne : pas de soumission, saut de ligne
    buf2 = Buffer()
    accepted2: list[str] = []
    buf2.accept_handler = lambda b: accepted2.append(b.text) or True
    buf2.text = "ligne 1\\"
    handler(SimpleNamespace(current_buffer=buf2))
    assert accepted2 == []
    assert buf2.text == "ligne 1\n"


# ---------- ttyguard ----------

def test_ttyguard_snapshot_restore_pty():
    """Sur un vrai pty : snapshot, passage en mode brut, restore -> état initial."""
    master, slave = os.openpty()
    try:
        before = termios.tcgetattr(slave)
        # simule un mode brut laissé par un prompt mal quitté
        raw = termios.tcgetattr(slave)
        raw[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(slave, termios.TCSANOW, raw)
        assert not (termios.tcgetattr(slave)[3] & termios.ECHO)
        ttyguard._saved = before  # snapshot fait AVANT le mode brut
        ttyguard.restore_stdin(slave)
        after = termios.tcgetattr(slave)
        assert after[3] & termios.ECHO
        assert after[3] & termios.ICANON
        assert after == before
    finally:
        ttyguard._saved = None
        os.close(master)
        os.close(slave)


def test_ttyguard_restore_idempotent_et_sans_tty(monkeypatch):
    monkeypatch.setattr(ttyguard, "_saved", None)
    ttyguard.restore_stdin()  # ne doit rien faire ni lever
    monkeypatch.setattr(ttyguard.termios, "tcsetattr",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("ko")))
    ttyguard._saved = [0, 0, 0, 0, 0, 0, b""]
    ttyguard.restore_stdin()  # l'erreur est avalée
    assert ttyguard._saved is None


def test_ttyguard_install_enregistre_signaux(monkeypatch):
    import signal as sig
    registered = {}
    monkeypatch.setattr(sig, "signal", lambda s, h: registered.__setitem__(s, h))
    monkeypatch.setattr(ttyguard, "_installed", False)
    monkeypatch.setattr(ttyguard, "snapshot_stdin", lambda: None)
    monkeypatch.setattr(ttyguard.atexit, "register", lambda f: None)
    ttyguard.install()
    assert sig.SIGTERM in registered and sig.SIGINT in registered
    # le handler restaure puis re-raise le signal avec le défaut
    killed = []
    monkeypatch.setattr(ttyguard.os, "kill", lambda pid, s: killed.append(s))
    monkeypatch.setattr(ttyguard, "_saved", [1, 2, 3, 0x8a3b, 0, 0, b""])
    termios.tcsetattr  # existe
    monkeypatch.setattr(ttyguard.termios, "tcsetattr", lambda *a, **k: None)
    registered[sig.SIGTERM](sig.SIGTERM, None)
    assert killed == [sig.SIGTERM]
    assert ttyguard._saved is None


# ---------- stdin=DEVNULL sur les sous-processus ----------

def test_voice_record_passe_stdin_devnull(monkeypatch):
    """record_wav (branche PCM stdout) ne doit pas hériter du tty en stdin."""
    from jarvis import voice
    from jarvis.config import VoiceConfig
    captured = {}
    real = asyncio.create_subprocess_exec

    async def spy(*a, **kw):
        captured.update(kw)
        raise RuntimeError("stop-test")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    cfg = VoiceConfig(record_cmd=["/bin/true", "-"])
    try:
        asyncio.run(voice.record_wav(0.1, cfg, Path("/tmp/x.wav")))
    except RuntimeError:
        pass
    finally:
        monkeypatch.setattr(asyncio, "create_subprocess_exec", real)
    assert captured.get("stdin") is asyncio.subprocess.DEVNULL


def test_voice_play_passe_stdin_devnull(monkeypatch, tmp_path):
    from jarvis import voice
    captured = {}
    real = asyncio.create_subprocess_exec

    async def spy(*a, **kw):
        captured.update(kw)
        raise RuntimeError("stop-test")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    monkeypatch.setattr(voice.shutil, "which", lambda b: f"/usr/bin/{b}")
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF")
    try:
        asyncio.run(voice.play_wav(wav, None))
    except RuntimeError:
        pass
    finally:
        monkeypatch.setattr(asyncio, "create_subprocess_exec", real)
    assert captured.get("stdin") is asyncio.subprocess.DEVNULL


def test_start_voice_server_popen_stdin_devnull(monkeypatch, tmp_path):
    from jarvis import voice
    from jarvis.config import VoiceServerConfig
    captured = {}

    class FakeProc:
        returncode = 1
        def poll(self): return 1
        def wait(self, timeout=None): return 1
        def terminate(self): pass
        def kill(self): pass

    def fake_popen(argv, **kw):
        captured.update(kw)
        return FakeProc()

    monkeypatch.setattr(voice.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(voice, "_log_path", lambda name: tmp_path / f"{name}.log")
    monkeypatch.setattr(voice, "resolve_model_path", lambda p: tmp_path / "m.gguf")
    (tmp_path / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(voice.shutil, "which", lambda b: f"/usr/bin/{b}")
    # pas de serveur qui répond -> state offline -> on tente de lancer
    async def fake_status(name, vcfg, timeout=2.0):
        return voice.VoiceServerStatus(name, "offline", "127.0.0.1", 1)
    monkeypatch.setattr(voice, "voice_server_status", fake_status)
    vcfg = VoiceServerConfig(binary="whisper-server", base_url="http://127.0.0.1:59997",
                             model="m.gguf")
    asyncio.run(voice.start_voice_server("stt", vcfg, ready_timeout=0.05))
    assert captured.get("stdin") is voice.subprocess.DEVNULL


def test_server_popen_stdin_devnull(monkeypatch, tmp_path):
    from jarvis import server
    captured = {}

    class FakeProc:
        def poll(self): return None

    def fake_popen(argv, **kw):
        captured.update(kw)
        return FakeProc()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_log_path", tmp_path / "llama-server-gui.log")

    async def fake_status(*a, **k):
        return server.ServerStatus("offline", "h", 1)
    monkeypatch.setattr(server, "get_status", fake_status)
    try:
        asyncio.run(server.start_server("/tmp/m.gguf", "http://127.0.0.1:59996/v1",
                                        ready_timeout=0.05))
    finally:
        server._proc = None
    assert captured.get("stdin") is server.subprocess.DEVNULL


# ---------- test PTY de bout en bout ----------

@pytest.mark.skipif(os.name != "posix", reason="pty POSIX uniquement")
def test_prompt_dans_thread_restaure_termios_en_pty(tmp_path):
    """Reproduit le bug exact : PromptSession (avec les vrais bindings de
    cli._make_session) prompté dans un thread daemon + asyncio.run, soumis
    par Entrée, puis sortie -> le tty doit retrouver ECHO/ICANON."""
    import pty
    import select
    import time

    src = str(Path(__file__).parent.parent / "src")
    script = tmp_path / "jarvis-pty-child.py"
    script.write_text(
        "import asyncio, os, sys, termios, threading\n"
        f"sys.path.insert(0, {src!r})\n"
        "os.environ['TERM'] = 'xterm-256color'\n"
        "from jarvis.cli import _make_session\n"
        "session = _make_session(None)\n"
        "async def _p():\n"
        "    box = {}\n"
        "    def run():\n"
        "        try:\n"
        "            box['t'] = session.prompt('[toi] > ')\n"
        "        except BaseException as e:\n"
        "            box['e'] = e\n"
        "    t = threading.Thread(target=run, daemon=True)\n"
        "    t.start()\n"
        "    while t.is_alive():\n"
        "        await asyncio.sleep(0.05)\n"
        "    return box\n"
        "box = asyncio.run(_p())\n"
        "print('RESULT', box, flush=True)\n"
        "print('AFTER_ECHO', bool(termios.tcgetattr(0)[3] & termios.ECHO), flush=True)\n"
    )
    code = str(script)
    pid, master = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, code])

    def drain(timeout):
        out = b""
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([master], [], [], 0.5)
            if r:
                try:
                    out += os.read(master, 65536)
                except OSError:
                    break
        return out

    try:
        initial = termios.tcgetattr(master)
        out = drain(20)
        assert b"[toi]" in out, out[-300:]
        during = termios.tcgetattr(master)
        if not (during[3] & termios.ECHO):
            pass  # prompt bien en mode brut pendant la saisie
        os.write(master, b"/quit\r")
        out += drain(10)
        deadline = time.time() + 15
        status = None
        while time.time() < deadline:
            w, status = os.waitpid(pid, os.WNOHANG)
            if w:
                break
            time.sleep(0.2)
        assert status is not None, "le process fils ne s'est pas terminé"
        after = termios.tcgetattr(master)
        assert after[3] & termios.ECHO, (
            f"RÉGRESSION: tty laissé en mode brut après le prompt "
            f"(lflag {after[3]:#x} au lieu de {initial[3]:#x})")
        assert after[3] & termios.ICANON
        assert b"RESULT {'t': '/quit'}" in out, out[-300:]
        assert b"AFTER_ECHO True" in out, out[-300:]
    finally:
        try:
            os.close(master)
        except OSError:
            pass
