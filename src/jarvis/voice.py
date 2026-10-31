"""Voix locale : STT (whisper.cpp server) et TTS (llama-server avec un modèle
de synthèse, ex. Voxtral 4B TTS).

Deux serveurs HTTP indépendants du llama-server principal :

- STT : `whisper-server -m <ggml whisper> --host H --port P` expose
  POST /inference (multipart: file=<wav>, response_format=json) et
  GET /health. Transcription d'un fichier audio en texte.
- TTS : `llama-server -m <tts gguf> --host H --port P` expose
  POST /v1/audio/speech (JSON OpenAI-compatible: input/voice/response_format)
  et, sur les builds récents, POST /tts (JSON {"input": ...} -> WAV).
  Le client essaie /v1/audio/speech puis retombe sur /tts.

Si aucun serveur ne répond sur base_url, start_voice_server() lance le
binaire configuré avec le GGUF choisi (résolu depuis ~/models si le chemin
est relatif) et attend le /health (chargement du modèle, peut être long).

Utilitaires audio (record micro / lecture WAV) en subprocess : arecord ou
ffmpeg pour capturer du 16 kHz mono, ffplay/aplay/paplay/play pour écouter.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import VoiceConfig, VoiceServerConfig

SAMPLE_RATE = 16000


class VoiceError(RuntimeError):
    pass


@dataclass
class VoiceServerStatus:
    name: str            # "stt" | "tts"
    state: str           # "online" | "managed" | "offline"
    host: str
    port: int
    detail: str = ""


def host_port(base_url: str) -> tuple[str, int]:
    from urllib.parse import urlparse
    u = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    return (u.hostname or "127.0.0.1", u.port or 80)


def resolve_model_path(model: str | Path, models_root: Path | None = None) -> Path:
    """Résout un chemin de modèle : ~ expansé, relatif -> models_root (~/models)."""
    p = Path(str(model)).expanduser()
    if not p.is_absolute():
        root = models_root or (Path.home() / "models")
        p = root / p
    return p


# ---------- gestion des processus serveurs voix ----------

_procs: dict[str, subprocess.Popen] = {}


def _log_path(name: str) -> Path:
    log_dir = Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis"))
    return log_dir / f"voice-{name}.log"


def _argv_for(name: str, vcfg: VoiceServerConfig, model_path: Path) -> list[str]:
    host, port = host_port(vcfg.base_url)
    if name == "stt":
        return [vcfg.binary, "--host", host, "--port", str(port),
                "-m", str(model_path), *vcfg.extra_args]
    return [vcfg.binary, "-m", str(model_path), "--host", host, "--port", str(port),
            "-ngl", "99", "--jinja", "-fa", "on", "--ctx-size", "8192",
            *vcfg.extra_args]


async def voice_server_status(name: str, vcfg: VoiceServerConfig,
                              timeout: float = 2.0) -> VoiceServerStatus:
    """GET /health sur le serveur voix. Ne lève jamais : offline si rien ne répond."""
    host, port = host_port(vcfg.base_url)
    proc = _procs.get(name)
    state = "managed" if proc is not None and proc.poll() is None else "offline"
    try:
        async with httpx.AsyncClient(base_url=vcfg.base_url.rstrip("/"),
                                     timeout=timeout) as http:
            resp = await http.get("/health")
            if resp.status_code == 200:
                return VoiceServerStatus(name, "online" if state == "offline" else "managed",
                                         host, port)
            return VoiceServerStatus(name, state, host, port, detail=f"HTTP {resp.status_code}")
    except (httpx.TransportError, httpx.HTTPError) as exc:
        return VoiceServerStatus(name, state, host, port,
                                 detail=type(exc).__name__)


async def start_voice_server(name: str, vcfg: VoiceServerConfig,
                             model_path: str | Path | None = None,
                             ready_timeout: float | None = None) -> VoiceServerStatus:
    """Lance le binaire voix avec le GGUF et attend que /health réponde."""
    if not vcfg.binary or not shutil.which(vcfg.binary):
        return VoiceServerStatus(name, "offline", *host_port(vcfg.base_url),
                                 detail=f"binaire introuvable : {vcfg.binary or '(vide)'}")
    model = resolve_model_path(model_path or vcfg.model)
    if not model.is_file():
        return VoiceServerStatus(name, "offline", *host_port(vcfg.base_url),
                                 detail=f"modèle introuvable : {model}")
    st = await voice_server_status(name, vcfg)
    if st.state != "offline":
        return st  # déjà en ligne (externe ou déjà lancé)

    host, port = host_port(vcfg.base_url)
    argv = _argv_for(name, vcfg, model)
    log_path = _log_path(name)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    log = open(log_path, "ab", buffering=0)
    log.write(f"\n--- {argv} ---\n".encode())
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    _procs[name] = proc

    deadline = asyncio.get_running_loop().time() + (ready_timeout or vcfg.ready_timeout)
    while asyncio.get_running_loop().time() < deadline:
        if proc.poll() is not None:
            _procs.pop(name, None)
            return VoiceServerStatus(name, "offline", host, port,
                                     detail=f"{vcfg.binary} mort (code {proc.returncode}), voir {log_path}")
        st = await voice_server_status(name, vcfg, timeout=1.0)
        if st.state != "offline":
            return st
        await asyncio.sleep(1.0)
    return VoiceServerStatus(name, "managed", host, port,
                             detail="démarrage en cours (chargement long)")


async def stop_voice_server(name: str, vcfg: VoiceServerConfig) -> VoiceServerStatus:
    """Arrête le processus lancé par jarvis (ne touche pas à un serveur externe)."""
    proc = _procs.pop(name, None)
    if proc is None:
        return await voice_server_status(name, vcfg)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    await asyncio.sleep(0.5)
    return await voice_server_status(name, vcfg)


async def stop_all_voice_servers(voice_cfg: VoiceConfig) -> None:
    for name, vcfg in (("stt", voice_cfg.stt), ("tts", voice_cfg.tts)):
        if name in _procs:
            await stop_voice_server(name, vcfg)


# ---------- clients ----------

class STTClient:
    """Transcription via whisper-server : POST /inference (multipart)."""

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url,
                                       timeout=httpx.Timeout(timeout, connect=10.0))

    async def transcribe(self, wav_path: str | Path, language: str | None = None) -> str:
        path = Path(wav_path)
        if not path.is_file():
            raise VoiceError(f"fichier audio introuvable : {path}")
        data: dict[str, str] = {"response_format": "json", "temperature": "0.0"}
        if language:
            data["language"] = language
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "audio/wav")}
            resp = await self._http.post("/inference", data=data, files=files)
        if resp.status_code != 200:
            raise VoiceError(f"whisper-server HTTP {resp.status_code}: {resp.text[:500]}")
        payload = resp.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        if text is None:
            raise VoiceError(f"réponse inattendue de /inference : {str(payload)[:500]}")
        return text.strip()

    async def aclose(self) -> None:
        await self._http.aclose()


class TTSClient:
    """Synthèse via llama-server : POST /v1/audio/speech, fallback POST /tts.

    Retourne toujours des octets WAV (response_format=wav).
    """

    def __init__(self, base_url: str, model: str = "tts", timeout: float = 300.0):
        # accepte base_url avec ou sans le suffixe /v1
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self._http = httpx.AsyncClient(base_url=self.root,
                                       timeout=httpx.Timeout(timeout, connect=10.0))

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Synthétise `text` et retourne le WAV (octets)."""
        if not text.strip():
            raise VoiceError("texte vide")
        payload: dict = {"input": text, "response_format": "wav"}
        if voice:
            payload["voice"] = voice
        resp = await self._http.post("/v1/audio/speech",
                                     json={**payload, "model": self.model})
        if resp.status_code == 404:
            resp = await self._http.post("/tts", json=payload)
        if resp.status_code != 200:
            raise VoiceError(f"llama-server TTS HTTP {resp.status_code}: "
                             f"{resp.text[:500]}")
        if not resp.content.startswith(b"RIFF"):
            raise VoiceError("la réponse du serveur TTS n'est pas un WAV")
        return resp.content

    async def voices(self) -> list[str]:
        """Voix disponibles (GET /v1/audio/voices) ; liste vide si non exposé."""
        try:
            resp = await self._http.get("/v1/audio/voices", timeout=5.0)
            if resp.status_code != 200:
                return []
            data = resp.json()
            voices = data.get("voices", data) if isinstance(data, dict) else data
            return [v.get("id", v) if isinstance(v, dict) else str(v) for v in voices]
        except (httpx.HTTPError, ValueError):
            return []

    async def aclose(self) -> None:
        await self._http.aclose()


# ---------- helpers haut niveau (démarrage auto + requête) ----------

async def ensure_voice_server(name: str, vcfg: VoiceServerConfig,
                              model_path: str | Path | None = None,
                              auto_start: bool = True) -> VoiceServerStatus:
    """Retourne le statut du serveur voix.

    Par défaut, tente de le démarrer (binaire + modèle de la config) s'il est
    hors ligne ; avec auto_start=False, se contente de lire le statut (utile
    côté navigateur, où lancer un processus depuis une requête HTTP serait
    surprenant — l'utilisateur démarre via le bouton du panneau).
    """
    st = await voice_server_status(name, vcfg)
    if st.state == "offline" and auto_start:
        st = await start_voice_server(name, vcfg, model_path)
    return st


async def transcribe_file(voice_cfg: VoiceConfig, wav_path: str | Path,
                          language: str | None = "fr") -> str:
    """Transcription d'un fichier audio via le serveur STT (démarré si besoin)."""
    st = await ensure_voice_server("stt", voice_cfg.stt)
    if st.state == "offline":
        raise VoiceError(f"serveur STT indisponible : {st.detail}")
    client = STTClient(voice_cfg.stt.base_url)
    try:
        return await client.transcribe(wav_path, language=language)
    finally:
        await client.aclose()


async def transcribe_bytes(voice_cfg: VoiceConfig, data: bytes, filename: str = "upload.wav",
                           language: str | None = "fr") -> str:
    """Transcription d'un upload navigateur (wav, webm, mp3…) : conversion
    éventuelle en WAV 16 kHz mono via ffmpeg, puis transcription."""
    import tempfile
    suffix = Path(filename).suffix or ".bin"
    tmp = Path(tempfile.gettempdir()) / f"jarvis-stt-{os.getpid()}-{abs(hash(data)) % 10**8}{suffix}"
    tmp.write_bytes(data)
    try:
        wav = await to_wav_16k_mono(tmp) if suffix.lower() != ".wav" else tmp
        st = await ensure_voice_server("stt", voice_cfg.stt, auto_start=False)
        if st.state == "offline":
            raise VoiceError(
                f"serveur STT hors ligne ({st.host}:{st.port}) — clique « Démarrer » "
                f"dans le panneau STT")
        return await transcribe_file(voice_cfg, wav, language=language)
    finally:
        for p in {tmp, tmp.with_suffix(".16k.wav")}:
            try:
                p.unlink()
            except OSError:
                pass


async def synthesize(voice_cfg: VoiceConfig, text: str, voice: str | None = None,
                     auto_start: bool = False) -> bytes:
    """Synthèse vocale via le serveur TTS. Retourne le WAV.

    auto_start=False par défaut côté interfaces (le serveur se démarre via le
    bouton du panneau) ; le CLI passe auto_start=True.
    """
    st = await ensure_voice_server("tts", voice_cfg.tts, auto_start=auto_start)
    if st.state == "offline":
        raise VoiceError(
            f"serveur TTS hors ligne ({st.host}:{st.port}) — clique « Démarrer » "
            f"dans le panneau TTS")
    client = TTSClient(voice_cfg.tts.base_url)
    try:
        return await client.synthesize(text, voice=voice)
    finally:
        await client.aclose()


async def list_voices(voice_cfg: VoiceConfig) -> list[str]:
    """Voix exposées par le serveur TTS ; [] si le serveur est hors ligne."""
    st = await voice_server_status("tts", voice_cfg.tts)
    if st.state == "offline":
        return []
    client = TTSClient(voice_cfg.tts.base_url)
    try:
        return await client.voices()
    finally:
        await client.aclose()


# ---------- utilitaires audio (record / play / conversion) ----------

def write_wav(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE,
              channels: int = 1) -> Path:
    """Écrit un WAV 16-bit à partir de PCM brut (little-endian)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return path


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    """PCM brut -> octets d'un fichier WAV complet (en mémoire)."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def record_wav(seconds: float, voice_cfg: VoiceConfig,
                     out_path: Path | None = None) -> Path:
    """Enregistre `seconds` de micro en WAV 16 kHz mono via record_cmd.

    La commande reçoit le chemin du fichier en dernier argument, ou "-" :
    dans ce cas elle produit du PCM s16le sur stdout (arecord/ffmpeg) et on
    écrit l'en-tête WAV nous-mêmes.
    """
    if out_path is None:
        import tempfile
        out_path = Path(tempfile.gettempdir()) / f"jarvis-rec-{os.getpid()}.wav"
    cmd = list(voice_cfg.record_cmd)
    if not cmd:
        raise VoiceError("record_cmd vide : pas de binaire d'enregistrement configuré")
    binary = shutil.which(cmd[0])
    if binary is None and cmd[0] == "arecord" and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "alsa", "-i", "default",
               "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"]
        binary = "ffmpeg"
    if binary is None:
        raise VoiceError(f"binaire d'enregistrement introuvable : {cmd[0]} "
                         f"(installe alsa-utils ou ffmpeg)")
    argv = [binary, *cmd[1:]]
    if argv[-1] == "-":
        argv = argv[:-1]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            pcm, err = await asyncio.wait_for(proc.communicate(), timeout=seconds + 10)
        except asyncio.TimeoutExpired:
            proc.kill()
            await proc.wait()
            raise VoiceError("enregistrement interrompu (timeout)")
        if proc.returncode != 0:
            raise VoiceError(f"{cmd[0]} a échoué : {err.decode(errors='replace')[:300]}")
        write_wav(out_path, pcm)
    else:
        argv.append(str(out_path))
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=seconds + 10)
        except asyncio.TimeoutExpired:
            proc.kill()
            await proc.wait()
            raise VoiceError("enregistrement interrompu (timeout)")
        if proc.returncode != 0 or not out_path.is_file():
            raise VoiceError(f"{cmd[0]} a échoué : {err.decode(errors='replace')[:300]}")
    return out_path


_PLAY_CANDIDATES = [
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["aplay", "-q"],
    ["paplay"],
    ["play", "-q"],
]


async def play_wav(path: str | Path, voice_cfg: VoiceConfig | None = None) -> str:
    """Joue un WAV avec le premier lecteur disponible. Retourne le binaire utilisé."""
    candidates = [list(voice_cfg.play_cmd)] if voice_cfg and voice_cfg.play_cmd else []
    candidates += _PLAY_CANDIDATES
    seen: set[str] = set()
    for cmd in candidates:
        binary = shutil.which(cmd[0])
        if binary is None or cmd[0] in seen:
            continue
        seen.add(cmd[0])
        proc = await asyncio.create_subprocess_exec(
            binary, *cmd[1:], str(path),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        if proc.returncode == 0:
            return cmd[0]
    raise VoiceError("aucun lecteur audio trouvé (ffplay/aplay/paplay/play)")


async def to_wav_16k_mono(src: Path, dst: Path | None = None) -> Path:
    """Convertit un fichier audio quelconque en WAV 16 kHz mono via ffmpeg."""
    if dst is None:
        dst = src.with_suffix(".16k.wav")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if src.suffix.lower() == ".wav":
            return src  # déjà un wav, whisper.cpp le lira tel quel
        raise VoiceError("ffmpeg introuvable pour convertir l'audio")
    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
        "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise VoiceError(f"ffmpeg a échoué : {err.decode(errors='replace')[:300]}")
    return dst
