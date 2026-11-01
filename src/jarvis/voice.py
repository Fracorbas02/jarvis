"""Voix locale : STT (whisper.cpp server ou llama-server + modèle audio type
Voxtral Mini 4B Realtime) et TTS (llama-server avec un modèle de synthese,
ex. Voxtral 4B TTS).

Deux serveurs HTTP indépendants du llama-server principal :

- STT, protocole "whisper" (défaut) : `whisper-server -m <ggml whisper>
  --host H --port P` expose POST /inference (multipart: file=<wav>,
  response_format=json) et GET /health.
- STT, protocole "voxtral" : `llama-server -m <gguf audio> --host H --port P`
  expose POST /v1/audio/transcriptions (multipart OpenAI-compatible :
  file=<wav>, model, response_format=json, éventuellement language) et
  GET /health. C'est le chemin des modèles audio de llama.cpp (Voxtral
  Mini 4B Realtime, Qwen3-ASR, etc.).
- TTS : `llama-server -m <tts gguf> --host H --port P` expose
  POST /v1/audio/speech (JSON OpenAI-compatible: input/voice/response_format)
  et, sur les builds récents, POST /tts (JSON {"input": ...} -> WAV).
  Le client essaie /v1/audio/speech puis retombe sur /tts.

Le protocole STT se règle dans config.toml ([voice.stt] protocol =
"whisper"|"voxtral") ; le client essaie l'endpoint du protocole configuré,
puis retombe sur l'autre si le serveur répond 404 (serveur déjà lancé
à la main avec l'autre binaire).

Si aucun serveur ne répond sur base_url, start_voice_server() lance le
binaire configuré avec le GGUF choisi (résolu depuis ~/models si le chemin
est relatif) et attend le /health (chargement du modèle, peut être long).

Utilitaires audio (record micro / lecture WAV) en subprocess : arecord ou
ffmpeg pour capturer du 16 kHz mono, ffplay/aplay/paplay/play pour écouter.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import VoiceConfig, VoiceServerConfig

SAMPLE_RATE = 16000


def _ffmpeg_input_args() -> list[str]:
    """Args d'entrée ffmpeg pour le micro : `-f pulse -i default` si PipeWire/
    PulseAudio est l'API dispo (système moderne sans ALSA direct), sinon
    `-f alsa -i default`."""
    import shutil
    if shutil.which("pactl") or shutil.which("pulseaudio"):
        return ["-f", "pulse", "-i", "default"]
    return ["-f", "alsa", "-i", "default"]


log = logging.getLogger("jarvis.voice")


class VoiceError(RuntimeError):
    pass


@dataclass
class VoiceServerStatus:
    name: str            # "stt" | "tts"
    state: str           # "online" | "managed" | "offline"
    host: str
    port: int
    detail: str = ""
    healthy: bool = False  # /health a répondu 200 (sinon le proc peut être mort)


def host_port(base_url: str) -> tuple[str, int]:
    from urllib.parse import urlparse
    u = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    return (u.hostname or "127.0.0.1", u.port or 80)



def resolve_model_path(model: str | Path, models_root: Path | None = None) -> Path:
    """Résout un chemin de modèle : ~ expansé ; relatif -> d'abord le cwd s'il
    existe, sinon models_root (~/models)."""
    p = Path(str(model)).expanduser()
    if p.is_absolute():
        return p
    if p.is_file():
        return p.resolve()
    root = models_root or (Path.home() / "models")
    return root / p


# ---------- gestion des processus serveurs voix ----------

_procs: dict[str, subprocess.Popen] = {}


# Erreurs de chargement d'un GGUF par un binaire trop vieux : on les remonte
# telles quelles dans le statut (au lieu du seul « mort (code N) »).
_MODEL_LOAD_ERRORS = (
    "unknown model architecture",
    "failed to load model",
    "error loading model",
    "unsupported model",
    "unknown architecture",
)


def _tail_log(log_path: Path, max_bytes: int = 4000) -> str:
    """Derniers octets d'un log de serveur voix (décodé best-effort)."""
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode(errors="replace").strip()
    except OSError:
        return ""


def _load_error_detail(log_path: Path) -> str:
    """Extrait la ligne d'erreur de chargement la plus parlante du log, ou "".

    Ex. « 0.00.219 E llama_model_load: error loading model: unknown model
    architecture: 'voxtral_realtime' » -> « error loading model: unknown model
    architecture: 'voxtral_realtime' » (on coupe le timestamp/préfixe de log).
    """
    tail = _tail_log(log_path)
    for line in reversed(tail.splitlines()):
        low = line.lower()
        hits = [(low.find(err), err) for err in _MODEL_LOAD_ERRORS if err in low]
        if hits:
            start = min(h[0] for h in hits)
            return line[start:start + 300].strip()
    return ""


def _log_path(name: str) -> Path:
    log_dir = Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis"))
    return log_dir / f"voice-{name}.log"


def _find_mmproj(model_path: Path) -> Path | None:
    """Retourne le mmproj*.gguf du même répertoire que `model_path`
    (BF16 préféré, puis Q8_0, puis le 1er trouvé), ou None."""
    try:
        cands = sorted(model_path.parent.glob("mmproj*.gguf"))
    except OSError:
        return None
    if not cands:
        return None
    for pref in ("bf16", "q8_0", "f16"):
        for c in cands:
            if pref in c.name.lower():
                return c
    return cands[0]


def _argv_for(name: str, vcfg: VoiceServerConfig, model_path: Path,
              gpu_args: list[str] | None = None) -> list[str]:
    host, port = host_port(vcfg.base_url)
    extra = [*(gpu_args or []), *vcfg.extra_args]
    if vcfg.device and "--device" not in extra:
        # --device attend le NOM ggml ("Vulkan1") ou un index — jamais "cuda:1"
        from .gpu import device_index
        dev = vcfg.device
        if not (dev[:-1].isalpha() and dev[-1:].isdigit()):  # pas un nom ggml
            dev = device_index(dev)
        extra += ["--device", dev]
    if name == "stt":
        if (vcfg.protocol or "whisper") == "voxtral":
            # llama-server + modèle audio (Voxtral Mini 4B Realtime, ...) :
            # sert POST /v1/audio/transcriptions. Pas de -fa (flash-attention)
            # par défaut sur ces architectures audio.
            # Le décodeur texte (arch mistral3) a besoin de son encodeur
            # audio (mmproj, arch voxtral_realtime) : s'il n'est pas déjà
            # dans extra_args, on prend le mmproj*.gguf du même répertoire
            # que le modèle (convention ggml-org : Voxtral-...-Q4_K_M.gguf
            # + mmproj-Voxtral-...-BF16.gguf).
            if "--mmproj" not in extra:
                mmproj = _find_mmproj(model_path)
                if mmproj is not None:
                    extra = [*extra, "--mmproj", str(mmproj)]
            return [vcfg.binary, "-m", str(model_path), "--host", host,
                    "--port", str(port), "-ngl", "99", "--jinja",
                    "--ctx-size", "8192", *extra]
        return [vcfg.binary, "--host", host, "--port", str(port),
                "-m", str(model_path), *extra]
    return [vcfg.binary, "-m", str(model_path), "--host", host, "--port", str(port),
            "-ngl", "99", "--jinja", "-fa", "on", "--ctx-size", "8192", *extra]


async def voice_server_status(name: str, vcfg: VoiceServerConfig,
                              timeout: float = 2.0) -> VoiceServerStatus:
    """GET /health sur le serveur voix. Ne lève jamais : offline si rien ne répond.

    `healthy` = /health a répondu 200. `state` vaut "managed" dès qu'un
    processus lancé par jarvis tourne, MÊME si /health ne répond pas encore
    (chargement) — ne pas confondre avec healthy.
    """
    host, port = host_port(vcfg.base_url)
    proc = _procs.get(name)
    state = "managed" if proc is not None and proc.poll() is None else "offline"
    try:
        async with httpx.AsyncClient(base_url=vcfg.base_url.rstrip("/"),
                                     timeout=timeout) as http:
            resp = await http.get("/health")
            if resp.status_code == 200:
                return VoiceServerStatus(name, "online" if state == "offline" else "managed",
                                         host, port, healthy=True)
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
    # Choix GPU global enregistré (pop-up 1er lancement / onglet Réglages) :
    # UN mécanisme par backend — env (CUDA_VISIBLE_DEVICES & co) pour
    # cuda/hip/cpu si la config du serveur ne précise pas déjà env, et
    # --device <nom ggml> pour vulkan (pas de variable d'env équivalente).
    # On ne MUTE PAS vcfg (partagé entre appels et interfaces) : on travaille
    # sur une copie via dataclasses.replace.
    from dataclasses import replace
    gpu_args: list[str] = []
    try:
        from .gpu import choice_args, choice_env
        gpu_env = choice_env()
        if gpu_env and not vcfg.env:
            vcfg = replace(vcfg, env=dict(gpu_env))
        if not vcfg.device:
            gpu_args = choice_args()
    except Exception:  # noqa: BLE001
        pass
    argv = _argv_for(name, vcfg, model, gpu_args)
    log_path = _log_path(name)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    log = open(log_path, "ab", buffering=0)
    try:
        log.write(f"\n--- {argv} ---\n".encode())
        env = {**os.environ, **vcfg.env} if vcfg.env else None
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    finally:
        log.close()
    _procs[name] = proc

    deadline = asyncio.get_running_loop().time() + (ready_timeout or vcfg.ready_timeout)
    while asyncio.get_running_loop().time() < deadline:
        if proc.poll() is not None:
            _procs.pop(name, None)
            detail = f"{vcfg.binary} mort (code {proc.returncode})"
            load_err = _load_error_detail(log_path)
            if load_err:
                detail += f" : {load_err}"
            detail += f", voir {log_path}"
            return VoiceServerStatus(name, "offline", host, port, detail=detail)
        # /health répond ? -> prêt. (state "managed" seul ne suffit pas :
        # le proc peut être vivant mais le modèle pas encore chargé)
        st = await voice_server_status(name, vcfg, timeout=1.0)
        if st.healthy:
            return st
        await asyncio.sleep(1.0)
        # le processus peut mourir pendant l'attente (mauvais args, port pris…)
        # sans que /health l'ait vu : re-vérifie après chaque tour
        if proc.poll() is not None:
            _procs.pop(name, None)
            detail = f"{vcfg.binary} mort (code {proc.returncode})"
            load_err = _load_error_detail(log_path)
            if load_err:
                detail += f" : {load_err}"
            detail += f", voir {log_path}"
            return VoiceServerStatus(name, "offline", host, port, detail=detail)
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

class RealtimeSTTClient:
    """Transcription d'un fichier audio via le websocket OpenAI-compatible
    POST /v1/realtime?intent=transcription de llama-server (modèles ASR
    streaming type Voxtral Mini 4B Realtime, dont la route batch
    /v1/audio/transcriptions n'existe pas).

    Protocole (stdlib uniquement, pas de dep websocket) : handshake HTTP
    Upgrade, puis trames WS texte JSON. Le client envoie session.update
    (modalities ["text"], input_audio_format pcm16, transcription objet),
    découpe le WAV 16 kHz mono en blocs input_audio_buffer.append (PCM16
    base64), termine par input_audio_buffer.commit, et collecte les
    événements jusqu'à conversation.item.input_audio_transcription.completed
    (ou response.done / error).
    """

    def __init__(self, base_url: str, timeout: float = 300.0, model: str = "stt",
                 chunk_ms: int = 200):
        from urllib.parse import urlparse
        u = urlparse(base_url if "://" in base_url else f"http://{base_url}")
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 80
        self.secure = (u.scheme or "http") == "https"
        self.model = model
        self.timeout = timeout
        self.chunk_ms = chunk_ms

    # ---------- mini client websocket (RFC 6455, trames texte/close/ping) ----------

    async def _ws_connect(self):
        import asyncio
        import base64 as _b64
        import os as _os
        import struct as _struct
        reader, writer = await asyncio.open_connection(self.host, self.port)
        key = _b64.b64encode(_os.urandom(16)).decode()
        scheme = "wss" if self.secure else "ws"
        req = (f"GET /v1/realtime?intent=transcription HTTP/1.1\r\n"
               f"Host: {self.host}:{self.port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
               f"Sec-WebSocket-Protocol: realtime\r\n\r\n")
        writer.write(req.encode())
        await writer.drain()
        # lit l'en-tête de réponse jusqu'à \r\n\r\n
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=10.0)
            if not chunk:
                raise VoiceError("websocket /v1/realtime : connexion fermée au handshake")
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if "101" not in status:
            raise VoiceError(f"websocket /v1/realtime refusé : {status} {head.decode(errors='replace')[:200]}")
        return reader, writer, rest

    @staticmethod
    async def _ws_recv(reader, pending: bytes):
        """Lit une trame WS texte/close. Retourne (opcode, payload, pending)."""
        import asyncio
        import struct as _struct
        while True:
            if len(pending) >= 2:
                b1, b2 = pending[0], pending[1]
                opcode = b1 & 0x0F
                masked = b2 & 0x80
                length = b2 & 0x7F
                offset = 2
                if length == 126:
                    if len(pending) < 4:
                        pending += await asyncio.wait_for(reader.read(65536), timeout=30.0)
                        continue
                    length = _struct.unpack(">H", pending[2:4])[0]
                    offset = 4
                elif length == 127:
                    if len(pending) < 10:
                        pending += await asyncio.wait_for(reader.read(65536), timeout=30.0)
                        continue
                    length = _struct.unpack(">Q", pending[2:10])[0]
                    offset = 10
                if len(pending) < offset + length:
                    pending += await asyncio.wait_for(reader.read(65536), timeout=30.0)
                    continue
                payload = pending[offset:offset + length]
                pending = pending[offset + length:]
                if opcode == 0x8:  # close
                    return 0x8, payload, pending
                if opcode == 0x9:  # ping -> pong
                    return 0x9, payload, pending
                if opcode in (0x1, 0x0, 0x2):
                    return opcode, payload, pending
            pending += await asyncio.wait_for(reader.read(65536), timeout=30.0)

    @staticmethod
    async def _ws_send_text(writer, text: str) -> None:
        import struct as _struct
        payload = text.encode()
        header = bytearray([0x81])  # FIN + texte
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)  # masqué (client -> serveur)
        elif n < 65536:
            header.append(0x80 | 126)
            header += _struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += _struct.pack(">Q", n)
        mask = b"\x12\x34\x56\x78"
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        writer.write(bytes(header) + masked)
        await writer.drain()

    # ---------- transcription d'un WAV via /v1/realtime ----------

    async def transcribe_wav(self, wav_path: str | Path, language: str | None = None) -> str:
        import asyncio
        import base64 as _b64
        import json as _json
        path = Path(wav_path)
        if not path.is_file():
            raise VoiceError(f"fichier audio introuvable : {path}")
        # lit le WAV (notre record_wav produit du 16 kHz mono 16-bit)
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise VoiceError("WAV attendu en 16-bit pour /v1/realtime")
            rate, channels = w.getframerate(), w.getnchannels()
            pcm = w.readframes(w.getnframes())
        if rate != SAMPLE_RATE or channels != 1:
            # conversion 16 kHz mono via ffmpeg (garde-fou pour un WAV externe)
            fixed = await to_wav_16k_mono(path)
            with wave.open(str(fixed), "rb") as w2:
                pcm = w2.readframes(w2.getnframes())

        try:
            reader, writer, pending = await asyncio.wait_for(
                self._ws_connect(), timeout=min(30.0, self.timeout))
        except VoiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VoiceError(f"/v1/realtime connexion impossible : {type(exc).__name__}: {exc}") from exc
        try:
            async def send(obj: dict) -> None:
                await self._ws_send_text(writer, _json.dumps(obj))

            await send({
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm16",
                    "turn_detection": None,
                    "input_audio_transcription": {
                        "model": self.model,
                        **({"language": language} if language else {}),
                    },
                },
            })
            chunk_bytes = max(1, SAMPLE_RATE * 2 * self.chunk_ms // 1000)
            for i in range(0, len(pcm), chunk_bytes):
                await send({"type": "input_audio_buffer.append",
                            "audio": _b64.b64encode(pcm[i:i + chunk_bytes]).decode()})
            await send({"type": "input_audio_buffer.commit"})

            texts: list[str] = []
            deadline = asyncio.get_running_loop().time() + self.timeout
            while asyncio.get_running_loop().time() < deadline:
                try:
                    opcode, payload, pending = await asyncio.wait_for(
                        self._ws_recv(reader, pending), timeout=max(1.0, deadline - asyncio.get_running_loop().time()))
                except TimeoutError:
                    break
                if opcode == 0x8:
                    break
                if opcode == 0x9:  # ping -> pong
                    import struct as _struct
                    header = bytearray([0x8A, 0x80 | len(payload)]) + b"\x00\x00\x00\x00"
                    writer.write(bytes(header) + payload)
                    await writer.drain()
                    continue
                try:
                    ev = _json.loads(payload.decode())
                except ValueError:
                    continue
                etype = ev.get("type", "")
                log.debug("ws /v1/realtime event: %s %s", etype, str(ev)[:200])
                if etype in ("conversation.item.input_audio_transcription.completed",
                             "transcription.completed"):
                    # champ réel du protocole : "transcript" au niveau de l'événement.
                    # Le completed FINAL contient TOUT le texte : il REMPLACE les
                    # deltas reçus avant lui (sinon affichage en double).
                    tr = ev.get("transcript")
                    if tr is None:
                        tr = ev.get("transcription")
                    if tr is None:
                        item = ev.get("item") or {}
                        content = item.get("content")
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("transcription"):
                                    tr = part["transcription"]
                                    break
                        if tr is None:
                            tr = item.get("transcription")
                    if isinstance(tr, dict):
                        tr = tr.get("text", "")
                    texts.clear()
                    texts.append(str(tr or ""))
                elif etype in ("transcript.text.delta", "transcription.text.delta",
                               "response.audio_transcript.delta",
                               "conversation.item.input_audio_transcription.delta"):
                    texts.append(str(ev.get("delta") or ev.get("text") or ""))
                elif etype == "error":
                    err = ev.get("error") or {}
                    raise VoiceError(f"/v1/realtime erreur : {err.get('message', ev)}")
                elif etype in ("response.done", "session.updated",
                               "input_audio_buffer.committed", "input_audio_buffer.speech_started",
                               "input_audio_buffer.speech_stopped"):
                    if etype == "response.done" and texts:
                        break
                # Voxtral Realtime envoie des deltas au fur et à mesure, puis un
                # completed FINAL qui contient TOUT le texte. Si on sort dès le
                # 1er delta non vide, on ne garde que le 1er mot. On attend donc
                # le completed (ou response.done / fermeture / timeout).
                if etype in ("conversation.item.input_audio_transcription.completed",
                             "transcription.completed") and any(t.strip() for t in texts):
                    break
            return " ".join(t for t in texts if t).strip()
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass


class STTClient:
    """Transcription d'un fichier audio.

    Deux protocoles, choisis par `protocol` (config [voice.stt]) :
    - "whisper" (défaut) : whisper-server, POST /inference (multipart) ;
    - "voxtral" : llama-server + modèle audio (Voxtral Mini 4B Realtime...),
      POST /v1/audio/transcriptions (multipart OpenAI-compatible).
    Si l'endpoint du protocole configuré répond 404, on retombe sur
    l'autre (le serveur peut avoir été lancé à la main avec l'autre
    binaire). Les deux renvoient {"text": ...} avec response_format=json.
    """

    def __init__(self, base_url: str, timeout: float = 300.0,
                 protocol: str = "whisper", model: str = "stt"):
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.protocol = (protocol or "whisper").lower()
        self.model = model
        self._timeout = timeout
        self._http = httpx.AsyncClient(base_url=self.root,
                                       timeout=httpx.Timeout(timeout, connect=10.0))

    async def _post_transcription(self, path: Path, language: str | None,
                                  endpoint: str, extra: dict[str, str]):
        data: dict[str, str] = {"response_format": "json", "temperature": "0.0",
                                **extra}
        if language:
            data["language"] = language
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "audio/wav")}
            return await self._http.post(endpoint, data=data, files=files)

    async def _post_chat_audio(self, path: Path, language: str | None) -> "httpx.Response":
        """Fallback chat multimodal : POST /v1/chat/completions avec le WAV en
        base64 dans un bloc input_audio (chemin des mod\u00e8les audio llama.cpp
        quand la route /v1/audio/transcriptions n'est pas enregistr\u00e9e)."""
        import base64
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        lang_hint = f" in {language}" if language else ""
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio",
                     "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text",
                     "text": f"Transcribe this audio{lang_hint} verbatim. "
                             "Output only the transcript."},
                ],
            }],
        }
        return await self._http.post("/v1/chat/completions", json=payload)

    @staticmethod
    def _text_from_chat(payload) -> str | None:
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text"]
                return "".join(parts).strip() or None
        except (KeyError, IndexError, TypeError):
            return None
        return None

    async def transcribe(self, wav_path: str | Path, language: str | None = None,
                         on_status=None) -> str:
        path = Path(wav_path)
        if not path.is_file():
            raise VoiceError(f"fichier audio introuvable : {path}")
        # Strat\u00e9gies selon le protocole, en cascade :
        #  voxtral : /v1/audio/transcriptions -> chat multimodal -> /inference
        #  whisper : /inference -> /v1/audio/transcriptions -> chat multimodal
        # Le chat multimodal ne sert que si le serveur a une capacit\u00e9 audio
        # (mmproj charg\u00e9) mais pas la route batch ; /inference couvre un
        # whisper-server lanc\u00e9 \u00e0 la main.
        if self.protocol == "voxtral":
            endpoints = [("/v1/audio/transcriptions", {"model": self.model}),
                         ("/inference", {})]
        else:
            endpoints = [("/inference", {}),
                         ("/v1/audio/transcriptions", {"model": self.model})]
        if on_status:
            on_status("transcription en cours…")
        last_err = ""
        for endpoint, extra in endpoints:
            try:
                resp = await self._post_transcription(path, language, endpoint, extra)
            except (httpx.TransportError, httpx.HTTPError) as exc:
                # serveur mort / connexion coupée (ReadError) : on essaie le
                # endpoint suivant (puis le websocket) au lieu de tout planter
                last_err = f"{endpoint} -> {type(exc).__name__}: {exc}"
                continue
            if resp.status_code == 200:
                payload = resp.json()
                text = payload.get("text") if isinstance(payload, dict) else None
                if text is None:
                    raise VoiceError(
                        f"r\u00e9ponse inattendue du serveur STT : {str(payload)[:500]}")
                return text.strip()
            if resp.status_code == 404:
                continue  # endpoint absent -> on essaie le suivant
            last_err = f"{endpoint} -> HTTP {resp.status_code}: {resp.text[:300]}"
            # 501 "does not support chat/completions" = le chat est INACTIF sur
            # ce serveur : inutile de tenter le fallback chat multimodal, on
            # passe directement au websocket /v1/realtime (modèles ASR streaming).
            if resp.status_code in (400, 501) and "chat" in resp.text.lower():
                last_err += " (chat inactif sur ce serveur)"
        # Dernier recours (modèles ASR streaming type Voxtral Realtime, sans
        # route batch ni chat multimodal) : websocket /v1/realtime. Toujours
        # tenté en protocole voxtral, même si les endpoints HTTP ont répondu.
        if self.protocol == "voxtral" or last_err:
            try:
                if on_status:
                    on_status("transcription en cours (websocket /v1/realtime)…")
                ws = RealtimeSTTClient(self.root, timeout=self._timeout,
                                       model=self.model)
                text = await ws.transcribe_wav(path, language=language)
                if text:
                    return text
                last_err += " ; /v1/realtime: transcription vide (pas d'événement completed)"
            except VoiceError as exc:
                last_err += f" ; /v1/realtime: {exc}"
            except Exception as exc:  # noqa: BLE001 — remonter le détail brut
                last_err += f" ; /v1/realtime: {type(exc).__name__}: {exc}"
        if last_err:
            raise VoiceError(f"serveur STT injoignable sur tous les endpoints : {last_err}")
        raise VoiceError("serveur STT HTTP 404 sur tous les endpoints connus "
                         "(/inference, /v1/audio/transcriptions)")

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
                          language: str | None = "fr", on_status=None) -> str:
    """Transcription d'un fichier audio via le serveur STT (démarré si besoin).
    Toute la cascade (batch -> chat multimodal -> WS /v1/realtime) est dans
    STTClient.transcribe ; on_status relaie les phases ("transcription en cours…")."""
    st = await ensure_voice_server("stt", voice_cfg.stt)
    if st.state == "offline":
        raise VoiceError(f"serveur STT indisponible : {st.detail}")
    client = STTClient(voice_cfg.stt.base_url, protocol=voice_cfg.stt.protocol)
    try:
        return await client.transcribe(wav_path, language=language, on_status=on_status)
    finally:
        await client.aclose()


async def transcribe_vad(voice_cfg: VoiceConfig, device: str | None = None,
                         language: str | None = "fr", max_seconds: float = 60.0,
                         on_chunk=None, on_status=None) -> tuple[str, Path, float]:
    """Enregistrement VAD (fin de parole auto) + transcription via le serveur
    STT (même cascade que transcribe_file). Retourne (texte, wav, rms)."""
    from dataclasses import replace
    vcfg = voice_cfg
    if device and device != "default":
        if device.startswith("pulse:"):
            source = device[len("pulse:"):]
            vcfg = replace(voice_cfg, record_cmd=[
                "ffmpeg", "-loglevel", "error", "-f", "pulse", "-i", source,
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"])
        else:
            vcfg = replace(voice_cfg, record_cmd=[
                "arecord", "-q", "-t", "raw", "-D", device,
                "-f", "S16_LE", "-r", "16000", "-c", "1", "-"])
    if on_status:
        on_status("parle")
    path = await record_wav_vad(vcfg, max_seconds=max_seconds, on_chunk=on_chunk,
                                on_status=on_status)
    rms = _pcm_rms(path.read_bytes()[44:])  # saute le header WAV
    st = await ensure_voice_server("stt", voice_cfg.stt)
    if st.state == "offline":
        raise VoiceError(f"serveur STT indisponible : {st.detail}")
    client = STTClient(voice_cfg.stt.base_url, protocol=voice_cfg.stt.protocol)
    try:
        text = await client.transcribe(path, language=language, on_status=on_status)
    finally:
        await client.aclose()
    if on_status:
        on_status("done")
    return text, path, rms


async def transcribe_bytes(voice_cfg: VoiceConfig, data: bytes, filename: str = "upload.wav",
                           language: str | None = "fr") -> str:
    """Transcription d'un upload navigateur (wav, webm, mp3…) : conversion
    éventuelle en WAV 16 kHz mono via ffmpeg, puis transcription."""
    import tempfile
    import uuid
    suffix = Path(filename).suffix or ".bin"
    tmp = Path(tempfile.gettempdir()) / f"jarvis-stt-{os.getpid()}-{uuid.uuid4().hex[:8]}{suffix}"
    tmp.write_bytes(data)
    wav = tmp if suffix.lower() == ".wav" else None
    try:
        if wav is None:
            wav = await to_wav_16k_mono(tmp)
        st = await ensure_voice_server("stt", voice_cfg.stt, auto_start=False)
        if st.state == "offline":
            raise VoiceError(
                f"serveur STT hors ligne ({st.host}:{st.port}) — clique « Démarrer » "
                f"dans le panneau STT")
        return await transcribe_file(voice_cfg, wav, language=language)
    finally:
        for f in {tmp, wav}:
            if f is not None:
                try:
                    f.unlink()
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
        # on garde le WAV dans le cache jarvis (pas /tmp, effacé au reboot) pour
        # pouvoir l'inspecter après coup : ~/.cache/jarvis/rec-<pid>-<ts>.wav
        cache = Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis"))
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            import tempfile
            cache = Path(tempfile.gettempdir())
        import time as _time
        out_path = cache / f"rec-{os.getpid()}-{_time.time_ns() // 1000}.wav"
    cmd = list(voice_cfg.record_cmd)
    if not cmd:
        raise VoiceError("record_cmd vide : pas de binaire d'enregistrement configuré")
    binary = shutil.which(cmd[0])
    if binary is None and cmd[0] == "arecord" and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-loglevel", "error", *_ffmpeg_input_args(),
               "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"]
        binary = "ffmpeg"
    if binary is None:
        raise VoiceError(f"binaire d'enregistrement introuvable : {cmd[0]} "
                         f"(installe alsa-utils ou ffmpeg)")
    argv = [binary, *cmd[1:]]
    stdout_to_file = argv[-1] != "-"

    async def _stop_after(proc, secs: float):
        """Laisse tourner `secs`, puis arrête le process proprement (SIGTERM,
        SIGKILL en dernier recours) et récupère sa sortie.

        Surveille la sortie en parallèle : si le binaire s'arrête de lui-même
        (fin de fichier, erreur), on ne bloque PAS les `secs` complètes — un
        enregistrement "6 s" ne doit pas produire un WAV de 0,05 s."""
        try:
            return await asyncio.wait_for(proc.communicate(), timeout=secs)
        except TimeoutError:
            pass
        if proc.returncode is None:
            proc.terminate()
            try:
                return await asyncio.wait_for(proc.communicate(), timeout=5)
            except TimeoutError:
                proc.kill()
                return await proc.communicate()
        return await proc.communicate()

    if not stdout_to_file:
        # "-" en dernier élément = sortie stdout du binaire (arecord/ffmpeg) :
        # on le GARDE (ffmpeg exige une sortie explicite, "At least one output
        # file must be specified" sinon).
        # -t <seconds> pour ffmpeg : il s'arrête de lui-même à la durée voulue
        # (sinon on le tue au bout de `seconds` et on perd le buffer interne :
        # c'est ce qui produisait des WAV de 0,05 s)
        if cmd[0] == "ffmpeg" or binary == "ffmpeg":
            argv = list(argv)
            if "-t" not in argv:
                # -t AVANT la sortie ("-") : avec une sortie stdout, -t en fin
                # d'arguments est pris pour un paramètre de sortie et ignoré
                # -> ffmpeg ne s'arrête jamais -> on le tue -> buffer perdu
                out_idx = argv.index("-")
                argv = argv[:out_idx] + ["-t", str(seconds)] + argv[out_idx:]
            # ffmpeg s'arrête seul à `seconds` : on attend juste un peu plus,
            # PAS seconds+10 (un micro muet ou un -t ignoré ne doit pas bloquer)
            wait_secs = seconds + 2.0
        else:
            wait_secs = seconds  # arecord : on coupe nous-mêmes à `seconds`
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        pcm, err = await _stop_after(proc, wait_secs)
        # arecord/ffmpeg coupés par SIGTERM renvoient un code non nul : ce n'est
        # pas une erreur tant qu'on a du PCM exploitable.
        if not pcm:
            raise VoiceError(f"{cmd[0]} a échoué : {err.decode(errors='replace')[:300]}")
        if pcm[:4] == b"RIFF":
            # arecord sans "-t raw" envoie un WAV complet sur stdout : on garde
            # le fichier tel quel (header inclus) au lieu de stripper 44 octets
            # à l'aveugle (un header WAV peut faire plus que 44 octets ->
            # WAV tronqué de 0 s dans ffplay)
            out_path.write_bytes(pcm)
        else:
            write_wav(out_path, pcm)
    else:
        argv.append(str(out_path))
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        _, err = await _stop_after(proc, seconds)
        if not out_path.is_file():
            raise VoiceError(f"{cmd[0]} a échoué : {err.decode(errors='replace')[:300]}")
    return out_path


async def record_wav_vad(voice_cfg: VoiceConfig, out_path: Path | None = None,
                         max_seconds: float = 60.0,
                         silence_seconds: float = 1.2,
                         silence_rms: float = 300.0,
                         min_seconds: float = 0.8,
                         on_chunk=None, on_status=None,
                         noise_floor: float | None = None) -> Path:
    """Enregistre le micro jusqu'à la fin de parole (VAD par seuil RMS),
    puis écrit le WAV 16 kHz mono. Utilisé par l'UI pour le "parler et
    attendre" : plus de durée fixe.

    - lit le PCM s16le 16 kHz mono sur stdout du record_cmd (arecord -t raw ou
      ffmpeg -f s16le -) bloc par bloc de 100 ms ;
    - garde les blocs ; dès que `silence_seconds` consécutives ont un RMS
      < silence_rms (et qu'on a au moins min_seconds d'audio), on arrête ;
    - plafond max_seconds (on tue alors le process proprement).
    - on_chunk(pcm_bytes) est appelé à chaque bloc (pour du streaming WS live).
    """
    import time as _time
    if out_path is None:
        cache = Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis"))
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            import tempfile
            cache = Path(tempfile.gettempdir())
        out_path = cache / f"rec-{os.getpid()}-{_time.time_ns() // 1000}.wav"

    cmd = list(voice_cfg.record_cmd)
    if not cmd:
        raise VoiceError("record_cmd vide : pas de binaire d'enregistrement configuré")
    binary = shutil.which(cmd[0])
    if binary is None and cmd[0] == "arecord" and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-loglevel", "error", *_ffmpeg_input_args(),
               "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"]
        binary = "ffmpeg"
    if binary is None:
        raise VoiceError(f"binaire d'enregistrement introuvable : {cmd[0]}")
    argv = [binary, *cmd[1:]]
    if argv[-1] != "-":
        raise VoiceError("record_wav_vad exige un record_cmd avec '-' en sortie (PCM stdout)")
    # ffmpeg : -t max comme plafond de sécurité (placé avant la sortie)
    if binary.endswith("ffmpeg") and "-t" not in argv:
        out_idx = argv.index("-")
        argv = argv[:out_idx] + ["-t", str(max_seconds + 5)] + argv[out_idx:]

    if on_status:
        on_status("parle")
    log.debug("record_wav_vad argv: %s", argv)
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    block = SAMPLE_RATE * 2 // 10  # 100 ms de PCM 16-bit mono
    buf = bytearray()
    silent = 0.0
    total = 0.0
    heard_speech = False
    # Auto-calibration IN-FLUX (pas de mesure séparée qui mangerait le début de
    # parole) : pendant la 1re seconde, on estime le bruit de fond (RMS min des
    # blocs) et le seuil devient max(silence_rms, floor*2.5). Avant calibration,
    # seuil fixe. noise_floor fourni de l'extérieur a priorité.
    floor = noise_floor
    threshold = silence_rms
    if floor is not None:
        threshold = max(silence_rms, floor * 2.5)
    calib_until = 1.0  # s
    calib_min = None
    try:
        deadline = asyncio.get_running_loop().time() + max_seconds
        while asyncio.get_running_loop().time() < deadline:
            chunk = await proc.stdout.read(block)
            if not chunk:
                break  # le process s'est arrêté
            buf += chunk
            if on_chunk is not None:
                try:
                    on_chunk(bytes(chunk))
                except Exception:  # noqa: BLE001 — le callback ne doit pas tuer l'enregistrement
                    pass
            rms = _pcm_rms(chunk)
            total += len(chunk) / 2 / SAMPLE_RATE
            if total <= 3.0:
                log.debug("vad bloc t=%.2f rms=%.0f seuil=%.0f heard=%s", total, rms, threshold, heard_speech)
            if floor is None and total <= calib_until:
                calib_min = rms if calib_min is None else min(calib_min, rms)
                if total >= calib_until:
                    floor = calib_min
                    threshold = max(silence_rms, (floor or 0) * 2.5)
                    log.debug("vad auto-calib: floor=%.0f seuil=%.0f", floor, threshold)
            if rms >= threshold:
                silent = 0.0
                if rms >= threshold * 2:
                    heard_speech = True
            else:
                silent += len(chunk) / 2 / SAMPLE_RATE
            # on ne coupe que si on a entendu de la parole (sinon on attend max_seconds)
            if heard_speech and total >= min_seconds and silent >= silence_seconds:
                if on_status:
                    on_status(f"fin de parole détectée ({total:.1f}s)")
                break  # fin de parole détectée
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                proc.kill()
                await proc.wait()
    if on_status:
        on_status(f"enregistrement terminé ({total:.1f}s)")
    if not buf:
        _, err = await proc.communicate()
        raise VoiceError(f"{cmd[0]} a échoué : {err.decode(errors='replace')[:300]}")
    write_wav(out_path, bytes(buf))
    return out_path


def _pcm_rms(chunk: bytes) -> float:
    """RMS d'un bloc de PCM 16-bit little-endian mono."""
    import math
    if not chunk:
        return 0.0
    n = len(chunk) // 2
    total = 0
    for i in range(0, len(chunk) - 1, 2):
        total += int.from_bytes(chunk[i:i + 2], "little", signed=True) ** 2
    return math.sqrt(total / max(1, n))


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
        try:
            await asyncio.wait_for(proc.wait(), timeout=600.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            continue
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
