"""Statut et contrôle du llama-server derrière la GUI.

Trois états possibles :
- "online"  : un serveur répond sur base_url (démarré ailleurs, ex. à la main) ;
- "managed" : la GUI a elle-même lancé le processus (start/stop possibles) ;
- "offline" : rien ne répond.

La GUI peut lancer llama-server avec un GGUF choisi parmi les modèles
découverts : llama-server -m <gguf> -ngl 99 --jinja -fa on --port <port>.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

# processus lancé par la GUI (None si le serveur est externe ou absent)
_proc: subprocess.Popen | None = None
_last_model: str | None = None  # GGUF servi par le dernier start_server géré
_log_path: Path = Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis")) / "llama-server-gui.log"


@dataclass
class ServerStatus:
    state: str                     # "online" | "managed" | "offline"
    host: str
    port: int
    models: list[str] = field(default_factory=list)   # modèles servis (GET /v1/models)
    detail: str = ""


def host_port(base_url: str) -> tuple[str, int]:
    """Extrait (host, port) d'une base_url type http://127.0.0.1:9931/v1."""
    u = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    return (u.hostname or "127.0.0.1", u.port or 80)


async def get_status(base_url: str, timeout: float = 2.0) -> ServerStatus:
    """Interroge GET {base_url}/models. Ne lève jamais : offline si rien ne répond."""
    host, port = host_port(base_url)
    if _proc is not None and _proc.poll() is None:
        state = "managed"
    else:
        state = "offline"
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as http:
            resp = await http.get("/models")
            if resp.status_code != 200:
                return ServerStatus(state, host, port, detail=f"HTTP {resp.status_code}")
            data = resp.json()
            # llama-server (OpenAI-compatible) renvoie {"object": "list", "data": [...]}
            items = data.get("data") if isinstance(data, dict) else data
            if items is None:
                items = data.get("models", []) if isinstance(data, dict) else []
            models = [m.get("id") or m.get("model") or m.get("name", "?")
                      for m in (items or []) if isinstance(m, dict)]
            return ServerStatus("online" if state == "offline" else "managed",
                                host, port, models=models)
    except (httpx.TransportError, httpx.HTTPError) as exc:
        return ServerStatus(state, host, port,
                            detail=str(exc.__class__.__name__))


def is_managed_running() -> bool:
    return _proc is not None and _proc.poll() is None


def last_managed_model() -> str | None:
    """GGUF servi par le dernier llama-server lancé (et géré) par Jarvis."""
    return _last_model


async def start_server(model_path: str | Path, base_url: str,
                       extra_args: list[str] | None = None,
                       ready_timeout: float = 120.0) -> ServerStatus:
    """Lance llama-server avec le GGUF choisi et attend que /v1/models réponde."""
    global _proc
    host, port = host_port(base_url)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    # Choix GPU global (onglet Réglages / pop-up 1er lancement) : un mécanisme
    # par backend — env (CUDA_VISIBLE_DEVICES & co) pour cuda/hip/cpu,
    # --device <nom ggml> pour vulkan (pas de variable d'env équivalente).
    # Les args du choix vont avant extra_args pour que la config puisse
    # surcharger ; jamais de --device brut type "vulkan:0".
    env = None
    gpu_args: list[str] = []
    try:
        from .gpu import choice_args, choice_env
        gpu_env = choice_env()
        if gpu_env:
            env = {**os.environ, **gpu_env}
        gpu_args = choice_args()
    except Exception:  # noqa: BLE001
        pass
    argv = [
        "llama-server", "-m", str(model_path),
        "--host", host, "--port", str(port),
        "-ngl", "99", "--jinja", "-fa", "on", "--ctx-size", "16384",
        *gpu_args,
        *(extra_args or []),
    ]
    log = open(_log_path, "ab", buffering=0)
    try:
        log.write(f"\n--- {argv} ---\n".encode())
        if env:
            log.write(f"--- env: {env} ---\n".encode())
        _last_model = str(model_path)
        _proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
    finally:
        log.close()
    deadline = asyncio.get_running_loop().time() + ready_timeout
    while asyncio.get_running_loop().time() < deadline:
        if _proc.poll() is not None:
            return ServerStatus("offline", host, port,
                                detail=f"llama-server mort (code {_proc.returncode}), voir {_log_path}")
        status = await get_status(base_url, timeout=1.0)
        if status.models:
            return status
        await asyncio.sleep(1.0)
    return ServerStatus("managed", host, port, detail="démarrage en cours (chargement long)")


async def stop_server(base_url: str) -> ServerStatus:
    """Arrête le llama-server lancé par la GUI (ne touche pas à un serveur externe)."""
    global _proc
    if _proc is None:
        return await get_status(base_url)
    if _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
            _proc.wait(timeout=5)
    _proc = None
    _last_model = None
    await asyncio.sleep(0.5)
    return await get_status(base_url)
