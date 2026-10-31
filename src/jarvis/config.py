"""Chargement de la configuration TOML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:9931/v1"
    model: str = "mistral-small-3.3-24b-instruct"
    api_key: str = "local"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: float = 300.0
    # Taille de contexte max affichée dans la TUI. Par défaut lue sur
    # GET /props de llama-server (n_ctx) : mettre une valeur ici ne sert que
    # de fallback si /props est indisponible. Ex: 32768 pour un ctx 32k.
    ctx_size: int | None = None
    # Kwargs passés au chat template à chaque requête (llama-server).
    # Ex : { enable_thinking = false } pour couper le raisonnement des modèles
    # thinking (Qwen3.6) et gagner la latence en usage interactif.
    chat_template_kwargs: dict = field(default_factory=dict)


@dataclass
class AgentConfig:
    system_prompt: str = "Tu es Jarvis, l'assistant local de Bastien."
    max_iterations: int = 8


@dataclass
class VoiceServerConfig:
    """Un serveur voix local (STT whisper.cpp ou TTS llama.cpp).

    binary        : exécutable lancé si aucun serveur ne répond (ex. whisper-server)
    base_url      : racine du serveur (le client ajoute /inference ou /v1/...)
    model         : GGUF à charger au démarrage (chemin relatif à ~/models ou absolu)
    extra_args    : arguments supplémentaires passés au binaire
    ready_timeout : attente max du /health (chargement du modèle, peut être long)
    """
    binary: str = ""
    base_url: str = ""
    model: str = ""
    extra_args: list[str] = field(default_factory=list)
    ready_timeout: float = 180.0
    device: str = ""        # nom ggml du device (ex. "Vulkan1", voir llama-server
                            # --list-devices) -> --device ; permet de répartir les
                            # modèles sur plusieurs GPU. Sinon choix global de
                            # l'onglet Réglages (gpu.json)
    env: dict[str, str] = field(default_factory=dict)  # ex. {"CUDA_VISIBLE_DEVICES": "1"}


@dataclass
class VoiceConfig:
    # binaire d'enregistrement micro (arecord par défaut, fallback ffmpeg) ;
    # "-" en dernier élément = le chemin du WAV temporaire y est concaténé
    record_cmd: list[str] = field(
        default_factory=lambda: ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-"])
    # lecteur audio pour les WAV produits (ffplay/aplay/paplay/play)
    play_cmd: list[str] = field(
        default_factory=lambda: ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"])
    stt: VoiceServerConfig = field(default_factory=lambda: VoiceServerConfig(
        binary="whisper-server",
        base_url="http://127.0.0.1:8178",
        model="~/models/stt/Q4_K_M.gguf",
    ))
    tts: VoiceServerConfig = field(default_factory=lambda: VoiceServerConfig(
        binary="llama-server",
        base_url="http://127.0.0.1:9932",
        model="~/models/tts/voxtral-4b-tts-q4_k.gguf",
    ))


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    # name -> argv du processus MCP server (stdio)
    mcp_servers: dict[str, list[str]] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    candidates = [
        Path(path) if path else None,
        Path(os.environ.get("JARVIS_CONFIG", "")) if os.environ.get("JARVIS_CONFIG") else None,
        Path("config.toml"),
    ]
    cfg_path = next((p for p in candidates if p and p.is_file()), None)
    if cfg_path is None:
        raise FileNotFoundError(
            "config.toml introuvable (ni JARVIS_CONFIG, ni ./config.toml). "
            "Copie config.example.toml en config.toml."
        )

    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = Config()
    cfg.llm = LLMConfig(**{k: v for k, v in raw.get("llm", {}).items()
                           if k in LLMConfig.__dataclass_fields__})
    cfg.agent = AgentConfig(**{k: v for k, v in raw.get("agent", {}).items()
                              if k in AgentConfig.__dataclass_fields__})
    raw_voice = raw.get("voice", {})
    for section in ("stt", "tts"):
        if section in raw_voice:
            setattr(cfg.voice, section, VoiceServerConfig(
                **{k: v for k, v in raw_voice[section].items()
                   if k in VoiceServerConfig.__dataclass_fields__}))
    for key in ("record_cmd", "play_cmd"):
        if key in raw_voice:
            setattr(cfg.voice, key, list(raw_voice[key]))
    for name, spec in raw.get("mcp", {}).get("servers", {}).items():
        cmd = spec.get("command", [])
        if isinstance(cmd, str):
            cmd = [cmd]
        cfg.mcp_servers[name] = list(cmd)
    return cfg
