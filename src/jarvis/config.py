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
    """Un serveur voix local (STT whisper.cpp ou llama.cpp, ou TTS llama.cpp).

    binary        : exécutable lancé si aucun serveur ne répond (ex. whisper-server)
    base_url      : racine du serveur (le client ajoute /inference ou /v1/...)
    model         : GGUF à charger au démarrage (chemin relatif à ~/models ou absolu)
    protocol      : pour le STT, "whisper" (whisper-server, POST /inference) ou
                    "voxtral" (llama-server + modèle audio type Voxtral Mini 4B
                    Realtime, POST /v1/audio/transcriptions) ; ignoré pour le TTS
    extra_args    : arguments supplémentaires passés au binaire
    ready_timeout : attente max du /health (chargement du modèle, peut être long)
    """
    binary: str = ""
    base_url: str = ""
    model: str = ""
    protocol: str = "whisper"
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
        default_factory=lambda: ["arecord", "-q", "-t", "raw", "-f", "S16_LE",
                                 "-r", "16000", "-c", "1", "-"])
    # lecteur audio pour les WAV produits (ffplay/aplay/paplay/play)
    play_cmd: list[str] = field(
        default_factory=lambda: ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"])
    stt: VoiceServerConfig = field(default_factory=lambda: VoiceServerConfig(
        binary="whisper-server",
        base_url="http://127.0.0.1:8178",
        model="~/models/stt/Q4_K_M.gguf",
        protocol="whisper",
    ))
    tts: VoiceServerConfig = field(default_factory=lambda: VoiceServerConfig(
        binary="llama-server",
        base_url="http://127.0.0.1:9932",
        model="~/models/tts/voxtral-4b-tts-q4_k.gguf",
    ))


@dataclass
class TTSAPIConfig:
    """TTS via l'API Mistral (Voxtral) au lieu du serveur local.

    enabled   : si true, les réponses lues à voix haute passent par l'API
    base_url  : racine de l'API (défaut https://api.mistral.ai)
    model     : modèle TTS (défaut voxtral-mini-tts-3-0 = Voxtral TTS 3, dernier ; ou 2603/latest)
    voice_id  : voix (preset "voice_592"/"gb_jane_neutral" ou voix clonée du workspace)
    response_format : mp3 (défaut) | wav | pcm | flac | opus
    La clé API n'est PAS ici : elle vit dans ~/.config/jarvis/mistral.json
    (définie dans Réglages) ou dans l'env MISTRAL_API_KEY.
    """
    enabled: bool = False
    base_url: str = "https://api.mistral.ai"
    model: str = "voxtral-mini-tts-3-0"  # Voxtral TTS 3 (dernier, 20/10/2026)
    voice_id: str = "voice_592"  # Marie, FR
    response_format: str = "mp3"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    tts_api: TTSAPIConfig = field(default_factory=TTSAPIConfig)
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

    try:
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise tomllib.TOMLDecodeError(
            f"{cfg_path}: {exc}. Clé dupliquée ? (souvent un bloc copié de "
            f"config.example.toml laissé sous le tien)") from exc
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
    raw_tts_api = raw.get("voice", {}).get("tts_api", {})
    if raw_tts_api:
        cfg.tts_api = TTSAPIConfig(
            **{k: v for k, v in raw_tts_api.items()
               if k in TTSAPIConfig.__dataclass_fields__})
    mcp_raw = raw.get("mcp", {})
    servers_raw = mcp_raw.get("servers", {}) if isinstance(mcp_raw, dict) else {}
    if not isinstance(servers_raw, dict):
        raise tomllib.TOMLDecodeError(
            f"[mcp] servers doit être une table ([mcp.servers.<nom>]...), "
            f"reçu: {servers_raw!r}.")
    for name, spec in servers_raw.items():
        if not isinstance(spec, dict):
            raise tomllib.TOMLDecodeError(
                f"[mcp.servers.{name}] doit être une table avec 'command = [...]' "
                f"(reçu: {spec!r}).")
        cmd = spec.get("command", [])
        if isinstance(cmd, str):
            cmd = [cmd]
        if not isinstance(cmd, list) or not all(isinstance(c, str) for c in cmd):
            raise tomllib.TOMLDecodeError(
                f"[mcp.servers.{name}] command doit être une liste de chaînes "
                f"(reçu: {cmd!r}).")
        cfg.mcp_servers[name] = list(cmd)
    return cfg
