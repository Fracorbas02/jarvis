"""Découverte des modèles installés localement, classés par mode d'usage.

La GUI ne connaît pas les modèles à l'avance : elle scanne les répertoires
connus (~/models, cache HuggingFace, plus $JARVIS_MODEL_DIRS) et classe chaque
fichier GGUF par heuristique :
- répertoire parent nommé stt/ ou tts/ (ex. ~/models/stt/, ~/models/tts/)
  -> mode forcé, même si le nom du fichier est opaque (Q4_K_M.gguf) ;
- sinon heuristique de nom : STT = whisper / ggml- (convention whisper.cpp),
  TTS = piper, kokoro, xtts, coqui, vits, melo, voxtral, tts générique ;
- LLM : tout le reste.

Les modes correspondent aux grandes familles de la GUI : chat LLM, synthèse
vocale (TTS), reconnaissance vocale (STT).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# mode -> (label affiché, icône Material, description)
MODES: dict[str, dict[str, str]] = {
    "llm": {
        "label": "Chat LLM",
        "icon": "smart_toy",
        "description": "Discussion avec le modèle servi par llama-server, avec outils MCP.",
    },
    "tts": {
        "label": "TTS",
        "icon": "record_voice_over",
        "description": "Synthèse vocale locale (phase voix).",
    },
    "stt": {
        "label": "STT",
        "icon": "mic",
        "description": "Reconnaissance vocale locale (phase voix).",
    },
    "settings": {
        "label": "Réglages",
        "icon": "settings",
        "description": "Réglages de l’application (GPU, etc.).",
    },
}

MODE_ORDER = ["llm", "tts", "stt", "settings"]

# whisper.cpp nomme ses modèles ggml-<taille>[.<lang>].bin(.gguf) : préfixe ggml- = STT
_STT_RE = re.compile(r"whisper|^ggml-", re.IGNORECASE)
_TTS_RE = re.compile(
    r"piper|kokoro|xtts|coqui|\bvits\b|melo|\btts\b|style-?tts|your-?tts|voxtral", re.IGNORECASE
)
# répertoire parent dédié à un mode voix (~/models/stt/, ~/models/tts/)
_DIR_MODE = {"stt": "stt", "tts": "tts"}


@dataclass
class ModelInfo:
    name: str          # nom du fichier
    path: Path         # chemin complet
    size_bytes: int
    mode: str          # "llm" | "tts" | "stt"

    @property
    def size_human(self) -> str:
        return format_size(self.size_bytes)


def format_size(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if size < 1024 or unit == "To":
            return f"{size:.1f} {unit}" if unit != "o" else f"{int(size)} o"
        size /= 1024
    return f"{size:.1f} To"


def classify_mode(filename: str, parent_dir: str | None = None) -> str:
    """Classe un fichier modèle dans un mode.

    Le nom du répertoire parent (ex. ~/models/stt) prime sur le nom du
    fichier : un ~/models/stt/Q4_K_M.gguf est un STT même sans indice dans
    son nom. Sinon heuristique sur le nom du fichier.
    """
    if parent_dir and parent_dir.lower() in _DIR_MODE:
        return _DIR_MODE[parent_dir.lower()]
    if _STT_RE.search(filename):
        return "stt"
    if _TTS_RE.search(filename):
        return "tts"
    return "llm"


def classify_path(path: Path) -> str:
    """Mode d'un fichier depuis son chemin complet (répertoire parent + nom)."""
    parent = None if str(path.parent) == str(path.anchor) else path.parent.name
    return classify_mode(path.name, parent)


def default_model_dirs() -> list[Path]:
    """Répertoires scannés par défaut : $JARVIS_MODEL_DIRS + ~/models + cache HF."""
    dirs: list[Path] = []
    env = os.environ.get("JARVIS_MODEL_DIRS", "")
    for part in env.split(os.pathsep):
        if part.strip():
            dirs.append(Path(part.strip()).expanduser())
    dirs.append(Path.home() / "models")
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    # dédoublonne en gardant l'ordre
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _iter_gguf(root: Path):
    """Itère les .gguf d'un répertoire racine (récursif, profondeur raisonnable)."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.gguf")):
        # un rglob part forcément de root : relative_to ne peut pas échouer,
        # le try/except ValueError était mort — on yield directement
        yield path


def discover_models(extra_dirs: list[Path] | None = None,
                    use_defaults: bool = True) -> dict[str, list[ModelInfo]]:
    """Scanne les répertoires de modèles et retourne {mode: [ModelInfo...]} trié par nom."""
    dirs = (default_model_dirs() if use_defaults else []) + list(extra_dirs or [])
    found: dict[str, list[ModelInfo]] = {m: [] for m in MODES}
    seen: set[Path] = set()
    for root in dirs:
        for path in _iter_gguf(root):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                size = rp.stat().st_size
            except OSError:
                size = 0
            mode = classify_path(rp)
            found[mode].append(
                ModelInfo(name=rp.name, path=rp, size_bytes=size, mode=mode)
            )
    for mode in found:
        found[mode].sort(key=lambda m: m.name.lower())
    return found
