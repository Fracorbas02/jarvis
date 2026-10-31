"""Détection des GPU disponibles et choix utilisateur persisté.

Le choix est stocké dans un JSON (~/.config/jarvis/gpu.json, ou
$XDG_CONFIG_HOME/jarvis/gpu.json) : {"device": "cuda:1", "env": {...}}.
L’app desktop propose ce GPU au premier lancement (pop-up) quand plusieurs
sont détectés et qu’aucun choix n’est enregistré ; l’onglet Réglages permet de le
changer ensuite.

Le choix est appliqué aux sous-processus lancés par jarvis (llama-server
principal + serveurs voix STT/TTS), au prochain démarrage, avec UN mécanisme
par backend (jamais les deux, sinon ils se contredisent) :
- cuda:N  -> env CUDA_VISIBLE_DEVICES=N
- hip:N   -> env HIP_VISIBLE_DEVICES=N + ROCR_VISIBLE_DEVICES=N
- vulkan:N -> argument --device N (llama.cpp/whisper.cpp ; il n’existe pas de
  variable d’env « VK_VISIBLE_DEVICES », sans --device le choix est ignoré)
- cpu     -> env CUDA/HIP/ROCR vides (masque les GPU, best-effort)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Gpu:
    device: str          # ex. "cuda:0", "hip:1", "vulkan:0", "cpu"
    name: str            # ex. "NVIDIA RTX 4090" / "AMD Radeon 890M"
    backend: str         # "cuda" | "hip" | "vulkan" | "cpu"


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    d = Path(base) / "jarvis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gpu_choice_path() -> Path:
    return config_dir() / "gpu.json"


def load_gpu_choice() -> dict:
    try:
        return json.loads(gpu_choice_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_gpu_choice(device: str, env: dict[str, str] | None = None) -> dict:
    data = {"device": device, "env": env or {}}
    gpu_choice_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return ""


def detect_gpus() -> list[Gpu]:
    """Liste les GPU détectés (best-effort) : nvidia-smi, then rocminfo,
    then lspci. Toujours au moins un CPU en fallback."""
    gpus: list[Gpu] = []
    # NVIDIA
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    for i, line in enumerate(l for l in out.splitlines() if l.strip()):
        gpus.append(Gpu(f"cuda:{i}", line.strip(), "cuda"))
    # AMD ROCm
    out = _run(["rocminfo"])
    if out:
        names = re.findall(r"Marketing Name:\s*(.+)", out)
        # chaque "gfx" distinct = un GPU ; on compte les agents GPU
        gfx = sorted(set(re.findall(r"gfx\d+", out)))
        for i, g in enumerate(gfx):
            name = names[i] if i < len(names) else g
            gpus.append(Gpu(f"hip:{i}", name.strip(), "hip"))
    # Vulkan (catch-all)
    out = _run(["vulkaninfo", "--summary"])
    if out and not gpus:
        # parser par bloc "GPUn:" pour lier deviceName et deviceType
        # (integrated vs discrete) — l’index vulkan:N suit l’ordre d’énumération
        # Vulkan, le même que --device N de llama.cpp.
        blocks = re.split(r"^GPU\d+:", out, flags=re.M)
        for i, b in enumerate(blocks[1:]):
            m = re.search(r"deviceName\s*=\s*(.+)", b)
            t = re.search(r"deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(\w+)", b)
            name = m.group(1).strip() if m else f"Vulkan GPU {i}"
            if t:
                name += f" [{t.group(1).lower()}]"
            gpus.append(Gpu(f"vulkan:{i}", name, "vulkan"))
    if not gpus:
        # lspci VGA
        out = _run(["lspci"])
        vgas = [l.split(": ", 1)[1] for l in out.splitlines() if "VGA" in l or "3D" in l]
        for i, n in enumerate(vgas):
            gpus.append(Gpu(f"vulkan:{i}", n.strip(), "vulkan"))
    gpus.append(Gpu("cpu", "CPU (pas de GPU)", "cpu"))
    return gpus


def env_for_device(device: str) -> dict[str, str]:
    """Variables d’env pour forcer ce device : CUDA_VISIBLE_DEVICES /
    HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES selon le backend.
    "cpu" masque tous les GPU (chaîne vide = aucun device visible)."""
    backend, _, idx = device.partition(":")
    idx = idx or "0"
    if backend == "cuda":
        return {"CUDA_VISIBLE_DEVICES": idx}
    if backend == "hip":
        return {"HIP_VISIBLE_DEVICES": idx, "ROCR_VISIBLE_DEVICES": idx}
    if backend == "cpu":
        return {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": ""}
    return {}  # vulkan : pas de variable standard fiable, llama-server détecte seul


def device_index(device: str) -> str:
    """Convertit "cuda:1" -> "1" pour un --device explicite : llama.cpp et
    whisper.cpp attendent un index (ou un nom de device interne), pas
    "cuda:1". Sans "backend:" on renvoie la valeur telle quelle."""
    _, _, idx = device.partition(":")
    return idx or device


def args_for_device(device: str) -> list[str]:
    """Arguments CLI pour forcer ce device quand l'env ne suffit pas.
    Vulkan n'a pas de variable d'env type CUDA_VISIBLE_DEVICES : le seul
    levier fiable est --device <index> (llama.cpp/whisper.cpp). On ne le
    renvoie QUE pour vulkan (pour cuda/hip, l'env masque déjà les autres GPU
    et --device <idx> serait alors invalide car le process ne voit plus que
    le device choisi, sous l'index 0)."""
    backend, _, idx = device.partition(":")
    if backend == "vulkan":
        return ["--device", idx or "0"]
    return []


def choice_env() -> dict[str, str]:
    """Env du choix GPU global enregistré (Réglages / pop-up 1er lancement).
    {} si aucun choix. Sert à passer env= au Popen des serveurs locaux."""
    try:
        choice = load_gpu_choice()
    except Exception:  # noqa: BLE001
        return {}
    if choice.get("env"):
        return dict(choice["env"])
    if choice.get("device"):
        return env_for_device(choice["device"])
    return {}


def choice_args() -> list[str]:
    """Args CLI (--device N pour vulkan) du choix GPU global enregistré.
    [] si aucun choix ou si le backend se force par env."""
    try:
        choice = load_gpu_choice()
    except Exception:  # noqa: BLE001
        return []
    if choice.get("device"):
        return args_for_device(choice["device"])
    return []
