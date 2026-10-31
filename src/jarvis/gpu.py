"""Détection des GPU disponibles et choix utilisateur persisté.

Le choix est stocké dans un JSON (~/.config/jarvis/gpu.json, ou
$XDG_CONFIG_HOME/jarvis/gpu.json) : {"device": "cuda:1", "env": {...}}.
L’app desktop propose ce GPU au premier lancement (pop-up) quand plusieurs
sont détectés et qu’aucun choix n’est enregistré ; l’onglet Réglages permet de le
changer ensuite.
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
        names = re.findall(r"deviceName\s*=\s*(.+)", out)
        for i, n in enumerate(dict.fromkeys(names)):
            gpus.append(Gpu(f"vulkan:{i}", n.strip(), "vulkan"))
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
    HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES selon le backend."""
    backend, _, idx = device.partition(":")
    idx = idx or "0"
    if backend == "cuda":
        return {"CUDA_VISIBLE_DEVICES": idx}
    if backend == "hip":
        return {"HIP_VISIBLE_DEVICES": idx, "ROCR_VISIBLE_DEVICES": idx}
    return {}
