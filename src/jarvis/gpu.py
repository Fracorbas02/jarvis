"""Détection des GPU disponibles et choix utilisateur persisté.

Le choix est stocké dans un JSON (~/.config/jarvis/gpu.json, ou
$XDG_CONFIG_HOME/jarvis/gpu.json) : {"device": "cuda:1", "env": {...}}.
L’app desktop propose ce GPU au premier lancement (pop-up) quand plusieurs
sont détectés et qu’aucun choix n’est enregistré ; l’onglet Réglages permet de le
changer ensuite.

Le device stocké est le NOM ggml du GPU quand llama-server est dispo
("Vulkan0", "Vulkan1", "CUDA0"… lu via `llama-server --list-devices`) —
c’est exactement ce que `--device` attend. Anciens formats "vulkan:N"
(index vulkaninfo) acceptés en entrée et convertis par correspondance de nom.

Le choix est appliqué aux sous-processus lancés par jarvis (llama-server
principal + serveurs voix STT/TTS), au prochain démarrage, avec UN mécanisme
par backend (jamais les deux, sinon ils se contredisent) :
- cuda / hip -> env CUDA_VISIBLE_DEVICES / HIP+ROCR_VISIBLE_DEVICES = index
- vulkan     -> argument --device <nom ggml> (Vulkan n’a pas de variable
  d’env « VK_VISIBLE_DEVICES » ; un index brut "0" est refusé par llama.cpp :
  "invalid device: 0")
- cpu        -> env CUDA/HIP/ROCR vides (masque les GPU, best-effort)
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
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        return ""


def list_llama_devices() -> list[tuple[str, str]]:
    """Devices ggml depuis `llama-server --list-devices` : [(nom, description)],
    ex. [("Vulkan0", "AMD Radeon RX 9070 XT (RADV GFX1201) (16304 MiB, …)")].
    Le nom ("Vulkan0", "CUDA1"…) est exactement ce que `--device` accepte.
    [] si llama-server absent ou option inconnue. L’ordre ggml peut différer
    de celui de vulkaninfo (la dGPU peut être Vulkan0)."""
    out = _run(["llama-server", "--list-devices"])
    devs: list[tuple[str, str]] = []
    for line in out.splitlines():
        m = re.match(r"\s*([A-Za-z]+\d+):\s*(.+)", line)
        if m:
            devs.append((m.group(1), m.group(2).strip()))
    return devs


def detect_gpus() -> list[Gpu]:
    """Liste les GPU détectés (best-effort). Source primaire : llama-server
    --list-devices (noms ggml directement utilisables en --device). Fallback :
    nvidia-smi, rocminfo, vulkaninfo, lspci. Toujours un CPU en fallback."""
    gpus: list[Gpu] = []
    # Primaire : devices ggml (llama.cpp)
    for dev, desc in list_llama_devices():
        backend = re.match(r"[A-Za-z]+", dev)
        gpus.append(Gpu(dev, desc, backend.group(0).lower() if backend else "vulkan"))
    if not gpus:
        _detect_gpus_fallback(gpus)
    gpus.append(Gpu("cpu", "CPU (pas de GPU)", "cpu"))
    return gpus


def _detect_gpus_fallback(gpus: list[Gpu]) -> None:
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


def _backend_index(device: str) -> tuple[str, str]:
    """("vulkan", "1") depuis "vulkan:1" (ancien format) OU "Vulkan1" (nom
    ggml). ("cpu", "") pour "cpu"."""
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", device)
    if m:
        return m.group(1).lower(), m.group(2)
    b, _, i = device.partition(":")
    return b.lower(), i


def env_for_device(device: str) -> dict[str, str]:
    """Variables d’env pour forcer ce device : CUDA_VISIBLE_DEVICES /
    HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES selon le backend.
    "cpu" masque tous les GPU (chaîne vide = aucun device visible).
    Vulkan -> {} (pas de variable d'env, on passe --device, voir args_for_device)."""
    backend, idx = _backend_index(device)
    idx = idx or "0"
    if backend == "cuda":
        return {"CUDA_VISIBLE_DEVICES": idx}
    if backend in ("hip", "rocm"):
        return {"HIP_VISIBLE_DEVICES": idx, "ROCR_VISIBLE_DEVICES": idx}
    if backend == "cpu":
        return {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": ""}
    return {}


def device_index(device: str) -> str:
    """Index numérique d'un device : "cuda:1" -> "1", "Vulkan1" -> "1".
    Pour --device quand on veut un index plutôt que le nom ggml."""
    return _backend_index(device)[1] or device


def _vulkaninfo_gpu_names() -> list[str]:
    """Noms des GPU dans l'ordre vulkaninfo (GPU0, GPU1…). Sert à convertir
    un ancien choix "vulkan:N" en nom ggml (l'ordre vulkaninfo diffère de
    celui de ggml : la dGPU peut être Vulkan0 chez ggml mais GPU1 chez
    vulkaninfo)."""
    out = _run(["vulkaninfo", "--summary"])
    names: list[str] = []
    for b in re.split(r"^GPU\d+:", out, flags=re.M)[1:]:
        m = re.search(r"deviceName\s*=\s*(.+)", b)
        if m:
            names.append(m.group(1).strip())
    return names


def args_for_device(device: str) -> list[str]:
    """Arguments CLI pour forcer ce device quand l'env ne suffit pas.

    llama.cpp refuse un index brut en --device ("invalid device: 0") : il
    faut le NOM ggml ("Vulkan0", "CUDA1"…). Donc pour vulkan on renvoie
    ["--device", <nom ggml>] — le nom exact si le device est déjà un nom
    ggml, sinon (ancien format "vulkan:N") via correspondance de nom entre
    vulkaninfo et `llama-server --list-devices`. Pour cuda/hip, l'env suffit
    et on ne renvoie rien (avec CUDA_VISIBLE_DEVICES=N le process ne voit
    plus que ce GPU, sous l'index 0 : un --device <idx> serait invalide)."""
    backend, idx = _backend_index(device)
    if backend != "vulkan":
        return []
    if re.fullmatch(r"[A-Za-z]+\d+", device):  # déjà un nom ggml "Vulkan1"
        return ["--device", device]
    # ancien format "vulkan:N" : N = index vulkaninfo -> nom via vulkaninfo,
    # puis nom ggml correspondant dans --list-devices
    try:
        names = _vulkaninfo_gpu_names()
        want = names[int(idx)]
        base = want.split("(")[0].strip().lower()
        for dev, desc in list_llama_devices():
            if base and base in desc.lower():
                return ["--device", dev]
    except Exception:  # noqa: BLE001 — best-effort
        pass
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
    """Args CLI (--device <nom ggml> pour vulkan) du choix GPU global
    enregistré. [] si aucun choix ou si le backend se force par env."""
    try:
        choice = load_gpu_choice()
    except Exception:  # noqa: BLE001
        return []
    if choice.get("device"):
        return args_for_device(choice["device"])
    return []
