"""MCP server systeme local : disques, memoire, uptime, GPU, processus, logs.

Tout passe par les binaires systeme usuels (df, free, ps, journalctl, nvidia-smi...)
ou par /proc ; aucune dépendance externe. Les outils manquants renvoient une
erreur explicite au lieu de planter.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("sysinfo")

_MAX_BYTES = 20_000


def _run(cmd: list[str], timeout: float = 15.0) -> str:
    if shutil.which(cmd[0]) is None:
        return f"ERREUR: '{cmd[0]}' introuvable dans PATH"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > _MAX_BYTES:
            out = out[:_MAX_BYTES] + f"\n... (tronqué à {_MAX_BYTES} octets)"
        return out.strip() or "(aucune sortie)"
    except subprocess.TimeoutExpired:
        return f"ERREUR: timeout après {timeout}s"


def _read_proc(path: str, limit: int = 4000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit).strip() or "(vide)"
    except OSError as exc:
        return f"ERREUR: lecture de {path} impossible ({exc.strerror})"


@mcp.tool()
def disk_usage(path: str = "/") -> str:
    """Espace disque (utilisé/libre) pour un point de montage, en Go."""
    return _run(["df", "-h", "--output=source,size,used,avail,pcent,target", path])


@mcp.tool()
def memory() -> str:
    """Mémoire vive et swap (RAM totale, utilisée, disponible) + charge."""
    out = ["== free -h ==", _run(["free", "-h"]), "", "== uptime ==",
           _run(["uptime"])]
    return "\n".join(out)


@mcp.tool()
def load() -> str:
    """Charge système (load average 1/5/15 min) et nombre de processus."""
    return _read_proc("/proc/loadavg") + "\n" + _read_proc("/proc/stat", 200).splitlines()[0]


@mcp.tool()
def gpu_status() -> str:
    """État des GPU : rocm-smi (AMD ROCm) / nvidia-smi, sinon sysfs amdgpu
    (iGPU/dGPU AMD sans ROCm, ex. Radeon 890M), sinon rocminfo, vulkaninfo,
    lspci (liste des cartes graphiques)."""
    for cmd in (["rocm-smi", "--showproductname", "--showmeminfo", "vram",
                 "--showtemp", "--showuse"],
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,"
                 "temperature.gpu,utilization.gpu", "--format=csv,noheader"],
                ["nvidia-smi"],
                ["rocminfo"]):
        if shutil.which(cmd[0]):
            out = _run(cmd, timeout=30.0)
            if out and not out.startswith("ERREUR"):
                amd = _amdgpu_sysfs()
                return out + (("\n\n== amdgpu (sysfs) ==\n" + amd) if amd else "")
    amd = _amdgpu_sysfs()
    if amd:
        return "== amdgpu (sysfs) ==\n" + amd
    if shutil.which("vulkaninfo"):
        return _run(["vulkaninfo", "--summary"], timeout=30.0)
    if shutil.which("lspci"):
        return _run(["lspci", "-k"], timeout=15.0)
    return ("ERREUR: aucun outil GPU trouvé (rocm-smi, nvidia-smi, rocminfo, "
            "vulkaninfo, lspci, /sys/class/drm)")


def _amdgpu_sysfs(base: Path = Path("/sys/class/drm")) -> str:
    """Lecture directe du driver amdgpu (sysfs) : fonctionne sans ROCm ni outil
    externe, typiquement sur un iGPU/dGPU AMD sous Linux (Radeon 890M, RX...)."""
    base = Path(base)
    lines: list[str] = []
    try:
        cards = sorted(d for d in base.iterdir()
                       if d.name.startswith("card") and d.name[4:].isdigit()
                       and (d / "device" / "gpu_busy_percent").exists())
    except OSError:
        return ""
    for card in cards:
        dev = card / "device"
        try:
            name = (dev / "product_name").read_text().strip()
        except OSError:
            try:
                uevent = (dev / "uevent").read_text()
                name = next((l.split("=", 1)[1].strip() for l in uevent.splitlines()
                             if l.startswith("PCI_NAME=")), card.name)
            except OSError:
                name = card.name
        try:
            busy = (dev / "gpu_busy_percent").read_text().strip()
        except OSError:
            busy = "?"
        try:
            vram_total = int((dev / "mem_info_vram_total").read_text().strip()) / 1024**2
            vram_used = int((dev / "mem_info_vram_used").read_text().strip()) / 1024**2
            vram = f"{vram_used:.0f}/{vram_total:.0f} Mo"
        except OSError:
            vram = "?"
        temp_txt = ""
        try:
            for h in (dev / "hwmon").iterdir():
                try:
                    t = int((h / "temp1_input").read_text().strip()) / 1000
                    temp_txt = f" | temp {t:.0f} °C"
                    break
                except OSError:
                    continue
        except OSError:
            pass
        lines.append(f"{name} [{card.name}] | VRAM {vram} | busy {busy}%{temp_txt}")
    return "\n".join(lines)



@mcp.tool()
def temperatures() -> str:
    """Températures (CPU/GPU) via sensors (lm-sensors), sinon /sys/class/thermal,
    sinon les hwmon /sys/class/hwmon (k10temp AMD, coretemp Intel, amdgpu...)."""
    if shutil.which("sensors"):
        out = _run(["sensors"], timeout=10.0)
        if out and not out.startswith("ERREUR"):
            return out
    lines = []
    try:
        zones = sorted(os.listdir("/sys/class/thermal"))
    except OSError:
        zones = []
    for zone in zones:
        if not zone.startswith("thermal_zone"):
            continue
        base = f"/sys/class/thermal/{zone}"
        try:
            with open(f"{base}/type") as f:
                name = f.read().strip()
            with open(f"{base}/temp") as f:
                millideg = int(f.read().strip())
            lines.append(f"{name}: {millideg / 1000:.1f} °C")
        except (OSError, ValueError):
            continue
    if not lines:
        try:
            for hw in sorted(os.listdir("/sys/class/hwmon")):
                hbase = f"/sys/class/hwmon/{hw}"
                try:
                    with open(f"{hbase}/name") as f:
                        hname = f.read().strip()
                except OSError:
                    continue
                for tf in sorted(os.listdir(hbase)):
                    if not (tf.startswith("temp") and tf.endswith("_input")):
                        continue
                    try:
                        with open(f"{hbase}/{tf}") as f:
                            millideg = int(f.read().strip())
                    except (OSError, ValueError):
                        continue
                    label = tf[:-6]
                    try:
                        with open(f"{hbase}/{label}_label") as f:
                            label = f.read().strip()
                    except OSError:
                        pass
                    lines.append(f"{hname} {label}: {millideg / 1000:.1f} °C")
        except OSError:
            pass
    if lines:
        return "\n".join(lines)
    return "ERREUR: ni 'sensors' ni /sys/class/thermal ni /sys/class/hwmon lisible"



@mcp.tool()
def processes(sort: str = "cpu", limit: int = 10) -> str:
    """Top processus. sort: cpu (défaut) ou mem. limit: nombre de lignes (max 50)."""
    sort = sort if sort in ("cpu", "mem") else "cpu"
    limit = max(1, min(limit, 50))
    col = "%mem" if sort == "mem" else "%cpu"
    out = _run(["ps", "-eo", "pid,comm,%cpu,%mem,rss,vsz,etime,args",
                f"--sort=-{col}"], timeout=15.0)
    lines = out.splitlines()
    return "\n".join(lines[:limit + 1])  # +1 pour l'en-tête


@mcp.tool()
def logs(service: str, lines: int = 50, since: str = "1h ago") -> str:
    """Derniers logs systemd d'un service via journalctl.
    service: nom d'unité (ex: NetworkManager, sshd, docker).
    lines: nombre de lignes (max 200). since: fenêtre ('1h ago', 'today', '2 days ago'...)."""
    if not re.fullmatch(r"[A-Za-z0-9@._-]+", service):
        return "ERREUR: nom de service invalide."
    lines = max(1, min(lines, 200))
    return _run(["journalctl", "-u", service, "-n", str(lines),
                 "--no-pager", "-o", "short-iso", "--since", since], timeout=30.0)


@mcp.tool()
def ports_listening() -> str:
    """Ports TCP/UDP en écoute locale avec le processus associé (ss -tulpn)."""
    if shutil.which("ss"):
        return _run(["ss", "-tulpn"])
    if shutil.which("netstat"):
        return _run(["netstat", "-tulpn"])
    return "ERREUR: ni 'ss' ni 'netstat' disponible"


@mcp.tool()
def network_interfaces() -> str:
    """Interfaces réseau, adresses IP et état (ip addr, ou /proc/net/dev en secours)."""
    if shutil.which("ip"):
        return _run(["ip", "-brief", "addr"]) + "\n\n== routes ==\n" + _run(["ip", "route"])
    return _read_proc("/proc/net/dev")


@mcp.tool()
def services(state: str = "failed") -> str:
    """Unités systemd. state: 'failed' (défaut), 'running' ou 'all'.
    En 'failed' : liste les services en échec (ce qui demande de l'attention)."""
    state = state if state in ("failed", "running", "all") else "failed"
    if state == "failed":
        return _run(["systemctl", "list-units", "--state=failed", "--no-legend",
                     "--no-pager"], timeout=30.0)
    flag = "--all" if state == "all" else "--state=running"
    return _run(["systemctl", "list-units", flag, "--type=service", "--no-pager",
                 "--no-legend"], timeout=30.0)


@mcp.tool()
def service_status(name: str) -> str:
    """État détaillé d'un service systemd (systemctl status) : actif, uptime,
    mémoire, dernières lignes de log. name: unité (ex: sshd, NetworkManager)."""
    if not re.fullmatch(r"[A-Za-z0-9@._-]+", name):
        return "ERREUR: nom d'unité invalide."
    return _run(["systemctl", "status", name, "--no-pager", "-n", "15", "-l"], timeout=30.0)


@mcp.tool()
def smart_disk(device: str = "/dev/sda") -> str:
    """Santé SMART d'un disque (smartctl -a) : modèle, série, heure de mise,
    température, erreurs, secteurs réalloués... device: /dev/sdX ou /dev/nvmeXnY."""
    if not re.fullmatch(r"/dev/[A-Za-z0-9]+", device):
        return "ERREUR: device invalide (ex: /dev/sda, /dev/nvme0n1)."
    return _run(["smartctl", "-a", device], timeout=30.0)


@mcp.tool()
def battery() -> str:
    """Batterie (/sys/class/power_supply) : capacité %, état (Charging/
    Discharging/Full), autonomie restante estimée si disponible. Vide sur fixe."""
    base = "/sys/class/power_supply"
    if not os.path.isdir(base):
        return "ERREUR: /sys/class/power_supply introuvable (machine fixe ?)."
    lines = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        fields = {}
        for f in ("type", "status", "capacity", "energy_now", "energy_full",
                  "power_now", "charge_now", "charge_full", "voltage_now"):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                fields[f] = _read_proc(p, 64).splitlines()[0]
        if fields.get("type") != "Battery":
            continue
        cap = fields.get("capacity", "?")
        status = fields.get("status", "?")
        line = f"{name}: {cap} % | {status}"
        # autonomie estimée : energy_now / power_now (µWh/µW -> h)
        try:
            now = float(fields.get("energy_now") or fields.get("charge_now") or 0)
            power = float(fields.get("power_now") or 0)
            if power > 0 and now > 0:
                hours = abs(now / power)
                line += f" | autonomie estimée ~{hours:.1f} h"
        except ValueError:
            pass
        lines.append(line)
    return "\n".join(lines) if lines else "Aucune batterie détectée (poste fixe)."


if __name__ == "__main__":
    mcp.run()  # stdio
