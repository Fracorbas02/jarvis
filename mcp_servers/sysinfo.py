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
    """État des GPU : nvidia-smi si présent, sinon rocminfo, sinon vulkaninfo,
    sinon lspci (liste des cartes graphiques)."""
    for cmd in (["nvidia-smi", "--query-gpu=name,memory.total,memory.used,"
                 "temperature.gpu,utilization.gpu", "--format=csv,noheader"],
                ["nvidia-smi"],
                ["rocminfo"]):
        if shutil.which(cmd[0]):
            return _run(cmd, timeout=30.0)
    if shutil.which("vulkaninfo"):
        return _run(["vulkaninfo", "--summary"], timeout=30.0)
    if shutil.which("lspci"):
        return _run(["lspci", "-k"], timeout=15.0)
    return "ERREUR: aucun outil GPU trouvé (nvidia-smi, rocminfo, vulkaninfo, lspci)"


@mcp.tool()
def temperatures() -> str:
    """Températures (CPU/GPU) via sensors (lm-sensors) ou /sys/class/thermal."""
    if shutil.which("sensors"):
        return _run(["sensors"], timeout=10.0)
    lines = []
    zones = sorted(os.listdir("/sys/class/thermal"))
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
        except OSError:
            continue
    if lines:
        return "\n".join(lines)
    return "ERREUR: ni 'sensors' ni /sys/class/thermal lisible"


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


if __name__ == "__main__":
    mcp.run()  # stdio
