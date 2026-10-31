"""MCP server Docker — inspection en lecture seule des conteneurs/images.

Commandes autorisées : ps, stats, logs, images, inspect, system df.
Aucune commande d'écriture (pas de run/stop/rm/exec).
"""

from __future__ import annotations

import re
import shutil
import subprocess

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("docker")

_MAX_BYTES = 20_000


def _run(args: list[str], timeout: float = 30.0) -> str:
    if shutil.which("docker") is None:
        return "ERREUR: 'docker' introuvable dans PATH"
    try:
        proc = subprocess.run(["docker"] + args, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > _MAX_BYTES:
            out = out[:_MAX_BYTES] + f"\n... (tronqué à {_MAX_BYTES} octets)"
        return out.strip() or "(aucune sortie)"
    except subprocess.TimeoutExpired:
        return f"ERREUR: timeout après {timeout}s"


def _check_name(name: str) -> str | None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        return "ERREUR: nom de conteneur/image invalide."
    return None


@mcp.tool()
def containers(all: bool = True) -> str:
    """Liste des conteneurs (docker ps). all=True inclut les arrêtés.
    Colonnes: ID, image, commande, statut, ports, noms."""
    return _run(["ps", "--format",
                 "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}"]
                + (["-a"] if all else []))


@mcp.tool()
def container_stats(name: str) -> str:
    """Stats temps réel (une mesure, --no-stream) d'un conteneur : CPU, mémoire,
    I/O réseau et disque. name: nom ou ID du conteneur."""
    err = _check_name(name)
    if err:
        return err
    return _run(["stats", "--no-stream", "--format",
                 "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}", name])


@mcp.tool()
def container_logs(name: str, lines: int = 100, since: str = "1h") -> str:
    """Logs d'un conteneur. lines: max 500. since: fenêtre ('10m', '1h', 'today'...)."""
    err = _check_name(name)
    if err:
        return err
    lines = max(1, min(lines, 500))
    return _run(["logs", "--tail", str(lines), "--since", since, name], timeout=60.0)


@mcp.tool()
def images() -> str:
    """Images présentes localement (docker images) : dépôt, tag, taille, date."""
    return _run(["images", "--format",
                 "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}\t{{.ID}}"])


@mcp.tool()
def container_inspect(name: str) -> str:
    """Configuration complète d'un conteneur (docker inspect) : réseau, volumes,
    env, healthcheck, limites. Sortie JSON tronquée."""
    err = _check_name(name)
    if err:
        return err
    return _run(["inspect", name], timeout=30.0)


@mcp.tool()
def disk_usage() -> str:
    """Espace disque utilisé par Docker (images, conteneurs, volumes, build cache)."""
    return _run(["system", "df", "-v"])


if __name__ == "__main__":
    mcp.run()  # stdio
