"""MCP server scan de ports — nmap en mode ciblé et non agressif.

Règles de sécurité intégrées :
- cibles limitées aux adresses privées/loopback par défaut (LAN, localhost) ;
  une cible publique exige allow_public=True explicite ;
- uniquement des scans TCP connect (-sT), jamais de scan SYN (-sS, root) ;
- ports limités à une plage fournie (défaut 1-1024), jamais un scan full 1-65535
  sans plage explicite ; vitesse prudente (-T3).
"""

from __future__ import annotations

import ipaddress
import re
import shutil
import subprocess

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("scan")

_MAX_BYTES = 20_000


def _is_private_target(target: str) -> bool:
    try:
        addr = ipaddress.ip_address(target)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False  # hostname : accepté seulement en allow_public ou .local/.lan


def _resolve_lan(target: str) -> bool:
    if _is_private_target(target):
        return True
    return target.endswith((".local", ".lan", ".home", ".internal"))


def _run(cmd: list[str], timeout: float = 120.0) -> str:
    if shutil.which(cmd[0]) is None:
        return f"ERREUR: '{cmd[0]}' introuvable dans PATH (installer nmap)"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > _MAX_BYTES:
            out = out[:_MAX_BYTES] + f"\n... (tronqué à {_MAX_BYTES} octets)"
        return out.strip() or "(aucune sortie)"
    except subprocess.TimeoutExpired:
        return f"ERREUR: timeout après {timeout}s"


@mcp.tool()
def port_scan(target: str, ports: str = "1-1024", allow_public: bool = False) -> str:
    """Scan TCP (connect scan, -sT) des ports d'une cible.
    target: IP/hostname. ports: plage nmap (défaut '1-1024', ex: '22,80,443' ou
    '1-65535'). allow_public: doit être True pour scanner hors LAN/localhost —
    par défaut seules les cibles privées (RFC1918, loopback, .local/.lan) sont
    acceptées."""
    target = target.strip()
    if not re.fullmatch(r"[A-Za-z0-9.:_-]+", target):
        return "ERREUR: cible invalide."
    if not allow_public and not _resolve_lan(target):
        return ("ERREUR: cible hors LAN/localhost refusée. "
                "Passe allow_public=True uniquement pour une cible que tu "
                "own/autorises explicitement.")
    if shutil.which("nmap") is None:
        return "ERREUR: 'nmap' introuvable dans PATH"
    return _run(["nmap", "-sT", "-T3", "--open", "-p", ports, target], timeout=180.0)


@mcp.tool()
def service_detect(target: str, ports: str = "1-1024", allow_public: bool = False) -> str:
    """Détection de versions des services (-sV) sur les ports ouverts d'une cible
    privée/LAN (mêmes règles que port_scan)."""
    target = target.strip()
    if not re.fullmatch(r"[A-Za-z0-9.:_-]+", target):
        return "ERREUR: cible invalide."
    if not allow_public and not _resolve_lan(target):
        return "ERREUR: cible hors LAN/localhost refusée (allow_public=True requis)."
    if shutil.which("nmap") is None:
        return "ERREUR: 'nmap' introuvable dans PATH"
    return _run(["nmap", "-sT", "-sV", "-T3", "--open", "-p", ports, target],
                timeout=300.0)


@mcp.tool()
def os_detect(target: str, allow_public: bool = False) -> str:
    """Empreinte OS (-O) d'une cible privée/LAN. Nécessite souvent les droits
    root pour les sondes raw ; en non-root, nmap retombe sur une détection
    limitée (résultat partiel ou refus — c'est normal)."""
    target = target.strip()
    if not allow_public and not _resolve_lan(target):
        return "ERREUR: cible hors LAN/localhost refusée (allow_public=True requis)."
    if shutil.which("nmap") is None:
        return "ERREUR: 'nmap' introuvable dans PATH"
    return _run(["nmap", "-sT", "-O", "-T3", target], timeout=180.0)


if __name__ == "__main__":
    mcp.run()  # stdio
