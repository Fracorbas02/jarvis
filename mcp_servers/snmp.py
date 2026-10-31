"""MCP server SNMP (v2c/v3) — interrogation de switchs/serveurs/imprimantes.

Par binaires net-snmp (snmpget/snmpwalk/snmpbulkwalk). Communauté en paramètre
ou via JARVIS_SNMP_COMMUNITY (défaut 'public'). Lecture seule : GET/WALK
uniquement, jamais de SET.
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

mcp = Server("snmp")

_MAX_BYTES = 20_000


def _run(cmd: list[str], timeout: float = 30.0) -> str:
    if shutil.which(cmd[0]) is None:
        return f"ERREUR: '{cmd[0]}' introuvable dans PATH (paquet net-snmp)"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > _MAX_BYTES:
            out = out[:_MAX_BYTES] + f"\n... (tronqué à {_MAX_BYTES} octets)"
        return out.strip() or "(aucune sortie)"
    except subprocess.TimeoutExpired:
        return f"ERREUR: timeout après {timeout}s"


def _target_ok(target: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9.:_-]+", target))


def _oid_ok(oid: str) -> bool:
    return bool(re.fullmatch(r"[0-9.]+", oid)) or oid.startswith("IF-MIB::")


@mcp.tool()
def snmp_get(target: str, oid: str, community: str = "") -> str:
    """SNMP GET (v2c) : valeur d'un OID sur une cible. community: chaîne SNMP
    (défaut: $JARVIS_SNMP_COMMUNITY ou 'public'). Ex d'OID: 1.3.6.0.2.1.1.3.0
    (sysUpTime), 1.3.6.0.2.1.1.5.0 (sysName)."""
    if not _target_ok(target) or not _oid_ok(oid):
        return "ERREUR: cible ou OID invalide."
    community = community or os.environ.get("JARVIS_SNMP_COMMUNITY", "public")
    return _run(["snmpget", "-v", "2c", "-c", community, "-t", "5", "-r", "1",
                 target, oid])


@mcp.tool()
def snmp_walk(target: str, oid: str = "1.3.6.0.2.1.1", community: str = "") -> str:
    """SNMP WALK (v2c) sous un OID — ex: 1.3.6.0.2.1.1 (infos système),
    1.3.6.0.2.1.2.2 (interfaces), 1.3.6.0.2.1.31.1.1.1 (ifTable étendu)."""
    if not _target_ok(target) or not _oid_ok(oid):
        return "ERREUR: cible ou OID invalide."
    community = community or os.environ.get("JARVIS_SNMP_COMMUNITY", "public")
    return _run(["snmpwalk", "-v", "2c", "-c", community, "-t", "5", "-r", "1",
                 target, oid], timeout=60.0)


@mcp.tool()
def snmp_interfaces(target: str, community: str = "") -> str:
    """Interfaces réseau d'un équipement SNMP : ifDescr, ifOperStatus, vitesses
    (ifHighSpeed en Mb/s), compteurs octets. OID ifTable : 1.3.6.0.2.1.2.2 +
    1.3.6.0.2.1.31.1.1.1.1."""
    if not _target_ok(target):
        return "ERREUR: cible invalide."
    community = community or os.environ.get("JARVIS_SNMP_COMMUNITY", "public")
    out = ["== ifDescr/ifOperStatus/ifSpeed ==",
           _run(["snmpwalk", "-v", "2c", "-c", community, target, "1.3.6.0.2.1.2.2"], timeout=60.0),
           "", "== ifName/ifHighSpeed ==",
           _run(["snmpwalk", "-v", "2c", "-c", community, target, "1.3.6.0.2.1.31.1.1.1"], timeout=60.0)]
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run()  # stdio
