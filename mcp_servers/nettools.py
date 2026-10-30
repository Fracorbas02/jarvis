"""MCP server exemple : outils réseau (ping, dig).

Interface stable : ajouter/retirer des outils ici ne change jamais le harnais.
Étendu plus tard par les vrais outils CyberLLM (scan, CVE, logs...).
"""

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

import shutil
import subprocess

mcp = Server("nettools")


def _run(cmd: list[str], timeout: float = 15.0) -> str:
    if shutil.which(cmd[0]) is None:
        return f"ERREUR: '{cmd[0]}' introuvable dans PATH"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (proc.stdout or "") + (proc.stderr or "") or "(aucune sortie)"
    except subprocess.TimeoutExpired:
        return f"ERREUR: timeout après {timeout}s"


@mcp.tool()
def ping(host: str, count: int = 4) -> str:
    """Ping ICMP vers un hôte. Retourne la sortie brute de ping."""
    return _run(["ping", "-c", str(max(1, min(count, 10))), host])


@mcp.tool()
def dig(query: str, record: str = "A", server: str | None = None) -> str:
    """Requête DNS via dig. record: A, AAAA, MX, TXT, PTR... server: résolveur optionnel."""
    cmd = ["dig", "+noall", "+answer", query, record]
    if server:
        cmd.append(f"@{server}")
    return _run(cmd)


@mcp.tool()
def traceroute(host: str) -> str:
    """Traceroute vers un hôte."""
    return _run(["traceroute", "-w", "2", "-q", "1", host], timeout=60.0)


if __name__ == "__main__":
    mcp.run()  # stdio
