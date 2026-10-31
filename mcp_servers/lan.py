"""MCP server découverte LAN — hôtes actifs, table ARP, baux DHCP locaux.

Lecture seule : /proc/net/arp, baux DHCP, et éventuellement arp-scan si
installé. Aucun scan agressif.
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

mcp = Server("lan")

_MAX_BYTES = 20_000


def _read(path: str, limit: int = _MAX_BYTES) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit).strip() or f"({path} vide)"
    except OSError as exc:
        return f"ERREUR: {path} illisible ({exc.strerror})"


def _run(cmd: list[str], timeout: float = 60.0) -> str:
    if shutil.which(cmd[0]) is None:
        return f"ERREUR: '{cmd[0]}' introuvable dans PATH"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (proc.stdout or "")[:_MAX_BYTES] + (proc.stderr or "")[:1000]
    except subprocess.TimeoutExpired:
        return f"ERREUR: timeout après {timeout}s"


@mcp.tool()
def arp_table() -> str:
    """Table ARP locale (/proc/net/arp) : IP ↔ MAC des voisins connus, avec
    l'interface. Vide = aucun voisin en cache (normal après un moment)."""
    return _read("/proc/net/arp")


@mcp.tool()
def dhcp_leases(path: str = "") -> str:
    """Baux DHCP actifs lus dans un fichier de leases (défaut:
    /var/lib/dhcp/dhcpd.leases, puis /var/lib/dhcpcd/, puis NetworkManager).
    path: fichier de leases explicite optionnel. Renvoie IP, MAC, hostname,
    expiration pour chaque bail actif."""
    found = None
    if path:
        found = path if os.path.isfile(path) else None
    else:
        nm_dir = "/var/lib/NetworkManager"
        if os.path.isdir(nm_dir):
            leases = sorted(f for f in os.listdir(nm_dir)
                            if f.startswith("dhcp-client") or "lease" in f.lower())
            if leases:
                found = os.path.join(nm_dir, leases[-1])
        if not found:
            for c in ("/var/lib/dhcp/dhcpd.leases", "/var/lib/dhcp/dhcpd6.leases",
                      "/var/lib/dhcpcd/dhcpcd.leases"):
                if os.path.isfile(c):
                    found = c
                    break
    if not found:
        return ("ERREUR: aucun fichier de leases trouvé (dhcpd/dhcpcd/NetworkManager). "
                "Passe path= explicitement.")
    text = _read(found)
    # parse simple des blocs "lease IP { ... }"
    blocks = re.findall(r"lease\s+(\d+\.\d+\.\d+\.\d+)\s*\{(.*?)\}", text, re.S)
    if not blocks:
        return f"{found}: aucun bail parsé ({len(text)} octets lus)."
    lines = [f"Baux DHCP dans {found} ({len(blocks)}):"]
    for ip, body in blocks:
        mac = re.search(r"hardware ethernet\s+([0-9a-fA-F:]{17})", body)
        host = re.search(r"client-hostname\s+\"([^\"]+)\"", body)
        end = re.search(r"ends\s+\d+\s+([\d/]+\s+[\d:]+)", body)
        lines.append(f"- {ip} | {mac.group(1) if mac else '?'} | "
                     f"{host.group(1) if host else '?'} | expire {end.group(1) if end else '?'}")
    return "\n".join(lines[:201])


@mcp.tool()
def lan_hosts(subnet: str = "") -> str:
    """Découverte douce des hôtes du LAN : arp-scan si installé, sinon ping
    sweep léger (1 paquet, timeout 1s) sur le sous-réseau donné ou déduit de
    l'interface principale. subnet: ex '192.168.1.0/24' (défaut: /24 du premier
    IPv4 privé trouvé)."""
    if subnet:
        if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+/\d{1,2}", subnet):
            return "ERREUR: subnet invalide (ex: 192.168.1.0/24)."
    else:
        subnet = _default_subnet()
        if not subnet:
            return "ERREUR: aucun IPv4 privé détecté pour deviner le LAN."
    if shutil.which("arp-scan"):
        return _run(["arp-scan", "--localnet", "-q"], timeout=90.0) \
            if subnet == _default_subnet() else _run(["arp-scan", subnet], timeout=90.0)
    # fallback : ping sweep via nmap si dispo, sinon fping
    if shutil.which("nmap"):
        return _run(["nmap", "-sn", "-T3", subnet], timeout=120.0)
    if shutil.which("fping"):
        import ipaddress
        net = ipaddress.ip_network(subnet, strict=False)
        hosts = [str(h) for h in net.hosts()][:254]
        return _run(["fping", "-a", "-g", subnet], timeout=120.0)
    return ("ERREUR: ni arp-scan, ni nmap, ni fping — impossible de scanner. "
            f"(LAN déduit: {subnet})")


def _default_subnet() -> str:
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # ne envoie rien, sert juste à lire l'IP source
        ip = s.getsockname()[0]
        s.close()
        if ip.startswith(("127.", "169.254.")):
            return ""
        parts = ip.split(".")
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    except OSError:
        return ""


if __name__ == "__main__":
    mcp.run()  # stdio
