"""MCP server utilitaires réseau/texte purs (stdlib uniquement, aucun appel
externe) : calcul IP (ipaddress), encodage (base64/hex/url), empreintes
(md5/sha1/sha256/sha512), heure (datetime/zoneinfo).
"""

from __future__ import annotations

import base64
import hashlib
import urllib.parse
from datetime import datetime, timezone
from ipaddress import ip_address, ip_network
from zoneinfo import ZoneInfo, available_timezones

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("netutils")


@mcp.tool()
def ip_calc(subnet: str) -> str:
    """Découpe un réseau CIDR ou une IP : réseau, broadcast, masque, nombre
    d'hôtes utilisables, première/dernière IP utilisable, classe, privé ou non.
    Ex: '192.168.1.0/24' ou '10.0.0.5'."""
    try:
        if "/" in subnet:
            net = ip_network(subnet.strip(), strict=False)
            hosts = list(net.hosts())
            out = [
                f"Réseau:    {net.network_address}/{net.prefixlen}",
                f"Masque:    {net.netmask}",
                f"Broadcast: {net.broadcast_address}",
                f"Hôtes:     {net.num_addresses - 2} utilisables (sur {net.num_addresses} adresses)",
            ]
            if hosts:
                out += [f"1ère IP:   {hosts[0]}", f"Dernière:  {hosts[-1]}"]
            out.append(f"Privé:     {'oui' if net.is_private else 'non'}")
            return "\n".join(out)
        addr = ip_address(subnet.strip())
        klass = ("A" if addr.version == 4 and addr < ip_address("128.0.0.0")
                 else "B" if addr.version == 4 and addr < ip_address("192.0.0.0")
                 else "C" if addr.version == 4 else "IPv6")
        return (f"Adresse:   {addr}\nEntier:    {int(addr)}\n"
                f"Version:   IPv{addr.version} | Classe: {klass}\n"
                f"Privée:    {'oui' if addr.is_private else 'non'}\n"
                f"Loopback:  {'oui' if addr.is_loopback else 'non'}\n"
                f"Lien-local: {'oui' if addr.is_link_local else 'non'}")
    except ValueError as exc:
        return f"ERREUR: adresse/réseau invalide ({exc})"


@mcp.tool()
def ip_convert(ip: str) -> str:
    """Convertit une IP en entier et inversement, et indique la notation
    hexadécimale complète (utile pour certains outils réseau)."""
    try:
        addr = ip_address(ip.strip())
    except ValueError as exc:
        return f"ERREUR: IP invalide ({exc})"
    return (f"IP:       {addr}\nEntier:   {int(addr)}\n"
            f"Hex:      {int(addr):x}"
            + (f"\nComplète: {addr.exploded}" if addr.version == 6 else ""))


@mcp.tool()
def encode(text: str, op: str = "b64encode") -> str:
    """Encode/décode du texte. op: b64encode, b64decode, hexencode, hexdecode,
    urlencode, urldecode, rot13."""
    try:
        if op == "b64encode":
            return base64.b64encode(text.encode()).decode()
        if op == "b64decode":
            return base64.b64decode(text.encode()).decode(errors="replace")
        if op == "hexencode":
            return text.encode().hex()
        if op == "hexdecode":
            return bytes.fromhex(text.strip()).decode(errors="replace")
        if op == "urlencode":
            return urllib.parse.quote(text)
        if op == "urldecode":
            return urllib.parse.unquote(text)
        if op == "rot13":
            import codecs
            return codecs.encode(text, "rot13")
        return (f"ERREUR: op inconnue '{op}'. Ops: b64encode, b64decode, "
                "hexencode, hexdecode, urlencode, urldecode, rot13")
    except Exception as exc:
        return f"ERREUR: {exc}"


@mcp.tool()
def hash_text(text: str, algo: str = "sha256") -> str:
    """Empreinte d'un texte. algo: md5, sha1, sha256 (défaut), sha512, sha3_256."""
    algos = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256,
             "sha512": hashlib.sha512, "sha3_256": hashlib.sha3_256}
    fn = algos.get(algo.lower())
    if fn is None:
        return f"ERREUR: algo inconnue '{algo}'. Disponibles: {', '.join(algos)}"
    h = fn(text.encode())
    return f"{algo.lower()}: {h.hexdigest()}"


@mcp.tool()
def time_now(tz: str = "Europe/Paris") -> str:
    """Heure actuelle dans un fuseau (ex: Europe/Paris, UTC, America/New_York),
    au format ISO 8601 + epoch Unix. Liste les fuseaux si tz='list'."""
    if tz == "list":
        zones = sorted(available_timezones())
        return f"{len(zones)} fuseaux disponibles, ex: {', '.join(zones[:15])} ..."
    try:
        now = datetime.now(ZoneInfo(tz))
    except Exception:
        try:
            now = datetime.now(ZoneInfo(tz)) if tz != "UTC" else datetime.now(timezone.utc)
        except Exception:
            return f"ERREUR: fuseau inconnu '{tz}' (essaie 'UTC' ou time_now(tz='list'))"
    utc = datetime.now(timezone.utc)
    return (f"Local ({tz}): {now.isoformat()}\n"
            f"UTC:          {utc.isoformat()}\n"
            f"Epoch:        {int(utc.timestamp())}")


@mcp.tool()
def time_convert(epoch: int, tz: str = "Europe/Paris") -> str:
    """Convertit un timestamp Unix (epoch) en date lisible, locale et UTC."""
    try:
        utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
        try:
            local = datetime.fromtimestamp(epoch, tz=ZoneInfo(tz))
            local_s = local.isoformat()
        except Exception:
            local_s = "(fuseau inconnu)"
        return f"Epoch {epoch} =\n  UTC:   {utc.isoformat()}\n  {tz}: {local_s}"
    except (OverflowError, OSError, ValueError) as exc:
        return f"ERREUR: epoch invalide ({exc})"


if __name__ == "__main__":
    mcp.run()  # stdio
