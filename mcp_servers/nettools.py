"""MCP server outils réseau : ping, dig, traceroute, whois, ss, ip, TLS, HTTP.

Interface stable : ajouter/retirer des outils ici ne change jamais le harnais.
Étendu plus tard par les vrais outils CyberLLM (scan, CVE, logs...).
"""

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

import re
import shutil
import subprocess

mcp = Server("nettools")

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


@mcp.tool()
def whois(query: str) -> str:
    """WHOIS d'un domaine ou d'une IP (enregistrement, registrar, dates)."""
    return _run(["whois", query], timeout=30.0)


@mcp.tool()
def listening_ports() -> str:
    """Ports TCP/UDP en écoute sur la machine locale (ss -tulpn)."""
    if shutil.which("ss"):
        return _run(["ss", "-tulpn"])
    return _run(["netstat", "-tulpn"])


@mcp.tool()
def connections() -> str:
    """Connexions réseau établies (ss -tnp) : état, local, distant, processus."""
    if shutil.which("ss"):
        return _run(["ss", "-tnp"])
    return _run(["netstat", "-tnp"])


@mcp.tool()
def ip_addr() -> str:
    """Adresses IP des interfaces + routes par défaut (ip -brief addr, ip route)."""
    out = _run(["ip", "-brief", "addr"])
    out += "\n\n== routes ==\n" + _run(["ip", "route"])
    return out


@mcp.tool()
def tls_info(host: str, port: int = 443) -> str:
    """Chaîne de certificats TLS d'un hôte : émetteur, sujet, dates de validité,
    SANs, algorithme de signature. Utile pour surveiller l'expiration."""
    if shutil.which("openssl") is None:
        return "ERREUR: 'openssl' introuvable dans PATH"
    proc = subprocess.run(
        ["openssl", "s_client", "-connect", f"{host}:{port}", "-servername", host],
        input="", capture_output=True, text=True, timeout=20)
    out = proc.stdout or ""
    err = proc.stderr or ""
    if "CONNECTED" not in err and not out:
        return f"ERREUR: connexion à {host}:{port} impossible ({err.strip()[:300]})"
    # certificat du serveur + chaîne
    cert = subprocess.run(
        ["openssl", "s_client", "-connect", f"{host}:{port}", "-servername", host,
         "-showcerts"],
        input="", capture_output=True, text=True, timeout=20).stdout
    lines = ["== s_client ==", (out + err).strip()[:4000], "", "== chaîne ==",
             cert[:_MAX_BYTES]]
    # dates de validité du certificat serveur
    m = re.search(r"-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----", cert, re.S)
    if m:
        p = subprocess.run(["openssl", "x509", "-noout", "-subject", "-issuer",
                            "-dates", "-ext", "subjectAltName"],
                           input="-----BEGIN CERTIFICATE-----" + m.group(1) + "-----END CERTIFICATE-----",
                           capture_output=True, text=True, timeout=10)
        lines += ["", "== certificat serveur ==", (p.stdout or p.stderr).strip()]
    return "\n".join(lines)


@mcp.tool()
def http_get(url: str, timeout: float = 15.0) -> str:
    """Requête HTTP(S) GET : statut, headers de réponse et début du corps
    (tronqué). url doit commencer par http:// ou https://."""
    if not url.startswith(("http://", "https://")):
        return "ERREUR: url doit commencer par http:// ou https://"
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, headers={"User-Agent": "jarvis-nettools/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(_MAX_BYTES + 1).decode(errors="replace")
            headers = "\n".join(f"{k}: {v}" for k, v in resp.headers.items())
            out = f"HTTP {resp.status} {resp.reason}\n{headers}\n\n{body}"
            if len(body) > _MAX_BYTES:
                out += f"\n... (corps tronqué à {_MAX_BYTES} octets)"
            return out
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {exc.reason} pour {url}"
    except urllib.error.URLError as exc:
        return f"ERREUR: {exc.reason}"


if __name__ == "__main__":
    mcp.run()  # stdio
