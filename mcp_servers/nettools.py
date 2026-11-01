"""MCP server outils réseau : ping, dig, traceroute, whois, ss, ip, TLS, HTTP.

Interface stable : ajouter/retirer des outils ici ne change jamais le harnais.
Étendu plus tard par les vrais outils CyberLLM (scan, CVE, logs...).
"""

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

import ipaddress
import os
import re
import shutil
import socket
import subprocess
import urllib.parse

mcp = Server("nettools")

_MAX_BYTES = 20_000


def _check_url_ssrf(url: str, allow_private: bool = False) -> str | None:
    """Garde SSRF : http(s) uniquement et l'hôte ne doit résoudre vers aucune IP
    loopback, link-local (169.254.169.254 métadonnées cloud), non routable ou
    privée (sauf allow_private=True). Toutes les IP résolues sont vérifiées."""
    if not url.startswith(("http://", "https://")):
        return "ERREUR: url doit commencer par http:// ou https://"
    host = urllib.parse.urlsplit(url).hostname or ""
    if not host:
        return "ERREUR: url sans hôte."
    try:
        addrs = [host]
        ipaddress.ip_address(host)
    except ValueError:
        try:
            addrs = [info[4][0] for info in socket.getaddrinfo(host, None)]
        except socket.gaierror:
            return f"ERREUR: hôte '{host}' non résoluble."
    # JARVIS_FETCH_ALLOW_PRIVATE=1 : override explicite (tests locaux) —
    # désactive les contrôles d'IP, jamais la vérification de schéma.
    if os.environ.get("JARVIS_FETCH_ALLOW_PRIVATE") == "1":
        return None
    for a in addrs:
        addr = ipaddress.ip_address(a)
        if addr.is_loopback or addr.is_link_local or addr.is_unspecified or addr.is_multicast:
            return f"ERREUR SSRF: '{host}' résout vers {addr} (loopback/link-local interdit)."
        if not allow_private and addr.is_private:
            return f"ERREUR SSRF: '{host}' résout vers {addr} (privée, allow_private=True pour un service interne)."
    return None


_HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:_%-]*")


def _check_host(host: str) -> str | None:
    """Valide un hôte/cible passé en argument positionnel à un binaire externe
    (ping, dig, whois...) : jamais de valeur commençant par '-' (sinon prise
    pour un flag : --help, --output=..., etc.), caractères sûrs uniquement."""
    if not host or not _HOST_RE.fullmatch(host):
        return f"ERREUR: hôte invalide ({host!r})."
    return None


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
    err = _check_host(host)
    if err:
        return err
    return _run(["ping", "-c", str(max(1, min(count, 10))), host])


@mcp.tool()
def dig(query: str, record: str = "A", server: str | None = None) -> str:
    """Requête DNS via dig. record: A, AAAA, MX, TXT, PTR... server: résolveur optionnel."""
    err = _check_host(query)
    if err:
        return err
    if not re.fullmatch(r"[A-Za-z0-9]{1,15}", record or ""):
        return f"ERREUR: type d'enregistrement invalide ({record!r})."
    cmd = ["dig", "+noall", "+answer", query, record]
    if server:
        serr = _check_host(server)
        if serr:
            return serr
        cmd.append(f"@{server}")
    return _run(cmd)


@mcp.tool()
def traceroute(host: str) -> str:
    """Traceroute vers un hôte."""
    err = _check_host(host)
    if err:
        return err
    return _run(["traceroute", "-w", "2", "-q", "1", host], timeout=60.0)


@mcp.tool()
def whois(query: str) -> str:
    """WHOIS d'un domaine ou d'une IP (enregistrement, registrar, dates)."""
    err = _check_host(query)
    if err:
        return err
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
    err = _check_host(host)
    if err:
        return err
    if not 1 <= port <= 65535:
        return f"ERREUR: port invalide ({port})."
    try:
        proc = subprocess.run(
            ["openssl", "s_client", "-connect", f"{host}:{port}", "-servername", host],
            input="", capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return f"ERREUR: timeout openssl s_client vers {host}:{port}"
    out = proc.stdout or ""
    err = proc.stderr or ""
    if "CONNECTED" not in err and not out:
        return f"ERREUR: connexion à {host}:{port} impossible ({err.strip()[:300]})"
    try:
        cert = subprocess.run(
            ["openssl", "s_client", "-connect", f"{host}:{port}", "-servername", host,
             "-showcerts"],
            input="", capture_output=True, text=True, timeout=20).stdout
    except subprocess.TimeoutExpired:
        cert = ""
    lines = ["== s_client ==", (out + err).strip()[:4000], "", "== chaîne ==",
             cert[:_MAX_BYTES]]
    pem_begin = "-----BEGIN CERTIFICATE-----"
    pem_end = "-----END CERTIFICATE-----"
    any_body = "(" + chr(46) + chr(42) + "?)"
    m = re.search(re.escape(pem_begin) + any_body + re.escape(pem_end), cert, re.S)
    if m:
        try:
            p = subprocess.run(["openssl", "x509", "-noout", "-subject", "-issuer",
                                "-dates", "-ext", "subjectAltName"],
                               input=pem_begin + m.group(1) + pem_end,
                               capture_output=True, text=True, timeout=10)
            lines += ["", "== certificat serveur ==", (p.stdout or p.stderr).strip()]
        except subprocess.TimeoutExpired:
            pass
    return "\n".join(lines)



@mcp.tool()
def bandwidth_test(server: str, duration: int = 10) -> str:
    """Test de débit via iperf3 (client TCP, upload+download) vers un serveur
    iperf3 -s que tu contrôles. server: hôte OBLIGATOIRE (aucun serveur public
    n'est contacté par défaut). duration: secondes par sens (max 30)."""
    err = _check_host(server or "")
    if err or not server:
        return "ERREUR: server requis (hôte d'un iperf3 -s, ex: 192.168.1.10)."
    if shutil.which("iperf3") is None:
        return "ERREUR: 'iperf3' introuvable dans PATH"
    duration = max(1, min(duration, 30))
    cmd = ["iperf3", "-c", server, "-t", str(duration), "-J"]
    out = _run(cmd, timeout=float(duration * 2 + 30))
    # sortie JSON : on extrait les débits lisibles
    try:
        import json as _json
        data = _json.loads(out)
        end = data.get("end", {})
        sum_sent = end.get("sum_sent", {})
        sum_recv = end.get("sum_received", {})
        def mbps(bps):
            return f"{bps / 1e6:.1f} Mb/s" if bps else "?"
        return (f"iperf3 vers {server} ({duration}s):\n"
                f"  Upload:   {mbps(sum_sent.get('bits_per_second', 0))}\n"
                f"  Download: {mbps(sum_recv.get('bits_per_second', 0))}")
    except Exception:
        return out  # sortie brute si le JSON n'est pas parsable


@mcp.tool()
def http_get(url: str, timeout: float = 15.0, allow_private: bool = False) -> str:
    """Requête HTTP(S) GET : statut, headers de réponse et début du corps
    (tronqué). url doit commencer par http:// ou https://. Garde SSRF :
    loopback/link-local toujours refusés ; IP privée refusée sauf
    allow_private=True (service interne)."""
    err = _check_url_ssrf(url, allow_private)
    if err:
        return err
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
