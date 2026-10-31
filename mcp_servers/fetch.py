"""MCP server fetch — récupère une page web et la convertit en texte lisible.

Stdlib uniquement : urllib pour le téléchargement, extraction du texte par
suppression des balises <script>/<style> et décodage des entités HTML.
Limite de taille configurable (défaut 100 ko de texte). Protocoles autorisés :
http/https uniquement.
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.parse
import urllib.request

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("fetch")

_MAX_BYTES = 100_000
_TIMEOUT = 20.0


def _to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|li|tr|h[1-6]|section|article)>", "\n\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


@mcp.tool()
def fetch_page(url: str, max_chars: int = 20000) -> str:
    """Télécharge une page http(s) et renvoie son texte brut (lisible, sans
    HTML). max_chars: plafond de caractères retournés (max 100000). Utile pour
    faire lire/résumer une doc ou un article par le LLM."""
    if not url.startswith(("http://", "https://")):
        return "ERREUR: url doit commencer par http:// ou https://"
    max_chars = max(100, min(max_chars, _MAX_BYTES))
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; jarvis-fetch/0.1)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    })
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(_MAX_BYTES * 3).decode(errors="replace")
            final_url = resp.geturl()
            status = resp.status
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {exc.reason} pour {url}"
    except urllib.error.URLError as exc:
        return f"ERREUR: {exc.reason}"
    if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
        return f"Type non texte ({ctype}) — page refusée pour {url}"
    text = _to_text(raw)
    header = f"URL: {final_url} (HTTP {status}, {ctype})"
    if len(text) > max_chars:
        return f"{header}\n\n{text[:max_chars]}\n... (tronqué à {max_chars} caractères)"
    return f"{header}\n\n{text}"


@mcp.tool()
def fetch_headers(url: str) -> str:
    """En-têtes HTTP d'une URL (méthode HEAD si possible, GET sinon) : statut,
    Content-Type, Content-Length, Last-Modified, Server... Sans le corps."""
    if not url.startswith(("http://", "https://")):
        return "ERREUR: url doit commencer par http:// ou https://"
    req = urllib.request.Request(url, method="HEAD", headers={
        "User-Agent": "jarvis-fetch/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            headers = "\n".join(f"{k}: {v}" for k, v in resp.headers.items())
            return f"HTTP {resp.status} {resp.reason} pour {resp.geturl()}\n{headers}"
    except urllib.error.HTTPError as exc:
        if exc.code == 405:  # HEAD refusé -> GET avec lecture minimale
            req = urllib.request.Request(url, headers={"User-Agent": "jarvis-fetch/0.1"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                headers = "\n".join(f"{k}: {v}" for k, v in resp.headers.items())
                return f"HTTP {resp.status} {resp.reason} pour {resp.geturl()}\n{headers}"
        return f"HTTP {exc.code} {exc.reason} pour {url}"
    except urllib.error.URLError as exc:
        return f"ERREUR: {exc.reason}"


if __name__ == "__main__":
    mcp.run()  # stdio
