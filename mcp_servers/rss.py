"""MCP server RSS/Atom — veille de flux sans API, stdlib uniquement (xml.etree).

rss_latest(feed_url, n) : les N derniers articles d'un flux (titre, date,
lien, résumé tronqué). rss_check(feed_url) : statut du flux (titre du site,
nombre d'articles, date du dernier).
"""

from __future__ import annotations

import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("rss")

_TIMEOUT = 20.0
_MAX_BYTES = 2_000_000


def _fetch_feed(url: str) -> ET.Element:
    if not url.startswith(("http://", "https://")):
        raise ValueError("url doit commencer par http:// ou https://")
    req = urllib.request.Request(url, headers={
        "User-Agent": "jarvis-rss/0.1", "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = resp.read(_MAX_BYTES)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"RSS: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"RSS: inaccessible ({exc.reason})") from exc
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"RSS: XML illisible ({exc})") from exc


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _entry_fields(item: ET.Element) -> dict:
    link_el = item.find("link")
    link = _text(link_el)
    if not link and link_el is not None:
        link = link_el.get("href", "")
    atom = "{http://www.w3.org/2005/Atom}"
    return {
        "title": _text(item.find("title")),
        "link": link,
        "published": (_text(item.find("published")) or _text(item.find("updated"))
                      or _text(item.find("pubDate")) or _text(item.find(atom + "updated"))),
        "summary": (_text(item.find("summary")) or _text(item.find("description"))
                    or _text(item.find(atom + "summary"))),
    }


@mcp.tool()
def rss_latest(feed_url: str, n: int = 10) -> str:
    """Les N derniers articles d'un flux RSS/Atom (max 50) : titre, date, lien,
    résumé tronqué à 300 caractères. Tri: ordre du flux (le plus récent en
    premier en général)."""
    n = max(1, min(n, 50))
    try:
        root = _fetch_feed(feed_url)
    except ValueError as exc:
        return f"ERREUR: {exc}"
    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    if not items:
        return f"Aucun article dans le flux {feed_url}."
    channel = root.find("channel")
    site = _text(channel.find("title")) if channel is not None else ""
    out = [f"Flux « {site or feed_url} » — {len(items)} article(s) au total, {min(n, len(items))} affiché(s):"]
    for item in items[:n]:
        f = _entry_fields(item)
        pub = f["published"][:19].replace("T", " ")
        out.append(f"- [{pub}] {f['title']}\n    {f['link']}\n    {f['summary'][:300]}")
    return "\n".join(out)


@mcp.tool()
def rss_check(feed_url: str) -> str:
    """Vérifie qu'un flux répond : titre du site, langue, nombre d'articles,
    date du premier article. Pour valider une URL de feed avant de s'abonner."""
    try:
        root = _fetch_feed(feed_url)
    except ValueError as exc:
        return f"ERREUR: {exc}"
    channel = root.find("channel")
    if channel is None:
        channel = root  # Atom : le root est le feed
    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    title = _text(channel.find("title"))
    desc = _text(channel.find("description")) or _text(channel.find("{http://www.w3.org/2005/Atom}subtitle"))
    first = _entry_fields(items[0])["published"][:19] if items else "?"
    return (f"Flux OK: « {title} »\nURL: {feed_url}\nDescription: {desc[:200]}\n"
            f"{len(items)} article(s), le plus récent: {first}")


if __name__ == "__main__":
    mcp.run()  # stdio
