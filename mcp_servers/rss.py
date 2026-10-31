"""MCP server RSS/Atom — veille de flux sans API, stdlib uniquement (xml.etree).

rss_latest(feed_url, n) : les N derniers articles d'un flux (titre, date,
lien, résumé tronqué). rss_check(feed_url) : statut du flux (titre du site,
nombre d'articles, date du dernier).
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("rss")

_TIMEOUT = 20.0
_MAX_BYTES = 2_000_000


def _check_url(url: str) -> None:
    """Garde SSRF : http(s) uniquement, et l'hôte ne doit résoudre vers aucune
    IP loopback, link-local (169.254.169.254), non routable ou privée — un flux
    RSS interne n'a pas de sens ici. Toutes les IP résolues sont vérifiées."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("url doit commencer par http:// ou https://")
    host = urllib.parse.urlsplit(url).hostname or ""
    if not host:
        raise ValueError("url sans hôte.")
    try:
        ipaddress.ip_address(host)
        addrs = [host]
    except ValueError:
        try:
            addrs = [info[4][0] for info in socket.getaddrinfo(host, None)]
        except socket.gaierror as exc:
            raise ValueError(f"hôte '{host}' non résoluble ({exc})")
    # JARVIS_FETCH_ALLOW_PRIVATE=1 : override explicite (tests locaux sur
    # 127.0.0.1) — désactive les contrôles d'IP, jamais la vérification de schéma.
    if os.environ.get("JARVIS_FETCH_ALLOW_PRIVATE") == "1":
        return
    for a in addrs:
        addr = ipaddress.ip_address(a)
        if (addr.is_loopback or addr.is_link_local or addr.is_unspecified
                or addr.is_multicast or addr.is_private):
            raise ValueError(f"SSRF: '{host}' résout vers {addr} (privée/loopback/link-local interdit).")


def _fetch_feed(url: str) -> ET.Element:
    _check_url(url)
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


_ATOM = "{http://www.w3.org/2005/Atom}"


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _find_any(el: ET.Element, *names: str) -> ET.Element | None:
    """Premier élément trouvé parmi les variantes RSS 2.0 et Atom namespacé.
    (Jamais de `or` sur des Element : un élément sans enfants est falsy.)"""
    for name in names:
        found = el.find(name)
        if found is None:
            found = el.find(_ATOM + name)
        if found is not None:
            return found
    return None


def _entry_link(item: ET.Element) -> str:
    """Lien d'une entrée : RSS <link> texte, ou Atom <link href=...> (rel=alternate
    prioritaire, sinon le premier)."""
    for link_el in item.findall("link") + item.findall(_ATOM + "link"):
        rel = link_el.get("rel", "alternate")
        href = link_el.get("href", "") or _text(link_el)
        if rel == "alternate" and href:
            return href
    for link_el in item.findall("link") + item.findall(_ATOM + "link"):
        href = link_el.get("href", "") or _text(link_el)
        if href:
            return href
    return ""


def _entry_fields(item: ET.Element) -> dict:
    return {
        "title": _text(_find_any(item, "title")),
        "link": _entry_link(item),
        "published": (_text(_find_any(item, "published", "pubDate"))
                      or _text(_find_any(item, "updated"))),
        "summary": (_text(_find_any(item, "summary", "description", "content"))
                    or _text(_find_any(item, "subtitle"))),
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
    items = root.findall(".//item") or root.findall(".//" + _ATOM + "entry")
    if not items:
        return f"Aucun article dans le flux {feed_url}."
    channel = root.find("channel")
    site_el = None
    if channel is not None:
        site_el = channel.find("title")
    if site_el is None:
        site_el = root.find(_ATOM + "title")
    if site_el is None:
        site_el = root.find("title")
    site = _text(site_el)
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
    items = root.findall(".//item") or root.findall(".//" + _ATOM + "entry")
    title = _text(_find_any(channel, "title"))
    desc = _text(_find_any(channel, "description", "subtitle"))
    first = _entry_fields(items[0])["published"][:19] if items else "?"
    return (f"Flux OK: « {title} »\nURL: {feed_url}\nDescription: {desc[:200]}\n"
            f"{len(items)} article(s), le plus récent: {first}")


if __name__ == "__main__":
    mcp.run()  # stdio
