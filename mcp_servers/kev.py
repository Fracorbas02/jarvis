"""MCP server CISA KEV — Known Exploited Vulnerabilities (catalogue public, sans clé).

API JSON : https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
Le catalogue (~2 Mo) est mis en cache 10 min en mémoire pour éviter de le
retélécharger à chaque appel.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("kev")

_KEV_URL = ("https://www.cisa.gov/sites/default/files/feeds/"
            "known_exploited_vulnerabilities.json")
_CACHE_TTL = 600.0
_cache: tuple[float, list[dict]] = (0.0, [])


def _catalog() -> list[dict]:
    global _cache
    now = time.monotonic()
    if now - _cache[0] < _CACHE_TTL and _cache[1]:
        return _cache[1]
    req = urllib.request.Request(_KEV_URL, headers={
        "User-Agent": "jarvis-kev/0.1", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"KEV: catalogue inaccessible ({exc})")
    vulns = data.get("vulnerabilities", [])
    _cache = (now, vulns)
    return vulns


def _fmt_line(v: dict, summary_len: int = 160) -> str:
    ransom = "RANSOMWARE" if v.get("knownRansomwareCampaignUse") == "Known" else "-"
    name = (v.get("vulnerabilityName") or "").replace("\n", " ")
    return (f"{v.get('cveID', '?')} | {v.get('vendorProject', '?')} / "
            f"{v.get('product', '?')} | ajouté {v.get('dateAdded', '?')} "
            f"| échéance {v.get('dueDate', '?')} | {ransom} | {name[:summary_len]}")


@mcp.tool()
def kev_recent(days: int = 30, limit: int = 20) -> str:
    """CVEs ajoutées au catalogue KEV (exploitées activement) sur les N derniers
    jours (max 365), triées de la plus récente. limit: max 100."""
    days = max(1, min(days, 365))
    limit = max(1, min(limit, 100))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    try:
        catalog = _catalog()
    except ValueError as exc:
        return f"ERREUR: {exc}"
    vulns = [v for v in catalog if (v.get("dateAdded") or "") >= cutoff]
    vulns.sort(key=lambda v: v.get("dateAdded") or "", reverse=True)
    if not vulns:
        return f"Aucune entrée KEV ajoutée ces {days} derniers jours."
    out = [f"{len(vulns)} entrée(s) KEV ajoutées ces {days} derniers jours "
           f"({min(limit, len(vulns))} affichées):"]
    out.extend(f"- {_fmt_line(v)}" for v in vulns[:limit])
    out.append("Utilise kev_info pour le détail d'une entrée.")
    return "\n".join(out)


@mcp.tool()
def kev_search(query: str, limit: int = 20) -> str:
    """Recherche dans le catalogue KEV (éditeur, produit, nom de vulnérabilité,
    CVE). limit: max 100. Résultats triés du plus récent ajout au plus ancien."""
    query = query.strip().lower()
    if not query:
        return "ERREUR: query vide."
    limit = max(1, min(limit, 100))
    try:
        catalog = _catalog()
    except ValueError as exc:
        return f"ERREUR: {exc}"
    hits = [v for v in catalog
            if query in (v.get("cveID", "") + " " + v.get("vendorProject", "") + " "
                        + v.get("product", "") + " " + v.get("vulnerabilityName", "")).lower()]
    hits.sort(key=lambda v: v.get("dateAdded") or "", reverse=True)
    if not hits:
        return f"Aucune entrée KEV pour « {query} »."
    out = [f"{len(hits)} entrée(s) KEV pour « {query} » ({min(limit, len(hits))} affichées):"]
    out.extend(f"- {_fmt_line(v)}" for v in hits[:limit])
    return "\n".join(out)


@mcp.tool()
def kev_info(cve_id: str) -> str:
    """Détail complet d'une entrée KEV : CVE, éditeur/produit, action requise
    (requiredAction), échéance (dueDate), usage ransomware, CWE, notes."""
    cve_id = cve_id.strip().upper()
    if not cve_id.startswith("CVE-"):
        return "ERREUR: identifiant attendu au format CVE-YYYY-NNNN."
    try:
        catalog = _catalog()
    except ValueError as exc:
        return f"ERREUR: {exc}"
    for v in catalog:
        if v.get("cveID", "").upper() == cve_id:
            out = [f"=== KEV {v.get('cveID')} ===",
                   f"Éditeur/produit: {v.get('vendorProject')} / {v.get('product')}",
                   f"Ajoutée: {v.get('dateAdded')} | Échéance: {v.get('dueDate')}",
                   f"Ransomware: {v.get('knownRansomwareCampaignUse')}",
                   f"Action requise: {v.get('requiredAction')}",
                   f"CWE: {', '.join(v.get('cwes') or []) or '-'}",
                   "", v.get("vulnerabilityName", "")]
            if v.get("notes"):
                out += ["", f"Notes: {v['notes']}"]
            return "\n".join(out)
    return f"{cve_id} n'est pas dans le catalogue KEV (pas d'exploitation connue documentée)."


if __name__ == "__main__":
    mcp.run()  # stdio
