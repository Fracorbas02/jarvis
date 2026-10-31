"""MCP server GHSA — GitHub Security Advisories via l'API REST publique.

Sans token : ~60 req/h en anonyme. Avec GITHUB_TOKEN (classic, scope public_repo
ou aucun scope pour les advisories publics) : 5000 req/h. Les advisories sont
toutes publiques en lecture.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("ghsa")

_GHSA_BASE = "https://api.github.com/advisories"
_MIN_INTERVAL = 1.2  # 60 req/h anonyme -> marge de sécurité
_last_call = 0.0


def _gh_get(url: str, timeout: float = 30.0) -> object:
    global _last_call
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    headers = {
        "User-Agent": "jarvis-ghsa/0.1",
        "Accept": "application/vnd.github+json",
    }
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise ValueError("GHSA: limite de débit GitHub atteinte (60 req/h sans token). "
                             "Définis GITHUB_TOKEN pour 5000 req/h.") from exc
        if exc.code == 404:
            raise ValueError("GHSA: aucune donnée (404).") from exc
        raise ValueError(f"GHSA: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"GHSA: inaccessible ({exc.reason})") from exc


def _fmt(a: dict, summary_len: int = 150) -> str:
    sev = ((a.get("severity") or "?")).upper()
    ghsa = a.get("ghsa_id", "?")
    cve = a.get("cve_id") or "-"
    summ = (a.get("summary") or "").replace("\n", " ")
    pkgs = ", ".join(f"{p.get('ecosystem', '?')}:{p.get('package', {}).get('name', '?')}"
                     for p in a.get("vulnerabilities", [])[:4])
    return f"{ghsa} | {cve} | {sev} | {a.get('published_at', '?')[:10]} | {pkgs} | {summ[:summary_len]}"


@mcp.tool()
def ghsa_search(query: str, ecosystem: str = "", limit: int = 20) -> str:
    """Recherche d'advisories GitHub (GHSA). query: texte libre (produit, CVE,
    mot-clé). ecosystem: filtre optionnel (npm, PyPI, Go, Maven, RubyGems...).
    limit: max 100. Triés du plus récent au plus ancien."""
    query = query.strip()
    if not query:
        return "ERREUR: query vide."
    limit = max(1, min(limit, 100))
    params = {"q": query, "per_page": limit, "sort": "published", "order": "desc"}
    if ecosystem:
        params["ecosystem"] = ecosystem.strip().lower()
    url = _GHSA_BASE + "?" + urllib.parse.urlencode(params)
    try:
        rows = _gh_get(url)
    except ValueError as exc:
        return f"ERREUR: {exc}"
    if not isinstance(rows, list) or not rows:
        return f"Aucun advisory pour « {query} »" + (f" ({ecosystem})" if ecosystem else "") + "."
    out = [f"{len(rows)} advisory(s) GHSA pour « {query} »"
           + (f" [{ecosystem}]" if ecosystem else "") + ":"]
    out.extend(f"- {_fmt(a)}" for a in rows)
    return "\n".join(out)


@mcp.tool()
def ghsa_for_package(package: str, ecosystem: str = "npm", limit: int = 20) -> str:
    """Advisories affectant un package précis (dépendance d'un de tes projets).
    package: nom exact (ex: 'lodash', 'requests'). ecosystem: npm (défaut),
    PyPI, Go, Maven, Cargo, NuGet... limit: max 100."""
    package = package.strip()
    if not package:
        return "ERREUR: nom de package vide."
    limit = max(1, min(limit, 100))
    params = {"package": package, "ecosystem": ecosystem.strip().lower(),
              "per_page": limit, "sort": "published", "order": "desc"}
    try:
        rows = _gh_get(_GHSA_BASE + "?" + urllib.parse.urlencode(params))
    except ValueError as exc:
        return f"ERREUR: {exc}"
    if not isinstance(rows, list) or not rows:
        return f"Aucun advisory pour {ecosystem}/{package}."
    out = [f"{len(rows)} advisory(s) pour {ecosystem}/{package} :"]
    out.extend(f"- {_fmt(a)}" for a in rows)
    out.append("Utilise ghsa_info pour le détail (versions affectées, correctif).")
    return "\n".join(out)


@mcp.tool()
def ghsa_info(ghsa_id: str) -> str:
    """Détail d'un advisory : GHSA/CVE, sévérité, CVSS, packages et versions
    affectées, versions corrigées (patched), références, date de publication."""
    ghsa_id = ghsa_id.strip().upper()
    if not ghsa_id.startswith("GHSA-"):
        return "ERREUR: identifiant attendu au format GHSA-xxxx-xxxx-xxxx."
    try:
        a = _gh_get(f"https://api.github.com/advisories/{ghsa_id}")
    except ValueError as exc:
        return f"ERREUR: {exc}"
    if not isinstance(a, dict) or not a.get("ghsa_id"):
        return f"Advisory {ghsa_id} introuvable."
    out = [f"=== {a.get('ghsa_id')} ===",
           f"CVE: {a.get('cve_id') or '-'} | Sévérité: {a.get('severity')} | "
           f"CVSS: {(a.get('cvss') or {}).get('score', '-')}",
           f"Publié: {a.get('published_at', '?')[:10]} | Modifié: {a.get('updated_at', '?')[:10]}",
           f"Résumé: {a.get('summary', '')}",
           "", a.get("description", "")[:1500]]
    vulns = a.get("vulnerabilities", [])
    if vulns:
        out += ["", f"Paquets affectés ({len(vulns)}):"]
        for v in vulns[:15]:
            pkg = v.get("package", {})
            out.append(f"  {v.get('ecosystem')}/{pkg.get('name')}: "
                       f"vulnérable {v.get('vulnerable_version_range', '?')} "
                       f"-> corrigé {v.get('patched_versions', '?') or 'jamais'}")
    refs = [r for r in a.get("references", []) if isinstance(r, dict) and r.get("url")]
    if refs:
        out += ["", "Références:"] + [f"  {r['url']}" for r in refs[:6]]
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run()  # stdio
