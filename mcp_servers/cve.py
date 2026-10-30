"""MCP server CVE — veille sur l'API NVD 2.0 (sans dépendance, stdlib uniquement).

Sans clé API, NVD limite à 5 requêtes / 30 s : un throttle côté client
sérialise et espace les appels. Clé optionnelle via la variable d'environnement
NVD_API_KEY (https://nvd.nist.gov/developers/request-key) — passe alors à
50 req / 30 s.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("cve")

_NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_API_KEY = os.environ.get("NVD_API_KEY", "")
# sans clé : 5 req/30s -> 6.5s d'intervalle ; avec clé : 50 req/30s -> 0.7s
_MIN_INTERVAL = 0.7 if _API_KEY else 6.5
_last_call = 0.0


def _nvd_get(params: dict, timeout: float = 30.0) -> dict:
    """Appel NVD throttlé. Lève une ValueError explicite en cas d'erreur API."""
    global _last_call
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()

    url = _NVD_BASE + "?" + urllib.parse.urlencode(params)
    headers = {
        "User-Agent": "jarvis-cve/0.1",
        "Accept": "application/json",
    }
    if _API_KEY:
        headers["apiKey"] = _API_KEY
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise ValueError(
                "NVD: limite de débit atteinte (5 req/30s sans clé API). "
                "Attends ~30s ou définis NVD_API_KEY."
            ) from exc
        if exc.code == 404:
            raise ValueError("NVD: aucune donnée pour cette requête.") from exc
        raise ValueError(f"NVD: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"NVD: inaccessible ({exc.reason})") from exc


def _desc(cve: dict) -> str:
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            return d.get("value", "")
    return ""


def _severity(cve: dict) -> str:
    """Score CVSS prioritaire : v3.1 > v3.0 > v2, label déduit pour la v2."""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        for m in cve.get("metrics", {}).get(key, []):
            data = m.get("cvssData", {})
            if data.get("baseScore") is not None:
                ver = data.get("version", "?")
                vec = data.get("vectorString", "")
                sev = data.get("baseSeverity")
                if not sev and key == "cvssMetricV2":
                    s = data["baseScore"]
                    sev = "HIGH" if s >= 7.0 else "MEDIUM" if s >= 4.0 else "LOW"
                return (f"{sev or '?'} {data['baseScore']:.1f} "
                        f"(CVSS:{ver} {vec})")
    return "NON ÉVALUÉE"


def _fmt_line(cve: dict, summary_len: int = 180) -> str:
    sev = _severity(cve)
    pub = (cve.get("published") or "")[:10]
    summary = _desc(cve).replace("\n", " ")
    return f"{cve['id']} | {sev.split(' ')[0]} {sev.split(' ')[1]} | {pub} | {summary[:summary_len]}"


@mcp.tool()
def cve_info(cve_id: str) -> str:
    """Détails complets d'une CVE précise (ex: CVE-2024-3094) : description,
    score CVSS et vecteur, CWE, produits affectés (CPE), références, dates."""
    cve_id = cve_id.strip().upper()
    if not cve_id.startswith("CVE-"):
        return "ERREUR: identifiant attendu au format CVE-YYYY-NNNN."
    data = _nvd_get({"cveId": cve_id})
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return f"Aucune donnée NVD pour {cve_id}."
    cve = vulns[0]["cve"]

    out = [f"=== {cve['id']} ===",
           f"Statut: {cve.get('vulnStatus', '?')} | "
           f"publiée: {cve.get('published', '?')[:10]} | "
           f"modifiée: {cve.get('lastModified', '?')[:10]}",
           f"Sévérité: {_severity(cve)}"]

    cwes = sorted({d["value"]
                   for w in cve.get("weaknesses", [])
                   for d in w.get("description", [])
                   if d.get("lang") == "en" and d.get("value", "").startswith("CWE")})
    if cwes:
        out.append("CWE: " + ", ".join(cwes))

    out.append("")
    out.append(_desc(cve))

    cpes = []
    for conf in cve.get("configurations", []):
        for node in conf.get("nodes", []):
            for cpe in node.get("cpeMatch", []):
                if cpe.get("vulnerable"):
                    cpes.append(cpe.get("criteria", ""))
    if cpes:
        shown = cpes[:12]
        out.extend(["", f"Produits affectés ({len(cpes)} CPE):"])
        out.extend(f"  {c}" for c in shown)
        if len(cpes) > len(shown):
            out.append(f"  ... et {len(cpes) - len(shown)} autres")

    refs = [r["url"] for r in cve.get("references", []) if r.get("url")]
    if refs:
        out.extend(["", "Références:"])
        out.extend(f"  {u}" for u in refs[:8])
    return "\n".join(out)


@mcp.tool()
def search_cve(keywords: str, limit: int = 10) -> str:
    """Recherche de CVEs par mots-clés (produit, technologie, editeur...).
    Retourne les limit premiers résultats avec sévérité et résumé."""
    limit = max(1, min(limit, 50))
    data = _nvd_get({"keywordSearch": keywords.strip(), "resultsPerPage": limit})
    total = data.get("totalResults", 0)
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return f"Aucune CVE trouvée pour « {keywords} »."
    out = [f"{total} CVE(s) NVD pour « {keywords} » ({len(vulns)} affichées):"]
    out.extend(f"- {_fmt_line(v['cve'])}" for v in vulns)
    out.append("Utilise cve_info pour le détail d'une entrée.")
    return "\n".join(out)


@mcp.tool()
def latest_cve(days: int = 7, limit: int = 20) -> str:
    """CVEs publiées au cours des N derniers jours (max 120), triées de la plus
    récente à la plus ancienne. Utile pour la veille quotidienne."""
    days = max(1, min(days, 120))
    limit = max(1, min(limit, 100))
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(days=days)
    fmt = "%Y-%m-%dT%H:%M:%S.000"
    data = _nvd_get({
        "pubStartDate": start.strftime(fmt),
        "pubEndDate": now.strftime(fmt),
        "resultsPerPage": limit,
    })
    vulns = [v["cve"] for v in data.get("vulnerabilities", [])]
    vulns.sort(key=lambda c: c.get("published") or "", reverse=True)
    if not vulns:
        return f"Aucune CVE publiée sur les {days} derniers jours."
    out = [f"Vulnérabilités publiées ces {days} derniers jours "
           f"({data.get('totalResults', len(vulns))} au total, {len(vulns)} affichées):"]
    out.extend(f"- {_fmt_line(c)}" for c in vulns)
    out.append("Utilise cve_info pour le détail d'une entrée.")
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run()  # stdio
