"""MCP server EPSS (FIRST) — score de probabilité d'exploitation à 30 jours.

API publique gratuite, sans clé : https://api.first.org/data/v1/epss
Limite ~300 req/min côté client ; les appels sont espacés d'au moins 0.3 s.
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

mcp = Server("epss")

_EPSS_BASE = "https://api.first.org/data/v1/epss"
_MIN_INTERVAL = 0.3
_last_call = 0.0


def _epss_get(params: dict, timeout: float = 30.0) -> dict:
    global _last_call
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    url = _EPSS_BASE + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "jarvis-epss/0.1", "Accept": "application/json"}
    token = os.environ.get("FIRST_API_KEY", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise ValueError("EPSS: limite de débit atteinte, réessaie dans une minute "
                             "(ou définis FIRST_API_KEY).") from exc
        raise ValueError(f"EPSS: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"EPSS: inaccessible ({exc.reason})") from exc


def _check_cve(cve: str) -> bool:
    return (cve.upper().startswith("CVE-") and len(cve) >= 12
            and all(c.isdigit() or c in "-." for c in cve.upper()))


@mcp.tool()
def epss_score(cves: str) -> str:
    """Score EPSS (probabilité d'exploitation à 30 jours, 0-1) et percentile
    pour une liste de CVEs séparées par des virgules (max 50, ex:
    "CVE-2024-3094,CVE-2021-44228"). Trié du plus exploitable au moins."""
    ids = [c.strip().upper() for c in cves.split(",") if c.strip()]
    ids = [c for c in ids if _check_cve(c)]
    if not ids:
        return "ERREUR: aucune CVE valide (format CVE-YYYY-NNNN, séparées par des virgules)."
    ids = ids[:50]
    try:
        data = _epss_get({"cve": ",".join(ids)})
    except ValueError as exc:
        return f"ERREUR: {exc}"
    rows = data.get("data", [])
    if not rows:
        return f"Aucun score EPSS pour {', '.join(ids)}."
    rows.sort(key=lambda r: float(r.get("epss") or 0), reverse=True)
    out = [f"Score EPSS (probabilité d'exploitation à 30 j) — {data.get('status', '')} "
           f"au {rows[0].get('date', '?')} :"]
    for r in rows:
        epss = float(r.get("epss") or 0)
        pct = float(r.get("percentile") or 0)
        out.append(f"- {r['cve']}: EPSS {epss:.4f} ({epss * 100:.2f} %) | "
                   f"percentile {pct * 100:.1f} %")
    missing = set(ids) - {r["cve"] for r in rows}
    if missing:
        out.append(f"Sans score: {', '.join(sorted(missing))}")
    return "\n".join(out)


@mcp.tool()
def epss_history(cve: str, days: int = 30) -> str:
    """Historique du score EPSS d'une CVE sur les N derniers jours (max 90),
    via scope=time-series. Utile pour voir si une CVE monte en flèche."""
    cve = cve.strip().upper()
    if not _check_cve(cve):
        return "ERREUR: identifiant attendu au format CVE-YYYY-NNNN."
    days = max(1, min(days, 90))
    try:
        data = _epss_get({"cve": cve, "scope": "time-series"})
    except ValueError as exc:
        return f"ERREUR: {exc}"
    rows = data.get("data", [])
    if not rows:
        return f"Aucun historique EPSS pour {cve}."
    ts = rows[0].get("time-series", [])
    if not ts:
        return f"Pas d'historique pour {cve} (score actuel {rows[0].get('epss')})."
    ts = ts[-days:]
    out = [f"Historique EPSS de {cve} ({len(ts)} jours, du plus ancien au plus récent):"]
    for point in ts:
        epss = float(point.get("epss") or 0)
        out.append(f"- {point.get('date')}: {epss:.4f} ({epss * 100:.2f} %)")
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run()  # stdio
