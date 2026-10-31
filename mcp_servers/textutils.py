"""MCP server utilitaires texte/dev purs (stdlib, aucun I/O) : diff unifié,
extraction JSON par chemin, test de regex avec groupes, conversions d'unités.
"""

from __future__ import annotations

import difflib
import json
import re

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("textutils")

_MAX = 20_000


@mcp.tool()
def diff(a: str, b: str, from_name: str = "a", to_name: str = "b") -> str:
    """Diff unifié entre deux textes (comme `diff -u`). a/b: contenus complets
    (ou chemins lus via files__read_file). Noms personnalisables pour l'en-tête."""
    if len(a) > _MAX * 4 or len(b) > _MAX * 4:
        return f"ERREUR: entrée trop grande (max {_MAX * 4} caractères par texte)."
    out = list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                    fromfile=from_name, tofile=to_name, lineterm=""))
    if not out:
        return "(identiques)"
    text = "\n".join(out)
    return text[:_MAX] + ("\n... (diff tronqué)" if len(text) > _MAX else "")


def _json_get(data, path: str):
    """Chemin simple : a.b[0].c — clés dict puis index liste."""
    cur = data
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            idx = int(part[1:-1])
            if not isinstance(cur, list) or idx >= len(cur):
                raise KeyError(f"index {part} hors limites")
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"clé '{part}' absente")
            cur = cur[part]
    return cur


@mcp.tool()
def json_query(json_text: str, path: str = "") -> str:
    """Extrait une valeur d'un JSON. path: chemin simple 'a.b[0].c' (vide =
    document entier pretty-printé). json_text: le JSON brut."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return f"ERREUR: JSON invalide ({exc})"
    if not path.strip():
        out = json.dumps(data, ensure_ascii=False, indent=2)
        return out[:_MAX] + ("\n... (tronqué)" if len(out) > _MAX else "")
    try:
        value = _json_get(data, path.strip())
    except (KeyError, IndexError, ValueError) as exc:
        return f"ERREUR: chemin '{path}' introuvable ({exc})"
    if isinstance(value, (dict, list)):
        out = json.dumps(value, ensure_ascii=False, indent=2)
        return out[:_MAX]
    return str(value)


@mcp.tool()
def regex_test(pattern: str, text: str, flags: str = "") -> str:
    """Teste une regex Python sur un texte : nombre de matchs, et pour chaque
    match les groupes capturés. flags: 'i' (insensible), 'm' (multiligne), 's'
    (dotall), combinaisons 'im'. text tronqué à 20000 caractères."""
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
    fl = 0
    for c in flags.lower():
        if c in flag_map:
            fl |= flag_map[c]
        else:
            return f"ERREUR: flag '{c}' inconnu (i, m, s)."
    text = text[:_MAX]
    try:
        rx = re.compile(pattern, fl)
    except re.error as exc:
        return f"ERREUR: regex invalide ({exc})"
    matches = list(rx.finditer(text))
    if not matches:
        return f"0 match pour {pattern!r}."
    out = [f"{len(matches)} match(s) pour {pattern!r}:"]
    for i, m in enumerate(matches[:50], 1):
        groups = m.groups()
        g = ", ".join(repr(g) for g in groups) if groups else "(pas de groupe)"
        out.append(f"{i}. span={m.span()} match={m.group(0)[:80]!r} groupes: {g}")
    if len(matches) > 50:
        out.append(f"... et {len(matches) - 50} autres matchs")
    return "\n".join(out)


# (valeur, unité) -> facteur vers l'unité de base ; conversions par table explicite
_UNITS = {
    # tailles (base: octet)
    "o": 1, "ko": 1024, "mo": 1024**2, "go": 1024**3, "to": 1024**4,
    "octet": 1, "kio": 1024, "mio": 1024**2, "gio": 1024**3, "tio": 1024**4,
    # débits (base: bit/s) — ko = 1000 bit/s (réseau), kio = 1024
    "b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3,
    "ko/s": 1024 * 8, "mo/s": 1024**2 * 8, "go/s": 1024**3 * 8,
    "kb/s": 1000 * 8, "mb/s": 1000**2 * 8, "gb/s": 1000**3 * 8,
    # durées (base: seconde)
    "ms": 1e-3, "s": 1, "min": 60, "h": 3600, "j": 86400,
    # températures : gérées à part (non linéaires)
    "c": None, "f": None, "k": None,
    # pourcentage
    "%": 0.01,
}


def _temp_convert(value: float, frm: str, to: str) -> float:
    to_c = {"c": lambda v: v, "f": lambda v: (v - 32) * 5 / 9, "k": lambda v: v - 273.15}[frm](value)
    return {"c": lambda v: v, "f": lambda v: v * 9 / 5 + 32, "k": lambda v: v + 273.15}[to](to_c)


@mcp.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Conversion d'unités. Tailles: o/ko/mo/go/to (1024) ou octet/kio/mio...
    Débits: b/kb/mb/gb (1000) ou kib/mib/gib, et ko/s, mo/s... Durées: ms/s/min/h/j.
    Températures: c/f/k. Pourcentage: '%' (ex: 50 % -> 0.5 avec to_unit='1').
    Ex: unit_convert(10, 'go', 'to') -> 0.009765625 to."""
    frm, to = from_unit.strip().lower(), to_unit.strip().lower()
    if frm in ("c", "f", "k") and to in ("c", "f", "k"):
        return f"{value} °{frm.upper()} = {_temp_convert(value, frm, to):.4g} °{to.upper()}"
    if frm not in _UNITS or _UNITS[frm] is None:
        return f"ERREUR: unité source '{from_unit}' inconnue. Voir la docstring."
    if to not in _UNITS or _UNITS[to] is None:
        return f"ERREUR: unité cible '{to_unit}' inconnue. Voir la docstring."
    base = value * _UNITS[frm]
    result = base / _UNITS[to]
    return f"{value:g} {from_unit} = {result:.6g} {to_unit}"


if __name__ == "__main__":
    mcp.run()  # stdio
