"""MCP server utilitaires texte/dev purs (stdlib, aucun I/O) : diff unifié,
extraction JSON par chemin, test de regex avec groupes, conversions d'unités.
"""

from __future__ import annotations

import difflib
import json
import re
import signal
import threading

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


_REGEX_TIMEOUT = 2.0  # secondes max d'exécution de la regex (anti-ReDoS)
_REGEX_MAX_TEXT = 2_000  # texte réduit : un pattern catastrophique reste borné


class _RegexTimeout(Exception):
    pass


def _regex_alarm(signum, frame):
    raise _RegexTimeout()


_WORKER_SRC = """
import json, re, sys
data = json.load(sys.stdin)
try:
    rx = re.compile(data["pattern"], data["flags"])
    out = [{"span": list(m.span()), "match": m.group(0)[:200],
            "groups": [None if g is None else g[:200] for g in m.groups()]}
           for m in rx.finditer(data["text"])]
    json.dump({"ok": True, "matches": out}, sys.stdout)
except re.error as exc:
    json.dump({"ok": False, "error": str(exc)}, sys.stdout)
"""


def _regex_finditer_timed(pattern: str, flags: int, text: str, timeout: float):
    """list(rx.finditer(text)) avec plafond de temps, sous forme de dicts.

    Thread principal : alarme SIGALRM (signal.signal/setitimer ne marchent
    QUE là). Ailleurs (le SDK MCP exécute les outils sync dans un thread
    worker anyio, où signal.signal lève ValueError) : la regex tourne dans
    un SOUS-PROCESSUS tué par le timeout du père — un join(timeout) sur un
    thread en calcul CPU pur ne respecte pas toujours son délai (observé :
    join(2s) bloqué 17s), alors que le timeout subprocess est géré par le
    noyau (SIGKILL), donc fiable.
    """
    if threading.current_thread() is threading.main_thread():
        rx = re.compile(pattern, flags)
        old = None
        try:
            old = signal.signal(signal.SIGALRM, _regex_alarm)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            return [{"span": list(m.span()), "match": m.group(0)[:200],
                     "groups": [None if g is None else g[:200] for g in m.groups()]}
                    for m in rx.finditer(text)]
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)
    import json as _json
    import subprocess as _sp
    import sys as _sys
    payload = _json.dumps({"pattern": pattern, "flags": flags, "text": text})
    try:
        proc = _sp.run([_sys.executable, "-c", _WORKER_SRC], input=payload,
                       capture_output=True, text=True, timeout=timeout)
    except _sp.TimeoutExpired:
        raise _RegexTimeout()
    if proc.returncode != 0:
        raise _RegexTimeout() if False else RuntimeError(
            f"worker regex mort (code {proc.returncode}): {proc.stderr[:200]}")
    data = _json.loads(proc.stdout or "{}")
    if not data.get("ok"):
        raise re.error(data.get("error", "erreur worker"))
    return data["matches"]



@mcp.tool()
def regex_test(pattern: str, text: str, flags: str = "") -> str:
    """Teste une regex Python sur un texte : nombre de matchs, et pour chaque
    match les groupes capturés. flags: 'i' (insensible), 'm' (multiligne), 's'
    (dotall), combinaisons 'im'. text tronqué à 2000 caractères, exécution
    plafonnée à 2 s (protection contre les regex à backtracking catastrophique)."""
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
    fl = 0
    for c in flags.lower():
        if c in flag_map:
            fl |= flag_map[c]
        else:
            return f"ERREUR: flag '{c}' inconnu (i, m, s)."
    text = text[:_REGEX_MAX_TEXT]
    try:
        re.compile(pattern, fl)  # valide d'abord (erreur propre si invalide)
    except re.error as exc:
        return f"ERREUR: regex invalide ({exc})"
    # plafond de temps anti-ReDoS : un pattern type (a+)+$ sur un texte long
    # bloquerait sinon le process stdio indéfiniment. SIGALRM dans le thread
    # principal, sous-processus tué par timeout sinon (chemin MCP : worker).
    try:
        matches = _regex_finditer_timed(pattern, fl, text, _REGEX_TIMEOUT)
    except _RegexTimeout:
        return (f"ERREUR: regex interrompue après {_REGEX_TIMEOUT:.0f}s "
                f"(backtracking trop coûteux — simplifie le pattern).")
    except re.error as exc:
        return f"ERREUR: regex invalide ({exc})"
    if not matches:
        return f"0 match pour {pattern!r}."
    out = [f"{len(matches)} match(s) pour {pattern!r}:"]
    for i, m in enumerate(matches[:50], 1):
        groups = m["groups"]
        g = ", ".join(repr(x) for x in groups) if groups else "(pas de groupe)"
        out.append(f"{i}. span={tuple(m['span'])} match={m['match'][:80]!r} groupes: {g}")
    if len(matches) > 50:
        out.append(f"... et {len(matches) - 50} autres matchs")
    return "\n".join(out)


# Conversions par dimension : chaque table ne contient QUE des unités de même
# dimension, sinon la conversion n'a pas de sens (5 s en Go est refusé).
_UNITS: dict[str, dict[str, float]] = {
    "taille": {  # base: octet
        "o": 1, "ko": 1024, "mo": 1024**2, "go": 1024**3, "to": 1024**4,
        "octet": 1, "kio": 1024, "mio": 1024**2, "gio": 1024**3, "tio": 1024**4,
    },
    "debit": {  # base: bit/s — ko/s = octets (1024) x8, kb/s = kilo réseau (1000)
        "b/s": 1, "kb/s": 1000, "mb/s": 1000**2, "gb/s": 1000**3,
        "kib/s": 1024, "mib/s": 1024**2, "gib/s": 1024**3,
        "ko/s": 1024 * 8, "mo/s": 1024**2 * 8, "go/s": 1024**3 * 8,
    },
    "duree": {  # base: seconde
        "ms": 1e-3, "s": 1, "min": 60, "h": 3600, "j": 86400,
    },
    "ratio": {  # base: 1 (sans unité)
        "%": 0.01, "1": 1, "ppm": 1e-6, "ppb": 1e-9,
    },
}

_TEMP = ("c", "f", "k")


def _dim_of(unit: str) -> str | None:
    if unit in _TEMP:
        return "temperature"
    for dim, table in _UNITS.items():
        if unit in table:
            return dim
    return None


def _temp_convert(value: float, frm: str, to: str) -> float:
    to_c = {"c": lambda v: v, "f": lambda v: (v - 32) * 5 / 9, "k": lambda v: v - 273.15}[frm](value)
    return {"c": lambda v: v, "f": lambda v: v * 9 / 5 + 32, "k": lambda v: v + 273.15}[to](to_c)


@mcp.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Conversion d'unités, même dimension obligatoire. Tailles: o/ko/mo/go/to
    (1024) ou octet/kio/mio/gio/tio. Débits: b/s/kb/s/mb/s/gb/s (1000) ou
    kib/s/mib/s/gib/s, ko/s/mo/s/go/s (1024). Durées: ms/s/min/h/j.
    Températures: c/f/k. Ratio: %/ppm/ppb vers '1' (ex: 50 % -> 0.5).
    Ex: unit_convert(10, 'go', 'to') -> 0.00976562 to. unit_convert(5, 's', 'go')
    -> ERREUR (dimensions différentes)."""
    frm, to = from_unit.strip().lower(), to_unit.strip().lower()
    dim_f, dim_t = _dim_of(frm), _dim_of(to)
    if dim_f is None:
        return f"ERREUR: unité source '{from_unit}' inconnue. Voir la docstring."
    if dim_t is None:
        return f"ERREUR: unité cible '{to_unit}' inconnue. Voir la docstring."
    if dim_f != dim_t:
        return (f"ERREUR: dimensions différentes ({frm} est {dim_f}, "
                f"{to} est {dim_t}) — conversion refusée.")
    if dim_f == "temperature":
        return f"{value:g} °{frm.upper()} = {_temp_convert(value, frm, to):.6g} °{to.upper()}"
    base = value * _UNITS[dim_f][frm]
    result = base / _UNITS[dim_t][to]
    return f"{value:g} {from_unit} = {result:.6g} {to_unit}"


if __name__ == "__main__":
    mcp.run()  # stdio
