"""MCP server fichiers/texte : recherche (ripgrep), lecture paginée, empreintes.

Lecture seule : aucun outil d'écriture ni de suppression. Les chemins sont
résolus puis confinés à la liste des répertoires autorisés (JARVIS_FILES_ROOTS,
séparés par ':' ; défaut: répertoire courant et $HOME).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("files")

_MAX_BYTES = 30_000
_MAX_MATCHES = 200


def _roots() -> list[Path]:
    env = os.environ.get("JARVIS_FILES_ROOTS", "")
    parts = [p for p in env.split(os.pathsep) if p]
    if not parts:
        parts = [os.getcwd(), os.path.expanduser("~")]
    return [Path(p).expanduser().resolve() for p in parts]


def _resolve(path: str) -> Path | None:
    """Résout un chemin et vérifie qu'il tombe dans un répertoire autorisé."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    try:
        p = p.resolve()
    except OSError:
        return None
    for root in _roots():
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    return None


def _clip(text: str) -> str:
    if len(text) > _MAX_BYTES:
        return text[:_MAX_BYTES] + f"\n... (tronqué à {_MAX_BYTES} octets)"
    return text


@mcp.tool()
def grep_files(pattern: str, path: str = ".", glob: str = "",
               max_matches: int = 100, context: int = 0) -> str:
    """Recherche plein texte (regex) dans des fichiers, via ripgrep si dispo,
    sinon grep récursif. path: répertoire de départ. glob: filtre de fichiers
    (ex: '*.py', '*.log'). max_matches: plafond de résultats (max 200).
    context: lignes de contexte autour de chaque match (0 par défaut)."""
    root = _resolve(path)
    if root is None:
        return f"ERREUR: '{path}' hors des répertoires autorisés ({':'.join(map(str, _roots()))})"
    if not root.exists():
        return f"ERREUR: '{path}' n'existe pas."
    max_matches = max(1, min(max_matches, _MAX_MATCHES))
    context = max(0, min(context, 10))
    if shutil.which("rg"):
        cmd = ["rg", "--no-heading", "--line-number", "--max-count", str(max_matches)]
        if context:
            cmd += ["-C", str(context)]
        if glob:
            cmd += ["--glob", glob]
        cmd += ["--", pattern, str(root)]
    else:
        cmd = ["grep", "-rIn", "--include", glob or "*", "-m", str(max_matches)]
        if context:
            cmd += ["-C", str(context)]
        cmd += ["-E", "--", pattern, str(root)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 1 and not out.strip():
            return f"Aucun résultat pour {pattern!r} dans {root}"
        return _clip(out.strip()) or "(aucune sortie)"
    except subprocess.TimeoutExpired:
        return "ERREUR: timeout après 30s (réduis le périmètre ou le glob)"


@mcp.tool()
def read_file(path: str, offset: int = 0, limit: int = 200) -> str:
    """Lecture paginée d'un fichier texte (lignes). offset: n° de ligne de
    départ (0 = début), limit: nombre de lignes (max 1000). Retourne aussi la
    taille totale en lignes pour paginer."""
    p = _resolve(path)
    if p is None:
        return f"ERREUR: '{path}' hors des répertoires autorisés ({':'.join(map(str, _roots()))})"
    if not p.is_file():
        return f"ERREUR: '{p}' n'est pas un fichier."
    offset = max(0, offset)
    limit = max(1, min(limit, 1000))
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"ERREUR: lecture impossible ({exc.strerror})"
    total = len(lines)
    chunk = lines[offset:offset + limit]
    header = (f"{p} — lignes {offset + 1}-{offset + len(chunk)} sur {total}"
              f"{'' if offset + limit >= total else ' (suite: augmente offset)'}")
    return header + "\n" + _clip("".join(chunk)).rstrip()


@mcp.tool()
def file_info(path: str) -> str:
    """Métadonnées d'un fichier : type (commande file), taille, permissions,
    propriétaire, dates, et empreintes md5/sha256 pour les fichiers < 64 Mo."""
    p = _resolve(path)
    if p is None:
        return f"ERREUR: '{path}' hors des répertoires autorisés ({':'.join(map(str, _roots()))})"
    if not p.exists():
        return f"ERREUR: '{p}' n'existe pas."
    st = p.stat()
    out = [f"Chemin: {p}", f"Taille: {st.st_size} octets",
           f"Mode: {oct(st.st_mode & 0o777)}"]
    if shutil.which("file"):
        proc = subprocess.run(["file", "-b", str(p)], capture_output=True, text=True)
        out.append(f"Type: {(proc.stdout or proc.stderr).strip()}")
    if st.st_size <= 64 * 1024 * 1024:
        h_md5, h_sha = hashlib.md5(), hashlib.sha256()
        try:
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h_md5.update(block)
                    h_sha.update(block)
            out += [f"MD5:    {h_md5.hexdigest()}", f"SHA256: {h_sha.hexdigest()}"]
        except OSError as exc:
            out.append(f"Empreintes: calcul impossible ({exc.strerror})")
    else:
        out.append("Empreintes: ignorées (fichier > 64 Mo)")
    return "\n".join(out)


@mcp.tool()
def list_dir(path: str = ".", pattern: str = "") -> str:
    """Liste le contenu d'un répertoire (fichiers + dossiers, tailles).
    pattern: filtre fnmatch optionnel (ex: '*.log')."""
    import fnmatch

    p = _resolve(path)
    if p is None:
        return f"ERREUR: '{path}' hors des répertoires autorisés ({':'.join(map(str, _roots()))})"
    if not p.is_dir():
        return f"ERREUR: '{p}' n'est pas un répertoire."
    try:
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except OSError as exc:
        return f"ERREUR: lecture impossible ({exc.strerror})"
    lines = []
    for e in entries:
        if pattern and not fnmatch.fnmatch(e.name, pattern):
            continue
        try:
            size = e.stat().st_size if e.is_file() else 0
            kind = "d" if e.is_dir() else "f"
            lines.append(f"{kind} {size:>12}  {e.name}{'/' if e.is_dir() else ''}")
        except OSError:
            lines.append(f"? {'?':>12}  {e.name}")
    if not lines:
        return f"(vide{', filtre ' + pattern if pattern else ''}) dans {p}"
    return f"{p} ({len(lines)} entrées):\n" + _clip("\n".join(lines))


if __name__ == "__main__":
    mcp.run()  # stdio
