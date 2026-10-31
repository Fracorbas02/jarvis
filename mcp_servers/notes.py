"""MCP server notes — bloc-notes persistant par sujet, complémentaire de
memory (qui est pour les faits courts).

Chaque sujet vit dans ~/.cache/jarvis/notes/<sujet>.md (JARVIS_NOTES_DIR pour
changer le répertoire). note_add(note, topic) ajoute une entrée horodatée,
note_list(topic) lit le sujet, note_topics() liste les sujets, note_search(q)
cherche dans tous les sujets.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("notes")

_DIR = Path(os.environ.get("JARVIS_NOTES_DIR", os.path.expanduser("~/.cache/jarvis/notes")))
_MAX_NOTE = 4000
_MAX_NOTES_PER_TOPIC = 500


def _safe_topic(topic: str) -> Path | None:
    topic = topic.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", topic):
        return None
    return _DIR / f"{topic}.md"


@mcp.tool()
def note_add(note: str, topic: str = "general") -> str:
    """Ajoute une note horodatée à un sujet (créé si besoin). topic: [a-z0-9_-],
    défaut 'general'."""
    path = _safe_topic(topic)
    if path is None:
        return "ERREUR: topic invalide (minuscules, chiffres, - et _, max 64 car.)."
    note = note.strip().replace("\r", "")[:_MAX_NOTE]
    if not note:
        return "ERREUR: note vide."
    _DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    existing = path.read_text(encoding="utf-8") if path.exists() else f"# Notes: {topic}\n\n"
    if not existing.startswith("#"):
        existing = f"# Notes: {topic}\n\n" + existing
    entries = [ln for ln in existing.splitlines() if ln.startswith("## [")]
    if len(entries) >= _MAX_NOTES_PER_TOPIC:
        return f"ERREUR: sujet '{topic}' plein ({_MAX_NOTES_PER_TOPIC} notes max)."
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## [{stamp}]\n{note}\n")
    return f"Note ajoutée à '{topic}' ({len(entries) + 1} notes) : {note[:80]}"


@mcp.tool()
def note_list(topic: str = "general", limit: int = 50) -> str:
    """Lit les notes d'un sujet, de la plus récente à la plus ancienne.
    limit: max 200 entrées affichées."""
    path = _safe_topic(topic)
    if path is None:
        return "ERREUR: topic invalide."
    if not path or not path.exists():
        return f"Aucune note pour '{topic}'."
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^## \[", text)
    entries = ["## [" + b for b in blocks[1:]]  # saute l'en-tête
    if not entries:
        return f"Sujet '{topic}' existe mais est vide."
    shown = entries[::-1][:max(1, min(limit, 200))]
    out = [f"Notes '{topic}' ({len(entries)} au total, {len(shown)} affichées, récentes d'abord) — {path}:"]
    out.extend(shown)
    return "\n".join(out)


@mcp.tool()
def note_topics() -> str:
    """Liste les sujets de notes existants avec leur nombre d'entrées."""
    if not _DIR.is_dir():
        return f"Aucun sujet (répertoire {_DIR} inexistant)."
    out = []
    for f in sorted(_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        n = len([ln for ln in text.splitlines() if ln.startswith("## [")])
        out.append(f"- {f.stem}: {n} note(s)")
    return f"{_DIR}:\n" + ("\n".join(out) if out else "(vide)")


@mcp.tool()
def note_search(query: str, limit: int = 20) -> str:
    """Recherche plein texte dans toutes les notes. limit: max 50 résultats."""
    query = query.strip().lower()
    if not query:
        return "ERREUR: query vide."
    if not _DIR.is_dir():
        return "Aucune note."
    limit = max(1, min(limit, 50))
    hits = []
    for f in sorted(_DIR.glob("*.md")):
        for m in re.finditer(r"(?ms)^## \[[^\]]+\]\n.*?(?=^## \[|\Z)", f.read_text(encoding="utf-8")):
            if query in m.group(0).lower():
                hits.append((f.stem, m.group(0).strip()))
    if not hits:
        return f"Aucune note ne contient « {query} »."
    out = [f"{len(hits)} note(s) contiennent « {query} » ({min(limit, len(hits))} affichées):"]
    for topic, block in hits[:limit]:
        out.append(f"--- {topic} ---\n{block[:600]}")
    return "\n\n".join(out)


if __name__ == "__main__":
    mcp.run()  # stdio
