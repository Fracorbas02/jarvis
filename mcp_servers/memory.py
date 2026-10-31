"""MCP server mémoire persistante de Jarvis : faits notés dans un fichier
Markdown (~/.cache/jarvis/memory.md par défaut, surchargeable via
JARVIS_MEMORY_FILE).

remember() ajoute un fait horodaté, recall() liste les faits (filtre texte
optionnel), forget() retire une entrée par son numéro. Le fichier peut être
injecté dans le prompt système pour que le LLM se souvienne entre sessions.
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

mcp = Server("memory")

_FILE = Path(os.environ.get("JARVIS_MEMORY_FILE",
                            os.path.expanduser("~/.cache/jarvis/memory.md")))
_MAX_FACT_LEN = 2000
_MAX_FACTS = 500


def _load() -> list[str]:
    if not _FILE.exists():
        return []
    text = _FILE.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.startswith("- [")]


def _save(facts: list[str]) -> None:
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    header = "# Mémoire de Jarvis\n\nFaits notés entre les sessions :\n\n"
    _FILE.write_text(header + "\n".join(facts) + "\n", encoding="utf-8")


@mcp.tool()
def remember(fact: str) -> str:
    """Note un fait en mémoire persistante (préférence, config, adresse, projet...).
    Le fait sera rappelé par recall() et peut être injecté dans le contexte."""
    fact = fact.strip().replace("\n", " ")[:_MAX_FACT_LEN]
    if not fact:
        return "ERREUR: fait vide."
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    facts = _load()
    facts.append(f"- [{stamp}] {fact}")
    if len(facts) > _MAX_FACTS:
        facts = facts[-_MAX_FACTS:]
    _save(facts)
    return f"Noté ({len(facts)} faits en mémoire) : {fact}"


@mcp.tool()
def recall(query: str = "", limit: int = 50) -> str:
    """Rappelle les faits mémorisés. query: filtre texte (insensible à la casse,
    vide = tout). limit: max 200. Chaque fait est numéroté pour forget()."""
    facts = _load()
    limit = max(1, min(limit, 200))
    if query:
        q = query.lower()
        facts = [f for f in facts if q in f.lower()]
    if not facts:
        return f"Aucun fait{' pour « {query} »' if query else ''} en mémoire ({_FILE})."
    shown = facts[-limit:]
    out = [f"Mémoire ({len(facts)} fait(s){', {len(shown)} affiché(s)' if len(facts) > limit else ''}) — {_FILE}:"]
    for i, f in enumerate(shown, 1):
        out.append(f"{i}. {f}")
    if len(facts) > limit:
        out.append(f"... et {len(facts) - limit} autres (augmente limit)")
    return "\n".join(out)


@mcp.tool()
def forget(number: int) -> str:
    """Supprime le fait n° `number` affiché par le dernier recall()."""
    facts = _load()
    if not 1 <= number <= len(facts):
        return f"ERREUR: numéro hors plage (1-{len(facts)})."
    removed = facts.pop(number - 1)
    _save(facts)
    return f"Oublié : {removed}"


if __name__ == "__main__":
    mcp.run()  # stdio
