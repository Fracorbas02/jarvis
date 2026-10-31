"""Compactage de l'historique de conversation.

Deux entrées :
- /compact côté CLI (commande manuelle) ;
- l'outil interne jarvis__compact, exposé au modèle : il peut décider de
  compacter lui-même quand la conversation devient longue et que le contexte
  risque de saturer.

Le résumé est produit par le LLM lui-même (chat non-stream), puis l'historique
est remplacé par [system, user(résumé), assistant(accusé)].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .agent import InternalTool

if TYPE_CHECKING:
    from .agent import Agent

COMPACT_SYSTEM = (
    "Tu es un résumeur de conversation. Tu dois produire un résumé fidèle et "
    "concis de la conversation fournie : faits établis, décisions prises, "
    "commandes exécutées et leurs résultats, sorties d'outils importantes, "
    "URLs, identifiants de CVE, adresses/hosts concernés, état actuel. "
    "Le résumé remplacera l'historique complet : ne perds aucune information "
    "nécessaire pour continuer la conversation. Réponds en français, en texte "
    "structuré (listes), sans commentary."
)


async def compact_history(agent: "Agent", history: list[dict]) -> tuple[str, list[dict]]:
    """Résume l'historique via le LLM et retourne (message, nouvel historique)."""
    convo = [m for m in history if m.get("role") != "system"]
    if len(convo) < 4:
        return ("historique déjà court (< 4 messages), rien à compacter", history)
    text = "\n\n".join(
        f"[{m.get('role')}] {m.get('content') or 'appel d outil'}" for m in convo
    )
    msg = await agent.llm.chat([
        {"role": "system", "content": COMPACT_SYSTEM},
        {"role": "user", "content": (
            "Résume la conversation suivante. Le résumé remplacera l'historique.\n\n"
            f"{text}"
        )},
    ], stream=False)
    summary = msg.get("content") or ""
    new_history = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": (
            f"[Contexte précédent, résumé automatiquement — {len(convo)} messages :]\n"
            f"{summary}"
        )},
        {"role": "assistant", "content": "Compris, je repars de ce résumé pour la suite."},
    ]
    return (
        f"contexte compacté : {len(convo)} messages → résumé de {len(summary)} car. "
        f"(~{max(1, len(summary) // 4)} tok estimés)",
        new_history,
    )


compact_tool = InternalTool(
    description=(
        "Compacter l'historique de la conversation en un résumé concis pour "
        "libérer du contexte. À utiliser quand la conversation est longue et "
        "que le contexte risque de saturer (voir l'usage affiché au prompt)."
    ),
    fn=lambda agent, history, args: compact_history(agent, history),
)
