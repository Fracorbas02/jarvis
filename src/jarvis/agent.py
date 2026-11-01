"""Boucle tool-calling : user -> LLM -> [tool call -> résultat -> LLM]* -> réponse finale."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from .llm import LLMClient
from .mcp import McpToolProvider

log = logging.getLogger("jarvis.agent")


@dataclass
class TurnTrace:
    """Observabilité du tour : ce qui a été appelé, combien d'itérations."""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None


@dataclass
class InternalTool:
    """Outil interne (hors MCP), exposé au modèle comme les outils MCP.

    fn: async (agent, history, args) -> (result: str, new_history: list | None).
    Si new_history est retourné, il remplace l'historique (ex. compactage).
    """
    description: str
    fn: object  # Callable[..., Awaitable[tuple[str, list | None]]]


class Agent:
    def __init__(self, llm: LLMClient, tools: McpToolProvider,
                 system_prompt: str, max_iterations: int = 8,
                 stream: bool = True,
                 internal_tools: dict[str, InternalTool] | None = None):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        # streaming par défaut : le raisonnement s'affiche en direct
        self.stream = stream
        self.internal_tools = internal_tools or {}

    def _tools_schemas(self) -> list[dict] | None:
        schemas = self.tools.schemas() if self.tools else []
        for name, t in self.internal_tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.description,
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        return schemas or None

    async def turn(self, history: list[dict], user_text: str,
                   on_tool_call=None, on_tool_result=None,
                   on_reasoning=None, on_content=None, on_llm_end=None
                   ) -> tuple[str, list[dict], TurnTrace]:
        """Exécute un tour complet. Retourne (réponse_texte, historique_maj, trace).

        L'historique géré par l'appelant inclut ce tour ; il suffit de le passer
        au tour suivant tel quel. En streaming, on_reasoning/on_content sont
        appelés fragment par fragment pendant la génération, et on_llm_end une
        fois par réponse LLM terminée (utile pour replier l'affichage du
        raisonnement même quand la réponse ne contient que des tool_calls).
        on_tool_call(name, args) précède l'exécution de l'outil ;
        on_tool_result(name, args, ok, duration_s, result) la suit.
        """
        if not any(m.get("role") == "system" for m in history):
            history = [{"role": "system", "content": self.system_prompt}, *history]
        history = [*history, {"role": "user", "content": user_text}]

        trace = TurnTrace()
        tools = self._tools_schemas()

        for iteration in range(self.max_iterations):
            trace.iterations = iteration + 1
            msg = await self.llm.chat(history, tools=tools, stream=self.stream,
                                      on_reasoning=on_reasoning,
                                      on_content=on_content)
            if on_llm_end:
                on_llm_end()

            if not msg.get("tool_calls"):
                history.append(msg)  # la réponse finale fait partie de l'historique
                return msg.get("content") or "", history, trace

            history.append(msg)
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name = fn["name"]
                args: dict = {}
                failed = False
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    result = f"ERREUR: arguments JSON invalides: {exc}"
                    failed = True
                else:
                    # loggue le nom de l'outil seulement : les args peuvent contenir
                    # des secrets (communauté SNMP, tokens...) et le log fichier est
                    # en DEBUG dans ~/.cache/jarvis/jarvis.log
                    log.info("outil %s appelé", name)
                    if on_tool_call:
                        on_tool_call(name, args)
                    start = time.perf_counter()
                    try:
                        if name in self.internal_tools:
                            # outil interne (ex. jarvis__compact) : peut remplacer l'historique
                            result, new_history = await self.internal_tools[name].fn(
                                self, history, args)
                            if new_history is not None:
                                history = new_history
                        else:
                            result = await self.tools.call(name, args)
                    except Exception as exc:  # noqa: BLE001 — serveur MCP mort, timeout...
                        # le tour ne doit pas mourir : le modèle voit l'erreur
                        # comme résultat d'outil et peut réagir
                        log.warning("outil %s a levé %s: %s", name, type(exc).__name__, exc)
                        result = f"ERREUR: {type(exc).__name__}: {exc}"
                        failed = True
                    duration = time.perf_counter() - start
                    # ok ne se déduit PAS du contenu : un outil qui réussit peut
                    # renvoyer un texte commençant par "ERREUR" (ex: whois sur un
                    # domaine inconnu). Seule une exception (ou des args JSON
                    # invalides) = échec.
                    if on_tool_result:
                        on_tool_result(name, args, not failed, duration, result)
                trace.tool_calls.append((name, json.dumps(args, ensure_ascii=False)))
                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        trace.error = "max_iterations atteint"
        return ("(arrêt: trop d'itérations d'outils)", history, trace)
