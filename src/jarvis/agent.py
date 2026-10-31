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


class Agent:
    def __init__(self, llm: LLMClient, tools: McpToolProvider,
                 system_prompt: str, max_iterations: int = 8,
                 stream: bool = True):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        # streaming par défaut : le raisonnement s'affiche en direct
        self.stream = stream

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
        tools = self.tools.schemas() if self.tools else None

        for iteration in range(self.max_iterations):
            trace.iterations = iteration + 1
            msg = await self.llm.chat(history, tools=tools, stream=self.stream,
                                      on_reasoning=on_reasoning,
                                      on_content=on_content)
            if on_llm_end:
                on_llm_end()

            if not msg.get("tool_calls"):
                return msg.get("content") or "", history, trace

            history.append(msg)
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    result = f"ERREUR: arguments JSON invalides: {exc}"
                else:
                    log.info("outil %s args=%s", name, args)
                    if on_tool_call:
                        on_tool_call(name, args)
                    start = time.perf_counter()
                    result = await self.tools.call(name, args)
                    duration = time.perf_counter() - start
                    if on_tool_result:
                        on_tool_result(name, args,
                                       not result.startswith("ERREUR"), duration, result)
                trace.tool_calls.append((name, json.dumps(args, ensure_ascii=False)))
                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        trace.error = "max_iterations atteint"
        return ("(arrêt: trop d'itérations d'outils)", history, trace)
