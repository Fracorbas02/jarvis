"""Pont entre la GUI et le cœur agentique existant (config + LLM + MCP + Agent).

Ne réécrit rien : réutilise LLMClient, McpToolProvider et Agent. La GUI fournit
un puits d'événements (sink) qui reçoit des dicts :

  {"type": "reasoning",  "text": ...}   fragment de pensée (streaming)
  {"type": "content",    "text": ...}   fragment de réponse (streaming)
  {"type": "tool_call",  "name": ..., "args": {...}}   outil appelé
  {"type": "done",       "reply": ..., "trace": {...}}
  {"type": "error",      "message": ...}

L'historique de conversation vit dans le backend : un tour = un appel à
stream_turn, la GUI n'a pas à gérer les messages.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Callable

from ..agent import Agent
from ..config import Config, load_config
from ..llm import LLMClient
from ..mcp import McpToolProvider

log = logging.getLogger("jarvis.gui.backend")

Sink = Callable[[dict], None]

# Paramètres acceptés par LLMClient.__init__ : on filtre la config avec, pour
# ne pas casser si LLMConfig gagne des champs réservés au CLI (ex. ctx_size).
_LLM_PARAMS = set(inspect.signature(LLMClient.__init__).parameters) - {"self"}


class GuiBackend:
    def __init__(self, config_path: str | None = None):
        self.cfg: Config = load_config(config_path)
        self.llm = LLMClient(**{k: v for k, v in self.cfg.llm.__dict__.items()
                                if k in _LLM_PARAMS})
        self.tools = McpToolProvider(self.cfg.mcp_servers)
        self.agent = Agent(self.llm, self.tools,
                           self.cfg.agent.system_prompt,
                           self.cfg.agent.max_iterations,
                           stream=True)
        self.history: list[dict] = []
        self.busy = False
        self._lock = asyncio.Lock()
        self.started = False

    async def start(self) -> None:
        if self.cfg.mcp_servers:
            await self.tools.start()
        self.started = True
        log.info("backend démarré, %d outils MCP", len(self.tools.names()))

    async def stop(self) -> None:
        # La fermeture des transports stdio MCP (anyio) échoue si elle n'a pas
        # lieu dans la même tâche que l'ouverture (harnais de test) : on ne
        # laisse pas ça faire échouer l'arrêt de la GUI.
        try:
            await self.tools.stop()
        except RuntimeError as exc:
            log.debug("fermeture MCP dégradée: %s", exc)
        await self.llm.aclose()
        self.started = False

    def tool_names(self) -> list[str]:
        return self.tools.names()

    def reset(self) -> None:
        self.history.clear()

    async def stream_turn(self, text: str, sink: Sink) -> None:
        """Un tour complet avec streaming des événements vers le sink."""
        if self.busy:
            sink({"type": "error", "message": "un tour est déjà en cours"})
            return
        async with self._lock:
            self.busy = True
        try:
            def on_reasoning(fragment: str) -> None:
                sink({"type": "reasoning", "text": fragment})

            def on_content(fragment: str) -> None:
                sink({"type": "content", "text": fragment})

            def on_tool_call(name: str, args: dict) -> None:
                sink({"type": "tool_call", "name": name, "args": args})

            try:
                reply, self.history, trace = await self.agent.turn(
                    self.history, text,
                    on_tool_call=on_tool_call,
                    on_reasoning=on_reasoning,
                    on_content=on_content,
                )
                sink({"type": "done", "reply": reply,
                      "trace": {"tool_calls": trace.tool_calls,
                                "iterations": trace.iterations,
                                "error": trace.error}})
            except Exception as exc:  # noqa: BLE001 — remonter à la GUI, ne pas tuer la boucle
                log.exception("tour en erreur")
                sink({"type": "error", "message": str(exc)})
        finally:
            self.busy = False
