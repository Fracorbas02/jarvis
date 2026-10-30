"""Agrégation de plusieurs MCP servers stdio en un fournisseur d'outils unifié.

Chaque serveur de la config est lancé en sous-processus stdio (protocole MCP).
Les outils sont exposés au LLM sous le nom "<server>__<tool>" pour éviter les
collisions entre serveurs.

C'est la frontière stratégique du projet : le harnais agent ne connaît que ce
contrat ; les serveurs MCP (et plus tard le modèle fine-tuné cybersécurité)
évoluent librement derrière.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpToolProvider:
    def __init__(self, servers: dict[str, list[str]]):
        # name -> argv de lancement
        self._servers = servers
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        # (full_name) -> (server_name, tool)
        self._tools: dict[str, tuple[str, object]] = {}

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        for name, argv in self._servers.items():
            params = StdioServerParameters(command=argv[0], args=list(argv[1:]))
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.list_tools()
            self._sessions[name] = session
            for tool in result.tools:
                full_name = f"{name}__{tool.name}"
                self._tools[full_name] = (name, tool)

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._sessions.clear()
        self._tools.clear()

    def schemas(self) -> list[dict]:
        """Schémas OpenAI "tools" pour l'ensemble des outils agrégés."""
        out = []
        for full_name, (_srv, tool) in self._tools.items():
            # SDK MCP >= 2 : champs snake_case ; <= 1.x : camelCase
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            out.append({
                "type": "function",
                "function": {
                    "name": full_name,
                    "description": tool.description or "",
                    "parameters": schema or {"type": "object", "properties": {}},
                },
            })
        return out

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def call(self, full_name: str, arguments: dict) -> str:
        if full_name not in self._tools:
            return f"ERREUR: outil inconnu '{full_name}'"
        server_name, _tool = self._tools[full_name]
        session = self._sessions[server_name]
        result = await session.call_tool(full_name.split("__", 1)[1],
                                         arguments=arguments or {})
        texts = [b.text for b in result.content if getattr(b, "type", "") == "text"]
        if texts:
            return "\n".join(texts)
        # is_error (v2) / isError (v1) : on remonte le détail si présent
        if getattr(result, "is_error", None) or getattr(result, "isError", None):
            return f"ERREUR outil: {json.dumps([getattr(b, 'model_dump', lambda: str(b))() for b in result.content])[:2000]}"
        return json.dumps([getattr(b, "model_dump", lambda: str(b))() for b in result.content])
