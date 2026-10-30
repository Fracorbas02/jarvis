"""Client minimal OpenAI-compatible pour llama-server (ou tout serveur compatible).

llama.cpp expose /v1/chat/completions avec support natif des tools si lancé
avec --jinja. On garde httpx plutôt que le SDK openai : moins de dépendances,
contrôle total sur le format, et le format Mistral tool-calling est fragile
à travers les couches d'abstraction.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "local",
                 temperature: float = 0.3, max_tokens: int = 4096,
                 timeout: float = 300.0, chat_template_kwargs: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_template_kwargs = chat_template_kwargs or {}
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Un tour de complétion. Retourne le message assistant normalisé :
        {"role": "assistant", "content": str|None, "tool_calls": [...]|None}."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = await self._http.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            raise LLMError(f"llama-server HTTP {resp.status_code}: {resp.text[:2000]}")
        data = resp.json()
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"réponse inattendue: {json.dumps(data)[:2000]}") from exc

        # Normalisation : s'assurer que tool_calls est absent plutôt que null,
        # et que le message est directement réinsérable dans l'historique.
        out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        return out

    async def aclose(self) -> None:
        await self._http.aclose()
