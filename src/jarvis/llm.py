"""Client minimal OpenAI-compatible pour llama-server (ou tout serveur compatible).

llama.cpp expose /v1/chat/completions avec support natif des tools si lancé
avec --jinja. On garde httpx plutôt que le SDK openai : moins de dépendances,
contrôle total sur le format, et le format Mistral tool-calling est fragile
à travers les couches d'abstraction.

Deux modes :
- non-streaming (défaut) : un POST, un message complet. Utilisé par les tests.
- streaming : deltas SSE avec callbacks on_reasoning/on_content pour afficher
  la pensée du modèle en direct (delta.reasoning_content) puis la réponse.
  Les tool_calls arrivent fragmentés (index/id/name/arguments par morceaux)
  et sont réassemblés.
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

    def _payload(self, messages: list[dict], tools: list[dict] | None,
                 stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True
            # demande l'usage réel (prompt/completion tokens) dans le chunk final
            payload["stream_options"] = {"include_usage": True}
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   stream: bool = False,
                   on_reasoning=None, on_content=None) -> dict:
        """Un tour de complétion. Retourne le message assistant normalisé :
        {"role": "assistant", "content": str|None, "tool_calls": [...]|None}.

        En streaming, on_reasoning/on_content reçoivent chaque fragment de
        raisonnement / de contenu dès qu'il arrive (affichage live).
        """
        payload = self._payload(messages, tools, stream)

        usage: dict | None = None
        if stream:
            msg = await self._chat_stream(payload, on_reasoning, on_content)
            usage = msg.pop("usage", None)
        else:
            resp = await self._http.post("/chat/completions", json=payload)
            if resp.status_code != 200:
                raise LLMError(f"llama-server HTTP {resp.status_code}: {resp.text[:2000]}")
            data = resp.json()
            try:
                msg = data["choices"][0]["message"]
            except (KeyError, IndexError) as exc:
                raise LLMError(f"réponse inattendue: {json.dumps(data)[:2000]}") from exc
            usage = data.get("usage")

        # Normalisation : s'assurer que tool_calls est absent plutôt que null,
        # et que le message est directement réinsérable dans l'historique.
        out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        if usage:
            out["usage"] = usage  # {"prompt_tokens": N, "completion_tokens": M, ...}
        return out

    async def _chat_stream(self, payload: dict, on_reasoning, on_content) -> dict:
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        usage: dict | None = None
        async with self._http.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:2000]
                raise LLMError(f"llama-server HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise LLMError(f"llama-server: {chunk['error']}")
                # chunk final "usage" (stream_options.include_usage) : pas de choices
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}

                reasoning = delta.get("reasoning_content")
                if reasoning:
                    if on_reasoning:
                        on_reasoning(reasoning)
                content = delta.get("content")
                if content:
                    content_parts.append(content)
                    if on_content:
                        on_content(content)

                # tool_calls fragmentés : réassemblage par index
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    acc = tool_calls_acc.setdefault(idx, {
                        "id": None, "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["function"]["arguments"] += fn["arguments"]

        msg: dict[str, Any] = {"role": "assistant",
                               "content": "".join(content_parts) or None}
        if tool_calls_acc:
            msg["tool_calls"] = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        if usage:
            msg["usage"] = usage
        return msg

    async def aclose(self) -> None:
        await self._http.aclose()
