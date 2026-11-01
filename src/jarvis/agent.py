"""Boucle tool-calling : user -> LLM -> [tool call -> résultat -> LLM]* -> réponse finale."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .llm import LLMClient, LLMError
from .mcp import McpToolProvider

log = logging.getLogger("jarvis.agent")

# Au-delà de ce ratio d'usage du contexte, l'agent compacte l'historique de
# lui-même avant l'appel LLM (le modèle n'a plus assez de place pour générer
# un tool call valide quand le contexte est plein : llama-server tronque alors
# les arguments JSON et renvoie HTTP 500 "Failed to parse tool call arguments").
_COMPACT_AUTO_THRESHOLD = 0.75
_CTX_NOTE_MARK = "[Usage du contexte :"


@dataclass
class TurnTrace:
    """Observabilité du tour : ce qui a été appelé, combien d'itérations."""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    tts_tags: list[str] = field(default_factory=list)  # [emotion=..][pace=..] extraits de la réponse


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
                 internal_tools: dict[str, InternalTool] | None = None,
                 ctx_max: int | None = None,
                 ctx_usage: Callable[[], Awaitable[int | None]] | None = None):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        # streaming par défaut : le raisonnement s'affiche en direct
        self.stream = stream
        self.internal_tools = internal_tools or {}
        # taille de contexte du serveur (n_ctx) + sonde d'usage réel : permettent
        # de compacter automatiquement avant que le contexte ne soit plein
        self.ctx_max = ctx_max
        self.ctx_usage = ctx_usage
        self._last_tts_tags: list[str] = []  # rempli par l'outil jarvis__voice_tone

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

    def _ctx_estimate(self, history: list[dict]) -> int:
        """Estimation ~4 car./token si l'usage réel du serveur est indisponible."""
        return sum(len(str(m.get("content") or "")) for m in history) // 4

    async def _maybe_auto_compact(self, history: list[dict]) -> tuple[list[dict], bool]:
        """Compacte l'historique si le contexte approche de la saturation.

        Sans cela, un long prompt en un seul tour remplit les 16k tokens : le
        modèle n'a plus la place de générer un tool call entier, llama-server
        tronque les arguments JSON et renvoie HTTP 500 (parse error), et le
        jarvis__compact du modèle n'a jamais pu s'exécuter.
        Retourne (history, compacté?).
        """
        if "jarvis__compact" not in self.internal_tools or self.ctx_max is None:
            return history, False
        usage = None
        if self.ctx_usage is not None:
            try:
                usage = await self.ctx_usage()
            except Exception:  # noqa: BLE001 — sonde best-effort
                usage = None
        est = usage if usage is not None else self._ctx_estimate(history)
        if est < int(self.ctx_max * _COMPACT_AUTO_THRESHOLD):
            return history, False
        log.info("contexte ~%s/%s tok : compaction auto avant l'appel LLM",
                 est, self.ctx_max)
        _msg, new_history = await self.internal_tools["jarvis__compact"].fn(
            self, history, {})
        return new_history if new_history is not None else history, True

    def _with_ctx_note(self, history: list[dict], used: int) -> list[dict]:
        """Injecte l'usage courant du contexte dans le system prompt (remplace
        une note précédente) : le modèle voit combien il lui reste et peut
        appeler jarvis__compact lui-même avant la saturation."""
        if self.ctx_max is None:
            return history
        note = (f"{_CTX_NOTE_MARK} ~{used}/{self.ctx_max} tokens. Si tu approches "
                f"de la limite, appelle jarvis__compact pour libérer du contexte.]")
        out = []
        for m in history:
            if m.get("role") == "system":
                content = str(m.get("content") or "")
                head = content.split(_CTX_NOTE_MARK)[0].rstrip()
                m = {"role": "system", "content": f"{head}\n{note}" if head else note}
            out.append(m)
        return out

    async def _chat(self, history: list[dict], tools: list[dict] | None,
                    on_reasoning, on_content, user_text: str
                    ) -> tuple[dict, list[dict]]:
        """Un appel LLM. Retourne (message, historique) — l'historique change
        si un rattrapage a compacté entre-temps.

        Rattrapage : si le serveur renvoie une erreur de tool call tronqué
        (contexte plein : llama-server coupe les arguments JSON et renvoie
        HTTP 500), on compacte l'historique et on réessaie une fois."""
        try:
            msg = await self.llm.chat(history, tools=tools, stream=self.stream,
                                      on_reasoning=on_reasoning,
                                      on_content=on_content)
            return msg, history
        except LLMError as exc:
            if ("tool call" not in str(exc).lower()
                    or "jarvis__compact" not in self.internal_tools):
                raise
            log.warning("tool call tronqué (%s) : compaction forcée + retry", exc)
            to_compact = history
            if (to_compact and to_compact[-1].get("role") == "user"
                    and to_compact[-1].get("content") == user_text):
                to_compact = to_compact[:-1]
            _msg, new_history = await self.internal_tools["jarvis__compact"].fn(
                self, to_compact, {})
            if new_history is not None:
                new_history = [*new_history, {"role": "user", "content": user_text}]
                msg = await self.llm.chat(new_history, tools=tools, stream=self.stream,
                                          on_reasoning=on_reasoning,
                                          on_content=on_content)
                return msg, new_history
            raise

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

        history, _compacted = await self._maybe_auto_compact(history)
        history = [*history, {"role": "user", "content": user_text}]

        trace = TurnTrace()
        tools = self._tools_schemas()

        for iteration in range(self.max_iterations):
            trace.iterations = iteration + 1
            if self.ctx_max is not None and self.ctx_usage is not None:
                try:
                    used = await self.ctx_usage()
                except Exception:  # noqa: BLE001 — sonde best-effort
                    used = None
                if used is None:
                    used = self._ctx_estimate(history)
                history = self._with_ctx_note(history, used)
            msg, history = await self._chat(history, tools, on_reasoning,
                                            on_content, user_text)
            if on_llm_end:
                on_llm_end()

            if not msg.get("tool_calls"):
                history.append(msg)  # la réponse finale fait partie de l'historique
                content = msg.get("content") or ""
                # tag d'émotion/pace en tête de réponse (Voxtral TTS 3) : extrait pour
                # le TTS, mais retiré du texte affiché/sauvegardé (jamais lu à l'écran)
                from .tts_tags import extract_lead_tags
                tags, clean = extract_lead_tags(content)
                if tags:
                    trace.tts_tags = tags
                    msg["content"] = clean
                    content = clean
                elif self._last_tts_tags:
                    # ton choisi via l'outil interne jarvis__voice_tone
                    trace.tts_tags = self._last_tts_tags
                self._last_tts_tags = []
                return content, history, trace

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
