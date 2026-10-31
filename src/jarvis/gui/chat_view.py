"""Vue de chat LLM : messages, streaming du raisonnement et des réponses, outils.

Un tour = un appel à GuiBackend.stream_turn ; les événements du puits mettent
à jour les widgets en direct (même tâche async, donc sûr) :
- reasoning -> expansion "Réflexion" en grisé, ouverte pendant le streaming ;
- tool_call -> lignes d'appel d'outil (nom + arguments) dans le bloc assistant ;
- content   -> markdown de la réponse, re-rendu fragment par fragment ;
- done/error -> finalisation (spinner off, notification).
"""

from __future__ import annotations

import json

from nicegui import ui

from .state import GuiState


class ChatView:
    def __init__(self, state: GuiState):
        self.state = state
        with ui.column().classes("w-full h-full gap-3") as self.root:
            self.scroll = ui.scroll_area().classes("w-full grow overflow-y-auto")
            with self.scroll:
                self.messages = ui.column().classes("w-full max-w-3xl mx-auto gap-4 py-2 px-1")
            self._welcome()
            with ui.row().classes("w-full max-w-3xl mx-auto items-center gap-2"):
                self.input = ui.input(placeholder="Message à Jarvis…").classes("grow").props(
                    "outlined dense rounded clearable")
                self.input.on("keydown.enter", self.send)
                self.spinner = ui.spinner(size="sm").classes("self-center")
                self.spinner.visible = False
                self.send_btn = ui.button(icon="send", on_click=self.send).props(
                    "round dense color=primary")
        state.reset_listeners.append(self.clear)
        self._set_busy(False)

    # ---------- UI de base ----------

    def _welcome(self) -> None:
        with self.messages:
            with ui.column().classes("w-full items-center gap-1 py-10 text-center"):
                ui.icon("smart_toy", size="48px").classes("text-[#4cc2ff] opacity-70")
                ui.label("Discute avec Jarvis").classes("text-xl font-semibold")
                ui.label("Le raisonnement du modèle s'affiche en direct, puis la réponse, puis les outils appelés.").classes(
                    "text-sm text-gray-500")

    def clear(self) -> None:
        if self.messages.is_deleted:
            return
        self.messages.clear()
        self._welcome()

    def _set_busy(self, busy: bool) -> None:
        self.send_btn.enabled = not busy
        self.input.enabled = not busy
        self.spinner.visible = busy

    def _scroll_bottom(self) -> None:
        self.scroll.scroll_to(percent=1.0, duration=0.15)

    # ---------- rendu des messages ----------

    def _add_user(self, text: str) -> None:
        with self.messages:
            with ui.row().classes("w-full justify-end"):
                with ui.card().classes("max-w-[80%] bg-[#1c2b3a] rounded-2xl px-4 py-2"):
                    ui.markdown(text).classes("text-sm")

    def _start_assistant_block(self) -> dict:
        """Crée le bloc assistant (raisonnement + outils + réponse) et le retourne."""
        block: dict = {}
        with self.messages:
            with ui.column().classes("w-full gap-2"):
                exp = ui.expansion("Réflexion", icon="psychology").classes(
                    "w-full text-sm text-gray-400 rounded-xl border border-[#232833]")
                exp.props("dense")
                with exp:
                    block["reasoning"] = ui.markdown("").classes(
                        "text-sm italic text-gray-500 reasoning-md")
                block["tools"] = ui.column().classes("w-full gap-1")
                with ui.card().classes("w-full bg-[#14171d] rounded-2xl px-4 py-3"):
                    block["content"] = ui.markdown("").classes("text-sm")
                exp.visible = False
        block["expansion"] = exp
        return block

    def _add_tool_call(self, container: ui.column, name: str, args: dict) -> None:
        with container:
            with ui.row().classes("items-center gap-2"):
                ui.chip(name, icon="build").props("dense color=secondary text-color=white").classes(
                    "text-xs mono")
                ui.label(json.dumps(args, ensure_ascii=False)).classes(
                    "text-xs text-gray-500 mono truncate")

    # ---------- envoi ----------

    async def send(self) -> None:
        backend = self.state.backend
        if backend is None or backend.busy:
            return
        text = (self.input.value or "").strip()
        if not text:
            return
        self.input.value = ""
        self._set_busy(True)
        self._add_user(text)
        block = self._start_assistant_block()
        self._scroll_bottom()

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        exp = block["expansion"]
        reasoning_md = block["reasoning"]
        content_md = block["content"]
        tools_col = block["tools"]

        def sink(event: dict) -> None:
            kind = event["type"]
            if kind == "reasoning":
                reasoning_parts.append(event["text"])
                if not exp.visible:
                    exp.visible = True
                    exp.value = True  # ouverte pendant le streaming
                reasoning_md.set_content("".join(reasoning_parts))
                self._scroll_bottom()
            elif kind == "content":
                content_parts.append(event["text"])
                content_md.set_content("".join(content_parts))
                self._scroll_bottom()
            elif kind == "tool_call":
                self._add_tool_call(tools_col, event["name"], event["args"])
                self._scroll_bottom()
            elif kind == "error":
                ui.notify(f"Erreur: {event['message']}", type="negative")
                with self.messages:
                    ui.label(f"Erreur : {event['message']}").classes("text-negative text-sm")
                self._scroll_bottom()

        await backend.stream_turn(text, sink)

        # finalisation
        if not content_parts and not reasoning_parts:
            content_md.set_content("*(pas de réponse)*")
        self._scroll_bottom()
        self._set_busy(False)
        self.input.run_method("focus")
