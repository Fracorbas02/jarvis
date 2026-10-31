"""Panneau de chat desktop : zone de texte avec tags + entrée + envoi.

Les événements du backend (drainés par app.py) arrivent ici :
- reasoning : fragments en grisé italique (tag "reasoning"), précédés d'un
  en-tête "réflexion" une fois par tour ;
- tool_call : ligne "→ nom {args}" en violet (tag "tool") ;
- content   : réponse rendue en Markdown (titres, **gras**, *italique*, `code`,
  listes, blocs ```) dans la zone "assistant", en-tête "Jarvis" une fois ;
- done/error : fin de tour (réactive la saisie) / message d'erreur.
"""

from __future__ import annotations

import json

import customtkinter as ctk

from .controller import DesktopController
from .markdown import parse_markdown
from .theme import ACCENT, BG, CODE_BG, PANEL_2, TEXT_DIM, TOOL, ERROR


class ChatPanel(ctk.CTkFrame):
    def __init__(self, master, ctrl: DesktopController, scale: float = 1.0):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.ctrl = ctrl
        self.scale = scale
        self.busy = False
        self._reasoning_started = False
        self._content_started = False
        self._content_parts: list[str] = []

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # en-tête
        ctk.CTkLabel(self, text="Chat LLM", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=ACCENT).grid(row=0, column=0, padx=16, pady=(12, 4),
                                             sticky="w")

        # zone de conversation
        # cursor="arrow" transmis au tkinter.Text interne : sinon le Text garde
        # le curseur X11 par défaut "xterm" (non scalé sous XWayland HiDPI,
        # donc minuscule par rapport au reste de la fenêtre).
        self.text = ctk.CTkTextbox(self, wrap="word", fg_color=PANEL_2,
                                   text_color="#e6edf3", border_width=0,
                                   font=ctk.CTkFont(size=13), activate_scrollbars=True,
                                   cursor="arrow")
        self.text.grid(row=1, column=0, padx=12, pady=8, sticky="nsew")
        self.text._canvas.configure(cursor="arrow")
        # CTkTextbox.tag_config interdit l'option font (scaling ctk) : couleurs
        # via tag_config, polices via le textbox interne. Les polices des tags
        # DOIVENT être scalées à la main (taille en pixels NÉGATIFS x scale),
        # sinon le texte du chat reste à l'échelle 1x pendant que le reste de
        # l'UI est à l'échelle de l'OS.
        self._tag("user", fg=ACCENT, size=13, styles=("bold",))
        self._tag("assistant", fg="#e6edf3")
        self._tag("assistant_head", fg=ACCENT, size=13, styles=("bold",))
        self._tag("reasoning", fg="#6e7681", size=12, styles=("italic",))
        self._tag("reasoning_head", fg="#6e7681", size=12,
                  styles=("italic", "bold"))
        self._tag("tool", fg=TOOL, size=12)
        self._tag("error", fg=ERROR)
        # styles du rendu Markdown de la réponse (voir markdown.py) — le code
        # passe en police monospace "DejaVu Sans Mono" (toujours dispo sous X11,
        # contrairement à une famille embarquée non enregistrée côté serveur X)
        self._tag("md_bold", size=13, styles=("bold",))
        self._tag("md_italic", size=13, styles=("italic",))
        self._tag("md_strike", size=13, styles=("overstrike",))
        self._tag("md_heading", fg=ACCENT, size=15, styles=("bold",))
        self._tag("md_code", fg="#d2a8ff", size=12, family="DejaVu Sans Mono")
        self._tag("md_code_block", fg="#d2a8ff", size=12, bg=CODE_BG,
                  family="DejaVu Sans Mono")
        self.text.configure(state="disabled")
        # marqueur de la zone Markdown re-rendue à chaque fragment "content"
        self.text._textbox.mark_set("md", "end")
        self.text._textbox.mark_gravity("md", "left")

    def _scaled_font(self, size: float, styles: tuple = (),
                     family: str = "Roboto") -> tuple:
        # tuple tk : (family, taille en pixels si négatif, *styles). La famille
        # "Roboto" (et non "TkDefaultFont") : sous XWayland, si la famille nommée
        # n'est pas enregistrée côté serveur X, Tk ignore tout le tuple et rend
        # en "fixed" par défaut (texte minuscule).
        return (family, -round(size * self.scale), *styles)

    def _tag(self, name: str, fg: str | None = None, size: float | None = None,
             styles: tuple = (), bg: str | None = None,
             family: str = "Roboto") -> None:
        kw: dict = {}
        if fg:
            kw["foreground"] = fg
        if bg:
            kw["background"] = bg
        if size:
            kw["font"] = self._scaled_font(size, styles, family)
        if kw:
            self.text._textbox.tag_config(name, **kw)

        # saisie
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")
        row.grid_columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(row, placeholder_text="Message à Jarvis…",
                                  height=34, fg_color=PANEL_2, border_color="#2a2e37")
        # curseur nommé sur le widget interne : "xterm" (défaut Entry sous X11)
        # n'est pas scalé sous XWayland HiDPI et apparaît minuscule
        self.entry._entry.configure(cursor="xterm")
        self.entry._canvas.configure(cursor="xterm")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self.send())
        self.send_btn = ctk.CTkButton(row, text="Envoyer", width=90, height=34,
                                      fg_color=ACCENT, text_color="#06121a",
                                      command=self.send)
        self.send_btn.grid(row=0, column=1)

        self._welcome()

    # ---------- affichage ----------

    def _insert(self, content: str, tag: str | None = None) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", content, tag)
        self.text.configure(state="disabled")
        self.text.see("end")

    def _welcome(self) -> None:
        self._insert("Discute avec Jarvis — le raisonnement du modèle s'affiche "
                     "en grisé, puis la réponse (en Markdown), puis les outils appelés.\n\n",
                     "reasoning")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text._textbox.mark_set("md", "end")
        self.text._textbox.mark_gravity("md", "left")
        self.text.configure(state="disabled")
        self._content_parts = []
        self._welcome()

    def _render_markdown(self) -> None:
        """Re-rend la zone Markdown (après le marqueur "md") depuis le buffer
        des fragments content reçus — appelé à chaque fragment pour que les
        marqueurs qui se ferment en cours de streaming soient pris en compte."""
        tb = self.text._textbox
        self.text.configure(state="normal")
        tb.delete("md", "end")
        for seg in parse_markdown("".join(self._content_parts)):
            tags = ["assistant"]
            if "heading" in seg.style:
                tags.append("md_heading")
            if "bold" in seg.style:
                tags.append("md_bold")
            if "italic" in seg.style:
                tags.append("md_italic")
            if "strike" in seg.style:
                tags.append("md_strike")
            if "code" in seg.style and "mono" in seg.style:
                tags.append("md_code_block")
            elif "code" in seg.style:
                tags.append("md_code")
            self.text.insert("end", seg.text, tuple(tags))
        self.text.configure(state="disabled")
        self.text.see("end")

    # ---------- envoi ----------

    def send(self) -> None:
        if self.busy or self.ctrl.backend is None:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.busy = True
        self.send_btn.configure(state="disabled")
        self.entry.configure(state="disabled")
        self._reasoning_started = False
        self._content_started = False
        self._content_parts = []
        self._insert(f"Toi › {text}\n\n", "user")
        self.ctrl.stream_turn(text)

    # ---------- événements du backend ----------

    def handle_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "reasoning":
            if not self._reasoning_started:
                self._reasoning_started = True
                self._insert("réflexion › ", "reasoning_head")
            self._insert(ev["text"], "reasoning")
        elif kind == "content":
            if not self._content_started:
                self._content_started = True
                self.text._textbox.mark_set("md", "end")
                self.text._textbox.mark_gravity("md", "left")
                self._content_parts = []
                self._insert("\nJarvis › ", "assistant_head")
            self._content_parts.append(ev["text"])
            self._render_markdown()
        elif kind == "tool_call":
            args = json.dumps(ev.get("args", {}), ensure_ascii=False)
            self._insert(f"\n→ {ev['name']} {args}\n", "tool")
        elif kind == "error":
            self._insert(f"\nerreur : {ev['message']}\n", "error")
            self._end_turn()
        elif kind == "done":
            self._insert("\n")
            self._end_turn()

    def _end_turn(self) -> None:
        self.busy = False
        self.send_btn.configure(state="normal")
        self.entry.configure(state="normal")
        self.entry.focus_set()
