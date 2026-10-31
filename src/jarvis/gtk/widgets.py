"""Widgets de la conversation GTK4 : bulles de messages, reasoning, outils.

Les réponses de l'assistant sont rendues en Markdown via le markup Pango
(voir markdown.py : titres, **gras**, *italique*, `code`, listes, blocs ```)
re-rendu à chaque fragment streamé ; le texte utilisateur est échappé.
Le raisonnement du modèle est repliable (Adw.ExpanderRow) et grisé/italique.
"""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, GLib, Gtk, Pango  # noqa: E402

from .markdown import to_pango  # noqa: E402


def _css() -> None:
    css = Gtk.CssProvider()
    css.load_from_string("""
    .jarvis-msg-user .jarvis-bubble { background-color: @accent_bg_color; }
    .jarvis-msg-user .jarvis-text { color: @accent_fg_color; }
    .jarvis-msg-assistant .jarvis-bubble { background-color: alpha(@window_fg_color, 0.06); }
    .jarvis-reasoning { color: alpha(@window_fg_color, 0.55); font-style: italic; }
    .jarvis-tool { color: @purple_3; }
    .jarvis-error { color: @error_color; }
    .jarvis-msg-user .jarvis-bubble { border-radius: 14px 14px 4px 14px; }
    .jarvis-msg-assistant .jarvis-bubble { border-radius: 14px 14px 14px 4px; }
    .jarvis-bubble { border-radius: 14px; }
    .jarvis-meta { font-size: 0.8em; color: alpha(@window_fg_color, 0.45); }
    .jarvis-home-title { font-size: 44px; font-weight: 700; }
    .jarvis-notif-badge {
        background-color: @error_color;
        color: #ffffff;
        border-radius: 10px;
        font-size: 0.75em;
        font-weight: 700;
        padding: 1px 5px;
        min-width: 16px;
    }
    .jarvis-recording {
        animation: jarvis-pulse 1.2s ease-in-out infinite;
        border: 2px solid @accent_bg_color;
    }
    @keyframes jarvis-pulse {
        0%   { box-shadow: 0 0 0 0 alpha(@accent_bg_color, 0.5); }
        70%  { box-shadow: 0 0 0 14px alpha(@accent_bg_color, 0); }
        100% { box-shadow: 0 0 0 0 alpha(@accent_bg_color, 0); }
    }
    .jarvis-home-bg {
        background: linear-gradient(160deg,
            alpha(@accent_bg_color, 0.10) 0%,
            alpha(@window_bg_color, 0) 45%,
            alpha(@accent_bg_color, 0.06) 100%);
    }
    .jarvis-mic-level { margin-top: 2px; }
    .jarvis-ctx-bar { min-height: 6px; }
    """
    )
    display = Gdk.Display.get_default()
    if display is not None:
        Gtk.StyleContext.add_provider_for_display(
            display, css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


class MessageBox(Gtk.Box):
    """Une bulle de conversation : rôle (user/assistant/system), contenu,
    reasoning repliable, ligne d'outils, métadonnées."""

    def __init__(self, role: str):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.role = role
        self.add_css_class(f"jarvis-msg-{role}")
        self.set_margin_start(48)
        self.set_margin_end(48)
        self.set_margin_top(6)
        self.set_margin_bottom(6)

        bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        bubble.add_css_class("jarvis-bubble")
        bubble.set_margin_start(12)
        bubble.set_margin_end(12)
        bubble.set_margin_top(8)
        bubble.set_margin_bottom(8)
        self.append(bubble)
        self._bubble = bubble

        self.reasoning_row: Adw.ExpanderRow | None = None
        self.reasoning_label: Gtk.Label | None = None
        if role == "assistant":
            self.reasoning_row = Adw.ExpanderRow(title="Raisonnement")
            self.reasoning_row.add_css_class("jarvis-reasoning")
            self.reasoning_label = Gtk.Label(xalign=0, wrap=True, selectable=True,
                                             use_markup=False)
            self.reasoning_label.add_css_class("jarvis-reasoning")
            self.reasoning_label.set_margin_start(8)
            self.reasoning_label.set_margin_end(8)
            self.reasoning_label.set_margin_bottom(6)
            self.reasoning_row.add_row(self.reasoning_label)
            self.reasoning_row.set_visible(False)
            bubble.append(self.reasoning_row)

        self.text_label = Gtk.Label(xalign=0, wrap=True, selectable=True,
                                    use_markup=True, wrap_mode=Pango.WrapMode.WORD_CHAR)
        self._md_buffer = ""
        if role == "user":
            self.text_label.add_css_class("jarvis-text")
        bubble.append(self.text_label)

        self.tools_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        bubble.append(self.tools_box)

        self.meta_label = Gtk.Label(xalign=0, wrap=True)
        self.meta_label.add_css_class("jarvis-meta")
        self.append(self.meta_label)

    def set_text(self, text: str) -> None:
        """Texte brut (rôle user/system) : échappé puis affiché en markup."""
        from xml.sax.saxutils import escape
        self._md_buffer = text
        self.text_label.set_markup(escape(text))
        self.text_label.set_visible(bool(text))

    def set_markdown(self, text: str) -> None:
        """Texte markdown (réponse assistant) rendu en markup Pango."""
        self._md_buffer = text
        self.text_label.set_markup(to_pango(text))
        self.text_label.set_visible(bool(text))

    def append_text(self, fragment: str) -> None:
        """Ajoute un fragment streamé et re-rend tout le Markdown : les
        marqueurs qui se ferment en cours de route (ex. "**bo" + "ld**")
        sont pris en compte au fragment suivant."""
        self._md_buffer += fragment
        self.text_label.set_markup(to_pango(self._md_buffer))
        self.text_label.set_visible(True)

    def append_reasoning(self, fragment: str) -> None:
        if self.reasoning_row is None:
            return
        assert self.reasoning_label is not None
        self.reasoning_row.set_visible(True)
        cur = self.reasoning_label.get_text()
        self.reasoning_label.set_text(cur + fragment)

    def collapse_reasoning(self, summary: str) -> None:
        if self.reasoning_row is None:
            return
        self.reasoning_row.set_title(summary or "Raisonnement")
        self.reasoning_row.set_expanded(False)

    def add_tool_line(self, name: str, args: dict, ok: bool | None = None,
                      duration: float | None = None, result_preview: str = "") -> Gtk.Label:
        import json as _json
        args_txt = _json.dumps(args, ensure_ascii=False)
        if len(args_txt) > 120:
            args_txt = args_txt[:117] + "..."
        if ok is None:
            line = f"\u26a1 {name}({args_txt})"
        else:
            mark = "\u2713" if ok else "\u2717"
            dur = f" {duration:.2f}s" if duration is not None else ""
            line = f"{mark} {name}({args_txt}){dur}"
        if result_preview and ok is False:
            line += f"  — {result_preview[:80]}"
        lbl = Gtk.Label(xalign=0, wrap=True, selectable=True)
        lbl.add_css_class("jarvis-tool")
        lbl.set_text(line)
        self.tools_box.append(lbl)
        return lbl

    def set_meta(self, text: str) -> None:
        self.meta_label.set_text(text)
        self.meta_label.set_visible(bool(text))


class MessagesColumn(Gtk.Box):
    """Colonne scrollable de MessageBox, avec autoscroll en bas."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._scroll = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
        self._scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._scroll.set_child(self)
        self._boxes: list[MessageBox] = []

    @property
    def widget(self) -> Gtk.ScrolledWindow:
        return self._scroll

    def add(self, box: MessageBox) -> None:
        self._boxes.append(box)
        self.append(box)
        self.scroll_to_end()

    def clear(self) -> None:
        for b in self._boxes:
            self.remove(b)
        self._boxes.clear()

    def scroll_to_end(self) -> None:
        vadj = self._scroll.get_vadjustment()
        GLib.idle_add(self._set_v, vadj, vadj.get_upper())

    @staticmethod
    def _set_v(vadj: Gtk.Adjustment, value: float) -> bool:
        vadj.set_value(max(0.0, value - vadj.get_page_size()))
        return GLib.SOURCE_REMOVE


def user_message(text: str) -> MessageBox:
    box = MessageBox("user")
    box.set_halign(Gtk.Align.END)
    box.set_text(text)
    return box


def assistant_message() -> MessageBox:
    box = MessageBox("assistant")
    box.set_halign(Gtk.Align.START)
    return box


def info_message(text: str) -> MessageBox:
    box = MessageBox("system")
    box.set_halign(Gtk.Align.CENTER)
    box.set_text(text)
    box.text_label.add_css_class("jarvis-meta")
    return box


def error_message(text: str) -> MessageBox:
    box = MessageBox("system")
    box.set_halign(Gtk.Align.CENTER)
    box.set_text(text)
    box.text_label.add_css_class("jarvis-error")
    return box
