"""Widgets de la conversation GTK4 : bulles de messages, reasoning, outils.

Les réponses de l'assistant sont rendues en Markdown complet (CommonMark +
GFM, voir markdown.py) via des widgets GTK natifs, re-rendus à chaque
fragment streamé :
- titres, paragraphes, inline (gras/italique/barré/code) -> Gtk.Label Pango,
  liens cliquables (<a href>, activate-link -> navigateur) ;
- listes puces/numérotées/tâches -> puces "•"/"1."/☑☐ indentées ;
- citations -> bloc indenté avec barre latérale ;
- blocs de code -> label mono sur fond sombre, sélectionnable ;
- tableaux GFM -> Gtk.Grid avec alignements et en-tête en gras ;
- images -> Gtk.Picture chargé en async (cache mémoire), avec repli sur le
  lien alt si le chargement échoue ;
- séparateur --- -> Gtk.Separator ;
- footnotes -> exposants cliquables + définitions en fin de message.

Le texte utilisateur (rôles user/system) reste échappé (pas de markup).
Le raisonnement du modèle est repliable (Adw.ExpanderRow) et grisé/italique.
"""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402
from xml.sax.saxutils import escape  # noqa: E402

from .markdown import (_safe_href, collect_defs, inline_to_pango, parse_blocks,
                       sanitize_pango, to_pango)  # noqa: E402

_HEADING_SIZE = {1: "xx-large", 2: "x-large", 3: "large", 4: "medium",
                 5: "small", 6: "small"}

# cache mémoire des images téléchargées : url -> (bytes, content_type)
_IMAGE_CACHE: dict[str, tuple[bytes, str]] = {}


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
    .jarvis-ctx-bar { min-height: 6px; }
    .jarvis-md-quote { border-left: 3px solid alpha(@accent_bg_color, 0.7);
                        padding-left: 8px; color: alpha(@window_fg_color, 0.75); }
    .jarvis-md-code { background-color: #0a0c10; border-radius: 6px; }
    .jarvis-md-table { border: 1px solid alpha(@window_fg_color, 0.15);
                       border-radius: 8px; }
    .jarvis-md-th { font-weight: bold; }
    .jarvis-md-hr { margin-top: 6px; margin-bottom: 6px; }
    .jarvis-md-img { border-radius: 8px; }
    .jarvis-md-task-done { color: alpha(@window_fg_color, 0.5); }
    """
    )
    display = Gdk.Display.get_default()
    if display is not None:
        Gtk.StyleContext.add_provider_for_display(
            display, css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


def _open_uri(uri: str) -> None:
    """Ouvre un lien dans le navigateur (best-effort, jamais bloquant)."""
    try:
        Gio.AppInfo.launch_default_for_uri(uri, None)
    except Exception:  # noqa: BLE001
        pass


def _label(markup: str, raw: str, selectable: bool = True) -> Gtk.Label:
    """Gtk.Label en markup Pango, avec filet de sécurité (repli texte brut)."""
    lbl = Gtk.Label(xalign=0, wrap=True, selectable=selectable,
                    use_markup=True, wrap_mode=Pango.WrapMode.WORD_CHAR)
    _set_markup_safe(lbl, markup, raw)
    return lbl


def _set_markup_safe(lbl: Gtk.Label, markup: str, raw: str) -> None:
    """set_markup avec filet : si Pango refuse le markup, texte brut (sinon
    set_markup ne fait que logger un warning Gtk et garde l'ancien contenu)."""
    markup = sanitize_pango(markup)
    try:
        ok = Pango.parse_markup(markup, -1, "\0")[0]
    except Exception:  # noqa: BLE001
        ok = False
    if ok:
        lbl.set_markup(markup)
        lbl.connect("activate-link", lambda _l, uri: (_open_uri(uri), True)[1])
    else:
        lbl.set_text(raw)


def _load_image_async(url: str, pic: Gtk.Picture, fallback: Gtk.Label) -> None:
    """Télécharge une image en thread (jamais le thread GTK) puis l'affiche."""
    import threading

    def _job() -> None:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "jarvis/0.1"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read(8 * 1024 * 1024)  # max 8 Mo
                ctype = r.headers.get("Content-Type", "")
            _IMAGE_CACHE[url] = (data, ctype)
            GLib.idle_add(_show, data, ctype)
        except Exception:  # noqa: BLE001 — réseau coupé, 404, etc.
            GLib.idle_add(_show_fail)

    def _show(data: bytes, ctype: str) -> bool:
        try:
            from gi.repository import GdkPixbuf
            loader = GdkPixbuf.PixbufLoader()
            loader.write(data)
            loader.close()
            pixbuf = loader.get_pixbuf()
            if pixbuf:
                # limite la largeur à ~480 px
                w = pixbuf.get_width()
                if w > 480:
                    h = round(pixbuf.get_height() * 480 / w)
                    pixbuf = pixbuf.scale_simple(480, h, GdkPixbuf.InterpType.BILINEAR)
                pic.set_paintable(Gdk.Texture.new_for_pixbuf(pixbuf))
                pic.set_visible(True)
                fallback.set_visible(False)
        except Exception:  # noqa: BLE001
            GLib.idle_add(_show_fail)
        return GLib.SOURCE_REMOVE

    def _show_fail() -> bool:
        pic.set_visible(False)
        fallback.set_visible(True)
        return GLib.SOURCE_REMOVE

    if url in _IMAGE_CACHE:
        data, ctype = _IMAGE_CACHE[url]
        GLib.idle_add(_show, data, ctype)
    else:
        pic.set_visible(True)
        fallback.set_visible(False)
        threading.Thread(target=_job, daemon=True).start()


def _render_block(parent: Gtk.Box, b, link_defs: dict, footnotes: dict,
                  depth: int = 0) -> None:
    """Ajoute le widget GTK d'un bloc parse_blocks dans parent."""
    if b.kind == "heading":
        markup = (f'<span size="{_HEADING_SIZE[b.level]}" weight="bold" '
                  f'foreground="#4cc2ff">{inline_to_pango(b.text, link_defs, footnotes)}</span>')
        parent.append(_label(markup, b.text))
    elif b.kind == "paragraph":
        markup = inline_to_pango(b.text, link_defs, footnotes)
        parent.append(_label(markup, b.text))
    elif b.kind == "code":
        # label markup : mono + petit, sur fond sombre posé par le CSS du Box
        markup = f'<span font-family="monospace" size="small">{escape(b.text)}</span>'
        lbl = _label(markup, b.text)
        lbl.set_hexpand(False)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.add_css_class("jarvis-md-code")
        box.set_margin_top(4)
        box.set_margin_bottom(4)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.append(lbl)
        parent.append(box)
    elif b.kind == "list":
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.set_margin_start(12 * depth)
        for idx, item in enumerate(b.items):
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            if item and item[0].kind == "task":
                mark = "☑" if item[0].checked else "☐"
                m = Gtk.Label(label=mark, xalign=0)
                if item[0].checked:
                    m.add_css_class("jarvis-md-task-done")
                row.append(m)
                content = item[0].text
                row.append(_label(inline_to_pango(content, link_defs, footnotes), content))
            else:
                bullet = f"{b.start + idx}." if b.ordered else "•"
                row.append(Gtk.Label(label=bullet, xalign=0))
                inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
                for sub in item:
                    _render_block(inner, sub, link_defs, footnotes, depth + 1)
                row.append(inner)
            box.append(row)
        parent.append(box)
    elif b.kind == "quote":
        qbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        qbox.add_css_class("jarvis-md-quote")
        qbox.set_margin_start(6)
        qbox.set_margin_top(2)
        qbox.set_margin_bottom(2)
        for sub in b.items:
            _render_block(qbox, sub, link_defs, footnotes, depth)
        parent.append(qbox)
    elif b.kind == "table":
        grid = Gtk.Grid()
        grid.add_css_class("jarvis-md-table")
        grid.set_margin_top(4)
        grid.set_margin_bottom(4)
        grid.set_column_spacing(12)
        grid.set_row_spacing(2)
        aligns = b.aligns or [None] * max(len(r) for r in b.rows)
        for r, row in enumerate(b.rows):
            for c, cell in enumerate(row):
                markup = inline_to_pango(cell, link_defs, footnotes)
                if r == 0:
                    markup = f"<b>{markup}</b>"
                lbl = _label(markup, cell)
                a = aligns[c] if c < len(aligns) else None
                lbl.set_halign({"l": Gtk.Align.START, "c": Gtk.Align.CENTER,
                                "r": Gtk.Align.END}.get(a or "l", Gtk.Align.START))
                if r == 0:
                    lbl.add_css_class("jarvis-md-th")
                grid.attach(lbl, c, r, 1, 1)
        parent.append(grid)
    elif b.kind == "hr":
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sep.add_css_class("jarvis-md-hr")
        parent.append(sep)
    elif b.kind == "image":
        if _safe_href(b.url) is None:
            # schéma non autorisé (javascript:, data:, file:...) -> juste le texte
            parent.append(_label(escape(b.text or b.url), b.text or b.url))
            return
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        pic = Gtk.Picture()
        pic.add_css_class("jarvis-md-img")
        pic.set_content_fit(Gtk.ContentFit.CONTAIN)
        pic.set_size_request(-1, 120)
        pic.set_visible(False)
        alt = _label(f'<a href="{escape(b.url)}">{escape(b.text or b.url)}</a>', b.text or b.url)
        alt.set_visible(False)
        box.append(pic)
        box.append(alt)
        parent.append(box)
        _load_image_async(b.url, pic, alt)
    elif b.kind == "footnote_def":
        markup = (f'<span size="small">[{b.footnote_id}] '
                  f'{inline_to_pango(b.text, link_defs, footnotes)}</span>')
        parent.append(_label(markup, f"[{b.footnote_id}] {b.text}"))


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

        # contenu markdown : un Box rempli de widgets par set_markdown/append_text
        # (text_label conservé pour compatibilité : 1er label du content_box)
        self.content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        if role == "user":
            self.content_box.add_css_class("jarvis-text")
        bubble.append(self.content_box)
        self._md_buffer = ""

        self.tools_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        bubble.append(self.tools_box)

        self.meta_label = Gtk.Label(xalign=0, wrap=True)
        self.meta_label.add_css_class("jarvis-meta")
        self.append(self.meta_label)

    @property
    def text_label(self) -> Gtk.Label:
        """Premier label du contenu (compat)."""
        child = self.content_box.get_first_child()
        return child if isinstance(child, Gtk.Label) else Gtk.Label()

    # ---------- contenu ----------

    def _clear_content(self) -> None:
        while (child := self.content_box.get_first_child()) is not None:
            self.content_box.remove(child)

    def set_text(self, text: str) -> None:
        """Texte brut (rôles user/system) : un seul label échappé."""
        self._md_buffer = text
        self._clear_content()
        if text:
            self.content_box.append(_label(escape(text), text))

    def set_markdown(self, text: str) -> None:
        """Texte markdown (réponse assistant) rendu en widgets GTK natifs."""
        self._md_buffer = text
        self._render_md(text)

    def append_text(self, fragment: str) -> None:
        """Ajoute un fragment streamé et re-rend tout le Markdown : les
        marqueurs/blocs qui se ferment en cours de route sont pris en compte
        au fragment suivant."""
        self._md_buffer += fragment
        self._render_md(self._md_buffer)

    def _render_md(self, text: str) -> None:
        self._clear_content()
        link_defs, footnotes = collect_defs(text)
        for b in parse_blocks(text):
            _render_block(self.content_box, b, link_defs, footnotes)
        if not self.content_box.get_first_child():
            self.content_box.append(_label("", ""))

    # ---------- reasoning / outils / meta ----------

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
            line = f"⚡ {name}({args_txt})"
        else:
            mark = "✓" if ok else "✗"
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
    child = box.content_box.get_first_child()
    if child is not None:
        child.add_css_class("jarvis-meta")
    return box


def error_message(text: str) -> MessageBox:
    box = MessageBox("system")
    box.set_halign(Gtk.Align.CENTER)
    box.set_text(text)
    child = box.content_box.get_first_child()
    if child is not None:
        child.add_css_class("jarvis-error")
    return box
