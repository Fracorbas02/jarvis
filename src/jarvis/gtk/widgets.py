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

import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402
from xml.sax.saxutils import escape  # noqa: E402

from .markdown import (_safe_href, collect_defs, highlight_code, inline_to_pango,
                       parse_blocks, render_math_png, sanitize_pango,
                       to_pango)  # noqa: E402

# WebKitGTK optionnel : si présent, le markdown est rendu en HTML complet
# (markdown-it-py -> HTML -> WebView). Sinon, repli sur les widgets natifs.
def _webkit_available() -> bool:
    try:
        import gi as _gi
        for ver in ("6.0", "4.1"):
            try:
                _gi.require_version("WebKit", ver)
                from gi.repository import WebKit  # noqa: F401
                return True
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return False


# Le rendu WebKitGTK est opt-in (JARVIS_WEBKIT=1) : par défaut on rend en widgets
# natifs, stables. Le chemin WebKit reste là, activable pour tester.
import os as _os
WEBKIT_AVAILABLE = _os.environ.get("JARVIS_WEBKIT", "") == "1" and _webkit_available()

if WEBKIT_AVAILABLE:
    from gi.repository import WebKit  # noqa: E402
    from .markdown_lib import markdown_to_html  # noqa: E402

_HEADING_SIZE = {1: "xx-large", 2: "x-large", 3: "large", 4: "medium",
                 5: "small", 6: "small"}

# cache mémoire des images téléchargées : url -> (bytes, content_type), borné
_IMAGE_CACHE: dict[str, tuple[bytes, str]] = {}
_IMAGE_CACHE_MAX = 50  # éviction FIFO au-delà (pas de croissance infinie)


def _css() -> None:
    css = Gtk.CssProvider()
    css.load_from_string("""
    .jarvis-msg-user .jarvis-bubble { background-color: @accent_bg_color; }
    .jarvis-msg-user .jarvis-text { color: @accent_fg_color; }
    .jarvis-msg-assistant .jarvis-bubble { background-color: alpha(@window_fg_color, 0.06); }
    .jarvis-reasoning { color: alpha(@window_fg_color, 0.55); font-style: italic; }
    .jarvis-tool { color: @purple_3; }
    .jarvis-tool-result { font-family: monospace; font-size: 0.85em;
                          background-color: alpha(@window_fg_color, 0.05);
                          border-radius: 6px; padding: 6px 8px;
                          margin-top: 2px; margin-bottom: 4px; }
    .jarvis-error { color: @error_color; }
    .jarvis-msg-user .jarvis-bubble { border-radius: 14px 14px 4px 14px; }
    .jarvis-msg-assistant .jarvis-bubble { border-radius: 14px 14px 14px 4px; }
    .jarvis-bubble { border-radius: 14px; }
    .jarvis-meta { font-size: 0.8em; color: alpha(@window_fg_color, 0.45); }
    .jarvis-home-title { font-size: 44px; font-weight: 700; }
    .jarvis-notif-badge {
        background-color: #e01b24;
        color: #ffffff;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 700;
        padding: 1px 6px;
        min-width: 18px;
        min-height: 18px;
    }
    .jarvis-ctx-bar { min-height: 6px; }
    .jarvis-msg-user .jarvis-bubble { background-color: @accent_bg_color; }
    .jarvis-msg-assistant .jarvis-bubble {
        background-color: alpha(@window_fg_color, 0.07);
        border: 1px solid alpha(@window_fg_color, 0.08);
    }
    .jarvis-bubble { border-radius: 16px; }
    .jarvis-msg-user .jarvis-bubble { border-radius: 16px 16px 4px 16px; }
    .jarvis-msg-assistant .jarvis-bubble { border-radius: 16px 16px 16px 4px; }
    .jarvis-md-quote { border-left: 3px solid @accent_bg_color;
                        padding-left: 10px; color: alpha(@window_fg_color, 0.7); }
    .jarvis-md-code-card { background-color: #0d1117; border-radius: 10px;
                           border: 1px solid alpha(@window_fg_color, 0.1); }
    .jarvis-md-code-lang { font-size: 0.75em; color: alpha(@window_fg_color, 0.5);
                           text-transform: uppercase; font-weight: 600; }
    .jarvis-md-code-copy { border-radius: 6px; }
    /* tableau : bordure sur le conteneur, cellules en right/bottom seulement
       (évite les bordures doublées entre cellules, GTK n'a pas border-collapse) */
    .jarvis-md-table {
        border: 1px solid alpha(@window_fg_color, 0.25);
        border-radius: 10px;
    }
    .jarvis-md-th {
        font-weight: 700; color: @accent_color;
        background-color: alpha(@accent_bg_color, 0.25);
        border-bottom: 2px solid alpha(@accent_bg_color, 0.8);
        border-right: 1px solid alpha(@window_fg_color, 0.15);
        padding: 4px 10px;
    }
    .jarvis-md-td {
        border-bottom: 1px solid alpha(@window_fg_color, 0.12);
        border-right: 1px solid alpha(@window_fg_color, 0.12);
        padding: 3px 10px;
    }
    .jarvis-tool-arrow { color: alpha(@window_fg_color, 0.5); }
    .jarvis-flat-list, .jarvis-flat-list row { background-color: transparent; }
    /* sélection souris dans les bulles : les labels ont set_selectable(True) ;
       en GTK4 il n'existe pas de propriété CSS "user-select" (propriété GTK3,
       ignorée avec une erreur "Theme parser error" au chargement du CSS) */
    .jarvis-msg-copy { opacity: 0.55; }
    .jarvis-msg-copy:hover { opacity: 1; }
    .jarvis-md-fn-flash { background-color: alpha(@accent_bg_color, 0.35);
                          border-radius: 6px; }
    .jarvis-md-hr { margin-top: 8px; margin-bottom: 8px;
                    color: alpha(@window_fg_color, 0.25); }
    .jarvis-md-img { border-radius: 10px; }
    .jarvis-md-task-done { color: alpha(@window_fg_color, 0.5); }
    .jarvis-md-math { background-color: alpha(@accent_bg_color, 0.12);
                      border-radius: 8px; padding: 8px; }
    .jarvis-md-fn { border-top: 1px solid alpha(@window_fg_color, 0.12);
                    margin-top: 8px; padding-top: 4px; }
    .jarvis-entry-modern { border-radius: 20px; padding: 6px 12px;
                           background-color: alpha(@window_fg_color, 0.06); }
    .jarvis-entry-bar { background-color: alpha(@window_fg_color, 0.05);
                        border: 1px solid alpha(@window_fg_color, 0.1);
                        border-radius: 24px; padding: 6px 8px; }
    .jarvis-entry-modern textview, .jarvis-entry-modern {
        background: transparent; border: none;
    }
    .jarvis-entry-modern text { caret-color: @window_fg_color; }
    """
    )
    display = Gdk.Display.get_default()
    if display is not None:
        Gtk.StyleContext.add_provider_for_display(
            display, css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


def _open_uri(uri: str) -> None:
    """Ouvre un lien externe (navigateur). Best-effort, jamais bloquant."""
    import subprocess
    try:
        Gio.AppInfo.launch_default_for_uri(uri, None)
        return
    except Exception:  # noqa: BLE001
        pass
    for cmd in (["xdg-open", uri], ["gio", "open", uri]):
        try:
            subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:  # noqa: BLE001
            continue


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
        def _on_link(lbl, uri):
            if uri.startswith("#"):
                return False  # ancre interne : laisser GTK gérer (scroll)
            _open_uri(uri)
            return True
        lbl.connect("activate-link", _on_link)
    else:
        lbl.set_text(raw)


def _load_image_async(url: str, pic: Gtk.Picture, fallback: Gtk.Label) -> None:
    """Télécharge une image en thread (jamais le thread GTK) puis l'affiche."""
    import threading

    def _job() -> None:
        try:
            from ..netguard import check_url_ssrf
            # même garde SSRF que les MCP fetch/rss : une URL d'image vient du
            # LLM (réponse markdown) et ne doit pas pouvoir viser 169.254.169.254
            # ou une IP privée/loopback
            err = check_url_ssrf(url)
            if err:
                GLib.idle_add(_show_fail)
                return
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "jarvis/0.1"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read(8 * 1024 * 1024)  # max 8 Mo
                ctype = r.headers.get("Content-Type", "")
            if len(_IMAGE_CACHE) >= _IMAGE_CACHE_MAX:
                _IMAGE_CACHE.pop(next(iter(_IMAGE_CACHE)))  # éviction FIFO
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


def _code_card(code: str, lang: str) -> Gtk.Widget:
    """Carte de bloc de code moderne : barre en-tête (langue + bouton copier)
    + code colorisé (highlight_code) dans un label mono sélectionnable."""
    card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
    card.add_css_class("jarvis-md-code-card")
    card.set_margin_top(6)
    card.set_margin_bottom(6)

    # barre d'en-tête : langue à gauche, copier à droite
    header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    header.set_margin_start(10)
    header.set_margin_end(6)
    header.set_margin_top(6)
    header.set_margin_bottom(4)
    lang_lbl = Gtk.Label(label=(lang or "text").upper(), xalign=0, hexpand=True)
    lang_lbl.add_css_class("jarvis-md-code-lang")
    header.append(lang_lbl)

    copy_btn = Gtk.Button(icon_name="edit-copy-symbolic",
                          tooltip_text="Copier le code")
    copy_btn.add_css_class("jarvis-md-code-copy")
    copy_btn.add_css_class("flat")

    def _copy(_b) -> None:
        ok = False
        try:
            cb = copy_btn.get_clipboard()  # clipboard associé au widget (display)
            cb.set_text(code)
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            # fallback CLI (Wayland sans clipboard GTK, etc.)
            import subprocess
            for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"],
                        ["xsel", "--clipboard", "--input"]):
                try:
                    subprocess.run(cmd, input=code.encode(), timeout=3, check=True)
                    ok = True
                    break
                except Exception:  # noqa: BLE001
                    continue
        copy_btn.set_icon_name("object-select-symbolic" if ok else "dialog-error-symbolic")
        GLib.timeout_add(1500, lambda: (copy_btn.set_icon_name("edit-copy-symbolic"),
                                        GLib.SOURCE_REMOVE)[1])

    copy_btn.connect("clicked", _copy)
    header.append(copy_btn)
    card.append(header)

    # code colorisé
    hl = highlight_code(code, lang)
    markup = f'<span font-family="monospace" size="small">{hl}</span>'
    lbl = _label(markup, code)
    lbl.set_hexpand(False)
    lbl.set_margin_start(10)
    lbl.set_margin_end(10)
    lbl.set_margin_bottom(8)
    lbl.set_selectable(True)
    lbl.set_wrap(False)          # jamais de retour à la ligne : le code garde ses lignes
    lbl.set_wrap_mode(Pango.WrapMode.NONE)
    lbl.set_ellipsize(Pango.EllipsizeMode.NONE)
    lbl.set_single_line_mode(False)
    lbl.set_max_width_chars(-1)
    # scroll horizontal si une ligne dépasse la largeur de la bulle
    scroll = Gtk.ScrolledWindow(hexpand=True)
    scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
    scroll.set_child(lbl)
    card.append(scroll)
    return card


def _math_widget(tex: str, block: bool = False) -> Gtk.Widget:
    """Équation rendue en image PNG (matplotlib.mathtext) dans un Gtk.Picture,
   chargée en thread ; fallback : label italique orange si échec."""
    import threading

    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.set_halign(Gtk.Align.CENTER if block else Gtk.Align.START)
    if block:
        box.add_css_class("jarvis-md-math")
        box.set_margin_top(6)
        box.set_margin_bottom(6)

    png = render_math_png(tex, fontsize=16.0 if block else 13.0)

    if png is not None:
        pic = Gtk.Picture()
        pic.set_content_fit(Gtk.ContentFit.SCALE_DOWN)
        pic.set_can_shrink(True)
        box.append(pic)

        def _job() -> None:
            # le rendu est déjà fait (cache) ; on décode juste le PNG
            GLib.idle_add(_show, png)

        def _show(data: bytes) -> bool:
            try:
                from gi.repository import GdkPixbuf
                loader = GdkPixbuf.PixbufLoader()
                loader.write(data)
                loader.close()
                pixbuf = loader.get_pixbuf()
                if pixbuf is not None:
                    if block and pixbuf.get_width() > 560:
                        h = round(pixbuf.get_height() * 560 / pixbuf.get_width())
                        pixbuf = pixbuf.scale_simple(560, h, GdkPixbuf.InterpType.BILINEAR)
                    pic.set_paintable(Gdk.Texture.new_for_pixbuf(pixbuf))
            except Exception:  # noqa: BLE001
                box.remove(pic)
                box.append(_label(
                    f'<span style="italic" foreground="#ffb454">'
                    f"{escape(tex)}</span>", tex))
            return GLib.SOURCE_REMOVE

        threading.Thread(target=_job, daemon=True).start()
    else:
        size = ' size="large"' if block else ""
        box.append(_label(
            f'<span style="italic" foreground="#ffb454"{size}>{escape(tex)}</span>',
            tex))
    return box


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
        parent.append(_code_card(b.text, b.lang))
    elif b.kind == "math":
        parent.append(_math_widget(b.text, block=True))
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
        grid.set_margin_top(6)
        grid.set_margin_bottom(6)
        grid.set_column_spacing(0)
        grid.set_row_spacing(0)
        grid.set_hexpand(True)
        ncols = max((len(r) for r in b.rows), default=0)
        aligns = b.aligns or [None] * ncols
        for r, row in enumerate(b.rows):
            for c, cell in enumerate(row):
                markup = inline_to_pango(cell, link_defs, footnotes)
                lbl = _label(markup, cell)
                a = aligns[c] if c < len(aligns) else None
                # alignement du TEXTE dans la cellule (xalign) — halign sur le
                # label serait sans effet puisque hexpand remplit toute la colonne
                lbl.set_xalign({"l": 0.0, "c": 0.5, "r": 1.0}.get(a or "l", 0.0))
                lbl.set_hexpand(True)  # la cellule remplit sa colonne, le texte wrappe
                lbl.add_css_class("jarvis-md-th" if r == 0 else "jarvis-md-td")
                grid.attach(lbl, c, r, 1, 1)
        # jamais de scrollbar horizontale : le tableau tient dans la bulle,
        # les cellules longues wrappent (largeur mini d'une colonne = plus long mot)
        scroll = Gtk.ScrolledWindow(hexpand=True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
        scroll.set_child(grid)
        parent.append(scroll)
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
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.add_css_class("jarvis-md-fn")
        # ancre nommée pour le scroll depuis l'exposant [^id]
        box.set_name(f"fn-{b.footnote_id}")
        box._jarvis_fn_id = b.footnote_id  # pour le flash au clic
        box.append(_label(markup, f"[{b.footnote_id}] {b.text}"))
        parent.append(box)


class MessageBox(Gtk.Box):
    """Une bulle de conversation : rôle (user/assistant/system), contenu,
    reasoning repliable, ligne d'outils, métadonnées."""

    def __init__(self, role: str):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.role = role
        self.set_hexpand(True)  # sinon halign START/END n'a aucun effet (box pleine largeur)
        self.add_css_class(f"jarvis-msg-{role}")
        self.set_margin_start(24)
        self.set_margin_end(24)
        self.set_margin_top(6)
        self.set_margin_bottom(6)

        bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        bubble.add_css_class("jarvis-bubble")
        bubble.set_margin_start(12)
        bubble.set_margin_end(12)
        bubble.set_size_request(320, -1)  # largeur mini : la webview a besoin d'une vraie largeur
        bubble.set_margin_top(8)
        bubble.set_margin_bottom(8)
        self.append(bubble)
        self._bubble = bubble

        self.reasoning_row: Adw.ExpanderRow | None = None
        self.reasoning_box: Gtk.Box | None = None
        self._reasoning_buf = ""
        if role == "assistant":
            self.reasoning_row = Adw.ExpanderRow(title="Raisonnement")
            self.reasoning_row.add_css_class("jarvis-reasoning")
            # le raisonnement est rendu en Markdown comme la réponse (content_box dédié)
            self.reasoning_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            self.reasoning_box.add_css_class("jarvis-reasoning")
            self.reasoning_box.set_margin_start(8)
            self.reasoning_box.set_margin_end(8)
            self.reasoning_box.set_margin_bottom(6)
            self.reasoning_row.add_row(self.reasoning_box)
            self.reasoning_row.set_visible(False)
            # Adw.ExpanderRow ne peut être placé que dans un Gtk.ListBox
            # (dans un Gtk.Box, GTK logue "can only be placed into GtkListBox")
            reasoning_list = Gtk.ListBox()
            reasoning_list.set_selection_mode(Gtk.SelectionMode.NONE)
            reasoning_list.add_css_class("jarvis-flat-list")
            reasoning_list.append(self.reasoning_row)
            bubble.append(reasoning_list)

        # contenu markdown : WebView WebKit si dispo (HTML complet), sinon un
        # Box rempli de widgets natifs (repli)
        self.content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        if role == "user":
            self.content_box.add_css_class("jarvis-text")
        # la bulle interne suit le halign de la MessageBox (START/END) et ne fait
        # pas toute la largeur — hexpand False pour que
        # le wrap du texte/webview se fasse à la largeur de la bulle, pas de la fenêtre
        bubble.set_hexpand(False)
        bubble.append(self.content_box)
        self._md_buffer = ""
        self._last_render = 0
        self._render_timer = None
        self.webview = None
        # lignes d'outils intercalées dans le flux : (offset dans _md_buffer au
        # moment de l'appel, widget entry) — triées à chaque re-rendu
        self._tool_entries: list[tuple[int, Gtk.Box]] = []
        self._stats: dict = {}
        if role == "assistant" and WEBKIT_AVAILABLE:
            self.webview = self._make_webview()
            # visible dès la création (hauteur mini 40px) : _render_md la rend
            # visible au 1er fragment de toute façon ; la cacher ici faisait
            # disparaître la bulle assistant si le 1er rendu était vide/échouait
            self.webview.set_visible(True)
            bubble.append(self.webview)

        self.tools_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        bubble.append(self.tools_box)

        # boutons d'action (bulles assistant uniquement) : infos (tokens/tok/s)
        # + copier le texte brut markdown
        if role == "assistant":
            actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
            actions.set_halign(Gtk.Align.END)
            self.info_btn = Gtk.Button(icon_name="info-symbolic",
                                       tooltip_text="Infos de la réponse (tokens, tok/s, durée)",
                                       valign=Gtk.Align.CENTER)
            self.info_btn.add_css_class("flat")
            self.info_btn.add_css_class("jarvis-msg-copy")
            self.info_btn.set_visible(False)  # affiché quand des stats sont connues
            self._info_pop = Gtk.Popover()
            self._info_lbl = Gtk.Label(xalign=0, wrap=True, selectable=True)
            self._info_lbl.set_max_width_chars(52)
            self._info_lbl.set_margin_top(10)
            self._info_lbl.set_margin_bottom(10)
            self._info_lbl.set_margin_start(12)
            self._info_lbl.set_margin_end(12)
            self._info_pop.set_child(self._info_lbl)
            self._info_pop.set_parent(self.info_btn)
            self.info_btn.connect("clicked", lambda b: self._info_pop.popup())
            copy_btn = Gtk.Button(icon_name="edit-copy-symbolic",
                                  tooltip_text="Copier la réponse",
                                  valign=Gtk.Align.CENTER)
            copy_btn.add_css_class("flat")
            copy_btn.add_css_class("jarvis-msg-copy")
            copy_btn.connect("clicked", lambda b: self.copy_content())
            actions.append(self.info_btn)
            actions.append(copy_btn)
            bubble.append(actions)

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
        if self.webview is not None:
            self.webview.set_visible(False)
        self.content_box.set_visible(True)
        self._clear_content()
        if text:
            self.content_box.append(_label(escape(text), text))
        # un texte vide ne doit pas laisser la bulle "fantôme"
        self.content_box.set_visible(bool(text))

    def set_markdown(self, text: str) -> None:
        """Texte markdown (réponse assistant) rendu en widgets GTK natifs."""
        self._md_buffer = text
        self._render_md(text)

    _RENDER_MIN_MS = 66  # ~15 Hz, comme la TUI (RENDER_THROTTLE)

    def append_text(self, fragment: str) -> None:
        """Ajoute un fragment streamé. Le re-rendu complet est throttlé à ~15 Hz :
        on bufferise et on ne re-rend que si le délai est écoulé (ou en fin de tour
        via flush). Les marqueurs/blocs non fermés sont repris au fragment suivant."""
        self._md_buffer += fragment
        now = GLib.get_monotonic_time()
        if self._last_render and now - self._last_render < self._RENDER_MIN_MS * 1000:
            if self._render_timer is None:
                self._render_timer = GLib.timeout_add(
                    self._RENDER_MIN_MS,
                    lambda: (self._flush_render(), GLib.SOURCE_REMOVE)[1])
            return
        self._flush_render()

    def _flush_render(self) -> None:
        self._render_timer = None
        self._last_render = GLib.get_monotonic_time()
        self._render_md(self._md_buffer)


    def flush_render(self) -> None:
        """Force le re-rendu immédiat (fin de tour)."""
        if self._render_timer is not None:
            GLib.source_remove(self._render_timer)
            self._render_timer = None
        self._flush_render()

    def _render_md(self, text: str) -> None:
        if self.webview is not None:
            try:
                self._render_webview(text)
                return
            except Exception:  # noqa: BLE001 — WebKit KO (version, sandbox...) : repli natif
                import traceback
                print("[jarvis] webview KO, repli rendu natif:", traceback.format_exc()[:300],
                      file=sys.stderr)
                self.webview = None  # ne plus réessayer
        self._render_native(text)

    def _render_webview(self, text: str) -> None:
        # rendu HTML complet via markdown-it-py + WebKit.
        # Toujours visible (hauteur mini 40px) même si le texte est vide :
        # en streaming, les 1ers fragments peuvent être vides et la bulle
        # ne doit pas disparaître (0px) avant le 1er contenu.
        self.content_box.set_visible(False)
        self.webview.set_visible(True)
        # hauteur auto = contenu HTML ; on ne touche qu'à la hauteur (la largeur
        # mini 300px est posée à la création et ne doit pas être reset à -1)
        self.webview.set_size_request(300, -1)
        self.webview.load_html(markdown_to_html(text or " "), "about:blank")
        # WebKit dimensionne après le chargement (parfois 2 cycles) : on force
        # deux re-layouts différés pour que la hauteur suive le contenu
        GLib.idle_add(lambda: (self.queue_resize(), GLib.SOURCE_REMOVE)[1])
        GLib.timeout_add(120, lambda: (self.queue_resize(), GLib.SOURCE_REMOVE)[1])

    def _render_native(self, text: str) -> None:
        self._clear_content()
        if self.webview is not None:
            self.webview.set_visible(False)
        self.content_box.set_visible(True)
        link_defs, footnotes = collect_defs(text)
        # Intercale chaque ligne d'outil à la position du flux où le LLM l'a
        # appelée : le buffer markdown est découpé aux offsets enregistrés
        pos = 0
        for off, entry in sorted(self._tool_entries, key=lambda t: t[0]):
            seg = text[pos:off]
            if seg:
                for b in parse_blocks(seg):
                    _render_block(self.content_box, b, link_defs, footnotes)
            self.content_box.append(entry)  # le widget est re-parenté à chaque rendu
            pos = off
        if text[pos:]:
            for b in parse_blocks(text[pos:]):
                _render_block(self.content_box, b, link_defs, footnotes)
        if not self.content_box.get_first_child():
            self.content_box.append(_label("", ""))
        self._connect_footnote_anchors()

    def _make_webview(self):
        """Crée le WebKit.WebView : charge le HTML markdown, intercepte les
        liens externes (script-message -> xdg-open)."""
        wv = WebKit.WebView()
        wv.set_vexpand(False)         # hauteur = contenu HTML (pas tout l'écran)
        wv.set_hexpand(True)          # prend la largeur de la bulle (min 320px)
        wv.set_halign(Gtk.Align.FILL)
        wv.set_size_request(300, 60)  # viewport WebKit non nul (largeur/hauteur mini)
        settings = wv.get_settings()
        try:
            settings.set_enable_javascript(True)
            settings.set_enable_write_console_messages_to_stdout(False)
        except Exception:  # noqa: BLE001
            pass
        # contexte utilisateur pour les script-messages (liens externes)
        try:
            ctx = wv.get_user_content_manager()
            ctx.register_script_message_handler("jarvis_link")
            ctx.connect("script-message-received::jarvis_link",
                        lambda _c, msg: _open_uri(msg.get_js_value().to_string()))
        except Exception:  # noqa: BLE001
            pass
        return wv

    def _connect_footnote_anchors(self) -> None:
        """Clic sur un exposant [^id] : scroll vers la définition fn-<id> dans la
        bulle et flash visuel (au lieu d'ouvrir une URI externe)."""
        def walk(w):
            if isinstance(w, Gtk.Label):
                def on_link(lbl, uri):
                    if uri.startswith("#fn-"):
                        fid = uri[4:]
                        target = self._find_footnote_box(fid)
                        if target is not None:
                            target.grab_focus()
                            target.add_css_class("jarvis-md-fn-flash")
                            GLib.timeout_add(1200, lambda: (
                                target.remove_css_class("jarvis-md-fn-flash"),
                                GLib.SOURCE_REMOVE)[1])
                            # scroll dans la colonne de messages
                            anc = self.get_ancestor(MessagesColumn)
                            if anc is not None:
                                GLib.idle_add(self._scroll_to_widget, anc, target)
                        return True  # on a géré
                    return False
                lbl.connect("activate-link", on_link)
            for c in w.get_children() if hasattr(w, "get_children") else []:
                walk(c)
        walk(self.content_box)

    def _find_footnote_box(self, fid: str):
        def walk(w):
            if getattr(w, "_jarvis_fn_id", None) == fid:
                return w
            for c in w.get_children() if hasattr(w, "get_children") else []:
                r = walk(c)
                if r is not None:
                    return r
            return None
        return walk(self.content_box)

    @staticmethod
    def _scroll_to_widget(column: "MessagesColumn", target: Gtk.Widget) -> bool:
        # scroll la ScrolledWindow de la colonne pour amener target en vue
        scroll = column.widget
        vadj = scroll.get_vadjustment()
        y = target.get_allocation().y if target.get_allocated_height() else 0
        # position relative approximative via translate_coordinates
        try:
            _ok, ty = target.translate_coordinates(column, 0, 0)
            if _ok:
                y = ty
        except Exception:  # noqa: BLE001
            pass
        vadj.set_value(max(0.0, min(y - 40, vadj.get_upper())))
        return GLib.SOURCE_REMOVE

    # ---------- reasoning / outils / meta ----------

    def append_reasoning(self, fragment: str) -> None:
        if self.reasoning_row is None:
            return
        self.reasoning_row.set_visible(True)
        self._reasoning_buf += fragment
        # re-rendu Markdown du raisonnement (comme la réponse), throttlé comme append_text
        now = GLib.get_monotonic_time()
        if self._last_render and now - self._last_render < self._RENDER_MIN_MS * 1000:
            if self._render_timer is None:
                self._render_timer = GLib.timeout_add(
                    self._RENDER_MIN_MS,
                    lambda: (self._flush_reasoning(), GLib.SOURCE_REMOVE)[1])
            return
        self._flush_reasoning()

    def _flush_reasoning(self) -> None:
        self._render_timer = None
        self._last_render = GLib.get_monotonic_time()
        assert self.reasoning_box is not None
        while (c := self.reasoning_box.get_first_child()) is not None:
            self.reasoning_box.remove(c)
        link_defs, footnotes = collect_defs(self._reasoning_buf)
        for b in parse_blocks(self._reasoning_buf):
            _render_block(self.reasoning_box, b, link_defs, footnotes)

    def flush_reasoning(self) -> None:
        if self._render_timer is not None:
            GLib.source_remove(self._render_timer)
            self._render_timer = None
        self._flush_reasoning()

    def collapse_reasoning(self, summary: str) -> None:
        if self.reasoning_row is None:
            return
        self.reasoning_row.set_title(summary or "Raisonnement")
        self.reasoning_row.set_expanded(False)

    def copy_content(self) -> None:
        """Copie le contenu brut (markdown) de la bulle dans le presse-papiers."""
        text = getattr(self, "_md_buffer", "") or ""
        try:
            cb = self.get_clipboard()
            cb.set_text(text)
        except Exception:  # noqa: BLE001 — fallback CLI
            import subprocess
            for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"],
                        ["xsel", "--clipboard", "--input"]):
                try:
                    subprocess.run(cmd, input=text.encode(), timeout=3, check=True)
                    break
                except Exception:  # noqa: BLE001
                    continue

    def add_tool_line(self, name: str, args: dict, ok: bool | None = None,
                      duration: float | None = None, result_preview: str = "") -> Gtk.Label:
        """Ligne d'outil dépliable au clic : la ligne (⚡/✓/✗ + nom + args) reste
        TOUJOURS visible ; un clic sur la ligne montre/cache le RÉSULTAT complet.
        (pas d'ExpanderRow : dans un Gtk.Box il est mal placé, et son titre vide
        cacherait la ligne d'outil tant que l'expander est replié)"""
        import json as _json
        args_txt = _json.dumps(args, ensure_ascii=False)
        if len(args_txt) > 120:
            args_txt = args_txt[:117] + "…"
        if ok is None:
            text = f"⚡ {name}({args_txt})"
        else:
            mark = "✓" if ok else "✗"
            dur = f" {duration:.2f}s" if duration is not None else ""
            text = f"{mark} {name}({args_txt}){dur}"
        entry = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        header.add_css_class("jarvis-tool")
        header.set_cursor_from_name("pointer")
        arrow = Gtk.Label(label="▸", xalign=0)
        arrow.add_css_class("jarvis-tool-arrow")
        lbl = Gtk.Label(xalign=0, wrap=True, selectable=True, hexpand=True)
        lbl.set_text(text)
        header.append(arrow)
        header.append(lbl)
        result_lbl = Gtk.Label(xalign=0, wrap=True, selectable=True)
        result_lbl.add_css_class("jarvis-tool-result")
        result_lbl.add_css_class("dim-label")
        result_lbl.set_text(result_preview or "…")
        result_lbl.set_visible(False)
        entry.append(header)
        entry.append(result_lbl)

        def _toggle(*_a) -> None:
            show = not result_lbl.get_visible()
            result_lbl.set_visible(show)
            arrow.set_text("▾" if show else "▸")

        click = Gtk.GestureClick()
        click.connect("pressed", _toggle)
        header.add_controller(click)
        entry._jarvis_result = result_lbl  # pour set_tool_result
        entry._jarvis_arrow = arrow
        entry._jarvis_name = name
        # position dans le flux = offset courant du buffer markdown : la ligne
        # reste à l'endroit où le LLM a appelé l'outil, pas en fin de bulle
        self._tool_entries.append((len(self._md_buffer), entry))
        if self.webview is not None:
            # mode WebKit : le contenu est un HTML dans la webview, on ne peut
            # pas intercaler en HTML -> les outils restent dessous la webview
            self.tools_box.append(entry)
        else:
            self._flush_render()  # re-rend tout de suite avec l'entrée à sa place
        return lbl

    def set_tool_result(self, lbl: Gtk.Label, ok: bool, duration: float,
                        result: str, args: dict) -> None:
        """Met à jour la ligne d'outil avec le statut final + le résultat complet,
        et déplie le résultat."""
        import json as _json
        args_txt = _json.dumps(args, ensure_ascii=False)
        if len(args_txt) > 120:
            args_txt = args_txt[:117] + "…"
        # retrouve l'entry parent (lbl -> header -> entry)
        entry = lbl.get_parent()
        while entry is not None and not hasattr(entry, "_jarvis_result"):
            entry = entry.get_parent()
        name = getattr(entry, "_jarvis_name", "?") if entry is not None else "?"
        mark = "✓" if ok else "✗"
        lbl.set_text(f"{mark} {name}({args_txt}) {duration:.2f}s")
        if entry is not None:
            entry._jarvis_result.set_text(result[:2000] or "(vide)")
            # déplie automatiquement pour voir le résultat
            entry._jarvis_result.set_visible(True)
            entry._jarvis_arrow.set_text("▾")

    def set_meta(self, text: str) -> None:
        self.meta_label.set_text(text)
        self.meta_label.set_visible(bool(text))

    def set_stats(self, stats: dict) -> None:
        """Stats du tour (événement done) : alimente le popover du bouton "i"."""
        self._stats = stats
        if not hasattr(self, "info_btn"):
            return
        gen = stats.get("gen_tokens")
        tps = stats.get("gen_tps")
        dur = stats.get("duration")
        prompt = stats.get("prompt_tokens")
        gen_s = stats.get("gen_seconds")
        real = stats.get("tokens_real")
        trace = stats.get("trace") or {}
        lines: list[str] = []
        if gen is not None:
            kind = "usage serveur" if real else "estimé (~4 car/tok)"
            lines.append(f"Tokens générés : {gen} ({kind})")
        if tps is not None:
            gtxt = f" — génération {gen_s:.1f} s" if gen_s else ""
            lines.append(f"Vitesse : {tps:.1f} tok/s{gtxt}")
        if prompt:
            lines.append(f"Prompt : {prompt} tokens")
        if dur is not None:
            n_tools = len(trace.get("tool_calls") or [])
            it = trace.get("iterations") or 0
            extra = f" · {n_tools} outil(s) · {it} itération(s)" if (n_tools or it) else ""
            lines.append(f"Tour total : {dur:.1f} s{extra}")
        if trace.get("error"):
            lines.append(f"Erreur : {trace['error']}")
        self._info_lbl.set_text("\n".join(lines) or "pas de stats")
        self.info_btn.set_visible(True)


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

    def scroll_to_end(self, force: bool = False) -> None:
        """Scroll en bas, SAUF si l'utilisateur est remonté (stick-to-bottom) :
        on ne scrolle que si on est déjà dans la dernière page (~150px du bas)."""
        vadj = self._scroll.get_vadjustment()
        if not force:
            at_bottom = vadj.get_value() + vadj.get_page_size() >= vadj.get_upper() - 150
            if not at_bottom:
                return  # l'utilisateur relit plus haut : on ne le ramène pas en bas
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
