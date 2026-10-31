"""Rendu Markdown pour la conversation GTK4 (Pango markup).

Pas de lib externe (le projet n'en a pas et GTK n'en embarque pas) : on parse
ici un sous-ensemble suffisant pour les réponses de LLM et on produit du
markup Pango (Gtk.Label.set_markup) :

- blocs : titres #..######, listes -/*/+ (puce •), blocs de code ``` ;
- inline : **gras**, *italique* / _italique_, ***gras italique***, `code`, ~~barré~~.

Le parseur est volontairement tolérant au streaming : un marqueur non fermé
(fréquent en plein stream, ex. "**bo" puis "ld**") est rendu tel quel, et
re-rendu au fragment suivant quand le marqueur se ferme.

API :
- parse_markdown(text) -> list[Segment] (texte brut + style) ;
- to_pango(text) -> str : markup Pango échappé (sûr pour set_markup).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class Segment:
    text: str
    style: frozenset[str] = frozenset()


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
# inline : délimiteurs ouvrants/fermants. `code` est prioritaire (pas de
# mise en forme imbriquée à l'intérieur).
_INLINE_RE = re.compile(
    r"(`[^`\n]*`)"                       # 1: `code`
    r"|((\*\*\*|___)[^\n]+?(\*\*\*|___))"  # 2: ***gras italique*** / ___...___
    r"|(\*\*[^*\n]+\*\*)"             # 3: **gras**
    r"|(__[^_\n]+__)"                    # 4: __gras__
    r"|(\*[^*\n]+\*)"                   # 5: *italique*
    r"|(_[^_\n]+_)"                      # 6: _italique_
    r"|(~~[^~\n]+~~)"                    # 7: ~~barré~~
)


def _parse_inline(text: str, base: frozenset[str] = frozenset()) -> list[Segment]:
    """Découpe une ligne en segments inline (gras/italique/code/barré)."""
    out: list[Segment] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append(Segment(text[pos:m.start()], base))
        tok = m.group(0)
        if tok.startswith("`"):
            out.append(Segment(tok[1:-1], base | {"code"}))
        elif tok.startswith("***") or tok.startswith("___"):
            out.append(Segment(tok[3:-3], base | {"bold", "italic"}))
        elif tok.startswith("**") or tok.startswith("__"):
            out.append(Segment(tok[2:-2], base | {"bold"}))
        elif tok.startswith("~~"):
            out.append(Segment(tok[2:-2], base | {"strike"}))
        else:
            out.append(Segment(tok[1:-1], base | {"italic"}))
        pos = m.end()
    if pos < len(text):
        out.append(Segment(text[pos:], base))
    return out or [Segment("", base)]


def parse_markdown(text: str) -> list[Segment]:
    """Parse un texte markdown en segments stylés (avec les sauts de ligne)."""
    segments: list[Segment] = []
    in_code = False
    # splitlines(keepends=True) conserve les "\n" sans ajouter de ligne vide
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        nl = line[len(body):]  # "\n" ou "" en fin de texte
        if body.strip().startswith("```"):
            # toute ligne commençant par ``` est un délimiteur de bloc
            # (ouvrant avec ou sans langue, ex. ```python ou ``` nu, et
            # fermant) : jamais affichée comme contenu
            in_code = not in_code
            continue
        if in_code:
            segments.append(Segment(line, frozenset({"mono", "code"})))
            continue
        m = _HEADING_RE.match(body)
        if m:
            level = len(m.group(1))
            style = frozenset({"heading", f"h{level}"})
            segments.extend(_parse_inline(m.group(2) + nl, style))
            continue
        m = _LIST_RE.match(body)
        if m:
            segments.append(Segment("• ", frozenset()))
            segments.extend(_parse_inline(m.group(1) + nl))
            continue
        segments.extend(_parse_inline(line))
    return segments


def _wrap(text: str, tag: str, **attrs: str) -> str:
    attr = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f"<{tag}{attr}>{text}</{tag}>"


def segments_to_pango(segments: list[Segment]) -> str:
    """Convertit des segments en markup Pango (texte échappé, balises sûres)."""
    out: list[str] = []
    for seg in segments:
        t = escape(seg.text)
        # Pango : \n dans un label = retour à la ligne (wrap activé)
        if "code" in seg.style and "mono" in seg.style:
            # bloc de code complet (plusieurs lignes) : fond sombre + mono
            t = _wrap(t, "span", background="#0a0c10", font_family="monospace")
        elif "code" in seg.style:
            # <tt> n'accepte aucun attribut en Pango markup -> <span>
            t = _wrap(t, "span", font_family="monospace", foreground="#d2a8ff")
        for tag, style in (("b", "bold"), ("i", "italic"), ("s", "strike")):
            if style in seg.style:
                t = _wrap(t, tag)
        for level in range(1, 7):
            if f"h{level}" in seg.style:
                size = {1: "x-large", 2: "large", 3: "medium", 4: "medium",
                        5: "small", 6: "small"}[level]
                t = _wrap(t, "span", size=size, weight="bold",
                          foreground="#4cc2ff")
                break
        out.append(t)
    return "".join(out)


def to_pango(text: str) -> str:
    """Markdown brut -> markup Pango (pour Gtk.Label.set_markup)."""
    return segments_to_pango(parse_markdown(text))

# Balises/attributs Pango autorisés en sortie de segments_to_pango ; toute
# autre balise (ex. <tt font-family=...>, que Pango refuse) est neutralisée
# par sanitize_pango().
_ALLOWED_TAGS = {
    "b": set(),
    "i": set(),
    "s": set(),
    "span": {"size", "weight", "foreground", "background", "font-family"},
}
_ATTR_RE = re.compile(r'([a-zA-Z-]+)="([^"]*)"')
_TAG_RE = re.compile(r'</?([a-zA-Z]+)((?:\s+[a-zA-Z-]+="[^"]*")*)\s*/?>')


def sanitize_pango(markup: str) -> str:
    """Garde uniquement les balises/attributs Pango supportés ; le reste est
    rendu littéral (échappé). Garantit que set_markup ne peut pas échouer."""
    def _sub(m: re.Match) -> str:
        tag = m.group(1).lower()
        attrs = m.group(2) or ""
        allowed = _ALLOWED_TAGS.get(tag)
        if allowed is None:
            return escape(m.group(0))
        closing = m.group(0).startswith("</")
        keep = [f'{a}="{v}"' for a, v in _ATTR_RE.findall(attrs) if a in allowed]
        inner = (" " + " ".join(keep)) if keep else ""
        return f"</{tag}{inner}>" if closing else f"<{tag}{inner}>"
    return _TAG_RE.sub(_sub, markup)
