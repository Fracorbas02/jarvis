"""Rendu Markdown minimal pour le panneau de chat desktop (tkinter.Text).

Pourquoi pas une lib externe : le chat affiche du texte en direct dans un
tkinter.Text via des tags, et aucune lib de markdown ne rend nativement dans
un Text widget. On parse ici un sous-ensemble suffisant pour les réponses de
LLM et on produit des segments (texte, style) que ChatPanel traduit en tags :

- blocs : titres #..######, listes -/*/+, blocs de code ```, paragraphes ;
- inline : **gras**, *italique* / _italique_, `code`, ~~barré~~.

Le parseur est volontairement tolérant : un marqueur non fermé (fréquent en
plein streaming, ex. "**bo" puis "ld**") est rendu tel quel, et re-rendu au
fragment suivant quand le marqueur se ferme.

API : parse_markdown(text) -> list[Segment] avec Segment(text, style) où style
est un frozenset de {"bold", "italic", "strike", "code", "heading", "mono"}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    text: str
    style: frozenset[str] = frozenset()


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
# inline : délimiteurs ouvrants/fermants. `code` est prioritaire (pas de
# mise en forme imbriquée à l'intérieur).
_INLINE_RE = re.compile(
    r"(`[^`\n]*`)|(\*\*[^*\n]+\*\*)|(__[^_\n]+__)|(\*[^*\n]+\*)|(_[^_\n]+_)|(~~[^~\n]+~~)"
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
            was_in_code = in_code
            in_code = not in_code
            # le délimiteur ouvrant (avec sa langue, ex. ```python) et le
            # fermant ``` ne sont pas affichés ; seule une ligne ``` isolée
            # dans un bloc ouvert est du contenu
            if not was_in_code and body.strip() != "```":
                continue
            if was_in_code and not in_code and body.strip() == "```":
                continue
            segments.append(Segment(body + nl, frozenset({"mono", "code"})))
            continue
        if in_code:
            segments.append(Segment(line, frozenset({"mono", "code"})))
            continue
        m = _HEADING_RE.match(body)
        if m:
            segments.extend(_parse_inline(m.group(2) + nl,
                                          frozenset({"heading", "bold"})))
            continue
        m = _LIST_RE.match(body)
        if m:
            segments.append(Segment("• ", frozenset()))
            segments.extend(_parse_inline(m.group(1) + nl))
            continue
        segments.extend(_parse_inline(line))
    return segments
