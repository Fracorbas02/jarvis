"""Parseur Markdown (CommonMark + GFM) pour la conversation GTK4.

Pas de lib externe : on parse ici et on produit des blocs typés que
widgets.py rend en widgets GTK natifs (labels Pango avec liens cliquables,
Gtk.Grid pour les tableaux, Gtk.Picture pour les images, etc.).

Blocs supportés :
- titres ATX #..###### et setext (=== / ---) ;
- séparateur horizontal (---, ***, ___) ;
- paragraphes ;
- listes à puces (- * +), numérotées (1.), tâches GFM (- [ ] / - [x]),
  imbriquées (indentation) ;
- citations > (imbriquées) ;
- blocs de code ``` fences (avec ou sans langue) et indentés (4 espaces) ;
- tableaux GFM (| ... | avec ligne d'alignement :---: etc.) ;
- définitions de liens de référence [id]: url "titre" et de footnotes
  [^id]: contenu.

Inline supporté :
- **gras**, *italique* / _italique_, ***les deux***, ~~barré~~, `code` ;
- [texte](url "titre"), [texte][id], [id][] (références), autolinks
  <https://...> et URLs nues https://... ;
- ![alt](url "titre") -> bloc image ;
- [^id] -> appel de footnote (exposant) ;
- échappements \\* \\_ \\` \\\\ etc.

Le parseur est tolérant au streaming : un fence/bloc non fermé reste ouvert
jusqu'au fragment suivant (re-rendu complet à chaque fragment).

API :
- parse_blocks(text) -> list[Block] ;
- inline_to_pango(text) -> markup Pango (liens <a href="..." title="...">) ;
- to_pango(text) -> markup d'un texte entier (compat, utilisé en repli) ;
- sanitize_pango(markup) -> markup dont seules les balises whitelistées
  subsistent (garantit que Pango.parse_markup réussit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

# Schémas d'URL autorisés dans les href (sinon le lien est rendu en texte) :
# http/https/mailto + ancres #... et chemins relatifs. Bloque javascript:,
# data:, file:, etc.
_ALLOWED_SCHEMES = ("http://", "https://", "mailto:", "#", "/", "./", "../")


def _esc_attr(text: str) -> str:
    """Échappe un attribut (href/title) : & < > ET guillemets."""
    return escape(text, {'"': "&quot;", "'": "&#x27;"})


def _safe_href(url: str) -> str | None:
    """Retourne l'URL si son schéma est autorisé (ou relatif/ancre), sinon None."""
    u = url.strip()
    if not u:
        return None
    low = u.lower()
    if low.startswith(_ALLOWED_SCHEMES):
        return u
    if "://" in low or low.split(":", 1)[0] in ("javascript", "data", "file", "vbscript"):
        return None
    # relatif sans schéma explicite
    if ":" not in low.split("/", 1)[0]:
        return u
    return None


# ---------- blocs ----------

@dataclass
class Block:
    kind: str  # heading|paragraph|code|list|quote|table|hr|image|footnote_def|task
    level: int = 0
    lang: str = ""
    text: str = ""            # code brut, texte du paragraphe, alt de l'image...
    url: str = ""             # image
    title: str = ""           # image / lien
    items: list = field(default_factory=list)   # list/quote: list de sous-blocs
    ordered: bool = False
    start: int = 1
    rows: list = field(default_factory=list)    # table: list[list[str]] (cells inline md)
    aligns: list = field(default_factory=list)  # table: "l"|"c"|"r"|None
    footnote_id: str = ""     # footnote_def
    checked: bool = False     # task


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_RE = re.compile(r"^(=+|-+)\s*$")
_HR_RE = re.compile(r"^\s{0,3}((?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$")
_FENCE_RE = re.compile(r"^\s{0,3}```+\s*([^\s`]*)\s*$")
_ULIST_RE = re.compile(r"^(\s{0,3})([-*+])\s+(.*)$")
_OLIST_RE = re.compile(r"^(\s{0,3})(\d{1,9})[.)]\s+(.*)$")
_TASK_RE = re.compile(r"^\[( |x|X)\]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
_LINK_DEF_RE = re.compile(r'^\s{0,3}\[([^\]]+)\]:\s*(\S+)(?:\s+"([^"]*)")?\s*$')
_FOOTNOTE_DEF_RE = re.compile(r'^\s{0,3}\[\^([^\]]+)\]:\s*(.*)$')
_INDENT_CODE_RE = re.compile(r"^(?: {4}|\t)(.*)$")
_IMG_LINE_RE = re.compile(
    r'^\s{0,3}!\[([^\]]*)\]\(([^\s"\')]+)(?:\s+"([^"]*)")?\)\s*$')


def _split_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _parse_table_aligns(sep_line: str) -> list:
    cells = _split_table_row(sep_line)
    out = []
    for c in cells:
        left = c.startswith(":")
        right = c.endswith(":")
        if left and right:
            out.append("c")
        elif right:
            out.append("r")
        elif left:
            out.append("l")
        else:
            out.append(None)
    return out


def _is_block_start(line: str, next_line: str | None) -> bool:
    s = line.strip()
    return bool(
        _HEADING_RE.match(line) or _FENCE_RE.match(s) or _HR_RE.match(line)
        or _ULIST_RE.match(line) or _OLIST_RE.match(line) or _QUOTE_RE.match(line)
        or _LINK_DEF_RE.match(line) or _FOOTNOTE_DEF_RE.match(line)
        or _IMG_LINE_RE.match(line)
        or ("|" in line and next_line is not None
            and _TABLE_SEP_RE.match(next_line))
    )


def parse_blocks(text: str) -> list[Block]:
    """Découpe le texte en blocs. Tolérant au streaming : fences, tableaux et
    listes non terminés sont émis tels quels (re-parsés au fragment suivant)."""
    lines = text.split("\n")
    blocks: list[Block] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        nxt = lines[i + 1] if i + 1 < n else None

        # définitions (footnote d'abord : [^id]: matche aussi la regex lien)
        m = _FOOTNOTE_DEF_RE.match(line)
        if m:
            blocks.append(Block(kind="footnote_def", footnote_id=m.group(1),
                                text=m.group(2)))
            i += 1
            continue
        m = _LINK_DEF_RE.match(line)
        if m:
            i += 1
            continue

        # fence code
        fm = _FENCE_RE.match(stripped)
        if fm:
            lang = fm.group(1)
            i += 1
            code_lines: list[str] = []
            while i < n and not _FENCE_RE.match(lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1  # fence fermant
            blocks.append(Block(kind="code", lang=lang, text="\n".join(code_lines)))
            continue

        # ligne vide
        if not stripped:
            i += 1
            continue

        # titre setext (ligne suivie de === ou ---) : prioritaire sur hr
        if nxt is not None and _SETEXT_RE.match(nxt.strip()) and stripped:
            level = 1 if nxt.strip().startswith("=") else 2
            blocks.append(Block(kind="heading", level=level, text=stripped))
            i += 2
            continue

        # hr
        if _HR_RE.match(line):
            blocks.append(Block(kind="hr"))
            i += 1
            continue

        # titre ATX
        m = _HEADING_RE.match(line)
        if m:
            blocks.append(Block(kind="heading", level=len(m.group(1)),
                                text=m.group(2)))
            i += 1
            continue

        # tableau GFM
        if "|" in line and nxt is not None and _TABLE_SEP_RE.match(nxt):
            header = _split_table_row(line)
            aligns = _parse_table_aligns(nxt)
            i += 2
            rows = [header]
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(_split_table_row(lines[i]))
                i += 1
            blocks.append(Block(kind="table", rows=rows, aligns=aligns))
            continue

        # liste (puces / numérotée / tâches) : les items d'un même bloc sont
        # ceux dont le marqueur a la même indentation ; une liste numérotée
        # qui reprend à 1 après des puces forme un bloc séparé (comme GFM).
        mu = _ULIST_RE.match(line)
        mo = _OLIST_RE.match(line)
        if mu or mo:
            first = mu or mo
            ordered = mo is not None
            start = int(mo.group(2)) if ordered else 1
            marker_indent = len(first.group(1))
            marker_kind = "u" if mu else "o"
            items: list[list[Block]] = []
            while i < n:
                cur = lines[i]
                mu2 = _ULIST_RE.match(cur)
                mo2 = _OLIST_RE.match(cur)
                kind2 = "u" if mu2 else ("o" if mo2 else None)
                if kind2 is None:
                    # continuation indentée du dernier item
                    if items and (len(cur) - len(cur.lstrip(" "))) >= marker_indent + 2:
                        items[-1].extend(parse_blocks(cur.strip()))
                        i += 1
                        continue
                    break
                indent2 = len((mu2 or mo2).group(1))
                if indent2 < marker_indent:
                    break
                if indent2 == marker_indent and kind2 != marker_kind:
                    # Changement de type de marqueur à la même indentation :
                    # si la liste courante est un item numéroté ISOLÉ (un seul
                    # item, commençant à 1 — cas fréquent des LLM : "1. titre"
                    # suivi de puces), on absorbe la suite dans cette liste,
                    # qui devient une liste à puces. Sinon, nouvelle liste.
                    if len(items) == 1 and ordered:
                        # item numéroté isolé ("1."/"2." titre de section d'un
                        # LLM) suivi de puces : on absorbe dans une liste à puces
                        marker_kind = "u"
                        ordered = False
                        start = 1
                    else:
                        break
                if indent2 > marker_indent:
                    # lignes imbriquées consécutives : une seule sous-liste
                    m2 = mu2 or mo2
                    marker = m2.group(2) if mu2 else m2.group(2) + "."
                    sub_lines = [" " * (indent2 - marker_indent) + marker + " " + m2.group(3)]
                    i += 1
                    while i < n:
                        c2 = lines[i]
                        mu3 = _ULIST_RE.match(c2)
                        mo3 = _OLIST_RE.match(c2)
                        if (mu3 or mo3) and len((mu3 or mo3).group(1)) == indent2:
                            m3 = mu3 or mo3
                            mk = m3.group(2) if mu3 else m3.group(2) + "."
                            sub_lines.append(" " * (indent2 - marker_indent) + mk + " " + m3.group(3))
                            i += 1
                        elif c2.strip() and (len(c2) - len(c2.lstrip(" "))) > indent2:
                            sub_lines.append(c2)
                            i += 1
                        else:
                            break
                    sub = parse_blocks("\n".join(sub_lines))
                    if len(sub) == 1 and sub[0].kind == "list":
                        items[-1].append(sub[0])
                    else:
                        items[-1].extend(sub)
                    continue
                content = (mu2 or mo2).group(3)
                tm = _TASK_RE.match(content)
                if tm:
                    items.append([Block(kind="task", text=tm.group(2),
                                        checked=tm.group(1).lower() == "x")])
                else:
                    items.append(parse_blocks(content))
                i += 1
            blocks.append(Block(kind="list", ordered=ordered, start=start,
                                items=items))
            continue

        # citation
        if _QUOTE_RE.match(line):
            q_lines: list[str] = []
            while i < n and _QUOTE_RE.match(lines[i]):
                q_lines.append(_QUOTE_RE.match(lines[i]).group(1))
                i += 1
            blocks.append(Block(kind="quote", items=parse_blocks("\n".join(q_lines))))
            continue

        # code indenté (4 espaces)
        if _INDENT_CODE_RE.match(line):
            code_lines = []
            while i < n and (_INDENT_CODE_RE.match(lines[i]) or not lines[i].strip()):
                if _INDENT_CODE_RE.match(lines[i]):
                    code_lines.append(_INDENT_CODE_RE.match(lines[i]).group(1))
                i += 1
            blocks.append(Block(kind="code", text="\n".join(code_lines)))
            continue

        # image seule sur la ligne -> bloc image
        im = _IMG_LINE_RE.match(line)
        if im:
            blocks.append(Block(kind="image", text=im.group(1), url=im.group(2),
                                title=im.group(3) or ""))
            i += 1
            continue

        # paragraphe : jusqu'à ligne vide ou prochain bloc
        para: list[str] = []
        while i < n:
            cur = lines[i]
            if not cur.strip():
                i += 1
                break
            if para and _is_block_start(cur, lines[i + 1] if i + 1 < n else None):
                break
            para.append(cur)
            i += 1
        blocks.append(Block(kind="paragraph", text="\n".join(para)))

    return blocks


# ---------- inline -> Pango ----------

# ordre : échappement, code, image inline, footnote, lien inline, lien ref
# [txt][id], ref courte [txt], autolink <...>, url nue, ***...***, **...**,
# *...*, ~~...~~
_INLINE_RE = re.compile(
    r"|<((?:https?://|mailto:)[^>\s]+)>"                           # 1: <url> autolink
    r"|(\\[\\`*_{}\[\]()#+\-.!>~|])"                              # 2: échappement
    r"|(`+)([^`\n]*?)\3"                                           # 3-4: `code`
    r"|!\[([^\]]*)\]\(([^\s\"\')]+)(?:\s+\"([^\"]*)\")?\)"            # 5-7: ![alt](url "t")
    r"|\[\^([^\]]+)\]"                                             # 8: [^id] footnote
    r"|\[([^\]]+)\]\(([^\s\"\')]+)(?:\s+\"([^\"]*)\")?\)"             # 9-11: [txt](url "t")
    r"|\[([^\]]+)\]\[([^\]]*)\]"                                   # 12-13: [txt][id]
    r"|\[([^\]]+)\]"                                               # 14: [txt] ref courte
    r"|(?<![\w<])((?:https?://|www\.)[^\s<>()]+[^\s<>().,;:!?\'\"\]<])"  # 15: url nue
    r"|(\*\*\*|___)(.+?)\16"                                       # 16-17: ***...***
    r"|(\*\*|__)(.+?)\18"                                          # 18-19: **...**
    r"|(\*|_)([^*_\n]+)\20"                                        # 20-21: *...* / _..._
    r"|(~~)(.+?)\22"                                               # 22-23: ~~...~~
)


def inline_to_pango(text: str, link_defs: dict | None = None,
                    footnotes: dict | None = None) -> str:
    """Texte inline markdown -> markup Pango (texte échappé, liens <a>)."""
    link_defs = link_defs or {}
    footnotes = footnotes or {}
    out: list[str] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append(escape(text[pos:m.start()]))
        if m.group(1):                       # <url> autolink (regex: http/https/mailto)
            url = m.group(1)
            out.append(f'<a href="{_esc_attr(url)}">{escape(url)}</a>')
        elif m.group(2):                     # échappement \x -> x
            out.append(escape(m.group(2)[1:]))
        elif m.group(3):                     # `code`
            out.append('<span font-family="monospace" foreground="#d2a8ff">'
                       f"{escape(m.group(4))}</span>")
        elif m.group(5) is not None:         # image inline -> alt cliquable
            alt, url, title = m.group(5), m.group(6), m.group(7) or ""
            safe = _safe_href(url)
            if safe is None:
                out.append(escape(m.group(0)))
            else:
                t = f' title="{_esc_attr(title)}"' if title else ""
                out.append(f'<a href="{_esc_attr(safe)}"{t}>{escape(alt or safe)}</a>')
        elif m.group(8):                     # footnote [^id] -> exposant
            fid = m.group(8)
            num = list(footnotes).index(fid) + 1 if fid in footnotes else fid
            out.append(f'<a href="#fn-{_esc_attr(fid)}">'
                       f'<span size="x-small" rise="6000">[{escape(str(num))}]</span></a>')
        elif m.group(9):                     # [txt](url "titre")
            txt, url, title = m.group(9), m.group(10), m.group(11) or ""
            safe = _safe_href(url)
            if safe is None:
                out.append(escape(m.group(0)))
            else:
                t = f' title="{_esc_attr(title)}"' if title else ""
                inner = inline_to_pango(txt, link_defs, footnotes)
                out.append(f'<a href="{_esc_attr(safe)}"{t}>{inner}</a>')
        elif m.group(12):                    # [txt][id] / [txt][]
            txt, ref = m.group(12), (m.group(13) or m.group(12)).lower()
            if ref in link_defs:
                url, title = link_defs[ref]
                safe = _safe_href(url)
                if safe is None:
                    out.append(escape(m.group(0)))
                else:
                    t = f' title="{_esc_attr(title)}"' if title else ""
                    inner = inline_to_pango(txt, link_defs, footnotes)
                    out.append(f'<a href="{_esc_attr(safe)}"{t}>{inner}</a>')
            else:
                out.append(escape(m.group(0)))
        elif m.group(14):                    # [txt] référence courte
            ref = m.group(14).lower()
            if ref in link_defs:
                url, title = link_defs[ref]
                safe = _safe_href(url)
                if safe is None:
                    out.append(escape(m.group(0)))
                else:
                    t = f' title="{_esc_attr(title)}"' if title else ""
                    out.append(f'<a href="{_esc_attr(safe)}"{t}>{escape(m.group(14))}</a>')
            else:
                out.append(escape(m.group(0)))
        elif m.group(15):                    # url nue
            url = m.group(15)
            href = url if url.lower().startswith(("http://", "https://")) else f"http://{url}"
            out.append(f'<a href="{_esc_attr(href)}">{escape(url)}</a>')
        elif m.group(16):                    # ***...***
            inner = inline_to_pango(m.group(17), link_defs, footnotes)
            out.append(f"<b><i>{inner}</i></b>")
        elif m.group(18):                    # **...**
            inner = inline_to_pango(m.group(19), link_defs, footnotes)
            out.append(f"<b>{inner}</b>")
        elif m.group(20):                    # *...* / _..._
            inner = inline_to_pango(m.group(21), link_defs, footnotes)
            out.append(f"<i>{inner}</i>")
        elif m.group(22):                    # ~~...~~
            inner = inline_to_pango(m.group(23), link_defs, footnotes)
            out.append(f"<s>{inner}</s>")
        pos = m.end()
    if pos < len(text):
        out.append(escape(text[pos:]))
    return "".join(out)


def collect_defs(text: str) -> tuple[dict, dict]:
    """Extrait les définitions de liens [id]: url et de footnotes [^id]: ..."""
    link_defs: dict[str, tuple[str, str]] = {}
    footnotes: dict[str, str] = {}
    for line in text.split("\n"):
        m = _FOOTNOTE_DEF_RE.match(line)
        if m:
            footnotes[m.group(1)] = m.group(2)
            continue
        m = _LINK_DEF_RE.match(line)
        if m:
            link_defs[m.group(1).lower()] = (m.group(2), m.group(3) or "")
    return link_defs, footnotes


def to_pango(text: str) -> str:
    """Markdown brut -> markup Pango d'un texte entier (un seul label).

    Rendu de repli (texte aplati) ; le rendu principal passe par
    parse_blocks + widgets GTK natifs."""
    link_defs, footnotes = collect_defs(text)
    parts: list[str] = []
    for b in parse_blocks(text):
        if b.kind == "heading":
            size = {1: "xx-large", 2: "x-large", 3: "large", 4: "medium",
                    5: "small", 6: "small"}[b.level]
            parts.append(f'<span size="{size}" weight="bold" foreground="#4cc2ff">'
                         f"{inline_to_pango(b.text, link_defs, footnotes)}</span>\n")
        elif b.kind == "code":
            parts.append('<span background="#0a0c10" font-family="monospace">'
                         f"{escape(b.text)}\n</span>")
        elif b.kind == "list":
            for idx, item in enumerate(b.items):
                bullet = f"{b.start + idx}. " if b.ordered else "• "
                for sub in item:
                    if sub.kind == "task":
                        parts.append("☑ " if sub.checked else "☐ ")
                        parts.append(inline_to_pango(sub.text, link_defs, footnotes))
                        parts.append("\n")
                    else:
                        parts.append(bullet)
                        parts.append(inline_to_pango(sub.text, link_defs, footnotes)
                                     if sub.kind == "paragraph" else escape(sub.text))
                        parts.append("\n")
                        bullet = "  "
        elif b.kind == "quote":
            for sub in b.items:
                parts.append("❝ " + (inline_to_pango(sub.text, link_defs, footnotes)
                                     if sub.kind == "paragraph" else escape(sub.text))
                             + "\n")
        elif b.kind == "table":
            for row in b.rows:
                parts.append(" | ".join(inline_to_pango(c, link_defs, footnotes)
                                        for c in row) + "\n")
        elif b.kind == "hr":
            parts.append("――――――――\n")
        elif b.kind == "image":
            safe = _safe_href(b.url)
            if safe is None:
                parts.append(escape(b.text or b.url) + "\n")
            else:
                parts.append(f'<a href="{_esc_attr(safe)}">{escape(b.text or safe)}</a>\n')
        elif b.kind == "footnote_def":
            parts.append(f'<span size="small">[{escape(b.footnote_id)}] '
                         f"{inline_to_pango(b.text, link_defs, footnotes)}</span>\n")
        elif b.kind == "paragraph":
            parts.append(inline_to_pango(b.text, link_defs, footnotes) + "\n")
    return "".join(parts)


# ---------- sanitize (sécurité set_markup) ----------

_ALLOWED_TAGS = {
    "b": set(),
    "i": set(),
    "s": set(),
    "u": set(),
    "span": {"size", "weight", "style", "foreground", "background",
             "font-family", "rise"},
    "a": {"href", "title"},
    "sub": set(),
    "sup": set(),
}
_ATTR_RE = re.compile(r'([a-zA-Z-]+)="([^"]*)"')
# une balise = <nom suivi d'attributs nom="valeur" (valeur pouvant contenir des
# espaces/échappés) ; on capture tout jusqu'au '>' final pour ne rien laisser
# passer (ex. " onclick=..." après une URL contenant un guillemet)
_TAG_RE = re.compile(r'<(/?)([a-zA-Z]+)((?:\s+[^>]*?)?)>')


def sanitize_pango(markup: str) -> str:
    """Whitelist stricte : seules des balises Pango complètes et bien formées
    de _ALLOWED_TAGS passent (attributs filtrés ; pour <a>, href validé par
    _safe_href). Toute autre balise est remplacée par son texte échappé, et
    tout '<' orphelin devient &lt;. Garantit que set_markup ne peut ni échouer
    ni propager un schéma dangereux."""
    out: list[str] = []
    i = 0
    n = len(markup)
    while i < n:
        lt = markup.find("<", i)
        if lt < 0:
            out.append(markup[i:])
            break
        out.append(markup[i:lt])
        m = _TAG_RE.match(markup, lt)
        if not m:
            out.append("&lt;")
            i = lt + 1
            continue
        closing = m.group(1) == "/"
        tag = m.group(2).lower()
        attrs = m.group(3) or ""
        allowed = _ALLOWED_TAGS.get(tag)
        pairs = _ATTR_RE.findall(attrs)
        consumed = "".join(f' {a}="{v}"' for a, v in pairs)
        ok = allowed is not None and consumed.strip() == attrs.strip()
        if ok:
            keep = []
            for a, v in pairs:
                if a not in allowed:
                    ok = False
                    break
                if tag == "a" and a == "href" and not closing and _safe_href(v) is None:
                    ok = False
                    break
                keep.append(f'{a}="{v}"')
            if ok:
                inner = (" " + " ".join(keep)) if keep else ""
                out.append(f"</{tag}{inner}>" if closing else f"<{tag}{inner}>")
                i = m.end()
                continue
        # balise refusée : on émet son texte échappé (guillemets inclus)
        out.append(_esc_attr(m.group(0)))
        i = m.end()
    return "".join(out)
