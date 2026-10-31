"""Tests unitaires du parseur Markdown GTK4 (sans GTK installé).

Couvre jarvis.gtk.markdown : parse_blocks (CommonMark + GFM) et
inline_to_pango / to_pango (markup Pango échappé, liens <a>), y compris
les cas limites du streaming (fence non fermé, marqueur non fermé).

Usage: uv run pytest tests/test_gtk_markdown.py -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jarvis.gtk.markdown import (Block, collect_defs, inline_to_pango,  # noqa: E402
                                 parse_blocks, sanitize_pango, to_pango)


def kinds(text: str) -> list[str]:
    return [b.kind for b in parse_blocks(text)]


# ---------- blocs ----------


def test_headings_atx_and_setext() -> None:
    assert kinds("# H1\n") == ["heading"]
    assert parse_blocks("###### H6\n")[0].level == 6
    b = parse_blocks("Titre\n===\n")
    assert b[0].kind == "heading" and b[0].level == 1
    b = parse_blocks("Sous-titre\n---\n")
    assert b[0].kind == "heading" and b[0].level == 2


def test_hr_vs_setext() -> None:
    assert kinds("---\n")[0] == "hr"
    assert kinds("***\n")[0] == "hr"
    assert kinds("___\n")[0] == "hr"
    # setext prioritaire sur hr quand une ligne précède
    assert kinds("texte\n---\n")[0] == "heading"


def test_paragraphs() -> None:
    assert kinds("ligne 1\nligne 2\n\nligne 3\n") == ["paragraph", "paragraph"]


def test_code_fence_with_and_without_lang() -> None:
    b = parse_blocks("```python\nx = 1\n```\n")
    assert b[0].kind == "code" and b[0].lang == "python" and b[0].text == "x = 1"
    b = parse_blocks("```\ncode\n```\naprès\n")
    assert [x.kind for x in b] == ["code", "paragraph"]
    assert "```" not in b[0].text


def test_code_fence_unclosed_streaming() -> None:
    b = parse_blocks("texte\n```py\npartial")
    assert b[-1].kind == "code" and "partial" in b[-1].text


def test_indented_code() -> None:
    b = parse_blocks("para\n\n    code indenté\n")
    assert b[-1].kind == "code" and "code indenté" in b[-1].text


def test_unordered_list_and_tasks() -> None:
    b = parse_blocks("- a\n- [ ] todo\n- [x] done\n")
    assert b[0].kind == "list" and not b[0].ordered
    assert b[0].items[1][0].kind == "task" and not b[0].items[1][0].checked
    assert b[0].items[2][0].kind == "task" and b[0].items[2][0].checked


def test_ordered_list_separate_block() -> None:
    # une vraie liste numérotée (2+ items) ne fusionne pas avec des puces
    b = parse_blocks("1. un\n2. deux\n- puce\n")
    assert [(x.kind, x.ordered, len(x.items)) for x in b] == [
        ("list", True, 2), ("list", False, 1)]


def test_numbered_section_absorbed_by_bullets() -> None:
    # cas fréquent des LLM : "1. Titre" suivi de puces -> une seule liste à puces
    b = parse_blocks("1. Mise en forme\n- **Gras**\n- *Italique*\n\n2. Liens\n- [x](https://a.b)\n")
    assert [(x.kind, x.ordered, len(x.items)) for x in b] == [
        ("list", False, 3), ("list", False, 2)]


def test_nested_list() -> None:
    b = parse_blocks("- parent\n  - enfant\n  - enfant2\n- autre\n")
    assert b[0].kind == "list" and len(b[0].items) == 2
    subs = [s for s in b[0].items[0] if s.kind == "list"]
    assert len(subs) == 1 and len(subs[0].items) == 2


def test_quote() -> None:
    b = parse_blocks("> citation **gras**\n> ligne 2\n")
    assert b[0].kind == "quote"
    assert b[0].items[0].kind == "paragraph"
    assert "gras" in b[0].items[0].text


def test_table_gfm() -> None:
    src = "| A | B | C |\n|:--|:-:|--:|\n| 1 | 2 | 3 |\n"
    b = parse_blocks(src)
    assert b[0].kind == "table"
    assert b[0].aligns == ["l", "c", "r"]
    assert len(b[0].rows) == 2 and b[0].rows[0] == ["A", "B", "C"]


def test_image_block() -> None:
    b = parse_blocks('![Alt text](https://x.y/img.png "Tip")\n')
    assert b[0].kind == "image"
    assert b[0].text == "Alt text" and b[0].url == "https://x.y/img.png"
    assert b[0].title == "Tip"


def test_link_and_footnote_defs() -> None:
    ld, fn = collect_defs('[r1]: https://ref.example "T"\n[^n1]: contenu\n')
    assert ld == {"r1": ("https://ref.example", "T")}
    assert fn == {"n1": "contenu"}
    b = parse_blocks("[^n1]: contenu\n")
    assert b[0].kind == "footnote_def" and b[0].footnote_id == "n1"


# ---------- inline -> Pango ----------


def test_inline_styles() -> None:
    p = inline_to_pango("**g** *i* ***b*** ~~s~~ `c`")
    assert "<b>g</b>" in p and "<i>i</i>" in p and "<b><i>b</i></b>" in p
    assert "<s>s</s>" in p and '<span font-family="monospace"' in p and ">c</span>" in p
    assert "**" not in p and "~~" not in p and "`" not in p


def test_inline_links() -> None:
    ld, _ = collect_defs('[r1]: https://ref.example "T"\n')
    p = inline_to_pango(
        '[txt](https://a.b "T") [r][r1] [r1][] [r1] <https://auto.link> '
        "https://bare.url/x?a=1&b=2 www.ex.com/z", ld)
    assert '<a href="https://a.b" title="T">txt</a>' in p
    assert p.count('<a href="https://ref.example" title="T">') == 3, p
    assert '<a href="https://auto.link">https://auto.link</a>' in p
    assert '<a href="https://bare.url/x?a=1&amp;b=2">' in p
    assert '<a href="http://www.ex.com/z">www.ex.com/z</a>' in p


def test_inline_image_is_link() -> None:
    p = inline_to_pango('![alt](https://img.x/y.png)')
    assert '<a href="https://img.x/y.png">alt</a>' in p
    assert "![" not in p


def test_footnote_ref_superscript() -> None:
    _, fn = collect_defs("[^n1]: contenu\n")
    p = inline_to_pango("texte[^n1] fin", {}, fn)
    assert '<a href="#fn-n1">' in p and 'rise="6000"' in p and "[1]" in p, p
    # sans définition connue : l'id est affiché tel quel
    p2 = inline_to_pango("texte[^zz] fin")
    assert "[zz]" in p2


def test_escapes() -> None:
    p = inline_to_pango(r"\*pas gras\* \_pas italique\_ \`pas code\`")
    assert "*pas gras*" in p and "_pas italique_" in p and "`pas code`" in p
    assert "<b>" not in p and "<i>" not in p


def test_xml_escaped_everywhere() -> None:
    p = inline_to_pango("a < b & c > d <script>alert(1)</script> `x<y&z`")
    assert "<script>" not in p and "&lt;script&gt;" in p
    assert "x&lt;y&amp;z" in p


def test_streaming_partial_inline() -> None:
    assert inline_to_pango("voici **bo") == "voici **bo"
    assert "<b>bold</b>" in inline_to_pango("voici **bold** fin")


# ---------- to_pango / sanitize ----------


def test_to_pango_full_document_tags_whitelisted() -> None:
    doc = ("# T\n\n**g** *i* `c` [l](https://a.b) https://x.y\n\n- a\n- [x] t\n\n"
           "> q\n\n```py\nx=1\n```\n\n| A |\n|---|\n| 1 |\n\n---\n\n![i](https://img)\n\n"
           "note[^n]\n\n[^n]: def\n")
    s = sanitize_pango(to_pango(doc))
    tags = set(re.findall(r"</?([a-zA-Z]+)", s))
    assert tags <= {"b", "i", "s", "span", "a"}, tags
    assert "<tt" not in s and "```" not in s and "|" not in s


def test_sanitize_neutralizes_invalid_tags() -> None:
    s = sanitize_pango('<script>x</script> <tt font-family="m">c</tt> <span size="large">ok</span>')
    assert "&lt;script&gt;" in s and "<script>" not in s
    assert "<tt" not in s and "&lt;tt" in s
    assert '<span size="large">' in s


def test_sanitize_keeps_allowed() -> None:
    s = sanitize_pango('<b>g</b><i>i</i><s>s</s><a href="https://x" title="t">l</a>'
                       '<span weight="bold" foreground="#fff" font-family="monospace">t</span>')
    assert "<b>g</b>" in s and '<a href="https://x" title="t">l</a>' in s
    assert 'font-family="monospace"' in s
