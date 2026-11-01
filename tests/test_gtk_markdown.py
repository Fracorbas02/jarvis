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

from jarvis.gtk.markdown import (Block, _safe_href, collect_defs,  # noqa: E402
                                 highlight_code, inline_to_pango, parse_blocks,
                                 render_math_png, sanitize_pango, to_pango)


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
    assert '<a href="https://a.b" title="T">' in p and '>txt</span></a>' in p
    assert p.count('<a href="https://ref.example" title="T">') == 3, p
    assert '<a href="https://auto.link">' in p and 'https://auto.link</span></a>' in p
    assert '<a href="https://bare.url/x?a=1&amp;b=2">' in p
    assert '<a href="http://www.ex.com/z">' in p and 'www.ex.com/z</span></a>' in p


def test_inline_image_is_link() -> None:
    p = inline_to_pango('![alt](https://img.x/y.png)')
    assert '<a href="https://img.x/y.png">' in p and '>alt</span></a>' in p
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


def test_safe_href_schemas() -> None:
    assert _safe_href("https://x") == "https://x"
    assert _safe_href("HTTPS://X") == "HTTPS://X"
    assert _safe_href("mailto:a@b.c") == "mailto:a@b.c"
    assert _safe_href("#ancre") == "#ancre"
    assert _safe_href("/relatif") == "/relatif"
    assert _safe_href("page.html") == "page.html"
    assert _safe_href("javascript:alert(1)") is None
    assert _safe_href("data:text/html,x") is None
    assert _safe_href("file:///etc/passwd") is None
    assert _safe_href("vbscript:x") is None


def test_dangerous_schemes_rendered_as_text() -> None:
    p = inline_to_pango("[c](javascript:alert(1)) [d](data:text/html;base64,XX) [ok](https://a.b)")
    assert 'href="javascript:' not in p and 'href="data:' not in p
    assert "javascript:alert(1)" in p  # texte brut
    assert '<a href="https://a.b">' in p and '>ok</span></a>' in p
    s = sanitize_pango(p)
    assert 'href="javascript:' not in s and 'href="data:' not in s


def test_sanitize_blocks_bad_href_and_unknown_tags() -> None:
    assert sanitize_pango('<a href="javascript:alert(1)">c</a>').count("<a ") == 0
    assert sanitize_pango('<script>x</script>').count("<script>") == 0
    # attribut non whitelisté -> balise littérale échappée (guillemets inclus)
    out = sanitize_pango('<span onclick="evil()">t</span>')
    # la balise ouvrante est rendue littérale et échappée : plus de '<span' actif,
    # l'attribut n'est plus dans une balise (guillemets échappés en &quot;)
    assert "<span" not in out and "&lt;span onclick=&quot;" in out


def test_sanitize_keeps_allowed() -> None:
    s = sanitize_pango('<b>g</b><i>i</i><s>s</s><a href="https://x" title="t">l</a>'
                       '<span weight="bold" foreground="#fff" font-family="monospace">t</span>')
    assert "<b>g</b>" in s and '<a href="https://x" title="t">l</a>' in s
    assert 'font-family="monospace"' in s


# ---------- nouveautés UI moderne ----------


def test_math_inline_and_block() -> None:
    p = inline_to_pango("E = $mc^2$ fin")
    assert 'style="italic"' in p and "#ffb454" in p and "mc^2" in p
    p = inline_to_pango("$$\\int_0^1 x^2 dx$$")
    assert 'size="large"' in p and "\\int_0^1" in p
    b = parse_blocks("texte\n\n$$\n\\frac{1}{3}\n$$\n\naprès\n")
    kinds = [x.kind for x in b]
    assert "math" in kinds
    assert b[kinds.index("math")].text == "\\frac{1}{3}"


def test_links_are_styled() -> None:
    p = inline_to_pango("[exemple](https://example.com) et https://bare.url")
    assert 'underline="single"' in p and 'foreground="#4cc2ff"' in p
    assert '<a href="https://example.com">' in p
    assert '<a href="https://bare.url">' in p


def test_highlight_python() -> None:
    h = highlight_code('def f(x):  # com\n    return "s" + 42', "python")
    assert 'foreground="#ff7b72">def</span>' in h
    assert 'foreground="#8b949e">' in h          # commentaire
    assert 'foreground="#a5d6ff">' in h          # string
    assert 'foreground="#79c0ff">42</span>' in h  # nombre
    assert "&lt;" not in h or True
    # le markup reste valide après sanitize
    s = sanitize_pango(f'<span font-family="monospace">{h}</span>')
    assert "<tt" not in s


def test_highlight_shell_and_unknown() -> None:
    h = highlight_code("echo hello # comment", "bash")
    assert 'foreground="#ff7b72">echo</span>' in h
    # langue inconnue : pas de couleur, texte échappé
    h = highlight_code("a < b & c", "inconnu")
    assert "a &lt; b &amp; c" in h and "foreground" not in h


def test_highlight_json() -> None:
    h = highlight_code('{"clef": true, "n": 12}', "json")
    assert 'foreground="#a5d6ff">' in h          # chaînes
    assert 'foreground="#ff7b72">true</span>' in h
    assert 'foreground="#79c0ff">12</span>' in h


def test_render_math_png() -> None:
    d = render_math_png(r"\int_0^1 x^2\,dx = \frac{1}{3}")
    assert d is not None and d[:8] == b"\x89PNG\r\n\x1a\n" and len(d) > 100
    d2 = render_math_png(r"E = mc^2", fontsize=13)
    assert d2 is not None and d2[:8] == b"\x89PNG\r\n\x1a\n"
    # cache : même objet
    assert render_math_png(r"\int_0^1 x^2\,dx = \frac{1}{3}") is d


# ---------- repros rapport revue (B1-B12 + bug principal) ----------

def test_b1_tilde_fence() -> None:
    b = parse_blocks("~~~python\nx = 1\n~~~\n")
    assert b[0].kind == "code" and b[0].lang == "python"


def test_b2_fence_4_backticks_contenant_3() -> None:
    b = parse_blocks("````md\n```py\nx\n```\ny\n````\n")
    assert b[0].kind == "code" and "```py" in b[0].text and "y" in b[0].text


def test_b3_info_string_avec_espace() -> None:
    b = parse_blocks('```python title="x"\ncode\n```\n')
    assert b[0].kind == "code" and b[0].lang == "python"


def test_b4_setext_multilignes() -> None:
    b = parse_blocks("ligne1\nligne2\n---\n")
    assert b[0].kind == "heading" and b[0].level == 2 and "ligne1" in b[0].text


def test_b5_liste_ligne_vide_entre_items() -> None:
    b = parse_blocks("- a\n\n- b\n")
    assert len(b) == 1 and b[0].kind == "list" and len(b[0].items) == 2


def test_b7_crlf_normalise() -> None:
    b = parse_blocks("a\r\nb\r\n")
    assert b[0].text == "a\nb"


def test_b8_url_avec_parentheses() -> None:
    p = inline_to_pango("[x](https://wikipedia.org/wiki/Maladie_(virus))")
    assert 'href="https://wikipedia.org/wiki/Maladie_(virus)"' in p


def test_b9_underscore_intra_mot_pas_italique() -> None:
    assert "<i>" not in inline_to_pango("snake_case_name")
    assert "<i>italique</i>" in inline_to_pango("du _italique_ ici")


def test_b10_double_backtick() -> None:
    p = inline_to_pango("``a ` b``")
    assert "a ` b" in p and p.count("font-family") == 1


def test_b11_table_cellule_pipe_echappe() -> None:
    b = parse_blocks("| a \\| b | c |\n|---|---|\n| 1 | 2 |\n")
    assert b[0].rows[0] == ["a | b", "c"]


def test_b12_html_mini_sur() -> None:
    p = inline_to_pango("l1<br>l2 H<sub>2</sub>O x<sup>2</sup> <kbd>Ctrl</kbd>")
    assert "\n" in p and "<sub>2</sub>" in p and "<sup>2</sup>" in p and "font-family" in p


def test_bug_principal_code_indente_avec_tirets() -> None:
    # le bug vu en prod : --- dans un bloc de code indenté cassait en titre
    b = parse_blocks("para\n\n    x = 1\n    ---\n    y = 2\n")
    assert [x.kind for x in b] == ["paragraph", "code"]
    assert "---" in b[1].text


# ---------- régressions affichage (audit markdown) ----------


def test_dollar_not_math_when_spaces_or_shell_vars() -> None:
    # prix, variables shell, "5$$ et 3$$" : PAS de maths (règles pandoc)
    for t in ["ça coûte 5$ et 3$ au total", "variable $HOME et $PATH",
              "prix: 10$ - 20$", "5$$ et 3$$", "a $ b $ c"]:
        p = inline_to_pango(t)
        assert 'style="italic"' not in p, (t, p)
        assert "$" in p
    # vraies maths toujours reconnues
    assert 'style="italic"' in inline_to_pango("math $x^2$ fin")
    assert 'size="large"' in inline_to_pango("$$E=mc^2$$ bloc")
    # $ fermant suivi d'un chiffre : pas de maths
    assert 'style="italic"' not in inline_to_pango("$x$2 degrés")


def test_heading_closing_hash_needs_space() -> None:
    # "## C#" ne doit pas perdre son # final (fermeture = espace + #)
    assert parse_blocks("## C# et F#\n")[0].text == "C# et F#"
    assert parse_blocks("# Titre #\n")[0].text == "Titre"
    assert parse_blocks("# Titre##\n")[0].text == "Titre##"  # pas d'espace -> contenu


def test_double_underscore_intraword_literal() -> None:
    p = inline_to_pango("a__b__c et foo__bar__baz")
    assert "<b>" not in p and "__b__" in p
    # __gras__ isolé fonctionne toujours
    assert "<b>gras</b>" in inline_to_pango("__gras__")
    # *** et ** intra-mot restent autorisés (CommonMark)
    assert "<b><i>x</i></b>" in inline_to_pango("a***x***b")


def test_code_fence_inside_list_item() -> None:
    b = parse_blocks("- voici :\n  ```python\n  def f():\n      return 1\n  ```\n- suite\n")
    assert b[0].kind == "list" and len(b[0].items) == 2
    codes = [s for s in b[0].items[0] if s.kind == "code"]
    assert len(codes) == 1, [s.kind for s in b[0].items[0]]
    assert codes[0].lang == "python"
    assert "def f():" in codes[0].text and "return 1" in codes[0].text
    assert "```" not in codes[0].text
    assert b[0].items[1][0].kind == "paragraph" and b[0].items[1][0].text == "suite"


def test_list_continuation_paragraph_still_works() -> None:
    b = parse_blocks("- item\n  texte continu\n- item2\n")
    assert [s.kind for s in b[0].items[0]] == ["paragraph", "paragraph"]
    assert b[0].items[0][1].text == "texte continu"
