"""Tests unitaires du rendu Markdown du chat desktop (sans affichage Tk).

Couvre jarvis.desktop.markdown.parse_markdown : titres, listes, inline
(gras/italique/code/barré), blocs de code, et cas limites du streaming
(marqueur non fermé, code non fermé).

Usage: uv run pytest tests/test_desktop_markdown.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jarvis.desktop.markdown import Segment, parse_markdown  # noqa: E402


def texts(segs: list[Segment]) -> str:
    return "".join(s.text for s in segs)


def test_plain_text_passthrough() -> None:
    segs = parse_markdown("bonjour le monde\n")
    assert texts(segs) == "bonjour le monde\n"
    assert all(not s.style for s in segs)


def test_bold_italic_strike() -> None:
    segs = parse_markdown("du **gras** et *italique* et ~~barré~~\n")
    by_text = {s.text: s.style for s in segs}
    assert by_text["gras"] == frozenset({"bold"})
    assert by_text["italique"] == frozenset({"italic"})
    assert by_text["barré"] == frozenset({"strike"})
    assert texts(segs) == "du gras et italique et barré\n"


def test_inline_code() -> None:
    segs = parse_markdown("appelle `echo__ping({})` maintenant\n")
    by_text = {s.text: s.style for s in segs}
    assert by_text["echo__ping({})"] == frozenset({"code"})
    # pas de mise en forme imbriquée dans le code
    segs = parse_markdown("`**pas gras**`\n")
    code_segs = [s for s in segs if s.style == frozenset({"code"})]
    assert code_segs and code_segs[0].text == "**pas gras**"


def test_heading() -> None:
    segs = parse_markdown("## Titre\n")
    assert segs[0].text == "Titre\n"
    assert "heading" in segs[0].style and "bold" in segs[0].style


def test_list_bullet() -> None:
    segs = parse_markdown("- premier\n* second\n")
    assert texts(segs) == "• premier\n• second\n"


def test_code_block() -> None:
    src = "avant\n```python\nprint(1)\n```\nAprès\n"
    segs = parse_markdown(src)
    # les délimiteurs ``` eux-mêmes ne sont pas affichés
    assert texts(segs) == "avant\nprint(1)\nAprès\n"
    block = [s for s in segs if "print(1)" in s.text]
    assert block and block[0].style == frozenset({"mono", "code"})


def test_streaming_marqueur_non_ferme_puis_ferme() -> None:
    # fragment 1 : "**bo" -> affiché tel quel ; fragment 2 : "ld**" -> gras
    s1 = parse_markdown("voici **bo")
    assert texts(s1) == "voici **bo"
    assert all("bold" not in s.style for s in s1)
    s2 = parse_markdown("voici **bold** fin\n")
    by_text = {s.text: s.style for s in s2}
    assert by_text["bold"] == frozenset({"bold"})


def test_streaming_code_block_non_ferme() -> None:
    s = parse_markdown("```py\npartial")
    assert texts(s) == "partial"
    assert all(s.style == frozenset({"mono", "code"}) for s in s)


def test_texte_multiligne_conserve_les_sauts() -> None:
    src = "ligne 1\nligne 2\n\nligne 3\n"
    assert texts(parse_markdown(src)) == src
