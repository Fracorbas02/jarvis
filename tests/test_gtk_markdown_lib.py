"""Tests du rendu Markdown 100% via markdown-it-py -> HTML (markdown_lib.py).

Valide que markdown_to_html produit un document HTML complet et sûr :
titres, styles inline, liens/autolinks, footnotes, tasklists, listes
ordonnées, citations, code highlighté, tableaux avec alignements, images,
équations en PNG data:URI, bouton copier JS, CSS — et que les schémas
dangereux (javascript:, data:text/html, onerror) sont neutralisés.

Usage: uv run pytest tests/test_gtk_markdown_lib.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jarvis.gtk.markdown_lib import markdown_to_html  # noqa: E402

DOC = """# Titre H1
## Sous-titre

Texte **gras** *ital* ***both*** ~~barré~~ `code` [lien](https://a.b "T") https://auto.link note[^n].

- [ ] tâche ouverte
- [x] tâche faite
1. un
2. deux

> citation **gras**

```python
def f(x): return x < 2  # com
```

| A | B |
|:--|--:|
| 1 | 2 |

$$
\\int_0^1 x^2 dx = \\frac{1}{3}
$$

[^n]: contenu de la note

![img](https://picsum.photos/100)
"""


def body(html: str) -> str:
    return html.split("<body>", 1)[1]


def test_document_complet() -> None:
    h = markdown_to_html(DOC)
    b = body(h)
    assert h.startswith("<!DOCTYPE html>")
    assert "color-scheme: dark" in h          # CSS embarqué
    assert "copy-btn" in h                      # JS bouton copier
    assert "<h1>Titre H1</h1>" in b
    assert "<h2>Sous-titre</h2>" in b
    assert "<strong>gras</strong>" in b and "<em>ital</em>" in b
    assert "<em><strong>both</strong></em>" in b
    assert "<s>barré</s>" in b and "<code>code</code>" in b
    assert '<a href="https://a.b" title="T">lien</a>' in b
    assert '<a href="https://auto.link">https://auto.link</a>' in b  # linkify
    assert "#fn1" in b and "footnote" in b      # footnote ref + def
    assert "task-list-item-checkbox" in b and "checked" in b
    assert "<ol>" in b and "<li>un</li>" in b
    assert "<blockquote>" in b
    assert 'class="language-python"' in b
    assert '<span style="color:#ff7b72">def</span>' in b   # highlight injecté
    assert "<table>" in b and 'text-align:left' in b and 'text-align:right' in b
    assert "data:image/png;base64," in b        # équation en image PNG
    assert '<img src="https://picsum.photos/100"' in b


def test_securite_schemas_et_html() -> None:
    h = markdown_to_html(
        "[x](javascript:alert(1)) [d](data:text/html;base64,XX) "
        "<script>alert(1)</script> <img src=x onerror=alert(1)> [ok](https://a.b)")
    b = body(h)
    # aucun lien/balise actif avec schéma dangereux ou event handler :
    # markdown-it n'émet pas de <a href="javascript:...">, bleach strippe le reste
    import re
    hrefs = re.findall(r'href="([^"]*)"', b)
    assert all(h_.startswith(("http://", "https://", "mailto:", "#")) for h_ in hrefs), hrefs
    assert not re.search(r"<\s*(script|img)[^>]*on", b)
    assert "<script>alert" not in b
    assert '<a href="https://a.b">ok</a>' in b
    # le texte des liens dangereux reste visible mais inerte (échappé)
    assert "javascript:alert(1)" in b


def test_streaming_bloc_non_ferme() -> None:
    # fence non fermé : markdown-it rend quand même le contenu en <pre>
    h = markdown_to_html("texte\n```py\npartial")
    b = body(h)
    assert "partial" in b and "```" not in b


def test_math_inline_reste_texte() -> None:
    h = markdown_to_html("E = $mc^2$ fin")
    assert "mc^2" in body(h)  # inline : pas d'image, texte présent


def test_repli_texte_si_matplotlib_absent() -> None:
    # une formule invalide ne doit pas faire planter le rendu
    h = markdown_to_html("$$\\invalidzzz{$$")
    assert "<!DOCTYPE html>" in h  # document produit quoi qu'il arrive
