"""Rendu Markdown 100% via markdown-it-py -> HTML, pour affichage WebKitGTK.

markdown-it-py ("gfm-like" + plugins footnote/tasklists + linkify) couvre
CommonMark + GFM : titres ATX/setext, tableaux avec alignements, listes
puces/numérotées/imbriquées, tâches, citations imbriquées, blocs de code
fences/indentés, hr, images, liens inline/références/autolinks/URLs nues,
footnotes avec backrefs, strikethrough, échappements...

Ce module ajoute par-dessus :
- coloration syntaxique des blocs <code> via highlight_code() (markdown.py),
  injectée en spans HTML inline ;
- équations $$...$$ remplacées par une <img src="data:image/png;base64,...">
  rendue par render_math_png() (matplotlib.mathtext) ;
- un CSS embarqué (thème sombre) ;
- un assainissement bleach : seules les balises/attrs sûres passent, les href
  javascript:/data: sont supprimés (markdown-it le fait déjà, bleach en
  double filet).

API : markdown_to_html(text) -> document HTML complet (str), prêt pour
WebKit.WebView.load_html(). Tolérant au streaming : on re-rend tout à chaque
fragment (markdown-it gère les blocs non fermés).
"""

from __future__ import annotations

import base64
import html as _html

from bleach.sanitizer import Cleaner
from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

from .markdown import highlight_code, render_math_png

_md = MarkdownIt("gfm-like",
                 options_update={"html": False, "linkify": True,
                                 "typographer": False})
_md.use(footnote_plugin).use(tasklists_plugin)

# double filet de sécurité HTML (markdown-it bloque déjà javascript:/data:)
from bleach.css_sanitizer import CSSSanitizer

_CSS_SANITIZER = CSSSanitizer(allowed_css_properties=[
    "text-align", "color", "background-color", "font-weight", "font-size",
])
_CLEANER = Cleaner(
    tags={"h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "hr", "ul", "ol", "li",
          "blockquote", "pre", "code", "table", "thead", "tbody", "tr", "th",
          "td", "strong", "em", "b", "i", "s", "del", "a", "img", "sup", "sub",
          "span", "section", "input", "div"},
    attributes={
        "a": ["href", "title"],
        "img": ["src", "alt", "title"],
        "th": ["style"], "td": ["style"],
        "input": ["type", "checked", "disabled", "class"],
        "li": ["class"], "ul": ["class"], "section": ["class"],
        "hr": ["class"], "code": ["class"], "span": ["style"],
        "div": ["class"],
    },
    protocols=["http", "https", "mailto"],
    css_sanitizer=_CSS_SANITIZER,
    strip=True,
)

_CSS = """
:root { color-scheme: dark; }
body {
  font-family: 'Inter', 'Cantarell', system-ui, sans-serif;
  font-size: 14px; line-height: 1.55; color: #e6edf3;
  margin: 0; padding: 2px 0; overflow-wrap: break-word;
}
h1 { font-size: 1.7em; color: #4cc2ff; margin: 0.5em 0 0.3em; }
h2 { font-size: 1.4em; color: #4cc2ff; margin: 0.5em 0 0.3em; }
h3 { font-size: 1.15em; color: #4cc2ff; margin: 0.5em 0 0.25em; }
h4, h5, h6 { color: #4cc2ff; margin: 0.4em 0 0.2em; }
p { margin: 0.35em 0; }
a { color: #4cc2ff; text-decoration: underline; }
code {
  font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.88em;
  background: rgba(210,168,255,0.12); color: #d2a8ff;
  border-radius: 4px; padding: 1px 5px;
}
pre {
  background: #0d1117; border: 1px solid rgba(230,237,243,0.1);
  border-radius: 10px; padding: 10px 12px; margin: 8px 0;
  overflow-x: auto; position: relative;
}
pre code { background: none; color: #e6edf3; padding: 0; font-size: 0.85em; }
pre .code-lang {
  position: absolute; top: 4px; right: 8px;
  font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.05em;
  color: rgba(230,237,243,0.45); font-weight: 600;
}
pre .copy-btn {
  position: absolute; top: 2px; right: 44px;
  background: rgba(230,237,243,0.08); border: none; border-radius: 6px;
  color: rgba(230,237,243,0.7); cursor: pointer; padding: 3px 7px; font-size: 0.75em;
}
pre .copy-btn:hover { background: rgba(76,194,255,0.25); color: #4cc2ff; }
blockquote {
  border-left: 3px solid #4cc2ff; margin: 6px 0; padding: 2px 0 2px 12px;
  color: rgba(230,237,243,0.72);
}
table { border-collapse: collapse; margin: 8px 0; display: block; overflow-x: auto; }
th, td { border: 1px solid rgba(230,237,243,0.18); padding: 6px 12px; }
th { background: rgba(76,194,255,0.18); color: #4cc2ff; font-weight: 700; }
tr:nth-child(even) td { background: rgba(230,237,243,0.03); }
ul, ol { margin: 0.3em 0; padding-left: 1.4em; }
li.task-list-item { list-style: none; }
li.task-list-item input { margin-right: 6px; accent-color: #4cc2ff; }
hr { border: none; border-top: 1px solid rgba(230,237,243,0.2); margin: 10px 0; }
img { max-width: 100%; border-radius: 10px; }
.footnotes { border-top: 1px solid rgba(230,237,243,0.15); margin-top: 10px;
              padding-top: 4px; font-size: 0.85em; color: rgba(230,237,243,0.7); }
.footnote-ref a { text-decoration: none; color: #4cc2ff; font-size: 0.75em;
                  vertical-align: super; }
.math-block { display: block; text-align: center; margin: 8px 0;
              background: rgba(76,194,255,0.08); border-radius: 8px; padding: 8px; }
.math-block img { max-width: 560px; }
"""

# JS embarqué : bouton copier (navigator.clipboard avec fallback execCommand),
# et liens externes interceptés -> postMessage vers Python (xdg-open)
_JS = """
document.addEventListener('click', function(e) {
  var a = e.target.closest('a');
  if (a && a.href && !a.href.startsWith('#') && !a.href.startsWith('mailto:')) {
    e.preventDefault();
    window.webkit.messageHandlers.jarvis_link.postMessage(a.href);
  }
  var btn = e.target.closest('.copy-btn');
  if (btn) {
    var pre = btn.closest('pre');
    var code = pre.querySelector('code');
    var txt = code ? code.innerText : pre.innerText;
    function done(ok) { btn.textContent = ok ? '✓ copié' : '✗'; setTimeout(function(){ btn.textContent = 'copier'; }, 1200); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(function(){ done(true); }, function(){ fallback(); });
    } else { fallback(); }
    function fallback() {
      var ta = document.createElement('textarea');
      ta.value = txt; document.body.appendChild(ta); ta.select();
      try { done(document.execCommand('copy')); } catch (err) { done(false); }
      document.body.removeChild(ta);
    }
  }
});
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('pre').forEach(function(pre) {
    var m = (pre.querySelector('code')||{}).className || '';
    var lang = m.replace('language-', '');
    if (lang) {
      var s = document.createElement('span');
      s.className = 'code-lang'; s.textContent = lang;
      pre.appendChild(s);
    }
    var b = document.createElement('button');
    b.className = 'copy-btn'; b.textContent = 'copier';
    pre.appendChild(b);
  });
});
"""


def markdown_to_html(text: str) -> str:
    """Markdown brut -> document HTML complet (str) pour WebKit.WebView.

    Pipeline : placeholders neutres pour les équations $$...$$ (non échappés
    par markdown-it car html=False) -> _md.render -> substitution des
    placeholders par des <img data:image/png> -> highlight des fences ->
    protection des data:image/png -> bleach -> restauration -> document final.
    """
    import re
    math_htmls: list[str] = []

    def _stash(m: re.Match) -> str:
        tex = m.group(1).strip()
        png = render_math_png(tex, fontsize=16.0)
        if png is None:
            html_img = f'<p class="math-fallback"><em>{_html.escape(tex)}</em></p>'
        else:
            b64 = base64.b64encode(png).decode()
            html_img = (f'<img class="math-block" src="data:image/png;base64,{b64}" '
                        f'alt="{_html.escape(tex)}">')
        math_htmls.append(html_img)
        return f"JARVISMATH{len(math_htmls) - 1}"

    # 1) stash équations bloc (placeholder neutre, markdown-it ne l'échappe pas)
    protected = re.sub(r"\$\$([^$]+)\$\$", _stash, text, flags=re.S)

    # 2) rendu markdown-it complet
    body = _md.render(protected)

    # 3) substitution des placeholders par les <img> math (le placeholder peut
    #    être dans un <p>...</p> : on le remplace, et on retire le <p> autour si
    #    le placeholder est seul dans son paragraphe)
    def _unstash(m: re.Match) -> str:
        return math_htmls[int(m.group(1))]

    body = re.sub(r"<p>JARVISMATH(\d+)</p>", lambda m: math_htmls[int(m.group(1))], body)
    body = re.sub(r"JARVISMATH(\d+)", _unstash, body)

    # 4) coloration des fences de code (post-traitement sur le HTML rendu)
    def _hl_pre(m: re.Match) -> str:
        lang = _html.unescape(m.group(1))
        code = _html.unescape(m.group(2))
        code = code[:-1] if code.endswith("\n") else code
        hl = highlight_code(code, lang)
        # markup Pango (foreground="...") -> style="color:..." pour le HTML
        hl = re.sub(r'<span foreground="([^"]+)">',
                    lambda mm: '<span style="color:' + mm.group(1) + '">', hl)
        return f'<pre><code class="language-{_html.escape(lang)}">{hl}</code></pre>'

    body = re.sub(r'<pre><code class="language-([^"]*)">(.*?)</code></pre>',
                  _hl_pre, body, flags=re.S)

    # 5) protéger les data:image/png (nos équations) avant bleach, restaurer après
    data_protected: list[str] = []

    def _prot(m: re.Match) -> str:
        data_protected.append(m.group(0))
        return f"JARVISDATAIMG{len(data_protected) - 1}"

    body = re.sub(r'<img class="math-block" src="data:image/png;base64,[A-Za-z0-9+/=]+"[^>]*>',
                  _prot, body)

    # 6) assainissement (double filet bleach)
    body = _CLEANER.clean(body)

    # 7) restauration des images data: + nettoyage cosmétique du ';'
    body = re.sub(r"JARVISDATAIMG(\d+)",
                  lambda m: data_protected[int(m.group(1))], body)
    body = re.sub(r'style="([^"]*?);"', r'style="\1"', body)

    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<style>{_CSS}</style></head><body>{body}"
        f"<script>{_JS}</script></body></html>"
    )
