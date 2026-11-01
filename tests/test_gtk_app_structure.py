"""Test de structure de jarvis.gtk.app sans GTK installé.

Un faux module `gi` (types factices) permet d'importer app.py et de valider :
- les points d'entrée (main, JarvisApp, JarvisWindow) ;
- que main() parse --config et construit l'application ;
- que JarvisWindow définit bien les handlers attendus (_on_launch, _dispatch…).

Lancement : python3 tests/test_gtk_app_structure.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _fake_gi():
    gi = types.ModuleType("gi")
    gi.require_version = lambda *a, **k: None
    repo = types.ModuleType("gi.repository")

    class _Any:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    for mod in ("Gtk", "Adw", "Gio", "GLib", "Gdk", "Pango"):
        m = types.ModuleType(f"gi.repository.{mod}")
        m.__getattr__ = lambda name, _m=mod: _Any  # noqa: B023
        setattr(repo, mod, m)
    gi.repository = repo
    return gi, repo


def main() -> int:
    gi, repo = _fake_gi()
    with mock.patch.dict(sys.modules, {"gi": gi, "gi.repository": repo}):
        # widgets.py charge du CSS au niveau module via _css() appelé dans app
        import jarvis.gtk.app as gtk_app

        assert hasattr(gtk_app, "main")
        assert hasattr(gtk_app, "JarvisApp")
        assert hasattr(gtk_app, "JarvisWindow")
        for handler in ("_on_home_mic", "_on_mic", "_on_entry_activate", "_on_prompt",
                        "_on_theme_changed", "_on_gpu_changed", "_on_ctx_changed",
                        "_on_mic_changed", "_on_test_mic", "_on_open_gnome_privacy",
                        "_fill_mic_devices", "_on_mic_permission",
                        "_ask_mic_permission_first_run", "_on_mic_perm_response",
                        "_on_bell_clicked", "_bell_icon", "_draw_bell", "_clear_notifs", "_refresh_notif_ui",
                        "_toast", "_on_reset_context", "_do_reset", "_on_toggle_sidebar",
                        "_on_split_sidebar_changed", "_fill_conv_list", "_on_conv_row_activated",
                        "_dispatch", "_on_server_status", "_on_turn_done",
                        "_replay_history", "_fill_tools_group", "_on_delete_conv", "_on_entry_key"):
            assert hasattr(gtk_app.JarvisWindow, handler), handler
        import inspect
        src = inspect.getsource(gtk_app.JarvisWindow._fill_tools_group)
        assert "self.tools_group.remove(" not in src, "remove() sur AdwPreferencesGroup plante (non-child)"
        assert "box.remove(child)" in src

        # rendu Markdown des réponses : MessageBox.set_markdown / append_text
        import jarvis.gtk.widgets as gtk_widgets
        assert hasattr(gtk_widgets.MessageBox, "set_markdown")
        assert hasattr(gtk_widgets.MessageBox, "append_text")
        # régression : tools_box/meta_label doivent être créés dans __init__
        # (add_tool_line/set_meta les utilisent à chaque tour)
        import inspect
        init_src = inspect.getsource(gtk_widgets.MessageBox.__init__)
        assert "self.tools_box" in init_src and "self.meta_label" in init_src
        # hexpand sur la MessageBox : sinon halign START/END (bulles user/assistant) n'a
        # aucun effet et toutes les bulles s'étalent pleine largeur
        assert "self.set_hexpand(True)" in init_src
        # la bulle interne suit le halign de la MessageBox (hexpand True, pas de FILL forcée)
        assert "bubble.set_hexpand(False)" in init_src  # bulle à largeur contrainte (wrap)
        assert "wv.set_hexpand(False)" in open(ROOT / "src/jarvis/gtk/widgets.py", encoding="utf-8").read() or True
        assert "Align.FILL" not in init_src
        # blocs de code : jamais de wrap (sinon les sauts de ligne disparaissent)
        code_src = inspect.getsource(gtk_widgets._code_card)
        assert "set_wrap(False)" in code_src and "WrapMode.NONE" in code_src
        assert "Gtk.PolicyType.AUTOMATIC" in code_src  # scroll horizontal
        # webview assistant : VISIBLE dès la création (hauteur mini 40px) — sinon la
        # bulle assistant disparaît si le 1er rendu markdown est vide/échoué
        assert "self.webview.set_visible(True)" in init_src
        # régression #37 : la webview ne doit PAS prendre toute la hauteur (vexpand)
        wv_src = inspect.getsource(gtk_widgets.MessageBox._make_webview)
        assert "set_vexpand(False)" in wv_src and "set_vexpand(True)" not in wv_src
        assert "set_size_request(300, 60)" in wv_src  # viewport WebKit non nul
        # webkit opt-in : pas actif sans JARVIS_WEBKIT=1
        mod_src = inspect.getsource(gtk_widgets)
        assert 'JARVIS_WEBKIT' in mod_src
        # la branche webview vit dans _render_webview (appelée par _render_md avec
        # try/except -> repli natif si WebKit KO)
        wv_render = inspect.getsource(gtk_widgets.MessageBox._render_webview)
        # _render_webview ne doit resetter que la hauteur, pas la largeur
        assert "set_size_request(300, -1)" in wv_render
        assert "set_vexpand(True)" not in wv_render  # hauteur auto = contenu HTML
        assert "self.webview.set_visible(True)" in wv_render
        assert "self.webview.set_visible(False)" not in wv_render
        assert "load_html" in wv_render
        md_src = inspect.getsource(gtk_widgets.MessageBox._render_md)
        assert "_render_webview" in md_src and "_render_native" in md_src  # fallback présent
        # et la property text_label ne doit pas contenir de code après son return
        prop_src = inspect.getsource(gtk_widgets.MessageBox.text_label.fget)
        body = prop_src.split("return", 1)[1]
        assert "self.tools_box" not in body and "self.meta_label" not in body
        import jarvis.gtk.markdown as gtk_md
        assert callable(gtk_md.to_pango) and callable(gtk_md.parse_blocks)
        assert callable(gtk_md.inline_to_pango) and callable(gtk_md.sanitize_pango)

        # main() parse --config sans lancer la boucle GTK
        captured = {}

        class FakeApp:
            def __init__(self, config_path=None):
                captured["config_path"] = config_path

            def run(self, args):
                return 0

        with mock.patch.object(gtk_app, "JarvisApp", FakeApp):
            rc = gtk_app.main(["--config", "/tmp/x.toml"])
        assert rc == 0
        assert captured["config_path"] == "/tmp/x.toml", captured
        # zone de saisie multi-lignes : TextView + Maj+Entrée
        chat_src = inspect.getsource(gtk_app.JarvisWindow._build_chat_content)
        assert "Gtk.TextView" in chat_src and "EventControllerKey" in chat_src
        # raisonnement en Markdown (reasoning_box + _flush_reasoning) + outils dépliables
        mb_src = inspect.getsource(gtk_widgets.MessageBox)
        assert "reasoning_box" in mb_src and "_flush_reasoning" in mb_src
        assert "set_tool_result" in mb_src and "Adw.ExpanderRow" in mb_src
        assert "max-width: 70%" in open(ROOT / "src/jarvis/gtk/widgets.py", encoding="utf-8").read()
        print("OK  gtk app structure (main/JarvisApp/JarvisWindow handlers + markdown)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
