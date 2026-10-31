"""Test de structure de jarvis.gtk.app sans GTK install\u00e9.

Un faux module `gi` (types factices) permet d'importer app.py et de valider :
- les points d'entr\u00e9e (main, JarvisApp, JarvisWindow) ;
- que main() parse --config et construit l'application ;
- que JarvisWindow d\u00e9finit bien les handlers attendus (_on_launch, _dispatch\u2026).

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
        # widgets.py charge du CSS au niveau module via _css() appel\u00e9 dans app
        import jarvis.gtk.app as gtk_app

        assert hasattr(gtk_app, "main")
        assert hasattr(gtk_app, "JarvisApp")
        assert hasattr(gtk_app, "JarvisWindow")
        for handler in ("_on_home_mic", "_on_mic", "_on_entry_activate", "_on_prompt",
                        "_on_theme_changed", "_on_gpu_changed", "_on_ctx_changed",
                        "_on_mic_changed", "_on_test_mic", "_on_open_gnome_privacy",
                        "_fill_mic_devices", "_dispatch",
                        "_on_server_status", "_on_turn_done", "_replay_history"):
            assert hasattr(gtk_app.JarvisWindow, handler), handler

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
        print("OK  gtk app structure (main/JarvisApp/JarvisWindow handlers)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
