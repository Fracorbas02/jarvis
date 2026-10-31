"""Tests des correctifs fenêtre desktop : redimensionnable + scaling écran.

- Partie unitaire (sans affichage) : detect_scale / adaptive_geometry.
- Partie intégration (sous Xvfb) : fenêtre resizable, PanedWindow à 2 volets,
  scaling appliqué (widgets + polices), géométrie dans l'écran.

Lancer avec deux DPI pour couvrir le scaling GNOME :
  xvfb-run -a uv run python tests/test_desktop_scaling.py --expect-scale 1
  xvfb-run -a -s "-screen 0 3840x2160x24 -dpi 192" \
      uv run python tests/test_desktop_scaling.py --expect-scale 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from jarvis.desktop import scaling as scaling_mod  # noqa: E402
from jarvis.desktop.scaling import (  # noqa: E402
    adaptive_geometry, adaptive_minsize, detect_scale, setup_window,
)


class StubRoot:
    def __init__(self, dpi: float, w: int, h: int):
        self._dpi, self._w, self._h = dpi, w, h

    def winfo_fpixels(self, _d: str) -> float:
        return self._dpi

    def winfo_screenwidth(self) -> int:
        return self._w

    def winfo_screenheight(self) -> int:
        return self._h


def test_unit() -> None:
    # environnement hermétique : ne pas lire le vrai ~/.config de la machine
    # (un monitors.xml GNOME réel fausserait les asserts de scale)
    import tempfile
    _old_xdg = os.environ.get("XDG_CONFIG_HOME")
    _tmp_xdg = tempfile.mkdtemp()
    os.environ["XDG_CONFIG_HOME"] = _tmp_xdg
    assert detect_scale(StubRoot(96, 1920, 1080)) == 1.0
    assert detect_scale(StubRoot(192, 3840, 2160)) == 2.0
    assert detect_scale(StubRoot(144, 2560, 1440)) == 1.5
    assert detect_scale(StubRoot(96, 1920, 1080), override=2.5) == 2.5
    assert detect_scale(StubRoot(384, 7680, 4320)) == 3.0  # clampé
    # hint d'env GNOME
    os.environ["GDK_SCALE"] = "2"
    try:
        assert detect_scale(StubRoot(96, 1920, 1080)) == 2.0
    finally:
        del os.environ["GDK_SCALE"]
    # hint gsettings GNOME (Wayland) : prioritaire sur un DPI X11 à 96
    orig = scaling_mod._gsettings_scale
    scaling_mod._gsettings_scale = lambda: 2.0
    try:
        assert detect_scale(StubRoot(96, 1920, 1080)) == 2.0
        # le DPI X11 à 192 gagne si plus fort
        assert detect_scale(StubRoot(192, 3840, 2160)) == 2.0
    finally:
        scaling_mod._gsettings_scale = orig
    # monitors.xml (GNOME >= 46 Wayland : scaling-factor gsettings supprimé)
    import tempfile
    monitors = (
        '<monitors version="2"><monitor><spec><connector>DP-1</connector></spec>'
        '<logicalmonitor><x>0</x><y>0</y><scale>2</scale>'
        '<monitor><spec><connector>DP-1</connector></spec></monitor>'
        '</logicalmonitor></monitor></monitors>'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(monitors)
        mpath = f.name
    try:
        assert scaling_mod._monitors_xml_scale(mpath) == 2.0
        orig_mx = scaling_mod._monitors_xml_scale
        scaling_mod._monitors_xml_scale = lambda path=None: 2.0
        try:
            assert detect_scale(StubRoot(96, 1920, 1080)) == 2.0
            assert detect_scale(StubRoot(192, 3840, 2160)) == 2.0
        finally:
            scaling_mod._monitors_xml_scale = orig_mx
        assert scaling_mod._monitors_xml_scale(mpath + ".missing") == 0.0
    finally:
        os.unlink(mpath)
    # env JARVIS_SCALE prioritaire sur tout (hors override)
    os.environ["JARVIS_SCALE"] = "2"
    try:
        assert detect_scale(StubRoot(96, 1920, 1080)) == 2.0
    finally:
        del os.environ["JARVIS_SCALE"]
    # Hyprland : ligne monitor avec scale 2 dans hyprland.conf
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        hyp = Path(d) / "hypr" / "hyprland.conf"
        hyp.parent.mkdir()
        hyp.write_text("monitor = eDP-1,2880x1920@60,0x0,2\n"
                       "# monitor = DP-1,1920x1080,0x0,1\n", encoding="utf-8")
        os.environ["XDG_CONFIG_HOME"] = d
        try:
            assert scaling_mod._hypr_scale() == 2.0
            assert detect_scale(StubRoot(96, 1920, 1080)) == 2.0
        finally:
            os.environ["XDG_CONFIG_HOME"] = _tmp_xdg
    assert scaling_mod._hypr_scale() == 0.0  # pas de conf dans ce dossier temporaire
    # géométrie centrée sur le moniteur du pointeur (multi-écran)
    import tempfile
    # format réel GNOME: width/height dans <monitor><mode><width/height>
    monitors2 = (
        '<monitors version="2">'
        # config active (primary=yes sur eDP-1), avec doublons d'historique
        '<monitor><spec><connector>eDP-1</connector></spec>'
        '<logicalmonitor><x>0</x><y>0</y><scale>2</scale>'
        '<monitor><primary>yes</primary><mode><width>1440</width><height>900</height></mode></monitor>'
        '</logicalmonitor></monitor>'
        '<monitor><spec><connector>DP-1</connector></spec>'
        '<logicalmonitor><x>1440</x><y>0</y><scale>1</scale>'
        '<monitor><mode><width>1920</width><height>1080</height></mode></monitor>'
        '</logicalmonitor></monitor>'
        # historique (anciennes configs, primary=no ou sans primary)
        '<monitor><spec><connector>HDMI-1</connector></spec>'
        '<logicalmonitor><x>1920</x><y>0</y><scale>1</scale>'
        '<monitor><primary>no</primary><mode><width>1920</width><height>1080</height></mode></monitor>'
        '</logicalmonitor></monitor>'
        '</monitors>'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(monitors2)
        mpath2 = f.name
    try:
        mons = scaling_mod._monitors_xml_layout(mpath2)
        # dédoublonné + primary=no filtré (historique)
        assert mons == [(0, 0, 1440, 900, 2.0), (1440, 0, 1920, 1080, 1.0)], mons

        class PtrRoot(StubRoot):
            def __init__(self, dpi, w, h, px, py):
                super().__init__(dpi, w, h)
                self._px, self._py = px, py
            def winfo_pointerxy(self):
                return self._px, self._py

        orig_layout = scaling_mod._monitors_xml_layout
        scaling_mod._monitors_xml_layout = lambda path=None: mons
        try:
            # pointeur (physiques) sur DP-1: logique x=1440, scale global 2 -> pxl=1500
            r = PtrRoot(96, 3840, 2160, 3000, 500)
            g = scaling_mod._center_geometry(r, 2.0)
            w, rest = g.split("x", 1)
            h, xy = rest.split("+", 1)
            x, y = xy.split("+", 1)
            # DP-1 logique 1920x1080 scale 1: fenêtre 1190x734 centrée en x=1440
            assert (int(w), int(h)) == (1190, 734), g
            assert (int(x), int(y)) == (1440 + (1920 - 1190) // 2, (1080 - 734) // 2), g
            # pointeur sur eDP-1 (scale 2): pxl=500 -> logique 250, dans eDP-1 0..1440
            r2 = PtrRoot(192, 3840, 2160, 500, 500)
            g2 = scaling_mod._center_geometry(r2, 2.0)
            w2, rest2 = g2.split("x", 1)
            h2, xy2 = rest2.split("+", 1)
            x2, y2 = xy2.split("+", 1)
            # eDP-1 logique 1440x900 scale 2: fenêtre min 440x280 -> 62% = 892x612
            assert (int(w2), int(h2)) == (892, 612), g2
            assert int(x2) == (1440 - 892) // 2 and 0 <= int(y2), g2
        finally:
            scaling_mod._monitors_xml_layout = orig_layout
    finally:
        os.unlink(mpath2)
    # géométrie adaptative
    assert adaptive_geometry(StubRoot(96, 1920, 1080)) == "1190x734+365+173"
    # petit écran -> minimum, x centré à 0 (pas négatif)
    assert adaptive_geometry(StubRoot(96, 800, 600)) == "880x560+0+20"
    # restaure l'environnement
    if _old_xdg is None:
        os.environ.pop("XDG_CONFIG_HOME", None)
    else:
        os.environ["XDG_CONFIG_HOME"] = _old_xdg
    print("OK  unit scaling/geometry")


def run_integration(expect_scale: float) -> None:
    import customtkinter as ctk
    from customtkinter.windows.widgets.scaling.scaling_tracker import ScalingTracker

    os.environ["JARVIS_CONFIG"] = str(ROOT / "tests" / "config.test.toml")
    from jarvis.desktop.app import DesktopApp

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    scale = detect_scale(root)
    assert abs(scale - expect_scale) < 0.01, f"scale={scale} attendu={expect_scale}"
    setup_window(root, scale)

    app = DesktopApp(root, str(ROOT / "tests" / "config.test.toml"), scale=scale)
    root.update_idletasks()

    # 1. fenêtre redimensionnable (resizable + maxsize déverrouillé au-delà de
    #    la taille courante : c'est CTk qui bloquait sinon)
    assert root.resizable() == (True, True), root.resizable()
    import tkinter as tk
    max_w, max_h = tk.Tk.maxsize(root)
    assert max_w >= root.winfo_screenwidth(), (max_w, root.winfo_screenwidth())
    # 2. deux volets (sidebar sur sash) + tailles mini respectées
    panes = app.pw.panes()
    assert len(panes) == 2, panes
    sw = app.sidebar.winfo_width()
    assert sw >= 200, f"sidebar trop étroite: {sw}"
    # 3. géométrie demandée = adaptative (unités logiques), dans l'écran
    geo = root.geometry()  # ex. "1190x734+365+173"
    gw, gh = (int(v) for v in geo.split("+")[0].split("x"))
    assert geo == adaptive_geometry(root), (geo, adaptive_geometry(root))
    assert gw >= adaptive_minsize()[0] and gh >= adaptive_minsize()[1], geo
    # 4. scaling effectif appliqué aux widgets (ctk ne scale pas auto sous Linux :
    #    get_widget_scaling = window_dpi(1) * widget_scaling)
    eff = ScalingTracker.get_widget_scaling(app.sidebar)
    assert abs(eff - expect_scale) < 0.01, eff
    # la police d'un vrai widget est scalée (ctk multiplie la taille via ce facteur)
    lbl = ctk.CTkLabel(app.main, text="x", font=ctk.CTkFont(size=13))
    root.update_idletasks()
    # spec de la police du widget tk interne : "Roboto -26 {...}" à scale 2
    inner_font = str(lbl._label.cget("font"))
    font_size = -int(inner_font.split()[1])
    assert font_size == round(13 * expect_scale), inner_font
    # 5. le chat est présent et la sidebar contient le segmented des modes
    assert app.chat.winfo_manager() == "grid"
    assert app.sidebar.mode_seg is not None
    # 6. les polices internes du chat (tags du textbox) suivent le scale :
    #    sinon le texte du chat reste à l'échelle 1x dans une UI à l'échelle OS
    tb = app.chat.text._textbox
    tag_font = str(tb.tag_cget("reasoning", "font"))
    assert f"-{round(12 * expect_scale)}" in tag_font, tag_font
    head_font = str(tb.tag_cget("user", "font"))
    assert f"-{round(13 * expect_scale)}" in head_font, head_font

    app.ctrl.stop()
    root.destroy()
    print(f"OK  intégration scale={scale} resizable panes={len(panes)} "
          f"sidebar={sw}px geom={gw}x{gh} font13->{font_size}px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-scale", type=float, default=float(os.environ.get("EXPECT_SCALE", "1")))
    args = parser.parse_args()
    test_unit()
    run_integration(args.expect_scale)
