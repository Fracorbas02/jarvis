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
    # géométrie adaptative
    assert adaptive_geometry(StubRoot(96, 1920, 1080)) == "1190x734+365+173"
    # petit écran -> minimum, x centré à 0 (pas négatif)
    assert adaptive_geometry(StubRoot(96, 800, 600)) == "880x560+0+20"
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
