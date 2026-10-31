"""Détection du facteur d'échelle de l'écran (HiDPI / scaling GNOME) et
géométrie de fenêtre adaptative.

Sous GNOME avec scaling d'affichage (200 %), une appli Tk non aware paraît
minuscule : on lit le DPI réel de l'écran (winfo_fpixels('1i') / 96), avec
hints d'env (GDK_SCALE, QT_SCREEN_SCALE_FACTORS) et override manuel --scale,
puis on applique ctk.set_widget_scaling / set_window_scaling AVANT de créer
les widgets (ça scale dimensions et polices).
"""

from __future__ import annotations

import os
from typing import Protocol

BASE_DPI = 96.0
MIN_SCALE, MAX_SCALE = 1.0, 3.0


class _Root(Protocol):
    def winfo_fpixels(self, distance: str) -> float: ...
    def winfo_screenwidth(self) -> int: ...
    def winfo_screenheight(self) -> int: ...


def _clamp(scale: float) -> float:
    return max(MIN_SCALE, min(MAX_SCALE, scale))


def detect_scale(root: _Root, override: float | None = None) -> float:
    """Facteur d'échelle de l'écran : override > env > DPI X11."""
    if override is not None:
        return _clamp(override)
    for var in ("GDK_SCALE", "QT_SCREEN_SCALE_FACTORS"):
        raw = os.environ.get(var, "")
        if raw:
            try:
                first = float(raw.split(",")[0].split(";")[0])
            except ValueError:
                continue
            if first > 1.0:
                return _clamp(first)
    dpi = root.winfo_fpixels("1i")
    if dpi > 0:
        return _clamp(dpi / BASE_DPI)
    return 1.0


def adaptive_geometry(root: _Root, width_ratio: float = 0.62,
                      height_ratio: float = 0.68,
                      min_w: int = 880, min_h: int = 560) -> str:
    """Géométrie de la fenêtre = fraction de l'écran, centrée, avec minimum."""
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w = max(int(sw * width_ratio), min_w)
    h = max(int(sh * height_ratio), min_h)
    x, y = max((sw - w) // 2, 0), max((sh - h) // 2, 0)
    return f"{w}x{h}+{x}+{y}"


def adaptive_minsize() -> tuple[int, int]:
    return 880, 560


def setup_window(root, scale: float) -> None:
    """Applique le scaling puis configure la fenêtre (géométrie adaptative,
    redimensionnable).

    Ordre important : CTk bloque maxsize à la taille courante dans son
    _set_scaling (appelé par set_window_scaling) — il faut donc déverrouiller
    maxsize APRÈS les set_*_scaling et AVANT geometry/minsize, sinon la
    fenêtre reste coincée à 600x500 et non redimensionnable.
    """
    import customtkinter as ctk

    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    root.maxsize(root.winfo_screenwidth(), root.winfo_screenheight())
    root.minsize(*adaptive_minsize())
    root.geometry(adaptive_geometry(root))
    root.resizable(True, True)
