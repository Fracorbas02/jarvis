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


def _gsettings_scale() -> float:
    """Échelle de l'OS lue via gsettings (utile sous GNOME Wayland, où le DPI
    X11 peut rester à 96). Best-effort : 0 si indisponible.
    Note : depuis GNOME 46, la clé scaling-factor n’existe plus —
    le scale est dans monitors.xml (voir _monitors_xml_scale)."""
    import subprocess
    for key in ("scaling-factor", "text-scaling-factor"):
        try:
            out = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", key],
                capture_output=True, text=True, timeout=2).stdout.strip()
            value = float(out.split()[-1]) if out else 0.0
            if value > 1.0:
                return value
        except Exception:  # noqa: BLE001 — best-effort
            continue
    return 0.0


def _monitors_xml_scale(path=None) -> float:
    """Échelle lue dans monitors.xml ($XDG_CONFIG_HOME/monitors.xml, défaut
    ~/.config/monitors.xml) : c'est là que GNOME >= 46 stocke le scale de
    chaque écran en Wayland (la clé gsettings scaling-factor y est supprimée),
    et aussi en X11 avec scaling fractionnaire. On prend le max des <scale>
    des <logicalmonitor>. Best-effort : 0 si absent/illisible."""
    import xml.etree.ElementTree as ET
    if path is None:
        cfg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config")
        path = os.path.join(cfg, "monitors.xml")
    try:
        root = ET.parse(path).getroot()
    except Exception:  # noqa: BLE001 — best-effort
        return 0.0
    scales = []
    for lm in root.iter("logicalmonitor"):
        el = lm.find("scale")
        if el is not None and el.text:
            try:
                scales.append(float(el.text))
            except ValueError:
                pass
    return max(scales) if scales else 0.0


def detect_scale(root: _Root, override: float | None = None) -> float:
    """Facteur d'échelle de l'OS : override > max(env GDK_SCALE/QT, gsettings
    GNOME, monitors.xml GNOME>=46, DPI X11). Le max des indices couvre X11
    (Xft.dpi) et Wayland (monitors.xml) où un seul des deux est renseigné."""
    if override is not None:
        return _clamp(override)
    hints = [1.0]
    for var in ("GDK_SCALE", "QT_SCREEN_SCALE_FACTORS"):
        raw = os.environ.get(var, "")
        if raw:
            try:
                hints.append(float(raw.split(",")[0].split(";")[0]))
            except ValueError:
                pass
    hints.append(_gsettings_scale())
    hints.append(_monitors_xml_scale())
    dpi = root.winfo_fpixels("1i")
    if dpi > 0:
        hints.append(dpi / BASE_DPI)
    return _clamp(max(hints))


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
