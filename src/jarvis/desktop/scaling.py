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


def _env_scale(var: str) -> float:
    """Valeur d'une variable d'env d'échelle (JARVIS_SCALE, GDK_SCALE,
    QT_SCREEN_SCALE_FACTORS) : premier segment avant ',' ou ';'. 0 si absente."""
    raw = os.environ.get(var, "")
    if not raw:
        return 0.0
    try:
        return float(raw.split(",")[0].split(";")[0])
    except ValueError:
        return 0.0


def _hypr_scale() -> float:
    """Scale Hyprland : lignes `monitor = <nom>,<res>,<pos>,<scale>` de
    $XDG_CONFIG_HOME/hypr/hyprland.conf. Best-effort : 0 si absent."""
    cfg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config")
    path = os.path.join(cfg, "hypr", "hyprland.conf")
    best = 0.0
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line.startswith("monitor"):
                    continue
                parts = [p.strip() for p in line.split("=", 1)[1].split(",")]
                if len(parts) >= 4:
                    try:
                        best = max(best, float(parts[3]))
                    except ValueError:
                        pass
    except OSError:
        return 0.0
    return best


def _sway_scale() -> float:
    """Scale Sway/wlroots via `swaymsg -t get_outputs -r` (champ `scale`).
    Best-effort : 0 si indisponible."""
    import json
    import subprocess
    try:
        out = subprocess.run(["swaymsg", "-t", "get_outputs", "-r"],
                             capture_output=True, text=True, timeout=2).stdout
        data = json.loads(out)
        scales = [float(o["scale"]) for o in data
                  if isinstance(o, dict) and o.get("scale")]
        return max(scales) if scales else 0.0
    except Exception:  # noqa: BLE001 — best-effort
        return 0.0


def detect_scale(root: _Root, override: float | None = None) -> float:
    """Facteur d'échelle de l'OS : override > max(env JARVIS_SCALE/GDK_SCALE/QT,
    gsettings GNOME, monitors.xml GNOME>=46, Hyprland, Sway, DPI X11).
    JARVIS_DEBUG_SCALE=1 affiche les hints détectés sur stderr."""
    if override is not None:
        return _clamp(override)
    env_hints = [_env_scale(v) for v in
                 ("JARVIS_SCALE", "GDK_SCALE", "QT_SCREEN_SCALE_FACTORS")]
    gs, mx, hy, sw = (_gsettings_scale(), _monitors_xml_scale(),
                      _hypr_scale(), _sway_scale())
    dpi = root.winfo_fpixels("1i")
    hints = [1.0, *env_hints, gs, mx, hy, sw]
    if dpi > 0:
        hints.append(dpi / BASE_DPI)
    scale = _clamp(max(hints))
    if os.environ.get("JARVIS_DEBUG_SCALE"):
        import sys
        print(f"[jarvis] scale: env={env_hints} gsettings={gs} monitors.xml={mx} "
              f"hypr={hy} sway={sw} dpi={dpi:.1f} -> scale={scale}",
              file=sys.stderr)
    return scale


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


def ensure_fonts(root) -> None:
    """S'assure que les polices embarquées de CTk (Roboto) sont connues de Tk.

    Sous X11, CTk copie ses .ttf dans ~/.fonts/ mais ne lance pas fc-cache :
    si le cache fontconfig de XWayland n'est pas rafraîchi, Tk ne voit que
    'fixed' (tkfont.families() == ['fixed']) et ignore tout spec 'Roboto -44'
    -> texte minuscule. On copie les .ttf CTk dans ~/.fonts/ (idempotent) et
    on lance fc-cache -f si dispo. Best-effort."""
    import shutil
    import subprocess
    from pathlib import Path as _P
    try:
        import customtkinter as ctk
        assets = _P(ctk.__file__).parent / "assets" / "fonts"
        dst_dir = _P(os.path.expanduser("~/.fonts"))
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = False
        if assets.is_dir():
            for ttf in assets.rglob("*.ttf"):
                dst = dst_dir / ttf.name
                if not dst.exists() or dst.stat().st_size != ttf.stat().st_size:
                    shutil.copy2(ttf, dst)
                    copied = True
        if copied:
            try:
                subprocess.run(["fc-cache", "-f"], capture_output=True, timeout=10)
            except Exception:  # noqa: BLE001
                pass
        # Enregistrer aussi le dossier dans le font path du SERVEUR X
        # (XWayland a son propre font path, indépendant de fontconfig) :
        # sans ça, tkfont.families() ne voit que 'fixed' et Tk ignore 'Roboto'.
        if os.environ.get("DISPLAY"):
            try:
                # mkfontscale génère fonts.dir/fonts.scale (obligatoire pour
                # que le serveur X accepte le dossier), puis on l'ajoute au
                # font path et on rehash.
                subprocess.run(["mkfontscale", str(dst_dir)],
                               capture_output=True, timeout=10)
                subprocess.run(["mkfontdir", str(dst_dir)],
                               capture_output=True, timeout=10)
                subprocess.run(["xset", "fp+", str(dst_dir)],
                               capture_output=True, timeout=5)
                subprocess.run(["xset", "fp", "rehash"],
                               capture_output=True, timeout=5)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 — best-effort
        pass


def setup_window(root, scale: float) -> None:
    """Applique le scaling puis configure la fenêtre (géométrie adaptative,
    redimensionnable).

    Sous GNOME Wayland scale 2 (XWayland non scalé, Xft.dpi=192) : Tk travaille
    en pixels physiques (winfo_screenwidth = 2880) ; CTk set_widget_scaling(scale)
    scale dimensions et polices (font 22 -> -44 px). --scale / JARVIS_SCALE forcent.

    Ordre important : CTk bloque maxsize à la taille courante dans son
    _set_scaling (appelé par set_window_scaling) — il faut donc déverrouiller
    maxsize APRÈS les set_*_scaling et AVANT geometry/minsize, sinon la
    fenêtre reste coincée à 600x500 et non redimensionnable.
    """
    import customtkinter as ctk

    ensure_fonts(root)
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    root.maxsize(root.winfo_screenwidth(), root.winfo_screenheight())
    root.minsize(*adaptive_minsize())
    root.geometry(adaptive_geometry(root))
    root.resizable(True, True)
