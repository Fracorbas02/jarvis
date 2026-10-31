"""Application desktop de Jarvis — point d'entrée `uv run jarvis-app`.

UI native customtkinter (dark mode) : sidebar (mode / modèle / serveur /
outils) + zone principale (chat LLM, TTS, STT). Toute la logique async
(LLM + MCP) tourne dans une boucle asyncio sur un thread dédié ; les
événements remontent au thread Tk via une queue (voir controller.py).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import queue
import sys
import tkinter as tk

import customtkinter as ctk

from .chat_panel import ChatPanel
from .controller import DesktopController, default_repo_root
from .scaling import _monitors_xml_layout, detect_scale, setup_window
from .sidebar import Sidebar
from .theme import ACCENT, BG, PANEL
from .voice_panel import VoicePanel

SIDEBAR_W = 280
SIDEBAR_MIN = 230
MAIN_MIN = 520


class DesktopApp:
    def __init__(self, root: ctk.CTk, config_path: str | None,
                 scale: float = 1.0):
        self.root = root
        self.scale = scale
        self.ctrl = DesktopController(config_path)

        # Fenêtre redimensionnable ; la sidebar est sur un sash déplaçable.
        root.resizable(True, True)
        sash = max(3, round(6 * scale))
        self.pw = tk.PanedWindow(root, orient="horizontal", bg=BG, bd=0,
                                 sashwidth=sash, sashpad=0, opaqueresize=True,
                                 handlepad=0)
        self.pw.pack(fill="both", expand=True)

        self.sidebar = Sidebar(self.pw, self.ctrl,
                               on_mode_change=self.show_mode,
                               on_reset=self.on_reset)
        self.pw.add(self.sidebar, width=int(SIDEBAR_W * scale),
                    minsize=int(SIDEBAR_MIN * scale))

        self.main = ctk.CTkFrame(self.pw, fg_color=BG, corner_radius=0)
        self.pw.add(self.main, minsize=int(MAIN_MIN * scale))
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # panneaux par mode
        self.panels: dict[str, ctk.CTkFrame] = {}
        self.chat = ChatPanel(self.main, self.ctrl, scale=scale)
        self.panels["llm"] = self.chat
        for mode in ("tts", "stt"):
            self.panels[mode] = VoicePanel(self.main, self.ctrl, mode)
        from .settings_panel import SettingsPanel
        self.panels["settings"] = SettingsPanel(self.main, self.ctrl)
        for f in self.panels.values():
            f.grid(row=0, column=0, sticky="nsew")
        self.show_mode(self.ctrl.mode)

        self.ctrl.start()
        self.root.after(50, self.drain_events)
        self.root.after(300, self._maybe_gpu_popup)
        self.root.after(800, self.ctrl.refresh_status)
        self.root.after(5000, self._periodic_status)
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

    def _maybe_gpu_popup(self) -> None:
        """Premier lancement : si plusieurs GPU détectés et aucun choix
        enregistré, pop-up MODAL demandant lequel utiliser (persisté en JSON).
        S'affiche au premier plan, centré sur la fenêtre principale."""
        try:
            from ..gpu import (detect_gpus, env_for_device, gpu_choice_path,
                               load_gpu_choice, save_gpu_choice)
            if load_gpu_choice().get("device"):
                return
            gpus = [g for g in detect_gpus() if g.backend != "cpu"]
            if len(gpus) < 2:
                return
            top = ctk.CTkToplevel(self.root)
            top.title("Choisir le GPU")
            top.transient(self.root)           # liée à la fenêtre mère
            top.attributes("-topmost", True)   # visible même en multi-écrans
            # centrée sur la fenêtre principale (pas sur l'écran virtuel entier)
            self.root.update_idletasks()
            w, h = 500, 300
            x = self.root.winfo_x() + max((self.root.winfo_width() - w) // 2, 0)
            y = self.root.winfo_y() + max((self.root.winfo_height() - h) // 2, 0)
            top.geometry(f"{w}x{h}+{x}+{y}")
            top.lift()
            top.focus_force()
            top.grab_set()                     # modal : oblige à répondre oui/non

            dev_by_label = {f"{g.name}  ({g.device})": g.device for g in gpus}
            ctk.CTkLabel(top,
                         text="Plusieurs GPU détectés. Lequel utiliser pour les modèles locaux (LLM, STT, TTS) ?",
                         font=ctk.CTkFont(size=14, weight="bold"),
                         wraplength=460).pack(pady=(16, 8))
            combo = ctk.CTkComboBox(top, values=list(dev_by_label), width=440)
            combo.set(next(iter(dev_by_label)))
            combo.pack(pady=4)
            status = ctk.CTkLabel(top, text="", font=ctk.CTkFont(size=11),
                                  text_color="#9aa4b0", wraplength=460)
            status.pack(pady=2)

            def _ok():
                dev = dev_by_label.get(combo.get(), combo.get())
                save_gpu_choice(dev, env_for_device(dev))
                status.configure(text=f"Enregistré : {dev} -> {gpu_choice_path()}\n"
                                      "Appliqué au prochain démarrage des serveurs locaux.")
                btn_ok.configure(state="disabled")
                btn_no.configure(state="disabled")
                top.after(2500, top.destroy)

            def _cancel():
                top.grab_release()
                top.destroy()

            btns = ctk.CTkFrame(top, fg_color="transparent")
            btns.pack(pady=10)
            btn_ok = ctk.CTkButton(btns, text="Oui, utiliser ce GPU",
                                   fg_color=ACCENT, text_color="#06121a",
                                   command=_ok)
            btn_ok.pack(side="left", padx=6)
            btn_no = ctk.CTkButton(btns, text="Non (plus tard)", command=_cancel)
            btn_no.pack(side="left", padx=6)
            top.protocol("WM_DELETE_WINDOW", _cancel)
        except Exception:  # noqa: BLE001 — jamais bloquant, mais on logue
            import traceback
            print("[jarvis] pop-up GPU impossible :", traceback.format_exc(),
                  file=sys.stderr)

    # ---------- événements ----------

    def drain_events(self) -> None:
        try:
            while True:
                self.handle_event(self.ctrl.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(50, self.drain_events)

    def handle_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "backend_ready":
            self.sidebar.set_tools(ev["tools"])
            self.ctrl.refresh_status()
        elif kind == "backend_error":
            self.show_error(ev["message"])
        elif kind in ("server_status", "server_started", "server_stopped"):
            self.sidebar.set_server_status(ev["state"], ev["host"], ev["port"],
                                           ev["models"], ev.get("detail", ""))
        elif kind == "voice_status":
            panel = self.panels.get(ev["mode"])
            if isinstance(panel, VoicePanel):
                panel.set_voice_status(ev["state"], ev["host"], ev["port"],
                                       ev.get("detail", ""))
        elif kind == "tts_done":
            panel = self.panels.get("tts")
            if isinstance(panel, VoicePanel):
                panel.on_tts_done(ev["path"])
        elif kind == "tts_error":
            panel = self.panels.get("tts")
            if isinstance(panel, VoicePanel):
                panel.on_tts_error(ev["message"])
        elif kind in ("stt_done", "stt_recorded"):
            panel = self.panels.get("stt")
            if isinstance(panel, VoicePanel) and kind == "stt_done":
                panel.on_stt_done(ev["text"])
        elif kind in ("stt_error", "wav_error"):
            panel = self.panels.get("stt") if kind == "stt_error" else self.panels.get("tts")
            if isinstance(panel, VoicePanel):
                panel.on_stt_error(ev["message"]) if kind == "stt_error" else \
                    panel.on_tts_error(ev["message"])
        elif kind in ("reasoning", "content", "tool_call", "done", "error"):
            self.chat.handle_event(ev)

    def show_error(self, message: str) -> None:
        ctk.CTkLabel(self.main, text=f"Erreur de configuration : {message}",
                     text_color="#f85149", wraplength=500).place(relx=0.5, rely=0.4,
                                                                 anchor="center")

    def _periodic_status(self) -> None:
        self.ctrl.refresh_status()
        for mode in ("tts", "stt"):
            if mode == self.ctrl.mode:
                self.ctrl.refresh_voice_status(mode)
        self.root.after(5000, self._periodic_status)

    # ---------- modes ----------

    def show_mode(self, mode: str) -> None:
        for m, panel in self.panels.items():
            if m == mode:
                panel.grid(row=0, column=0, sticky="nsew")
            else:
                panel.grid_remove()
        if mode in ("tts", "stt"):
            panel = self.panels[mode]
            if isinstance(panel, VoicePanel):
                panel.refresh_status()

    def on_reset(self) -> None:
        self.chat.clear()

    def on_quit(self) -> None:
        from ..voice import stop_all_voice_servers
        if self.ctrl.backend is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    stop_all_voice_servers(self.ctrl.backend.cfg.voice),
                    self.ctrl.loop).result(timeout=10)
            except Exception:  # noqa: BLE001 — arrêt best-effort
                pass
        self.ctrl.stop()
        self.root.destroy()


def _debug_scale(root, scale: float, app=None) -> None:
    """Dump de diagnostic du scaling (JARVIS_DEBUG_SCALE=1) : toujours sur
    stderr + /tmp/jarvis-debug.log, jamais bloquant (try/except)."""
    import sys
    import traceback
    import customtkinter as _ctk
    lines: list[str] = []
    try:
        from customtkinter.windows.widgets.scaling.scaling_tracker import ScalingTracker
        lbl = _ctk.CTkLabel(root, text="x", font=_ctk.CTkFont(size=13))
        root.update_idletasks()
        lines.append(
            f"[jarvis] debug: scale={scale} "
            f"widget_scaling={ScalingTracker.get_widget_scaling(lbl)} "
            f"window_scaling={ScalingTracker.get_window_scaling(root)} "
            f"screen={root.winfo_screenwidth()}x{root.winfo_screenheight()} "
            f"dpi={root.winfo_fpixels('1i'):.1f} "
            f"tk_scaling={root.tk.call('tk', 'scaling')} "
            f"label13_font={lbl._label.cget('font')}")
        lbl.destroy()
    except Exception:
        lines.append("[jarvis] debug: ERREUR dans le dump:")
        lines.extend(traceback.format_exc().splitlines())
    if app is not None:
        try:
            def walk(w, depth=0):
                if depth > 4:
                    return
                for c in w.winfo_children():
                    if isinstance(c, _ctk.CTkLabel):
                        txt = str(c.cget("text"))[:20]
                        lines.append(
                            f"[jarvis] debug: label {txt!r} font={c._label.cget('font')}")
                    walk(c, depth + 1)
            walk(app.root)
            tb = app.chat.text._textbox
            lines.append(
                f"[jarvis] debug: chat default font={tb.cget('font')} "
                f"tag reasoning font={tb.tag_cget('reasoning', 'font')}")
            lines.append(
                f"[jarvis] debug: root winfo={root.winfo_width()}x{root.winfo_height()} "
                f"req={root.winfo_reqwidth()}x{root.winfo_reqheight()} "
                f"screen={root.winfo_screenwidth()}x{root.winfo_screenheight()} "
                f"pointerxy={root.winfo_pointerxy()} "
                f"geometry={root.geometry()} "
                f"monitors={_monitors_xml_layout()} "
                f"screenmm={root.winfo_screenmmwidth()}x{root.winfo_screenmmheight()}mm "
                f"pw={app.pw.winfo_width()}x{app.pw.winfo_height()} "
                f"sidebar_w={app.sidebar.winfo_width()} chat_w={app.chat.winfo_width()}")
            try:
                import tkinter.font as tkfont
                fams = sorted(tkfont.families())
                lines.append(f"[jarvis] debug: font families: Roboto={'Roboto' in fams} "
                             f"n={len(fams)} sample={fams[:8]}")
                import glob as _glob
                home_fonts = _glob.glob(os.path.expanduser("~/.fonts/**/*.ttf"), recursive=True) + \
                             _glob.glob(os.path.expanduser("~/.fonts/*.ttf"))
                lines.append(f"[jarvis] debug: ~/.fonts ttf={home_fonts}")
                xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
                lines.append(f"[jarvis] debug: XDG_DATA_HOME={xdg} "
                             f"fonts dir exists={os.path.isdir(os.path.join(xdg, 'fonts'))}")
                lines.append(f"[jarvis] debug: DISPLAY={os.environ.get('DISPLAY')} "
                             f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')}")
                import subprocess as _sp
                try:
                    out = _sp.run(["xlsfonts"], capture_output=True, text=True, timeout=5).stdout
                    n = len(out.splitlines())
                    rob = [l for l in out.splitlines() if "oboto" in l][:3]
                    lines.append(f"[jarvis] debug: xlsfonts n={n} roboto={rob}")
                except Exception as e:
                    lines.append(f"[jarvis] debug: xlsfonts erreur: {e}")
                for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
                    f = tkfont.nametofont(name)
                    lines.append(
                        f"[jarvis] debug: named font {name}: actual={f.actual()} "
                        f"measure(JARVIS)={f.measure('JARVIS')}px")
                # tester un spec pixel -44 avec famille connue vs Roboto
                for spec in (("Roboto", -44, "bold"), ("DejaVu Sans", -44, "bold"), ("fixed", -44)):
                    ff = tkfont.Font(family=spec[0], size=spec[1], weight=spec[2] if len(spec) > 2 else "normal")
                    lines.append(f"[jarvis] debug: Font{spec} actual={ff.actual()} measure={ff.measure('JARVIS')}px")
                    ff.destroy()
            except Exception:
                pass
            # hauteur réelle d'un label JARVIS (titre sidebar)
            import tkinter.font as _tkf
            for c in app.sidebar.winfo_children():
                if isinstance(c, _ctk.CTkLabel) and c.cget("text") == "JARVIS":
                    spec = c._label.cget("font")
                    f = _tkf.Font(font=spec)
                    lines.append(
                        f"[jarvis] debug: label JARVIS winfo_h={c.winfo_height()} "
                        f"reqh={c.winfo_reqheight()} spec={spec} "
                        f"font_actual={f.actual()} measure={f.measure('JARVIS')}px")
                    break
        except Exception:
            lines.append("[jarvis] debug: ERREUR scan widgets:")
            lines.extend(traceback.format_exc().splitlines())
    text = "\n".join(lines)
    sys.stderr.write(text + "\n")
    sys.stderr.flush()
    try:
        with open("/tmp/jarvis-debug.log", "w", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass


def _apply_cursor_scaling(root, scale: float) -> None:
    """Curseur de la souris uniforme dans toute la fenêtre.

    Deux écueils sous X11/XWayland HiDPI :
    - les widgets tk bruts (tkinter.Text du chat, tkinter.Entry de la saisie)
      ont un curseur nommé par défaut ("xterm") que libXcursor ne scale pas
      quand XCURSOR_SIZE n'est pas fixé avant le chargement de la lib ;
    - les canvas CTk laissés sur cursor="" rendent le curseur système brut
      (souvent minuscule) au lieu d'hériter du curseur de la racine.
    On force donc un curseur nommé sur la racine (puis propagé par héritage)
    et on remplace les curseurs vides des widgets internes par "arrow".
    """
    import tkinter as tk

    # default pour tous les widgets créés ensuite (pop-up, etc.) ; les
    # widgets qui posent leur curseur explicitement (boutons "hand2") gardent
    # leur valeur (config widget > base d'options)
    root.option_add("*cursor", "arrow", 90)

    def _walk(w) -> None:
        try:
            if str(w.cget("cursor")) == "":
                w.configure(cursor="arrow")
        except tk.TclError:
            pass
        for c in w.winfo_children():
            _walk(c)

    root.configure(cursor="arrow")
    root.after(100, lambda: _walk(root))


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis-app",
                                     description="Application desktop de Jarvis")
    parser.add_argument("--config", default=None, help="chemin vers config.toml")
    parser.add_argument("--scale", type=float, default=None,
                        help="facteur d'échelle UI (défaut: auto, détecté depuis "
                             "l'écran — utile avec le scaling GNOME)")
    args = parser.parse_args()

    # Les commandes MCP de la config sont relatives à la racine du dépôt :
    # s'y placer si on trouve config.toml à la racine (lancement hors du repo).
    repo = default_repo_root()
    if args.config is None and (repo / "config.toml").is_file():
        os.chdir(repo)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Curseur souris proportionnel au scale de l'écran, AVANT la fenêtre
    # principale : libXcursor lit XCURSOR_SIZE à la première utilisation de
    # chaque connexion X (info display mise en cache par Display*). Une
    # fenêtre sonde jetable calcule le scale (DPI) ; la vraie fenêtre
    # lira ensuite XCURSOR_SIZE avec la bonne valeur. Sous XWayland scale 2
    # sans cette variable, le curseur reste en 24px (minuscule sur HiDPI).
    if "XCURSOR_SIZE" not in os.environ:
        try:
            import tkinter as _tk
            _probe = _tk.Tk()
            _s = detect_scale(_probe, args.scale)
            _probe.destroy()
            os.environ["XCURSOR_SIZE"] = str(int(24 * _s))
        except Exception:  # noqa: BLE001
            pass

    root = ctk.CTk()
    root.title("Jarvis")

    # Scaling HiDPI (GNOME) + fenêtre adaptative et redimensionnable.
    # setup_window applique set_*_scaling AVANT la création des widgets
    # (scale dimensions + polices) et déverrouille maxsize (CTk le bloque).
    scale = detect_scale(root, args.scale)
    setup_window(root, scale)
    root.configure(fg_color=BG)
    _apply_cursor_scaling(root, scale)

    app = DesktopApp(root, args.config, scale=scale)
    if os.environ.get("JARVIS_DEBUG_SCALE"):
        _debug_scale(root, scale, app)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
