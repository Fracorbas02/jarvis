"""Application desktop de Jarvis — point d'entrée `uv run jarvis-app`.

UI native customtkinter (dark mode) : sidebar (mode / modèle / serveur /
outils) + zone principale (chat LLM, TTS, STT). Toute la logique async
(LLM + MCP) tourne dans une boucle asyncio sur un thread dédié ; les
événements remontent au thread Tk via une queue (voir controller.py).
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import tkinter as tk

import customtkinter as ctk

from .chat_panel import ChatPanel
from .controller import DesktopController, default_repo_root
from .scaling import detect_scale, setup_window
from .sidebar import Sidebar
from .theme import BG, PANEL
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
        # PanedWindow = Tk brut : ses dimensions (sashwidth, width, minsize) sont
        # en unités LOGIQUES (sous Wayland scale 2, 1 unité = 2 px physiques),
        # donc on ne multiplie PAS par scale ici (CTk le fait déjà pour ses widgets).
        root.resizable(True, True)
        sash = max(3, round(6))
        self.pw = tk.PanedWindow(root, orient="horizontal", bg=BG, bd=0,
                                 sashwidth=sash, sashpad=0, opaqueresize=True,
                                 handlepad=0)
        self.pw.pack(fill="both", expand=True)

        self.sidebar = Sidebar(self.pw, self.ctrl,
                               on_mode_change=self.show_mode,
                               on_reset=self.on_reset)
        self.pw.add(self.sidebar, width=SIDEBAR_W, minsize=SIDEBAR_MIN)

        self.main = ctk.CTkFrame(self.pw, fg_color=BG, corner_radius=0)
        self.pw.add(self.main, minsize=MAIN_MIN)
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # panneaux par mode
        self.panels: dict[str, ctk.CTkFrame] = {}
        self.chat = ChatPanel(self.main, self.ctrl, scale=scale)
        self.panels["llm"] = self.chat
        for mode in ("tts", "stt"):
            self.panels[mode] = VoicePanel(self.main, self.ctrl, mode)
        for f in self.panels.values():
            f.grid(row=0, column=0, sticky="nsew")
        self.show_mode(self.ctrl.mode)

        self.ctrl.start()
        self.root.after(50, self.drain_events)
        self.root.after(800, self.ctrl.refresh_status)
        self.root.after(5000, self._periodic_status)
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

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
                                           ev["models"])
        elif kind in ("reasoning", "content", "tool_call", "done", "error"):
            self.chat.handle_event(ev)

    def show_error(self, message: str) -> None:
        ctk.CTkLabel(self.main, text=f"Erreur de configuration : {message}",
                     text_color="#f85149").place(relx=0.5, rely=0.4,
                                                  anchor="center")

    def _periodic_status(self) -> None:
        self.ctrl.refresh_status()
        self.root.after(5000, self._periodic_status)

    # ---------- modes ----------

    def show_mode(self, mode: str) -> None:
        for m, panel in self.panels.items():
            if m == mode:
                panel.grid(row=0, column=0, sticky="nsew")
            else:
                panel.grid_remove()

    def on_reset(self) -> None:
        self.chat.clear()

    def on_quit(self) -> None:
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
                f"screenmm={root.winfo_screenmmwidth()}x{root.winfo_screenmmheight()}mm "
                f"pw={app.pw.winfo_width()}x{app.pw.winfo_height()} "
                f"sidebar_w={app.sidebar.winfo_width()} chat_w={app.chat.winfo_width()}")
            # hauteur réelle d'un label JARVIS (titre sidebar)
            for c in app.sidebar.winfo_children():
                if isinstance(c, _ctk.CTkLabel) and c.cget("text") == "JARVIS":
                    lines.append(
                        f"[jarvis] debug: label JARVIS winfo_h={c.winfo_height()} "
                        f"reqh={c.winfo_reqheight()} font={c._label.cget('font')}")
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

    root = ctk.CTk()
    root.title("Jarvis")

    # Scaling HiDPI (GNOME) + fenêtre adaptative et redimensionnable.
    # setup_window applique set_*_scaling AVANT la création des widgets
    # (scale dimensions + polices) et déverrouille maxsize (CTk le bloque).
    scale = detect_scale(root, args.scale)
    setup_window(root, scale)
    root.configure(fg_color=BG)

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
