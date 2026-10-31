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
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
