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
from pathlib import Path

import customtkinter as ctk

from ..gui.models import MODES
from .chat_panel import ChatPanel
from .controller import DesktopController, default_repo_root
from .sidebar import Sidebar
from .theme import ACCENT, BG, PANEL_2, TEXT_DIM


class DesktopApp:
    def __init__(self, root: ctk.CTk, config_path: str | None):
        self.root = root
        self.ctrl = DesktopController(config_path)

        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(root, self.ctrl,
                               on_mode_change=self.show_mode,
                               on_reset=self.on_reset)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.main = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # panneaux par mode : chat LLM branché, TTS/STT en placeholder
        self.panels: dict[str, ctk.CTkFrame] = {}
        self.chat = ChatPanel(self.main, self.ctrl)
        self.panels["llm"] = self.chat
        for mode in ("tts", "stt"):
            meta = MODES[mode]
            f = ctk.CTkFrame(self.main, fg_color=BG, corner_radius=0)
            ctk.CTkLabel(f, text=f"{meta['label']} — {meta['description']}",
                         font=ctk.CTkFont(size=16), text_color=TEXT_DIM).place(
                relx=0.5, rely=0.5, anchor="center")
            self.panels[mode] = f
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
                     text_color="#f85149", wraplength=500).place(relx=0.5, rely=0.4,
                                                                 anchor="center")

    def _periodic_status(self) -> None:
        self.ctrl.refresh_status()
        self.root.after(5000, self._periodic_status)

    # ---------- modes ----------

    def show_mode(self, mode: str) -> None:
        for m, panel in self.panels.items():
            if m == mode:
                panel.tkraise()
            else:
                panel.grid_remove()
        # re-grid the visible one (tkraise ne gère pas grid)
        for m, panel in self.panels.items():
            if m == mode:
                panel.grid(row=0, column=0, sticky="nsew")

    def on_reset(self) -> None:
        self.chat.clear()

    def on_quit(self) -> None:
        self.ctrl.stop()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis-app",
                                     description="Application desktop de Jarvis")
    parser.add_argument("--config", default=None, help="chemin vers config.toml")
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
    root.geometry("1100x700")
    root.minsize(860, 560)
    root.configure(fg_color=BG)

    app = DesktopApp(root, args.config)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
