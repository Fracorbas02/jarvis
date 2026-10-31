"""Application desktop de Jarvis — point d'entrée `uv run jarvis-app`.

UI native customtkinter (dark mode) : sidebar (mode / modèle / serveur /
outils) + zone principale (chat LLM, TTS, STT). Toute la logique async
(LLM + MCP) tourne dans une boucle asyncio sur un thread dédié ; les
événements remontent au thread Tk via une queue (voir controller.py).
"""

from __future__ import annotations

import argparse

import customtkinter as ctk

ACCENT = "#4cc2ff"
BG = "#0d0f13"
PANEL = "#16181d"
PANEL_2 = "#121419"


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis-app",
                                     description="Application desktop de Jarvis")
    parser.add_argument("--config", default=None, help="chemin vers config.toml")
    args = parser.parse_args()

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Jarvis")
    root.geometry("1100x700")
    root.minsize(860, 560)
    root.configure(fg_color=BG)

    ctk.CTkLabel(root, text="Jarvis — application desktop (scaffold)",
                 font=ctk.CTkFont(size=20, weight="bold")).pack(expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
