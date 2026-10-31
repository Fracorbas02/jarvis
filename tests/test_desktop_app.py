"""Test d'intégration de l'application desktop sous Xvfb, contre le mock LLM.

Pilote la vraie fenêtre customtkinter : attend que le backend soit prêt,
envoie un message dans le chat, vérifie que le raisonnement grisé, l'appel
d'outil et la réponse streamés apparaissent dans la zone de texte.

Usage: xvfb-run -a uv run python tests/test_desktop_app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import customtkinter as ctk  # noqa: E402

from jarvis.desktop.app import DesktopApp  # noqa: E402
from mock_llm import start_mock  # noqa: E402

CHATMOCK = ROOT / "tests" / "config.chatmock.toml"


def main() -> int:
    srv, thread = start_mock(18082)
    os.environ["JARVIS_CONFIG"] = str(CHATMOCK)
    try:
        ctk.set_appearance_mode("dark")
        root = ctk.CTk()
        root.title("Jarvis-test")
        root.geometry("1100x700")
        app = DesktopApp(root, str(CHATMOCK))

        result = {"code": 1, "detail": ""}
        state = {"sent": False, "t0": 0.0}

        def finish(code: int, detail: str) -> None:
            result["code"] = code
            result["detail"] = detail
            root.quit()

        def step() -> None:
            if not state["sent"]:
                if app.ctrl.backend is not None and app.ctrl.backend.started:
                    app.chat.entry.insert(0, "fais un ping vers 127.0.0.1")
                    app.chat.send()
                    state["sent"] = True
                    state["t0"] = time.time()
                    root.after(300, step)
                else:
                    root.after(300, step)
                return
            content = app.chat.text.get("1.0", "end")
            if "Résultat du ping" in content:
                ok = "réflexion" in content and "echo__ping" in content
                # switch de mode vers le panneau TTS
                app.show_mode("tts")
                root.update_idletasks()
                tts_visible = app.panels["tts"].winfo_ismapped()
                def all_labels(w):
                    out = [w.cget("text")] if isinstance(w, ctk.CTkLabel) else []
                    for c in w.winfo_children():
                        out.extend(all_labels(c))
                    return out
                labels = all_labels(app.panels["tts"])
                has_empty_state = any("Modèles détectés" in str(t) for t in labels)
                ok = ok and tts_visible and has_empty_state
                finish(0 if ok else 1,
                       f"reçu: réflexion={'réflexion' in content} "
                       f"outil={'echo__ping' in content} "
                       f"réponse={'Résultat du ping' in content} "
                       f"tts_visible={tts_visible} empty_state={has_empty_state}")
                return
            if time.time() - state["t0"] > 60:
                finish(1, f"timeout, contenu={content[-400:]!r}")
                return
            root.after(300, step)

        root.after(500, step)
        root.mainloop()
        app.ctrl.stop()
        root.destroy()
        print("OK" if result["code"] == 0 else "FAIL", "chat desktop:", result["detail"])
        return result["code"]
    finally:
        srv.shutdown()
        thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
