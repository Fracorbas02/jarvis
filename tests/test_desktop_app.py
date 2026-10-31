"""Test d'intégration de l'application desktop sous Xvfb, contre le mock LLM.

Pilote la vraie fenêtre customtkinter : attend que le backend soit prêt,
envoie un message dans le chat, vérifie que le raisonnement grisé, l'appel
d'outil et la réponse streamés apparaissent dans la zone de texte, puis un
2e tour « markdown » vérifie le rendu Markdown (gras/titre/code sans
marqueurs) et le curseur nommé de la zone de chat.

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
                # --- rendu Markdown de la réponse (2e tour) ---
                tb = app.chat.text._textbox
                md_ok = {"n": 0}

                def md_step() -> None:
                    tb2 = app.chat.text._textbox
                    full = app.chat.text.get("1.0", "end")
                    has_md = ("texte en gras" in full and "**texte en gras**" not in full
                              and "item un" in full and "• item un" in full
                              and "print('hi')" in full and "```" not in full)
                    # tags markdown appliqués dans la zone du texte
                    tag_ok = False
                    for tag in ("md_bold", "md_heading", "md_code_block"):
                        if tb2.tag_ranges(tag):
                            tag_ok = True
                            break
                    # curseur nommé sur la zone de chat (pas le défaut xterm)
                    cursor_ok = str(tb2.cget("cursor")) == "arrow"
                    if has_md and tag_ok and cursor_ok:
                        # switch de mode vers le panneau TTS (vérif d'affichage)
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
                               f"markdown={has_md} tags={tag_ok} cursor={cursor_ok} "
                               f"tts_visible={tts_visible} empty_state={has_empty_state}")
                        return
                    md_ok["n"] += 1
                    if md_ok["n"] > 60 or time.time() - state["t0"] > 90:
                        finish(1, f"markdown: has_md={has_md} tag_ok={tag_ok} "
                                  f"cursor={cursor_ok} contenu={full[-400:]!r}")
                        return
                    root.after(300, md_step)

                app.chat.entry.configure(state="normal")
                app.chat.entry.insert(0, "donne une réponse en markdown")
                app.chat.busy = False
                app.chat.send()
                state["t0"] = time.time()
                root.after(300, md_step)
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
