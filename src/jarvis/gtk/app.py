"""Application GTK4 / libadwaita de Jarvis (`uv run jarvis-gtk`).

Refonte compl\u00e8te de l'UI, sans toucher au reste du projet :
- page d'accueil : un seul bouton \u00ab Lancer \u00bb (d\u00e9marre le llama-server si
  besoin, puis ouvre la conversation) ;
- page conversation : on parle au mod\u00e8le (bouton micro -> STT -> tour LLM),
  le prompt texte vit dans le menu (Ctrl+P) ;
- onglet r\u00e9glages : th\u00e8me, GPU, taille de contexte, outils MCP.

Toute la logique async est dans backend.GtkBackend (thread asyncio + queue) ;
les \u00e9v\u00e9nements remontent au thread GTK via GLib.idle_add.
"""

from __future__ import annotations

import argparse
import logging
import queue

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

from ..gpu import detect_gpus, load_gpu_choice
from .backend import (GtkBackend, load_ui_settings, save_ui_settings)
from .widgets import (MessagesColumn, assistant_message, error_message,
                      user_message, _css)

log = logging.getLogger("jarvis.gtk")

APP_ID = "io.github.Fracorbas02.Jarvis"


class JarvisWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application, backend: GtkBackend):
        super().__init__(application=app, title="Jarvis")
        self.set_default_size(960, 640)
        self.backend = backend
        self.settings = load_ui_settings()
        self.tts_enabled = bool(self.settings.get("tts", False))
        self._assistant_box = None
        self._tool_lines: dict[str, Gtk.Label] = {}
        self._busy = False
        self._status_state = "unknown"
        self._ctx = self.settings.get("ctx_size", 16384)

        _css()
        self._build_actions()
        self._build_ui()
        self._apply_theme()
        self._poll_events()
        backend.start()
        GLib.timeout_add_seconds(2, self._refresh_status)
        self._refresh_status()

    # ---------- actions ----------

    def _build_actions(self) -> None:
        app = self.get_application()

        def add(name: str, cb, accel: str | None = None, stateful: bool = False,
                state=None):
            act = Gio.SimpleAction.new_stateful(name, None, state) if stateful else \
                Gio.SimpleAction.new(name, None)
            if stateful:
                act.connect("activate", cb)
            else:
                act.connect("activate", lambda a, p: cb())
            app.add_action(act)
            if accel:
                app.set_accels_for_action(f"app.{name}", [accel])

        add("prompt", self._on_prompt, "<Ctrl>p")
        add("new-session", self._on_new_session, "<Ctrl>n")
        add("sessions", self._on_sessions, "<Ctrl>o")
        add("quit", lambda: self.close(), "<Ctrl>q")
        add("about", self._on_about)
        add("tts", self._on_tts_toggle, stateful=True,
            state=GLib.Variant.new_boolean(self.tts_enabled))
        add("home", lambda: self.stack.set_visible_child_name("home"), "<Alt>Home")

    # ---------- UI ----------

    def _build_ui(self) -> None:
        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)

        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)

        # menu principal (prompt, sessions, tts, \u00e0 propos)
        menu = Gio.Menu()
        menu.append("Prompt\u2026", "app.prompt")
        menu.append("Nouvelle conversation", "app.new-session")
        menu.append("Conversations\u2026", "app.sessions")
        menu.append_section(None)
        menu.append("Lire les r\u00e9ponses \u00e0 voix haute", "app.tts")
        menu.append_section(None)
        menu.append("\u00c0 propos", "app.about")
        menu_button = Gtk.MenuButton(menu_model=menu, icon_name="open-menu-symbolic",
                                     tooltip_text="Menu")
        header.pack_start(menu_button)

        self.stack = Adw.ViewStack()
        switcher = Adw.ViewSwitcher(stack=self.stack)
        header.set_title_widget(switcher)

        # statut serveur (popover) + micro
        self.status_button = Gtk.Button(icon_name="network-server-symbolic",
                                        tooltip_text="Statut du serveur LLM")
        self.status_button.connect("clicked", self._on_status_clicked)
        header.pack_end(self.status_button)
        self._build_status_popover()

        self.mic_button = Gtk.Button(icon_name="audio-input-microphone-symbolic",
                                     tooltip_text="Parler \u00e0 Jarvis (micro)")
        self.mic_button.connect("clicked", self._on_mic)
        header.pack_end(self.mic_button)

        # pages
        self.stack.add_titled_with_icon(self._build_home(), "home", "Accueil",
                                        "go-home-symbolic")
        self.stack.add_titled_with_icon(self._build_chat(), "chat", "Conversation",
                                        "emblem-chat-symbolic")
        self.stack.add_titled_with_icon(self._build_settings(), "settings", "R\u00e9glages",
                                        "emblem-system-symbolic")
        toolbar.set_content(self.stack)

        self.toast_overlay = Adw.ToastOverlay()
        self.toast_overlay.set_child(toolbar)
        self.set_content(self.toast_overlay)

    def _build_home(self) -> Gtk.Widget:
        clamp = Adw.Clamp(maximum_size=640, tightening_threshold=480)
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18,
                       vexpand=True, hexpand=True, valign=Gtk.Align.CENTER,
                       halign=Gtk.Align.CENTER)
        page.set_margin_top(48)
        page.set_margin_bottom(48)

        title = Gtk.Label(label="Jarvis", halign=Gtk.Align.CENTER)
        title.add_css_class("jarvis-home-title")
        page.append(title)

        sub = Gtk.Label(label="Assistant local — LLM + outils MCP + voix",
                        halign=Gtk.Align.CENTER)
        sub.add_css_class("dim-label")
        page.append(sub)

        self.home_status = Gtk.Label(label="Vérification du serveur…",
                                     halign=Gtk.Align.CENTER, wrap=True)
        self.home_status.add_css_class("dim-label")
        page.append(self.home_status)

        self.launch_button = Gtk.Button(label="Lancer", halign=Gtk.Align.CENTER)
        self.launch_button.add_css_class("suggested-action")
        self.launch_button.add_css_class("pill")
        page.append(self.launch_button)
        self.launch_button.connect("clicked", self._on_launch)

        # choix du modèle uniquement si le serveur est hors ligne et qu'on a
        # plusieurs modèles locaux (sinon Lancer prend le premier / celui servi)
        self.home_model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,
                                      spacing=8, halign=Gtk.Align.CENTER)
        self.home_model_box.set_visible(False)
        lbl = Gtk.Label(label="Modèle :")
        lbl.add_css_class("dim-label")
        self.home_model_box.append(lbl)
        self.home_model_combo = Gtk.ComboBoxText()
        for m in self.backend.models_llm:
            self.home_model_combo.append(str(m.path), f"{m.name}  ({m.size_human})")
        if self.backend.models_llm:
            self.home_model_combo.set_active(0)
        self.home_model_box.append(self.home_model_combo)
        page.append(self.home_model_box)

        hint = Gtk.Label(label="Le prompt texte est dans le menu (Ctrl+P)",
                         halign=Gtk.Align.CENTER)
        hint.add_css_class("dim-label")
        hint.add_css_class("caption")
        page.append(hint)

        clamp.set_child(page)
        return clamp

    def _build_chat(self) -> Gtk.Widget:
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, vexpand=True)
        self.messages = MessagesColumn()
        outer.append(self.messages.widget)

        bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
                         halign=Gtk.Align.CENTER, margin_top=8, margin_bottom=12)
        self.chat_mic = Gtk.Button(icon_name="audio-input-microphone-symbolic",
                                   tooltip_text="Parler à Jarvis (enregistre puis transcrit)")
        self.chat_mic.add_css_class("pill")
        self.chat_mic.connect("clicked", self._on_mic)
        bottom.append(self.chat_mic)
        self.chat_hint = Gtk.Label(label="Cliquez sur le micro pour parler",
                                   halign=Gtk.Align.CENTER)
        self.chat_hint.add_css_class("dim-label")
        self.chat_hint.add_css_class("caption")
        bottom.append(self.chat_hint)
        outer.append(bottom)
        return outer

    def _build_settings(self) -> Gtk.Widget:
        scroll = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        clamp = Adw.Clamp(maximum_size=720, tightening_threshold=560)
        clamp.set_margin_top(24)
        clamp.set_margin_bottom(24)
        clamp.set_margin_start(16)
        clamp.set_margin_end(16)
        page = Adw.PreferencesPage()
        clamp.set_child(page)
        scroll.set_child(clamp)

        # --- Apparence ---
        g_appearance = Adw.PreferencesGroup(title="Apparence")
        row_theme = Adw.ComboRow(title="Thème")
        theme_model = Gtk.StringList.new(["Système", "Sombre", "Clair"])
        row_theme.set_model(theme_model)
        theme_map = {"system": 0, "dark": 1, "light": 2}
        row_theme.set_selected(theme_map.get(self.settings.get("theme", "system"), 0))
        row_theme.connect("notify::selected", self._on_theme_changed)
        g_appearance.add(row_theme)
        page.add(g_appearance)

        # --- Modèle / GPU / contexte ---
        g_model = Adw.PreferencesGroup(title="Modèle local (llama-server)")
        self.gpu_row = Adw.ComboRow(title="GPU utilisé par les serveurs locaux",
                                    subtitle="Appliqué au prochain démarrage des serveurs (LLM, STT, TTS)")
        self._gpu_devices = detect_gpus()
        gpu_model = Gtk.StringList()
        for g in self._gpu_devices:
            gpu_model.append(f"{g.name}  ({g.device})")
        self.gpu_row.set_model(gpu_model)
        current_gpu = load_gpu_choice().get("device", "") or self.settings.get("gpu", "")
        sel = 0
        for i, g in enumerate(self._gpu_devices):
            if g.device == current_gpu:
                sel = i
                break
        self.gpu_row.set_selected(sel)
        self.gpu_row.connect("notify::selected", self._on_gpu_changed)
        g_model.add(self.gpu_row)

        self.ctx_row = Adw.SpinRow(title="Taille de contexte (tokens)",
                                   subtitle="Passée en --ctx-size au démarrage du serveur depuis l'accueil",
                                   adjustment=Gtk.Adjustment(
                                       value=int(self._ctx), lower=4096, upper=262144,
                                       step_increment=1024, page_increment=8192))
        self.ctx_row.connect("notify::value", self._on_ctx_changed)
        g_model.add(self.ctx_row)

        info = Adw.ActionRow(title="Serveur LLM (config.toml)",
                             subtitle=self.backend.cfg.llm.base_url)
        g_model.add(info)
        page.add(g_model)

        # --- Outils MCP (peuplé quand backend_ready arrive) ---
        self.tools_group = Adw.PreferencesGroup(title="Outils MCP disponibles")
        page.add(self.tools_group)

        # --- Voix ---
        g_voice = Adw.PreferencesGroup(title="Voix")
        row_tts = Adw.SwitchRow(title="Lire les réponses à voix haute (TTS)",
                                active=self.tts_enabled)
        row_tts.connect("notify::active", self._on_tts_switch)
        g_voice.add(row_tts)
        page.add(g_voice)

        return scroll

    def _build_status_popover(self) -> None:
        self.status_popover = Gtk.Popover()
        self.status_popover.set_parent(self.status_button)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10,
                      margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)
        box.set_size_request(280, -1)
        self.pop_status_label = Gtk.Label(xalign=0, wrap=True)
        box.append(self.pop_status_label)
        self.pop_model_combo = Gtk.ComboBoxText()
        for m in self.backend.models_llm:
            self.pop_model_combo.append(str(m.path), f"{m.name}  ({m.size_human})")
        if self.backend.models_llm:
            self.pop_model_combo.set_active(0)
        box.append(self.pop_model_combo)
        self.pop_button = Gtk.Button(label="Démarrer le serveur")
        self.pop_button.connect("clicked", self._on_pop_start_stop)
        box.append(self.pop_button)
        self.status_popover.set_child(box)

    # ---------- réglages ----------

    def _apply_theme(self) -> None:
        sm = Adw.StyleManager.get_default()
        scheme = {"dark": Adw.ColorScheme.FORCE_DARK,
                  "light": Adw.ColorScheme.FORCE_LIGHT}.get(
            self.settings.get("theme", "system"), Adw.ColorScheme.DEFAULT)
        sm.set_color_scheme(scheme)

    def _on_theme_changed(self, row, _pspec) -> None:
        idx = row.get_selected()
        theme = {0: "system", 1: "dark", 2: "light"}[idx]
        save_ui_settings({"theme": theme})
        self.settings["theme"] = theme
        self._apply_theme()

    def _on_gpu_changed(self, row, _pspec) -> None:
        idx = row.get_selected()
        if idx < 0 or idx >= len(self._gpu_devices):
            return
        device = self._gpu_devices[idx].device
        result = self.backend.apply_gpu(device)
        self._toast(f"GPU enregistré : {result['device']}")

    def _on_ctx_changed(self, row, _pspec) -> None:
        val = int(row.get_value())
        self._ctx = val
        save_ui_settings({"ctx_size": val})

    def _on_tts_toggle(self, action, _param) -> None:
        new = not action.get_state().get_boolean()
        action.set_state(GLib.Variant.new_boolean(new))
        self.tts_enabled = new
        save_ui_settings({"tts": new})

    def _on_tts_switch(self, row, _pspec) -> None:
        self.tts_enabled = row.get_active()
        save_ui_settings({"tts": self.tts_enabled})
        app = self.get_application()
        act = app.lookup_action("tts")
        if act is not None:
            act.set_state(GLib.Variant.new_boolean(self.tts_enabled))

    # ---------- page d'accueil ----------

    def _on_launch(self, _btn) -> None:
        if self._status_state in ("online", "managed"):
            self.stack.set_visible_child_name("chat")
            return
        # hors ligne : démarrer le serveur avec le modèle choisi
        model_path = None
        if self.home_model_combo.get_active_id():
            model_path = self.home_model_combo.get_active_id()
        elif self.backend.models_llm:
            model_path = str(self.backend.models_llm[0].path)
        if not model_path:
            self._toast("Aucun modèle GGUF trouvé (~/models, cache HF, $JARVIS_MODEL_DIRS)")
            return
        self.launch_button.set_sensitive(False)
        self.launch_button.set_label("Démarrage du modèle…")
        self.home_status.set_text("Chargement du modèle (peut être long)…")
        self.backend.start_server(model_path, ctx_size=int(self._ctx))

    def _on_status_clicked(self, _btn) -> None:
        self._refresh_status()
        self.status_popover.popup()

    def _on_pop_start_stop(self, _btn) -> None:
        if self._status_state in ("online", "managed") and self._server_managed():
            self.backend.stop_server()
        elif self._status_state == "online":
            self._toast("Serveur externe — arrêt impossible depuis Jarvis")
        else:
            model_path = self.pop_model_combo.get_active_id()
            if model_path:
                self.backend.start_server(model_path, ctx_size=int(self._ctx))

    def _server_managed(self) -> bool:
        from ..gui import server as server_mod
        return server_mod.is_managed_running()

    # ---------- conversation ----------

    def _on_mic(self, _btn) -> None:
        self.chat_mic.set_sensitive(False)
        self.mic_button.set_sensitive(False)
        self._set_chat_hint("Enregistrement…")
        self.backend.stt_record_and_transcribe(6.0)

    def _on_prompt(self, *_args) -> None:
        dialog = Adw.Dialog.new()
        dialog.set_title("Prompt")
        dialog.set_content_width(560)
        dialog.set_content_height(360)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10,
                      margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)
        scroll = Gtk.ScrolledWindow(vexpand=True, hexpand=True, min_content_height=180)
        self.prompt_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.prompt_view.set_top_margin(8)
        self.prompt_view.set_bottom_margin(8)
        self.prompt_view.set_left_margin(8)
        self.prompt_view.set_right_margin(8)
        scroll.set_child(self.prompt_view)
        box.append(scroll)
        send = Gtk.Button(label="Envoyer", halign=Gtk.Align.END)
        send.add_css_class("suggested-action")
        send.connect("clicked", lambda b: self._send_prompt(dialog))
        box.append(send)
        dialog.set_child(box)
        dialog.present(self)

    def _send_prompt(self, dialog: Adw.Dialog) -> None:
        buf = self.prompt_view.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), True).strip()
        if not text:
            return
        dialog.close()
        self._send_turn(text)

    def _on_new_session(self, *_args) -> None:
        self.backend.new_session()

    def _on_sessions(self, *_args) -> None:
        self.backend.list_sessions()

    def _on_about(self, *_args) -> None:
        d = Adw.AboutDialog(application_name="Jarvis",
                            developer_name="Fracorbas02",
                            version="0.1.0",
                            comments="Assistant local : LLM auto-hébergé + outils MCP + voix.")
        d.present(self)

    def _send_turn(self, text: str) -> None:
        if self._busy:
            self._toast("Un tour est déjà en cours")
            return
        self.stack.set_visible_child_name("chat")
        self.messages.add(user_message(text))
        self._assistant_box = assistant_message()
        self.messages.add(self._assistant_box)
        self._busy = True
        self._set_busy_ui(True)
        self.backend.stream_turn(text)

    # ---------- événements du backend ----------

    def _poll_events(self) -> None:
        GLib.timeout_add(30, self._drain_events)

    def _drain_events(self) -> bool:
        try:
            while True:
                event = self.backend.events.get_nowait()
                GLib.idle_add(self._dispatch, event)
        except queue.Empty:
            pass
        return GLib.SOURCE_CONTINUE

    def _dispatch(self, event: dict) -> bool:
        t = event["type"]
        if t == "backend_ready":
            tools = event.get("tools") or []
            self._toast(f"Prêt — {len(tools)} outils MCP")
            for name in tools:
                self.tools_group.add(Adw.ActionRow(title=name))
            if event.get("session_id"):
                self._replay_history()
        elif t == "backend_error":
            self._toast(f"Erreur : {event['message']}")
        elif t == "server_status":
            self._on_server_status(event)
        elif t in ("server_started", "server_stopped"):
            self._on_server_status(event)
            if t == "server_started" and event.get("models"):
                self.stack.set_visible_child_name("chat")
                self.launch_button.set_sensitive(True)
                self.launch_button.set_label("Ouvrir la conversation")
        elif t == "reasoning":
            if self._assistant_box is not None:
                self._assistant_box.append_reasoning(event["text"])
                self.messages.scroll_to_end()
        elif t == "content":
            if self._assistant_box is not None:
                self._assistant_box.append_text(event["text"])
                self.messages.scroll_to_end()
        elif t == "tool_call":
            if self._assistant_box is not None:
                self._tool_lines[event["name"]] = self._assistant_box.add_tool_line(
                    event["name"], event["args"])
        elif t == "tool_result":
            lbl = self._tool_lines.pop(event["name"], None)
            if lbl is not None and self._assistant_box is not None:
                import json as _json
                args_txt = _json.dumps(event["args"], ensure_ascii=False)
                if len(args_txt) > 120:
                    args_txt = args_txt[:117] + "..."
                mark = "\u2713" if event["ok"] else "\u2717"
                lbl.set_text(f"{mark} {event['name']}({args_txt}) "
                             f"{event['duration']:.2f}s")
                if not event["ok"]:
                    lbl.set_tooltip_text(event["result"][:400])
        elif t == "done":
            self._on_turn_done(event)
        elif t == "error":
            self.messages.add(error_message(event["message"]))
            self._busy = False
            self._set_busy_ui(False)
        elif t == "stt_recording":
            self._set_chat_hint(f"Enregistrement… {event['seconds']:.0f}s")
        elif t == "stt_done":
            self._set_chat_hint("")
            self.chat_mic.set_sensitive(True)
            self.mic_button.set_sensitive(True)
            if event["text"].strip():
                self._send_turn(event["text"].strip())
            else:
                self._toast("Aucune parole transcrite")
        elif t == "voice_error":
            self._set_chat_hint("")
            self.chat_mic.set_sensitive(True)
            self.mic_button.set_sensitive(True)
            self._toast(event["message"])
        elif t == "session_new":
            self.messages.clear()
            self._assistant_box = None
            self._toast("Nouvelle conversation")
        elif t == "session_loaded":
            self.messages.clear()
            self._replay_history()
            self.stack.set_visible_child_name("chat")
        elif t == "sessions_list":
            self._show_sessions_dialog(event["sessions"])
        return GLib.SOURCE_REMOVE

    def _on_turn_done(self, event: dict) -> None:
        box = self._assistant_box
        trace = event.get("trace") or {}
        if box is not None:
            if trace.get("error"):
                box.set_meta(trace["error"])
            box.collapse_reasoning("Raisonnement replié")
        self._tool_lines.clear()
        self._busy = False
        self._set_busy_ui(False)
        if self.tts_enabled and event.get("reply"):
            self.backend.tts_speak(event["reply"])

    def _on_server_status(self, event: dict) -> None:
        self._status_state = event.get("state", "unknown")
        models = ", ".join(event.get("models") or []) or "—"
        ctx = event.get("ctx") or {}
        ctx_txt = ""
        if ctx.get("n_ctx"):
            ctx_txt = f"  ·  contexte {ctx['used']}/{ctx['n_ctx']} tok"
        labels = {"online": "en ligne", "managed": "géré par Jarvis",
                  "offline": "hors ligne"}
        state_txt = labels.get(self._status_state, self._status_state)
        host = event.get("host", "")
        port = event.get("port", "")
        txt = f"{state_txt} — {host}:{port} — {models}{ctx_txt}"
        self.home_status.set_text(txt)
        self.pop_status_label.set_text(txt + (f"\n{event.get('detail') or ''}"
                                              if event.get("detail") else ""))
        if self._status_state == "offline":
            self.launch_button.set_label("Lancer")
            self.launch_button.set_sensitive(True)
            self.home_model_box.set_visible(len(self.backend.models_llm) > 1)
            self.pop_button.set_label("Démarrer le serveur")
        else:
            self.launch_button.set_label("Ouvrir la conversation")
            self.launch_button.set_sensitive(True)
            self.home_model_box.set_visible(False)
            self.pop_button.set_label("Arrêter le serveur (géré)"
                                       if self._server_managed() else "Serveur externe")

    def _refresh_status(self) -> bool:
        self.backend.refresh_status()
        return GLib.SOURCE_CONTINUE

    def _replay_history(self) -> None:
        """Rejoue l'historique de la session reprise (user/assistant uniquement)."""
        for m in self.backend.history:
            role = m.get("role")
            content = m.get("content")
            if role == "user" and content:
                self.messages.add(user_message(str(content)))
            elif role == "assistant" and content:
                box = assistant_message()
                box.set_text(str(content))
                self.messages.add(box)

    def _show_sessions_dialog(self, sessions: list[dict]) -> None:
        dialog = Adw.Dialog.new()
        dialog.set_title("Conversations")
        dialog.set_content_width(520)
        dialog.set_content_height(420)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                      margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)
        scroll = Gtk.ScrolledWindow(vexpand=True)
        listbox = Gtk.ListBox()
        listbox.add_css_class("boxed-list")
        for s in sessions:
            row = Gtk.ListBoxRow()
            inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2,
                            margin_top=8, margin_bottom=8, margin_start=10, margin_end=10)
            inner.append(Gtk.Label(label=s.get("title", "?"), xalign=0))
            sub = Gtk.Label(label=f"{s.get('updated', '')}  ·  {s.get('count', 0)} messages",
                            xalign=0)
            sub.add_css_class("dim-label")
            sub.add_css_class("caption")
            inner.append(sub)
            row.set_child(inner)
            row.sid = s.get("id")
            listbox.append(row)
        listbox.connect("row-activated", lambda lb, row: self._load_session_from(row, dialog))
        scroll.set_child(listbox)
        box.append(scroll)
        dialog.set_child(box)
        dialog.present(self)

    def _load_session_from(self, row, dialog) -> None:
        dialog.close()
        self.backend.load_session(row.sid)

    # ---------- divers ----------

    def _set_busy_ui(self, busy: bool) -> None:
        self.chat_mic.set_sensitive(not busy)
        self.mic_button.set_sensitive(not busy)
        if not busy:
            self._set_chat_hint("Cliquez sur le micro pour parler")

    def _set_chat_hint(self, text: str) -> None:
        self.chat_hint.set_text(text)

    def _toast(self, text: str) -> None:
        self.toast_overlay.add_toast(Adw.Toast(title=text, timeout=4))

    def do_close_request(self) -> bool:
        self.backend.stop()
        return False  # laisse GTK fermer


class JarvisApp(Adw.Application):
    def __init__(self, config_path: str | None = None):
        super().__init__(application_id=APP_ID,
                         flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.config_path = config_path
        self.backend = GtkBackend(config_path)

    def do_activate(self) -> None:
        win = self.get_active_window()
        if win is None:
            win = JarvisWindow(self, self.backend)
        win.present()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-gtk",
                                     description="Interface GTK4 de Jarvis")
    parser.add_argument("--config", default=None,
                        help="chemin vers config.toml (défaut: JARVIS_CONFIG ou ./config.toml)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = JarvisApp(config_path=args.config)
    return app.run([])


if __name__ == "__main__":
    raise SystemExit(main())
