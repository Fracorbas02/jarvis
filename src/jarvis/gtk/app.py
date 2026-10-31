"""Application GTK4 / libadwaita de Jarvis (`uv run jarvis-app`).

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
from .backend import APP_ID, GtkBackend, load_ui_settings, save_ui_settings
from .widgets import (MessagesColumn, assistant_message, error_message,
                      user_message, _css)

log = logging.getLogger("jarvis.gtk")

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
        self.mic_device = self.settings.get("mic", "default")

        _css()
        self._build_actions()
        self._build_ui()
        self._apply_theme()
        self._poll_events()
        backend.start()
        if not self.settings.get("mic_permission_asked"):
            GLib.idle_add(self._ask_mic_permission_first_run)
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

        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)

        # menu principal (prompt, sessions, tts, \u00e0 propos)
        menu = Gio.Menu()
        menu.append("Prompt\u2026", "app.prompt")
        menu.append("Nouvelle conversation", "app.new-session")
        menu.append("Conversations\u2026", "app.sessions")
        section = Gio.Menu()
        section.append("Lire les r\u00e9ponses \u00e0 voix haute", "app.tts")
        menu.append_section(None, section)
        section2 = Gio.Menu()
        section2.append("\u00c0 propos", "app.about")
        menu.append_section(None, section2)
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

        # Accueil : un seul gros bouton micro. Si le serveur LLM est hors
        # ligne, il est d'abord démarré (modèle choisi dans le popover
        # statut) et la conversation s'ouvre quand le modèle est servi.
        self.home_mic_button = Gtk.Button(halign=Gtk.Align.CENTER,
                                          tooltip_text="Parler à Jarvis")
        self.home_mic_button.add_css_class("suggested-action")
        self.home_mic_button.add_css_class("pill")
        mic_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6,
                          halign=Gtk.Align.CENTER)
        mic_icon = Gtk.Image(icon_name="audio-input-microphone-symbolic", pixel_size=48)
        mic_box.append(mic_icon)
        mic_box.append(Gtk.Label(label="Parler à Jarvis"))
        self.home_mic_button.set_child(mic_box)
        page.append(self.home_mic_button)
        self.home_mic_button.connect("clicked", self._on_home_mic)

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

        # barre du bas : micro + champ texte + envoyer
        bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
                         halign=Gtk.Align.CENTER, margin_top=8, margin_bottom=12,
                         margin_start=16, margin_end=16)
        self.chat_mic = Gtk.Button(icon_name="audio-input-microphone-symbolic",
                                   tooltip_text="Parler à Jarvis (enregistre puis transcrit)")
        self.chat_mic.add_css_class("pill")
        self.chat_mic.connect("clicked", self._on_mic)
        bottom.append(self.chat_mic)
        self.chat_entry = Gtk.Entry(hexpand=True, placeholder_text="Message…")
        self.chat_entry.add_css_class("pill")
        self.chat_entry.connect("activate", self._on_entry_activate)
        bottom.append(self.chat_entry)
        send_btn = Gtk.Button(icon_name="go-next-symbolic", tooltip_text="Envoyer")
        send_btn.add_css_class("pill")
        send_btn.connect("clicked", self._on_entry_activate)
        bottom.append(send_btn)
        outer.append(bottom)
        hint_row = Gtk.Box(halign=Gtk.Align.CENTER, margin_bottom=8)
        self.chat_hint = Gtk.Label(label="", halign=Gtk.Align.CENTER)
        self.chat_hint.add_css_class("dim-label")
        self.chat_hint.add_css_class("caption")
        hint_row.append(self.chat_hint)
        outer.append(hint_row)
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

        # --- Voix : micro + TTS ---
        g_voice = Adw.PreferencesGroup(title="Voix")
        self.mic_row = Adw.ComboRow(title="Micro utilisé pour parler à Jarvis",
                                    subtitle="Liste des cartes d'entrée ALSA (arecord -L)")
        self.mic_row.set_model(Gtk.StringList.new(["default"]))
        self.mic_row.set_selected(0)
        self.mic_row.connect("notify::selected", self._on_mic_changed)
        g_voice.add(self.mic_row)
        row_tts = Adw.SwitchRow(title="Lire les réponses à voix haute (TTS)",
                                active=self.tts_enabled)
        row_tts.connect("notify::active", self._on_tts_switch)
        g_voice.add(row_tts)
        page.add(g_voice)

        # --- Voix API Mistral (Voxtral) ---
        g_api = Adw.PreferencesGroup(
            title="Voix API Mistral (Voxtral)",
            description="TTS cloud, meilleure qualité que le local. La clé est "
                        "stockée localement (~/.config/jarvis/mistral.json, 0600), "
                        "jamais dans config.toml. Utilisée quand « Lire les réponses » "
                        "est actif et que cette option est cochée ; sinon le TTS local "
                        "sert de secours.")
        page.add(g_api)

        # clé API (champ masqué + enregistrer)
        self.api_key_row = Adw.ActionRow(title="Clé API Mistral",
                                         subtitle=self._api_key_subtitle())
        self.api_key_entry = Gtk.PasswordEntry(
            hexpand=True, placeholder_text="Colle ta clé api.mistral.ai…")
        self.api_key_entry.set_show_peek_icon(True)
        btn_save_key = Gtk.Button(label="Enregistrer", valign=Gtk.Align.CENTER)
        btn_save_key.add_css_class("suggested-action")
        btn_save_key.connect("clicked", self._on_save_api_key)
        self.api_key_row.add_suffix(self.api_key_entry)
        self.api_key_row.add_suffix(btn_save_key)
        g_api.add(self.api_key_row)

        # activer l'API comme source TTS
        tts_api = self.backend.cfg.tts_api
        self.api_switch_row = Adw.SwitchRow(
            title="Utiliser l'API Mistral pour le TTS",
            subtitle="sinon le serveur TTS local est utilisé",
            active=tts_api.enabled)
        self.api_switch_row.connect("notify::active", self._on_tts_api_switch)
        g_api.add(self.api_switch_row)

        # modèle TTS : rempli d'abord avec la liste statique, puis dynamiquement
        # via GET /v1/models (bouton Rafraîchir + auto après'enregistrement de la clé)
        self.api_model_row = Adw.ComboRow(
            title="Modèle TTS",
            subtitle="liste rafraîchie via l'API (GET /v1/models) ; voxtral-mini-tts-3-0 = dernier (Voxtral TTS 3)")
        from ..mistral_tts import TTS_MODELS
        self._api_model_ids = list(TTS_MODELS)
        self.api_model_row.set_model(Gtk.StringList.new(self._api_model_ids))
        try:
            self.api_model_row.set_selected(self._api_model_ids.index(tts_api.model))
        except ValueError:
            self.api_model_row.set_selected(0)
        self.api_model_row.connect("notify::selected", self._on_tts_api_model)
        g_api.add(self.api_model_row)

        row_refresh_models = Adw.ActionRow(
            title="Lister les modèles TTS du compte (API)",
            subtitle="GET /v1/models, filtre 'tts' — pour voir les nouveaux modèles (ex. voxtral-mini-tts-3-0)")
        btn_refresh_models = Gtk.Button(label="Rafraîchir", valign=Gtk.Align.CENTER)
        btn_refresh_models.connect("clicked",
                                   lambda b: self.backend.list_mistral_tts_models())
        row_refresh_models.add_suffix(btn_refresh_models)
        g_api.add(row_refresh_models)

        # voix : presets + voix du workspace (rafraîchie via l'API)
        self.api_voice_row = Adw.ComboRow(
            title="Voix",
            subtitle="presets Mistral + voix clonées du workspace (bouton Rafraîchir)")
        self._api_voice_ids: list[str] = []
        from ..mistral_tts import preset_voice_options
        presets = preset_voice_options()
        self._fill_api_voices(presets, current=tts_api.voice_id)
        self.api_voice_row.connect("notify::selected", self._on_tts_api_voice)
        g_api.add(self.api_voice_row)

        row_refresh = Adw.ActionRow(
            title="Lister les voix du compte (API)",
            subtitle="récupère presets + voix clonées via GET /v1/audio/voices")
        btn_refresh = Gtk.Button(label="Rafraîchir", valign=Gtk.Align.CENTER)
        btn_refresh.connect("clicked", lambda b: self.backend.list_mistral_voices())
        row_refresh.add_suffix(btn_refresh)
        g_api.add(row_refresh)

        row_test_api = Adw.ActionRow(title="Tester la voix",
                                     subtitle="synthétise un extrait via l'API et le joue")
        btn_test_api = Gtk.Button(label="Tester", valign=Gtk.Align.CENTER)
        btn_test_api.connect("clicked", lambda b: self.backend.test_mistral_tts())
        row_test_api.add_suffix(btn_test_api)
        g_api.add(row_test_api)

        # --- Autorisations système ---
        g_perm = Adw.PreferencesGroup(title="Autorisations système",
                                      description="GNOME contrôle l'accès au micro "
                                                  "par application (Paramètres > Confidentialité)")
        self.mic_perm_row = Adw.ActionRow(
            title="Autorisation micro",
            subtitle="Interrogation du portail système…")
        self.mic_perm_row.set_subtitle("Statut inconnu")
        g_perm.add(self.mic_perm_row)
        row_test = Adw.ActionRow(title="Tester le micro",
                                 subtitle="Enregistre 0,5 s pour vérifier l'autorisation")
        btn_test = Gtk.Button(label="Tester", valign=Gtk.Align.CENTER)
        btn_test.connect("clicked", self._on_test_mic)
        row_test.add_suffix(btn_test)
        g_perm.add(row_test)
        row_gnome = Adw.ActionRow(title="Ouvrir les réglages Confidentialité de GNOME",
                                  subtitle="Gérer les autorisations micro des applications")
        btn_gnome = Gtk.Button(label="Ouvrir", valign=Gtk.Align.CENTER)
        btn_gnome.connect("clicked", self._on_open_gnome_privacy)
        row_gnome.add_suffix(btn_gnome)
        g_perm.add(row_gnome)
        page.add(g_perm)

        # liste les micros + interroge l'autorisation dès l'ouverture de l'onglet
        self.backend.list_input_devices()
        self.backend.check_mic_permission()

        return scroll

    def _on_mic_permission(self, state: str) -> None:
        labels = {"yes": "Autorisée (portail système)",
                  "no": "Refusée — ouvre Paramètres > Confidentialité > Micro",
                  "unknown": "Non demandée / portail indisponible — le test "
                             "d'enregistrement déclenchera la demande système"}
        if hasattr(self, "mic_perm_row"):
            self.mic_perm_row.set_subtitle(labels.get(state, state))

    def _fill_mic_devices(self, devices: list, error: str) -> None:
        """Remplit le ComboRow des micros depuis arecord -L."""
        model = Gtk.StringList()
        labels: list[str] = []
        for dev, desc in devices:
            model.append(desc)
            labels.append(dev)
        self.mic_row.set_model(model)
        self._mic_devices = labels
        # resélectionne le micro enregistré
        try:
            idx = labels.index(self.mic_device)
        except ValueError:
            idx = 0
        self.mic_row.set_selected(idx)
        if error:
            self.mic_row.set_subtitle(error)

    def _on_mic_changed(self, row, _pspec) -> None:
        idx = row.get_selected()
        devices = getattr(self, "_mic_devices", ["default"])
        if 0 <= idx < len(devices):
            self.mic_device = devices[idx]
            save_ui_settings({"mic": self.mic_device})

    def _on_test_mic(self, _btn) -> None:
        self._toast("Test du micro…")
        self.backend.test_mic(getattr(self, "mic_device", "default"))

    def _on_open_gnome_privacy(self, _btn) -> None:
        import subprocess
        for uri in ("settings://privacy", "gnome-control-center privacy"):
            try:
                if uri.startswith("settings://"):
                    subprocess.Popen(["gio", "open", uri])
                else:
                    subprocess.Popen(uri.split())
                return
            except Exception:  # noqa: BLE001
                continue
        self._toast("Ouvre manuellement : Paramètres > Confidentialité")

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

    # ---------- autorisation micro (1er lancement) ----------

    def _ask_mic_permission_first_run(self) -> bool:
        self._mic_perm_dialog = Adw.AlertDialog.new(
            "Autoriser le micro ?",
            "Jarvis fonctionne à la voix : il enregistre le micro pour te "
            "transcrire (STT local), puis répond. Aucune donnée ne quitte "
            "ta machine.\n\nL'autorisation se gère dans Paramètres GNOME > "
            "Confidentialité > Micro, tu peux la révoquer à tout moment.")
        self._mic_perm_dialog.add_response("later", "Plus tard")
        self._mic_perm_dialog.add_response("allow", "Autoriser le micro")
        self._mic_perm_dialog.set_response_appearance(
            "allow", Adw.ResponseAppearance.SUGGESTED)
        self._mic_perm_dialog.set_default_response("allow")
        self._mic_perm_dialog.set_close_response("later")
        self._mic_perm_dialog.connect("response", self._on_mic_perm_response)
        self._mic_perm_dialog.present(self)
        return GLib.SOURCE_REMOVE

    def _on_mic_perm_response(self, dialog, response: str) -> None:
        save_ui_settings({"mic_permission_asked": True})
        self.settings["mic_permission_asked"] = True
        if response == "allow":
            # test réel : enregistre 0,5 s pour déclencher la vraie demande
            # d'autorisation système et vérifier que ça marche
            self._toast("Test du micro…")
            self.backend.test_mic(getattr(self, "mic_device", "default"))
    # ---------- voix API Mistral ----------

    def _api_key_subtitle(self) -> str:
        st = self.backend.mistral_api_key_status()
        if not st["has_key"]:
            return "aucune clé enregistrée (env MISTRAL_API_KEY non définie non plus)"
        src = "MISTRAL_API_KEY (env)" if st["source"] == "env" else "mistral.json local"
        return f"clé définie ({src}, {st['key_len']} car.) — colle une nouvelle pour changer"

    def _on_save_api_key(self, _btn) -> None:
        key = self.api_key_entry.get_text().strip()
        saved = self.backend.save_mistral_api_key(key)
        self.api_key_entry.set_text("")
        self.api_key_row.set_subtitle(self._api_key_subtitle())
        self._toast("Clé API enregistrée localement (mistral.json, 0600)"
                    if saved else "Clé API supprimée")
        # après une nouvelle clé, on peut lister voix ET modèles du compte
        if saved:
            self.backend.list_mistral_voices()
            self.backend.list_mistral_tts_models()

    def _on_tts_api_switch(self, row, _pspec) -> None:
        self.backend.set_tts_api(enabled=row.get_active())
        self._toast("TTS API Mistral " + ("activée" if row.get_active() else "désactivée")
                    + " (le TTS local reste le secours)")

    def _on_tts_api_model(self, row, _pspec) -> None:
        idx = row.get_selected()
        models = getattr(self, "_api_model_ids", [])
        if 0 <= idx < len(models):
            self.backend.set_tts_api(model=models[idx])

    def _fill_api_models(self, models: list[str], current: str = "") -> None:
        """Remplit le ComboRow des modèles TTS (liste dynamique API)."""
        if not models:
            return
        self._api_model_ids = list(models)
        self.api_model_row.set_model(Gtk.StringList.new(self._api_model_ids))
        sel = 0
        if current in self._api_model_ids:
            sel = self._api_model_ids.index(current)
        self.api_model_row.set_selected(sel)

    def _fill_api_voices(self, voices: list[tuple[str, str]], current: str = "") -> None:
        """Remplit le ComboRow des voix. voices = [(label, voice_id)]."""
        model = Gtk.StringList()
        self._api_voice_ids = []
        for label, vid in voices:
            model.append(f"{label}  [{vid}]")
            self._api_voice_ids.append(vid)
        self.api_voice_row.set_model(model)
        sel = 0
        if current:
            for i, vid in enumerate(self._api_voice_ids):
                if vid == current:
                    sel = i
                    break
        if self._api_voice_ids:
            self.api_voice_row.set_selected(sel)

    def _on_tts_api_voice(self, row, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._api_voice_ids):
            self.backend.set_tts_api(voice_id=self._api_voice_ids[idx])

    # ---------- page d'accueil ----------

    def _on_home_mic(self, _btn) -> None:
        # Accueil = micro uniquement : si le serveur LLM est hors ligne, on
        # le démarre d'abord (modèle choisi dans le popover statut), la
        # conversation s'ouvre quand le modèle est servi ; sinon on y va.
        if self._status_state not in ("online", "managed"):
            model_path = self.pop_model_combo.get_active_id()
            if not model_path and self.backend.models_llm:
                model_path = str(self.backend.models_llm[0].path)
            if model_path:
                self.home_status.set_text("Démarrage du modèle (peut être long)…")
                self.backend.start_server(model_path, ctx_size=int(self._ctx))
            else:
                self._toast("Aucun modèle GGUF trouvé "
                            "(~/models, cache HF, $JARVIS_MODEL_DIRS)")
            return
        self.stack.set_visible_child_name("chat")
        self._start_mic_turn()

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
        from .. import server as server_mod
        return server_mod.is_managed_running()

    # ---------- conversation ----------

    def _on_mic(self, _btn) -> None:
        self._start_mic_turn()

    def _start_mic_turn(self) -> None:
        self.stack.set_visible_child_name("chat")
        self.chat_mic.set_sensitive(False)
        self.mic_button.set_sensitive(False)
        self.chat_entry.set_sensitive(False)
        self._set_chat_hint("Enregistrement…")
        self.backend.stt_record_and_transcribe(6.0, device=getattr(self, "mic_device", "default"))

    def _on_entry_activate(self, _widget) -> None:
        text = self.chat_entry.get_text().strip()
        if not text:
            return
        self.chat_entry.set_text("")
        self._send_turn(text)

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
            self._set_busy_ui(False)
            if event["text"].strip():
                self._send_turn(event["text"].strip())
            else:
                self._toast("Aucune parole transcrite")
        elif t == "voice_error":
            self._set_chat_hint("")
            self._set_busy_ui(False)
            self._toast(event["message"])
        elif t == "mic_devices":
            self._fill_mic_devices(event.get("devices") or [], event.get("error") or "")
        elif t == "mic_test":
            if event.get("ok"):
                self._toast("Micro OK — autorisation et périphérique fonctionnent")
            else:
                self._toast(f"Micro indisponible : {event.get('message', '')}")
        elif t == "mic_permission":
            self._on_mic_permission(event.get("state", "unknown"))
        elif t == "mistral_tts_models":
            models = event.get("models") or []
            if models:
                current = self.backend.cfg.tts_api.model
                self._fill_api_models(models, current=current)
                self._toast(f"{len(models)} modèles TTS récupérés depuis l'API")
            if event.get("error"):
                self._toast(f"Modèles API : {event['error']}")
        elif t == "mistral_voices":
            voices = [(f"{v.get('name', v['id'])} ({v.get('language', '')}{', ' + v['gender'] if v.get('gender') else ''})"
                       + (" [custom]" if v.get("kind") == "custom" else ""),
                       v["id"]) for v in event.get("voices") or []]
            if voices:
                current = self.backend.cfg.tts_api.voice_id
                self._fill_api_voices(voices, current=current)
                self._toast(f"{len(voices)} voix récupérées depuis l'API")
            elif event.get("error"):
                self._toast(f"Voix API : {event['error']}")
        elif t == "tts_fallback":
            self._toast(f"TTS local utilis\u00e9 en secours : {event.get('message', '')}")
        elif t == "mistral_tts_test":
            self._toast(("TTS API OK — " + event.get("message", ""))
                        if event.get("ok")
                        else f"TTS API en erreur : {event.get('message', '')}")
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
            self.pop_button.set_label("Démarrer le serveur")
        else:
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
                box.set_markdown(str(content))
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
        self.chat_entry.set_sensitive(not busy)

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
    parser = argparse.ArgumentParser(prog="jarvis-app",
                                     description="Interface GTK4 de Jarvis")
    parser.add_argument("--config", default=None,
                        help="chemin vers config.toml (défaut: JARVIS_CONFIG ou ./config.toml)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # les requ\u00eates de statut (httpx) polluent le terminal toutes les 2 s
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = JarvisApp(config_path=args.config)
    return app.run([])


if __name__ == "__main__":
    raise SystemExit(main())
