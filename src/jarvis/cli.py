"""REPL Jarvis : affichage terminal riche (voir tui.py).

Le thinking du modèle et la réponse sont streamés en direct ; la réponse est
rendue en Markdown dans un panneau (tui.Display).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .compact import compact_history, compact_tool
from .config import load_config
from .llm import LLMClient
from .mcp import McpToolProvider
from .sessions import SessionStore
from .tui import Display
from .ttyguard import install as install_tty_guard, restore_stdin
from .voice import (VoiceError, play_wav, record_wav, record_wav_vad,
                    stop_all_voice_servers, synthesize, transcribe_file,
                    voice_server_status)

HELP = """\
[bold]commandes[/]
  /help            cette aide
  /tools           liste les outils MCP avec leur description
  /thinking        réaffiche le raisonnement du dernier tour
  /tool N          résultat complet du N-ième outil du dernier tour
  /history         liste les derniers messages envoyés
  /undo            retire le dernier échange de l'historique
  /copy            copie la dernière réponse dans le presse-papier
  /save [fichier]  exporte la conversation en Markdown
  /sessions        liste les conversations passées (reprise auto au démarrage)
  /load N          reprend la conversation n°N de /sessions
  /new             démarre une nouvelle conversation (archive l'actuelle)
  /compact         résume l'historique via le LLM pour libérer du contexte
                   (le modèle peut aussi appeler l'outil jarvis__compact seul)
  /reset           efface l'historique de conversation
  /verbose         bascule les logs console en DEBUG (fichier: ~/.cache/jarvis/jarvis.log)
  /tts <texte>     synthétise le texte (API Mistral si [voice.tts_api] enabled, sinon local)
  /stt [fichier]   transcrit un WAV ; sans argument, enregistre 5 s de micro d'abord
  /voice           statut des serveurs voix (STT whisper-server, TTS llama-server)
  /ttsapi <texte>  TEST : synthétise via l'API Mistral (clé mistral.json / MISTRAL_API_KEY)
                   avec le modèle+voix configurés, joue le résultat
  /apimodels       TEST : liste les modèles TTS du compte Mistral (GET /v1/models, filtre tts)
  /apivoices       TEST : liste les voix du compte Mistral (GET /v1/audio/voices)
  /quit            quitte (ou Ctrl+D)"""

COMMANDS = ["/help", "/tools", "/thinking", "/tool", "/history", "/undo",
            "/copy", "/save", "/sessions", "/load", "/new", "/compact",
            "/reset", "/verbose", "/tts", "/stt", "/voice", "/ttsapi",
            "/apimodels", "/apivoices", "/quit"]


def _state_dir() -> Path:
    return Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis"))


class _SlashCompleter(Completer):
    """Complétion Tab des commandes / uniquement en début de saisie."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd in COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))


def _make_session(hist_file: Path | None) -> PromptSession:
    """Session de prompt : historique persistant, complétion /, multiligne.

    - Maj+Entrée (Shift+Enter) insère un saut de ligne : le terminal l'envoie
      en CSI u (\x1b[13;2u) quand le mode clavier kitty est actif (kitty,
      wezterm, ghostty, foot...) ; on enregistre la séquence dans le parser
      de prompt_toolkit pour la capter même sans négociation.
    - Entrée soumet — sauf si la ligne finit par "\\" : continuation
      (fonctionne sur tous les terminaux, fallback universel).
    """
    # Maj+Entrée en CSI u (protocole clavier kitty) -> on la mappe sur F24,
    # touche fonction inutilisée, que l'on binde à l'insertion d'un saut de ligne
    ANSI_SEQUENCES["\x1b[13;2u"] = Keys.F24
    kb = KeyBindings()

    @kb.add(Keys.F24)
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter")
    def _enter(event):
        buf = event.current_buffer
        if buf.text.rstrip().endswith("\\"):
            buf.text = buf.text.rstrip()[:-1] + "\n"  # continuation
        else:
            # Buffer.submit() n'existe plus dans prompt_toolkit 3.x : un
            # AttributeError ici casse le chemin de sortie propre de
            # l'application et laisse le terminal en mode brut (plus d'echo).
            buf.validate_and_handle()

    return PromptSession(
        history=FileHistory(str(hist_file)) if hist_file is not None else None,
        completer=_SlashCompleter(),
        key_bindings=kb,
        enable_history_search=True,
        complete_while_typing=False,
    )


def _hist_file() -> Path | None:
    try:
        _state_dir().mkdir(parents=True, exist_ok=True)
        return _state_dir() / "history"
    except OSError:
        return None


async def _prompt(session: PromptSession, prompt_text) -> str:
    """session.prompt() dans un thread daemon.

    Appelé directement dans la coroutine, prompt_toolkit détecterait la running
    event loop et lirait stdin par petits chunks (mode asyncio), ce qui
    découpe les séquences clavier CSI u (Maj+Entrée) avant le parser. Dans un
    thread sans loop, il retombe en lecture synchrone et le parsing marche.
    """
    box: dict = {}

    def run():
        try:
            box["text"] = session.prompt(prompt_text)
        except BaseException as exc:  # noqa: BLE001 — propagé au main
            box["exc"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while t.is_alive():
        await asyncio.sleep(0.05)
    if "exc" in box:
        raise box["exc"]
    return box.get("text", "")


def _make_console() -> Console:
    """Console rich : couleurs forcées en vrai terminal (sauf NO_COLOR).

    Nettoie os.environ (lu par défaut par rich) : COLUMNS/LINES figeraient la
    largeur, et TERM=dumb ferait retomber rich sur 80x25 au lieu de lire la
    taille réelle du tty. La taille des panneaux live est de toute façon
    relue par ioctl à chaque rendu (tui._LiveRegion.tty_width).
    """
    if os.environ.get("NO_COLOR"):
        return Console(no_color=True)
    os.environ.pop("COLUMNS", None)
    os.environ.pop("LINES", None)
    if os.environ.get("TERM", "") == "dumb":
        os.environ["TERM"] = "xterm-256color"
    if sys.stdout.isatty():
        return Console(force_terminal=True, color_system="truecolor")
    return Console()


async def _server_info(base_url: str, ctx_fallback: int | None) -> tuple[str, bool, int | None]:
    """GET /v1/models + /slots : (statut, ok, taille de contexte max).

    La taille du contexte est lue dynamiquement (n_ctx de /slots, sinon
    /props) : si llama-server est relancé avec un --ctx-size 32k, la TUI le
    reflète sans rien changer. Fallback : ctx_size de la config.
    """
    status, ok = "indisponible", False
    ctx = ctx_fallback
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            root = base_url.rstrip("/").rsplit("/v1", 1)[0]
            r = await http.get(f"{base_url.rstrip('/')}/models")
            if r.status_code == 200:
                ids = [m.get("id", "?") for m in r.json().get("data", [])]
                status, ok = (", ".join(ids) if ids else "modèle chargé"), True
            else:
                status = f"HTTP {r.status_code}"
            for path, keys in (("/slots", ("n_ctx",)), ("/props", ("n_ctx", "context_window"))):
                try:
                    p = await http.get(f"{root}{path}")
                    if p.status_code != 200:
                        continue
                    data = p.json()
                    slots = data if isinstance(data, list) else [data]
                    for k in keys:
                        vals = [s[k] for s in slots if isinstance(s, dict) and s.get(k)]
                        if vals:
                            ctx = max(vals)
                            break
                    if ctx != ctx_fallback:
                        break
                except Exception:  # noqa: BLE001 — endpoint optionnel
                    continue
    except Exception as exc:  # noqa: BLE001 — juste un statut
        status = f"{type(exc).__name__}: {exc}"
    return status, ok, ctx


async def _ctx_usage(base_url: str) -> int | None:
    """Usage réel du contexte : max des n_prompt_tokens de /slots (llama-server)."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as http:
            root = base_url.rstrip("/").rsplit("/v1", 1)[0]
            r = await http.get(f"{root}/slots")
            if r.status_code == 200:
                slots = r.json()
                vals = [s.get("n_prompt_tokens", 0) for s in slots if isinstance(s, dict)]
                return max(vals) if vals else None
    except Exception:  # noqa: BLE001 — fallback sur l'estimation
        pass
    return None


def _est_tokens(history: list[dict]) -> int:
    """Estimation grossière du contexte utilisé : ~4 caractères par token."""
    return sum(len(str(m.get("content") or "")) for m in history) // 4


def _notify(body: str) -> None:
    """Notification desktop en fin de tour long.

    OSC 777 (kitty, foot, wezterm, ghostty) + OSC 9 (iTerm2, wezterm, konsole
    récent) : les terminaux qui ne connaissent pas une séquence l'ignorent.
    """
    if not sys.stdout.isatty():
        return
    sys.stdout.write(f"\x1b]777;notify;Jarvis;{body}\x07")
    sys.stdout.write(f"\x1b]9;{body}\x07")
    sys.stdout.flush()


def _banner(cfg, tools: McpToolProvider, server: tuple[str, bool], ctx_max: int | None) -> Panel:
    n = len(tools.names())
    servers = ", ".join(sorted(cfg.mcp_servers)) or "aucun"
    srv_style = "green" if server[1] else "red"
    ctx_txt = f"{ctx_max:,} tok max".replace(",", " ") if ctx_max else "inconnue"
    body = Text.from_markup(
        f"[bold]assistant local[/]\n"
        f"[dim]modèle[/]  {cfg.llm.model} @ {cfg.llm.base_url}\n"
        f"[{srv_style}]serveur[/]  {server[0]}\n"
        f"[dim]contexte[/]  {ctx_txt}\n"
        f"[dim]outils[/]  {n} chargés ({servers})\n"
        f"[dim]/help[/] pour les commandes"
    )
    return Panel(body, title="[bold cyan]JARVIS[/]", border_style="cyan",
                 box=box.ROUNDED, padding=(0, 1))


def _setup_logging(verbose: bool = False) -> None:
    """Console en WARNING par défaut (lisible), DEBUG complet dans un fichier.

    Les logs httpx/jarvis restent disponibles pour le debug post-mortem dans
    ~/.cache/jarvis/jarvis.log (ou $JARVIS_LOG_DIR/jarvis.log).
    """
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)
    log_dir = Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "jarvis.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
    except OSError:
        pass  # répertoire non writable : la console suffit, on ne crash pas


def _set_console_log_level(level: int) -> None:
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)


def _last_assistant_reply(history: list[dict]) -> str | None:
    for m in reversed(history):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return None


def _copy_to_clipboard(text: str) -> str | None:
    """Copie via l'outil dispo (wayland/X11/macOS). Retourne le nom de l'outil."""
    for tool, args in (("wl-copy", []), ("xclip", ["-selection", "clipboard"]),
                       ("xsel", ["--clipboard", "--input"]), ("pbcopy", [])):
        if shutil.which(tool):
            subprocess.run([tool, *args], input=text.encode(), check=False)
            return tool
    return None


def _with_system(history: list[dict], system_prompt: str) -> list[dict]:
    """Remet le system prompt courant en tête (il peut avoir changé en config)."""
    return [{"role": "system", "content": system_prompt},
            *[m for m in history if m.get("role") != "system"]]


def _export_markdown(history: list[dict], path: Path) -> int:
    """Exporte la conversation en Markdown. Retourne le nombre de messages écrits."""
    out: list[str] = [f"# Session jarvis — {datetime.now():%Y-%m-%d %H:%M}\n"]
    n = 0
    for m in history:
        role, content = m.get("role"), m.get("content")
        if role == "user" and content:
            out.append(f"\n## [toi]\n\n{content}\n")
            n += 1
        elif role == "assistant" and content:
            out.append(f"\n## [jarvis]\n\n{content}\n")
            n += 1
        elif role == "tool":
            out.append(f"\n> outil `{m.get('tool_call_id', '')}` : {str(content)[:200]}\n")
    path.write_text("\n".join(out), encoding="utf-8")
    return n


async def _voice_status_lines(cfg) -> list[str]:
    """Lignes de statut des deux serveurs voix (STT / TTS)."""
    lines = []
    for name, vcfg in (("stt", cfg.voice.stt), ("tts", cfg.voice.tts)):
        st = await voice_server_status(name, vcfg)
        if name == "stt":
            proto = getattr(vcfg, "protocol", "whisper")
            label = ("STT llama-server (voxtral)" if proto == "voxtral"
                     else "STT whisper-server")
        else:
            label = "TTS llama-server"
        lines.append(f"[dim]{label}[/] {st.host}:{st.port} — "
                     f"{'en ligne' if st.state == 'online' else 'géré par jarvis' if st.state == 'managed' else 'hors ligne'}"
                     + (f" ({st.detail})" if st.detail and st.state == "offline" else ""))
    return lines


async def _cmd_tts(cfg, console, args: str) -> None:
    """/tts <texte> : synthétise et joue. API Mistral si [voice.tts_api] enabled
    et clé enregistrée (mistral.json / MISTRAL_API_KEY), sinon serveur TTS local."""
    text = args.strip()
    if not text:
        console.print("[dim]usage: /tts <texte à synthétiser>[/]")
        return
    tts_api = getattr(cfg, "tts_api", None)
    audio: bytes | None = None
    ext = "wav"
    src = "serveur TTS local"
    api_error: str | None = None
    t0 = time.monotonic()
    if tts_api is not None and tts_api.enabled:
        # API choisie : on tente, et en cas d'échec on retombe sur le local
        from .mistral_tts import MistralTTSClient, load_api_key
        key = load_api_key()
        if key:
            client = MistralTTSClient(key, base_url=tts_api.base_url,
                                      model=tts_api.model)
            try:
                audio = await client.synthesize(
                    text, voice_id=tts_api.voice_id or None,
                    response_format=tts_api.response_format or "mp3")
                ext = tts_api.response_format or "mp3"
                src = f"API Mistral ({tts_api.model}, {tts_api.voice_id})"
            except Exception as exc:  # noqa: BLE001 — réseau/401/403/404
                api_error = str(exc)
                console.print(f"[yellow]API Mistral en échec ({exc}) "
                              f"— fallback sur le TTS local[/]")
            finally:
                await client.aclose()
        else:
            api_error = "clé API absente (mistral.json / MISTRAL_API_KEY)"
            console.print(f"[yellow]{api_error} — fallback sur le TTS local[/]")
    if audio is None:
        try:
            audio = await synthesize(cfg.voice, text, auto_start=True)
        except Exception as exc:  # noqa: BLE001 — VoiceError
            detail = f" (API aussi en échec: {api_error})" if api_error else ""
            console.print(f"[bold red]erreur TTS:[/] {exc}{detail}")
            return
    out = _state_dir() / f"tts-{datetime.now():%H%M%S}.{ext}"
    out.write_bytes(audio)
    console.print(f"[dim]{ext.upper()} écrit : {out} ({len(audio) / 1024:.0f} Ko, "
                  f"{time.monotonic() - t0:.1f}s, via {src})[/]")
    try:
        player = await play_wav(out, cfg.voice)
        console.print(f"[dim]lecture via {player}[/]")
    except VoiceError as exc:
        console.print(f"[yellow]lecture impossible:[/] {exc}")


async def _cmd_stt(cfg, console, args: str) -> None:
    """/stt [fichier.wav] : transcription ; sans argument, enregistre 5 s de micro."""
    path = Path(args.strip()).expanduser() if args.strip() else None
    if path is None:
        def _phase(msg: str) -> None:
            console.print(f"[dim]{msg}[/]")
        console.print("[dim]enregistrement depuis le micro, jusqu'à la fin de parole (VAD)…[/]")
        try:
            path = await record_wav_vad(cfg.voice, max_seconds=60.0, on_status=_phase)
        except VoiceError as exc:
            console.print(f"[bold red]erreur enregistrement:[/] {exc}")
            return
        console.print(f"[dim]WAV enregistré : {path}[/]")
    elif not path.is_file():
        console.print(f"[bold red]erreur:[/] fichier introuvable : {path}")
        return
    try:
        console.print("[dim]transcription en cours…[/]")
        text = await transcribe_file(cfg.voice, path, language="fr")
    except VoiceError as exc:
        console.print(f"[bold red]erreur STT:[/] {exc}")
        return
    console.print(f"[bold cyan]transcription[/] : {text or '(vide)'}")


async def _cmd_ttsapi(cfg, console, args: str) -> None:
    """/ttsapi <texte> : TEST de l'API Mistral TTS (clé mistral.json ou env),
    avec le modèle/voix de [voice.tts_api]. Ne touche pas au réglage enabled."""
    from .mistral_tts import (MistralTTSClient, MistralTTSError, has_api_key,
                              load_api_key)
    text = args.strip() or "Bonjour, ceci est un test de la voix API Mistral."
    tts_api = getattr(cfg, "tts_api", None)
    if tts_api is None:
        console.print("[bold red]erreur:[/] pas de section [voice.tts_api] dans la config")
        return
    if not has_api_key():
        console.print("[bold red]erreur:[/] aucune clé API Mistral "
                      "(~/.config/jarvis/mistral.json ou export MISTRAL_API_KEY) — "
                      "définis-la dans l'app (Réglages > Voix API Mistral) ou en env")
        return
    console.print(f"[dim]API {tts_api.base_url} · modèle {tts_api.model} · "
                  f"voix {tts_api.voice_id} · format {tts_api.response_format} · "
                  f"clé: {'env' if __import__('os').environ.get('MISTRAL_API_KEY') else 'mistral.json'}[/]")
    try:
        t0 = time.monotonic()
        client = MistralTTSClient(load_api_key(), base_url=tts_api.base_url,
                                  model=tts_api.model)
        try:
            audio = await client.synthesize(
                text, voice_id=tts_api.voice_id or None,
                response_format=tts_api.response_format or "mp3")
        finally:
            await client.aclose()
        ext = tts_api.response_format or "mp3"
        out = _state_dir() / f"ttsapi-{datetime.now():%H%M%S}.{ext}"
        out.write_bytes(audio)
        console.print(f"[green]OK[/] {len(audio) / 1024:.0f} Ko en "
                      f"{time.monotonic() - t0:.1f}s -> {out}")
        try:
            player = await play_wav(out, cfg.voice)
            console.print(f"[dim]lecture via {player}[/]")
        except VoiceError as exc:
            console.print(f"[yellow]lecture impossible:[/] {exc}")
    except MistralTTSError as exc:
        console.print(f"[bold red]erreur API:[/] {exc}")


async def _cmd_apimodels(cfg, console) -> None:
    """/apimodels : TEST GET /v1/models (filtre tts) sur le compte Mistral."""
    from .mistral_tts import (MistralTTSClient, MistralTTSError, TTS_MODELS,
                              has_api_key, load_api_key)
    if not has_api_key():
        console.print("[bold red]erreur:[/] aucune clé API Mistral — "
                      "liste statique de secours :")
        for m in TTS_MODELS:
            console.print(f"  {m}")
        return
    tts_api = cfg.tts_api
    client = MistralTTSClient(load_api_key(), base_url=tts_api.base_url,
                              model=tts_api.model, timeout=20.0)
    try:
        models = await client.list_tts_models()
    except MistralTTSError as exc:
        console.print(f"[bold red]erreur API:[/] {exc}")
        console.print("[dim]liste statique de secours :[/]")
        for m in TTS_MODELS:
            console.print(f"  {m}")
        return
    finally:
        await client.aclose()
    console.print(f"[green]{len(models)} modèle(s) TTS[/] sur {tts_api.base_url} "
                  f"(modèle courant: {tts_api.model}) :")
    for m in models:
        mark = " [dim]<-- courant[/]" if m == tts_api.model else ""
        console.print(f"  {m}{mark}")


async def _cmd_apivoices(cfg, console) -> None:
    """/apivoices : TEST GET /v1/audio/voices sur le compte Mistral."""
    from .mistral_tts import (MistralTTSClient, MistralTTSError, has_api_key,
                              load_api_key)
    if not has_api_key():
        console.print("[bold red]erreur:[/] aucune clé API Mistral")
        return
    tts_api = cfg.tts_api
    client = MistralTTSClient(load_api_key(), base_url=tts_api.base_url,
                              model=tts_api.model, timeout=20.0)
    try:
        voices = await client.list_voices()
    except MistralTTSError as exc:
        console.print(f"[bold red]erreur API:[/] {exc}")
        return
    finally:
        await client.aclose()
    console.print(f"[green]{len(voices)} voix[/] (voix courante: {tts_api.voice_id}) :")
    for v in voices:
        mark = " [dim]<-- courante[/]" if v.id == tts_api.voice_id else ""
        kind = "custom" if v.kind == "custom" else "preset"
        lang = f" {v.language}" if v.language else ""
        gender = f", {v.gender}" if v.gender else ""
        console.print(f"  {v.id}  {v.name} [{kind}{lang}{gender}]{mark}")


async def amain() -> int:
    install_tty_guard()  # snapshot termios : restaure le tty même en cas de sortie anormale
    _setup_logging()
    try:
        cfg = load_config()
    except FileNotFoundError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2
    except tomllib.TOMLDecodeError as exc:
        print(f"config: TOML invalide: {exc}", file=sys.stderr)
        return 2

    llm = LLMClient(**{k: v for k, v in cfg.llm.__dict__.items() if k != "ctx_size"})
    tools = McpToolProvider(cfg.mcp_servers)
    if cfg.mcp_servers:
        await tools.start()
    agent = Agent(llm, tools, cfg.agent.system_prompt, cfg.agent.max_iterations,
                  stream=True, internal_tools={"jarvis__compact": compact_tool})

    console = _make_console()
    display = Display(console)
    server = await _server_info(cfg.llm.base_url, cfg.llm.ctx_size)
    ctx_max = server[2]
    display.set_header(_banner(cfg, tools, (server[0], server[1]), ctx_max))
    hist_file = _hist_file()
    # prompt_toolkit négocie lui-même le protocole clavier kitty (CSI u) avec
    # le terminal : c'est ce qui rend Maj+Entrée distinguishable d'Entrée sur
    # kitty/wezterm/ghostty/foot. Ailleurs, fallback = "\\" en fin de ligne.
    session = _make_session(hist_file)
    store = SessionStore(_state_dir())
    # reprend la dernière conversation, comme un contexte qui reste
    sid = store.latest_id()
    if sid is not None:
        history = _with_system(store.load(sid), cfg.agent.system_prompt)
        meta = next(s for s in store.list() if s["id"] == sid)
        console.print(f"[dim]session reprise : « {meta['title']} » "
                      f"({meta['count']} messages, {meta['updated']}) — "
                      f"/sessions pour en changer, /new pour repartir à zéro[/]")
    else:
        sid = SessionStore.new_id()
        history = []
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGWINCH, display.on_resize)
    except (NotImplementedError, AttributeError):
        pass  # Windows / pas de signaux : le repaint suivra au prochain token
    verbose = False
    last_interrupt = 0.0
    try:
        while True:
            prompt = "\n\x1b[1;36m[toi] ›\x1b[0m"
            prompt_plain = "\n[toi] › "
            if ctx_max:
                usage = await _ctx_usage(cfg.llm.base_url)
                est = usage if usage is not None else _est_tokens(history)
                prompt += f" \x1b[2m(ctx ~{est}/{ctx_max})\x1b[0m"
                prompt_plain += f" (ctx ~{est}/{ctx_max})"
            prompt += " "
            prompt_plain += " "
            try:
                if sys.stdin.isatty():
                    user = (await _prompt(session, ANSI(prompt))).strip()
                else:
                    # pas de tty (pipe/test) : input() simple, prompt_toolkit y plante
                    user = input(prompt_plain).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                # Ctrl+C : annule la saisie en cours (on sort le prompt du thread) ;
                # deux Ctrl+C rapprochés quittent
                try:
                    session.app.exit(exception=KeyboardInterrupt())
                except Exception:  # noqa: BLE001
                    pass
                now = time.monotonic()
                if now - last_interrupt < 1.0:
                    break
                last_interrupt = now
                console.print("^C")
                continue
            last_interrupt = 0.0
            if not user:
                continue
            if user == "/quit":
                break
            if user == "/help":
                console.print(HELP)
                continue
            if user == "/reset":
                history.clear()
                display.reset_blocks()  # sinon les blocs reviennent au prochain resize
                store.save(sid, history)
                console.print("[dim]historique réinitialisé[/]")
                continue
            if user == "/tools":
                table = Table(box=box.SIMPLE_HEAD, title="Outils MCP",
                              title_style="bold", header_style="bold")
                table.add_column("Outil")
                table.add_column("Description")
                for s in tools.schemas():
                    fn = s["function"]
                    desc = (fn.get("description") or "").strip().splitlines()
                    table.add_row(fn["name"], desc[0] if desc else "")
                console.print(table)
                continue
            if user == "/verbose":
                verbose = not verbose
                _set_console_log_level(logging.DEBUG if verbose else logging.WARNING)
                console.print(f"[dim]niveau de log console: "
                              f"{'DEBUG' if verbose else 'WARNING'} "
                              f"(log complet: ~/.cache/jarvis/jarvis.log)[/]")
                continue
            if user == "/thinking":
                display.show_reasoning()
                continue
            if user == "/undo":
                # retire le dernier échange : tout depuis le dernier message user
                while history and history[-1].get("role") not in ("user", "system"):
                    history.pop()
                if history and history[-1].get("role") == "user":
                    history.pop()
                    store.save(sid, history)
                    console.print("[dim]dernier échange retiré[/]")
                else:
                    console.print("[dim]rien à retirer[/]")
                continue
            if user == "/history":
                users = [m["content"] for m in history
                         if m.get("role") == "user" and m.get("content")]
                if not users:
                    console.print("[dim]aucun message pour l'instant[/]")
                else:
                    start = max(1, len(users) - 9)
                    for i, u in enumerate(users[-10:], start=start):
                        console.print(f"[dim]{i}.[/] {str(u)[:100]}")
                continue
            if user == "/copy":
                reply = _last_assistant_reply(history)
                if not reply:
                    console.print("[dim]pas encore de réponse à copier[/]")
                elif tool := _copy_to_clipboard(reply):
                    console.print(f"[dim]réponse copiée via {tool} ({len(reply)} car.)[/]")
                else:
                    console.print("[bold red]erreur:[/] aucun outil de presse-papier "
                                  "(wl-copy/xclip/xsel/pbcopy)")
                continue
            if user.startswith("/save"):
                parts = user.split(maxsplit=1)
                path = (Path(parts[1]) if len(parts) > 1
                        else _state_dir() / f"session-{datetime.now():%Y%m%d-%H%M%S}.md")
                try:
                    n = _export_markdown(history, path)
                except OSError as exc:
                    console.print(f"[bold red]erreur:[/] {exc}")
                else:
                    console.print(f"[dim]{n} messages exportés dans {path}[/]")
                continue
            if user == "/sessions":
                sessions = store.list()
                if not sessions:
                    console.print("[dim]aucune session enregistrée[/]")
                else:
                    for i, s in enumerate(sessions, 1):
                        mark = " [dim](actuelle)[/]" if s["id"] == sid else ""
                        console.print(f"[dim]{i}.[/] {s['updated']}  {s['title']}"
                                      f"  ({s['count']} msg){mark}")
                continue
            if user.startswith("/load"):
                parts = user.split()
                sessions = store.list()
                if len(parts) == 2 and parts[1].isdigit() and 1 <= int(parts[1]) <= len(sessions):
                    sid = sessions[int(parts[1]) - 1]["id"]
                    history = _with_system(store.load(sid), cfg.agent.system_prompt)
                    display.reset_blocks()
                    console.print(f"[dim]session « {SessionStore.title_of(history)} » chargée "
                                  f"({len(history)} messages)[/]")
                else:
                    console.print(f"[dim]usage: /load N (1..{len(sessions)}) — voir /sessions[/]")
                continue
            if user == "/new":
                store.save(sid, history)  # archive l'actuelle
                sid = SessionStore.new_id()
                history = []
                display.reset_blocks()
                console.print("[dim]nouvelle conversation démarrée[/]")
                continue
            if user == "/compact":
                t0 = time.monotonic()
                try:
                    result, history = await compact_history(agent, history)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[bold red]erreur:[/] {exc}")
                    continue
                store.save(sid, history)
                console.print(f"[dim]{result} ({time.monotonic() - t0:.1f}s)[/]")
                continue
            if user.startswith("/tool"):
                parts = user.split()
                if len(parts) == 2 and parts[1].isdigit():
                    display.show_tool(int(parts[1]))
                else:
                    console.print(f"[dim]usage: /tool N "
                                  f"(1..{max(len(display.last_tool_results), 1)})[/]")
                continue
            if user.startswith("/ttsapi"):
                await _cmd_ttsapi(cfg, console, user[len("/ttsapi"):])
                continue
            if user.startswith("/tts"):
                await _cmd_tts(cfg, console, user[len("/tts"):])
                continue
            if user == "/apimodels":
                await _cmd_apimodels(cfg, console)
                continue
            if user == "/apivoices":
                await _cmd_apivoices(cfg, console)
                continue
            if user.startswith("/stt"):
                await _cmd_stt(cfg, console, user[len("/stt"):])
                continue
            if user == "/voice":
                for line in await _voice_status_lines(cfg):
                    console.print(line)
                console.print("[dim]/tts <texte> pour synthétiser, /stt [wav] pour transcrire[/]")
                continue

            snapshot = list(history)  # agent.turn mute history : on restaure si ça plante
            t0 = time.monotonic()
            try:
                reply, history, trace = await agent.turn(
                    history, user,
                    on_tool_call=display.tool_call,
                    on_tool_result=display.tool_result,
                    on_reasoning=display.reasoning,
                    on_content=display.content,
                    on_llm_end=display.end_llm_stream,
                )
                if reply and not display.content_started:
                    display.print_content(reply)
                display.end_turn()
                store.save(sid, history)  # la session persiste entre les lancements
                if time.monotonic() - t0 > 3.0:
                    _notify("réponse prête")  # tour long : notif si la fenêtre est ailleurs
            except KeyboardInterrupt:
                history = snapshot
                display.end_turn()
                console.print("[yellow]interrompu (Ctrl+C)[/]")
                continue
            except Exception as exc:  # noqa: BLE001 — le REPL ne doit pas mourir
                history = snapshot
                display.end_turn()
                console.print(f"[bold red]erreur:[/] {exc}")
                continue
            if trace.error:
                console.print(f"[bold yellow]warn:[/] {trace.error}")
    finally:
        restore_stdin()  # le terminal doit retrouver echo/icanon quoi qu'il arrive
        await tools.stop()
        await llm.aclose()
        await stop_all_voice_servers(cfg.voice)
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
