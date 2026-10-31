"""REPL Jarvis : affichage terminal riche (voir tui.py).

Le thinking du modèle et la réponse sont streamés en direct ; la réponse est
rendue en Markdown dans un panneau (tui.Display).
"""

from __future__ import annotations

import asyncio
import logging
import os
import readline
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .config import load_config
from .llm import LLMClient
from .mcp import McpToolProvider
from .sessions import SessionStore
from .tui import Display

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
  /reset           efface l'historique de conversation
  /verbose         bascule les logs console en DEBUG (fichier: ~/.cache/jarvis/jarvis.log)
  /quit            quitte (ou Ctrl+D)"""

COMMANDS = ["/help", "/tools", "/thinking", "/tool", "/history", "/undo",
            "/copy", "/save", "/sessions", "/load", "/new", "/reset",
            "/verbose", "/quit"]


def _state_dir() -> Path:
    return Path(os.environ.get("JARVIS_LOG_DIR", Path.home() / ".cache" / "jarvis"))


def _setup_readline() -> Path | None:
    """Historique persistant des saisies + complétion Tab des commandes /."""
    hist = _state_dir() / "history"
    try:
        _state_dir().mkdir(parents=True, exist_ok=True)
        readline.read_history_file(hist)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)

    def completer(text: str, state: int):
        matches = [c for c in COMMANDS if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    try:
        readline.parse_and_bind("tab: complete")          # GNU readline
    except Exception:                                     # libedit (macOS)
        readline.parse_and_bind("bind ^I rl_complete")
    return hist


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


async def amain() -> int:
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
                  stream=True)

    console = _make_console()
    display = Display(console)
    server = await _server_info(cfg.llm.base_url, cfg.llm.ctx_size)
    ctx_max = server[2]
    display.set_header(_banner(cfg, tools, (server[0], server[1]), ctx_max))
    hist_file = _setup_readline()
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
            prompt = "\n[bold cyan][toi] › [/]"
            if ctx_max:
                usage = await _ctx_usage(cfg.llm.base_url)
                est = usage if usage is not None else _est_tokens(history)
                prompt += f"[dim](ctx ~{est}/{ctx_max})[/] "
            try:
                user = console.input(prompt).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                # Ctrl+C : annule la saisie en cours ; deux Ctrl+C rapprochés quittent
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
            if user.startswith("/tool"):
                parts = user.split()
                if len(parts) == 2 and parts[1].isdigit():
                    display.show_tool(int(parts[1]))
                else:
                    console.print(f"[dim]usage: /tool N "
                                  f"(1..{max(len(display.last_tool_results), 1)})[/]")
                continue

            snapshot = list(history)  # agent.turn mute history : on restaure si ça plante
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
        if hist_file is not None:
            try:
                readline.write_history_file(hist_file)
            except OSError:
                pass
        await tools.stop()
        await llm.aclose()
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
