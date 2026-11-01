"""Protection de l'état du terminal (termios) à la sortie de l'application.

prompt_toolkit restaure normalement le tty en sortie de prompt, mais un
chemin de sortie anormal (exception dans un key binding, crash, SIGTERM,
kill) peut laisser le terminal en mode brut : plus d'echo, plus de mode
ligne — le shell paraît « mort » après fermeture de l'app.

`install()` sauvegarde les attributs de stdin à l'entrée (tant qu'il est
en mode normal) et les restaure :
- à la sortie du process (atexit, couvre sys.exit et la fin normale) ;
- sur SIGTERM/SIGINT (re-émis ensuite avec le handler par défaut) ;
- sur demande explicite via `restore_stdin()` (à appeler dans le finally
  du point d'entrée, couvre les exceptions remontées hors de la boucle).
"""

from __future__ import annotations

import atexit
import os
import signal
import sys

try:
    import termios
except ImportError:  # Windows : pas de termios
    termios = None  # type: ignore[assignment]

_saved: list | None = None
_installed = False


def snapshot_stdin() -> None:
    """Sauvegarde les termios actuels de stdin (uniquement si c'est un tty)."""
    global _saved
    if termios is None or not sys.stdin.isatty():
        return
    try:
        _saved = termios.tcgetattr(sys.stdin.fileno())
    except (termios.error, OSError, ValueError):
        _saved = None


def restore_stdin(fileno: int | None = None) -> None:
    """Restaure les termios sauvegardés. Idempotent, ne lève jamais.

    `fileno` permet de restaurer un descripteur précis (ex. un pty de test) ;
    par défaut on cible stdin.
    """
    global _saved
    if termios is None or _saved is None:
        return
    saved, _saved = _saved, None  # consommé quoi qu'il arrive
    if fileno is not None:
        fd = fileno
    else:
        try:
            fd = sys.stdin.fileno()
        except (OSError, ValueError, AttributeError):
            return  # stdin redirigé (pas de tty) : rien à restaurer
    try:
        termios.tcsetattr(fd, termios.TCSANOW, saved)
    except (termios.error, OSError, ValueError):
        pass


def install() -> None:
    """Snapshot + restore à la sortie du process et sur SIGTERM/SIGINT.

    À appeler en tout début de point d'entrée (avant que quiconque ne mette
    le tty en mode brut), une seule fois.
    """
    global _installed
    if _installed or termios is None:
        return
    _installed = True
    snapshot_stdin()
    atexit.register(restore_stdin)

    def _on_signal(signum, _frame):
        restore_stdin()
        # puis comportement par défaut du signal (exit 130/143)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass  # pas dans le thread principal : atexit couvre déjà la sortie
