"""MCP server git en lecture seule : status, log, diff, branches, remotes.

Aucune commande d'écriture (pas de commit/add/push) : le LLM observe l'état du
dépôt, l'humain décide. Le dépôt est celui du répertoire courant ou de
JARVIS_GIT_REPO (chemin absolu ou relatif).
"""

from __future__ import annotations

import os
import shutil
import subprocess

try:
    from mcp.server.mcpserver import MCPServer as Server  # SDK MCP >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server      # SDK MCP 1.x

mcp = Server("git")

_MAX_BYTES = 30_000


def _repo() -> str:
    return os.environ.get("JARVIS_GIT_REPO") or os.getcwd()


def _git(args: list[str], timeout: float = 20.0) -> str:
    if shutil.which("git") is None:
        return "ERREUR: 'git' introuvable dans PATH"
    repo = _repo()
    cmd = ["git", "-C", repo] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"ERREUR: timeout après {timeout}s"
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return f"ERREUR git: {out.strip()}"
    if len(out) > _MAX_BYTES:
        out = out[:_MAX_BYTES] + f"\n... (tronqué à {_MAX_BYTES} octets)"
    return out.strip() or "(aucune sortie)"


@mcp.tool()
def git_status() -> str:
    """État du dépôt courant : branche, suivi amont, fichiers modifiés/ajoutés/
    non suivis (git status --short + nom de branche)."""
    branch = _git(["branch", "--show-current"]) or "(détaché)"
    upstream = _git(["rev-parse", "--abbrev-ref", "@{upstream}"])
    short = _git(["status", "--short", "--branch"])
    return f"Dépôt: {_repo()}\nBranche: {branch}" + (f" (amont: {upstream})" if not upstream.startswith("ERREUR") else "") + f"\n\n{short}"


@mcp.tool()
def git_log(n: int = 10) -> str:
    """Historique des N derniers commits (max 50) : hash court, date, auteur, sujet."""
    n = max(1, min(n, 50))
    return _git(["log", f"-{n}", "--pretty=format:%h | %ad | %an | %s",
                 "--date=short"])


@mcp.tool()
def git_diff(what: str = "worktree", path: str = "") -> str:
    """Diff du dépôt. what: 'worktree' (modifs non indexées, défaut), 'staged'
    (indexées), 'HEAD' (tout vs dernier commit), ou un ref (ex: 'main',
    'HEAD~3'). path: restreint à un fichier/répertoire optionnel."""
    if what == "worktree":
        args = ["diff"]
    elif what == "staged":
        args = ["diff", "--cached"]
    elif what == "HEAD":
        args = ["diff", "HEAD"]
    else:
        args = ["diff", what]
    if path:
        args += ["--", path]
    return _git(args, timeout=30.0)


@mcp.tool()
def git_branches() -> str:
    """Branches locales et distantes, avec la branche courante marquée."""
    return _git(["branch", "-a", "-v"])


@mcp.tool()
def git_remotes() -> str:
    """Remotes configurés (fetch/push URLs)."""
    return _git(["remote", "-v"])


@mcp.tool()
def git_show(ref: str, path: str = "") -> str:
    """Contenu d'un fichier à une révision donnée (git show ref:path), ou le
    détail d'un commit si path est vide. ref: hash, branche, tag, HEAD~2..."""
    if not ref or any(c in ref for c in " ;&|`$(){}[]<>\\\"'"):
        return "ERREUR: ref invalide."
    if path:
        return _git(["show", f"{ref}:{path}"], timeout=30.0)
    return _git(["show", "--stat", "--pretty=format:%h | %ad | %an | %s", "--date=short", ref])


if __name__ == "__main__":
    mcp.run()  # stdio
