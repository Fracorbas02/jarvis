"""Tests des MCP servers locaux (sans LLM ni réseau externe) : chaque serveur
est lancé en sous-processus stdio via McpToolProvider, comme en production.

Valide que les outils répondent et renvoient le format attendu (y compris les
cas d'erreur : chemin hors racines, entrée invalide...).

Usage: uv run python tests/test_mcp_servers.py  (depuis la racine du repo)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jarvis.mcp import McpToolProvider  # noqa: E402

SERVERS = {name: [sys.executable, str(ROOT / "mcp_servers" / f"{name}.py")]
           for name in ("nettools", "netutils", "sysinfo", "files", "git", "memory", "echo")}


async def call_all(provider: McpToolProvider,
                   calls: list[tuple[str, dict]]) -> list[tuple[str, str]]:
    """Appelle chaque (name, args) et retourne [(name, result), ...] dans l'ordre
    (une liste, pas un dict : plusieurs appels peuvent viser le même outil)."""
    return [(name, await provider.call(name, args)) for name, args in calls]


def at(results: list[tuple[str, str]], name: str, index: int = 0) -> str:
    """Résultat du index-ième appel de l'outil `name`."""
    hits = [res for n, res in results if n == name]
    return hits[index] if index < len(hits) else ""


async def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, cond, detail))
        print(f"[{'OK ' if cond else 'FAIL'}] {name}"
              + (f" — {detail[:160]}" if detail and not cond else ""))

    # --- netutils : pur stdlib, déterministe ---
    p = McpToolProvider({"netutils": SERVERS["netutils"]})
    await p.start()
    r = await call_all(p, [
        ("netutils__ip_calc", {"subnet": "192.168.1.0/24"}),
        ("netutils__ip_calc", {"subnet": "10.0.0.5"}),
        ("netutils__ip_calc", {"subnet": "pas-une-ip"}),
        ("netutils__ip_convert", {"ip": "1.2.3.4"}),
        ("netutils__encode", {"text": "hello", "op": "b64encode"}),
        ("netutils__encode", {"text": "aGVsbG8=", "op": "b64decode"}),
        ("netutils__encode", {"text": "a b&c", "op": "urlencode"}),
        ("netutils__encode", {"text": "x", "op": "op-inconnue"}),
        ("netutils__hash_text", {"text": "abc", "algo": "sha256"}),
        ("netutils__hash_text", {"text": "abc", "algo": "nope"}),
        ("netutils__time_now", {"tz": "UTC"}),
        ("netutils__time_now", {"tz": "Fuseau/Inconnu"}),
        ("netutils__time_convert", {"epoch": 0, "tz": "UTC"}),
    ])
    await p.stop()
    check("ip_calc CIDR réseau/broadcast/254 hôtes",
          "192.168.1.0/24" in at(r, "netutils__ip_calc", 0)
          and "192.168.1.255" in at(r, "netutils__ip_calc", 0)
          and "254" in at(r, "netutils__ip_calc", 0))
    check("ip_calc IP seule privée",
          "10.0.0.5" in at(r, "netutils__ip_calc", 1)
          and "Privée:    oui" in at(r, "netutils__ip_calc", 1))
    check("ip_calc invalide -> ERREUR", at(r, "netutils__ip_calc", 2).startswith("ERREUR"))
    check("ip_convert entier 1.2.3.4 = 16909060",
          "16909060" in at(r, "netutils__ip_convert"))
    check("b64encode hello = aGVsbG8=", at(r, "netutils__encode", 0).strip() == "aGVsbG8=")
    check("b64decode aGVsbG8= = hello", "hello" in at(r, "netutils__encode", 1))
    check("urlencode espace/&", "a%20b%26c" in at(r, "netutils__encode", 2))
    check("encode op inconnue -> ERREUR", "ERREUR" in at(r, "netutils__encode", 3))
    check("sha256('abc') connu",
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
          in at(r, "netutils__hash_text", 0))
    check("hash algo inconnu -> ERREUR", "ERREUR" in at(r, "netutils__hash_text", 1))
    check("time_now UTC contient UTC et Epoch",
          "UTC" in at(r, "netutils__time_now", 0) and "Epoch" in at(r, "netutils__time_now", 0))
    check("time_now fuseau inconnu -> ERREUR", "ERREUR" in at(r, "netutils__time_now", 1))
    check("time_convert epoch 0 = 1970-01-01", "1970-01-01" in at(r, "netutils__time_convert"))

    # --- files : confinement aux racines ---
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        (tmp_p / "a.log").write_text(
            "ligne1 erreur ici\nligne2 ok\nligne3 erreur encore\n", encoding="utf-8")
        (tmp_p / "sub").mkdir()
        (tmp_p / "sub" / "b.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
        os.environ["JARVIS_FILES_ROOTS"] = tmp
        p = McpToolProvider({"files": SERVERS["files"]})
        await p.start()
        r = await call_all(p, [
            ("files__grep_files", {"pattern": "erreur", "path": tmp}),
            ("files__grep_files", {"pattern": "print", "path": tmp, "glob": "*.py"}),
            ("files__read_file", {"path": str(tmp_p / "a.log")}),
            ("files__read_file", {"path": str(tmp_p / "a.log"), "offset": 1, "limit": 1}),
            ("files__file_info", {"path": str(tmp_p / "a.log")}),
            ("files__list_dir", {"path": tmp, "pattern": "*.log"}),
            ("files__read_file", {"path": "/etc/passwd"}),  # hors racines -> refus
        ])
        await p.stop()
        del os.environ["JARVIS_FILES_ROOTS"]
        check("grep trouve 'erreur' dans a.log",
              "erreur" in at(r, "files__grep_files", 0)
              and "a.log" in at(r, "files__grep_files", 0))
        check("grep glob *.py ne touche que b.py",
              "b.py" in at(r, "files__grep_files", 1)
              and "a.log" not in at(r, "files__grep_files", 1))
        check("read_file retourne les 3 lignes + total",
              "ligne1" in at(r, "files__read_file", 0)
              and "sur 3" in at(r, "files__read_file", 0))
        check("read_file offset=1 limit=1 -> ligne2 seule",
              "ligne2" in at(r, "files__read_file", 1)
              and "ligne1" not in at(r, "files__read_file", 1))
        check("file_info sha256 + taille présents",
              "SHA256:" in at(r, "files__file_info")
              and "Taille:" in at(r, "files__file_info"))
        check("list_dir filtre *.log",
              "a.log" in at(r, "files__list_dir")
              and "b.py" not in at(r, "files__list_dir"))
        check("read_file hors racines -> ERREUR",
              at(r, "files__read_file", 2).startswith("ERREUR")
              and "autorisés" in at(r, "files__read_file", 2))

    # --- memory : fichier temporaire isolé ---
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["JARVIS_MEMORY_FILE"] = str(Path(tmp) / "mem.md")
        p = McpToolProvider({"memory": SERVERS["memory"]})
        await p.start()
        r = await call_all(p, [
            ("memory__remember", {"fact": "Bastien habite à Lyon"}),
            ("memory__remember", {"fact": "Projet jarvis en Python"}),
            ("memory__recall", {"query": "lyon"}),
            ("memory__recall", {"query": ""}),
            ("memory__recall", {"query": "inexistant"}),
            ("memory__forget", {"number": 1}),
            ("memory__recall", {"query": ""}),
        ])
        await p.stop()
        del os.environ["JARVIS_MEMORY_FILE"]
        check("remember confirme (2 faits)",
              "Noté" in at(r, "memory__remember", 1)
              and "2 faits" in at(r, "memory__remember", 1))
        check("recall filtre 'lyon'",
              "Lyon" in at(r, "memory__recall", 0)
              and "jarvis" not in at(r, "memory__recall", 0))
        check("recall vide = tous les faits",
              "Lyon" in at(r, "memory__recall", 1)
              and "jarvis" in at(r, "memory__recall", 1))
        check("recall sans match", "Aucun fait" in at(r, "memory__recall", 2))
        check("forget supprime le n°1", "Oublié" in at(r, "memory__forget"))
        check("après forget, 1 fait restant sans Lyon",
              "1 fait" in at(r, "memory__recall", 3)
              and "Lyon" not in at(r, "memory__recall", 3))

    # --- git : sur le repo courant (lecture seule) ---
    p = McpToolProvider({"git": SERVERS["git"]})
    await p.start()
    r = await call_all(p, [
        ("git__git_status", {}),
        ("git__git_log", {"n": 3}),
        ("git__git_branches", {}),
        ("git__git_remotes", {}),
        ("git__git_diff", {"what": "HEAD"}),
        ("git__git_show", {"ref": "HEAD; rm -rf /"}),
    ])
    await p.stop()
    check("git_status indique une branche", "Branche:" in at(r, "git__git_status"))
    check("git_log 3 commits au format hash | date | auteur | sujet",
          at(r, "git__git_log").count("\n") >= 2 and "|" in at(r, "git__git_log"))
    check("git_branches liste main", "main" in at(r, "git__git_branches"))
    check("git_remotes répond (liste ou ERREUR propre)", at(r, "git__git_remotes") != "")
    check("git_show injection de commande -> ERREUR", "ERREUR" in at(r, "git__git_show"))

    # --- sysinfo : outils présents dans le sandbox ---
    p = McpToolProvider({"sysinfo": SERVERS["sysinfo"]})
    await p.start()
    r = await call_all(p, [
        ("sysinfo__disk_usage", {"path": "/"}),
        ("sysinfo__memory", {}),
        ("sysinfo__load", {}),
        ("sysinfo__ports_listening", {}),
        ("sysinfo__processes", {"sort": "cpu", "limit": 5}),
        ("sysinfo__network_interfaces", {}),
        ("sysinfo__gpu_status", {}),
        ("sysinfo__temperatures", {}),
    ])
    await p.stop()
    check("disk_usage / répond", "/" in at(r, "sysinfo__disk_usage"))
    check("memory contient free et uptime",
          "free" in at(r, "sysinfo__memory").lower()
          or "Mem:" in at(r, "sysinfo__memory"))
    check("load non vide et sans ERREUR",
          at(r, "sysinfo__load").strip() != ""
          and not at(r, "sysinfo__load").startswith("ERREUR"))
    check("ports_listening répond (ss/netstat ou ERREUR propre)",
          at(r, "sysinfo__ports_listening").strip() != "")
    check("processes top 5 (en-têtes ps ou ERREUR propre)",
          "PID" in at(r, "sysinfo__processes")
          or "COMMAND" in at(r, "sysinfo__processes")
          or at(r, "sysinfo__processes").startswith("ERREUR"))
    check("network_interfaces non vide", at(r, "sysinfo__network_interfaces").strip() != "")
    check("gpu_status répond (données ou ERREUR propre)",
          at(r, "sysinfo__gpu_status").strip() != "")
    check("temperatures répond (données ou ERREUR propre)",
          at(r, "sysinfo__temperatures").strip() != "")

    # --- nettools : outils existants + nouveaux ---
    p = McpToolProvider({"nettools": SERVERS["nettools"]})
    await p.start()
    names = p.names()
    r = await call_all(p, [
        ("nettools__ping", {"host": "127.0.0.1", "count": 1}),
        ("nettools__http_get", {"url": "ftp://exemple"}),
        ("nettools__tls_info", {"host": "127.0.0.1", "port": 1}),
        ("nettools__listening_ports", {}),
        ("nettools__ip_addr", {}),
    ])
    await p.stop()
    for n in ("nettools__ping", "nettools__dig", "nettools__traceroute", "nettools__whois",
              "nettools__listening_ports", "nettools__connections", "nettools__ip_addr",
              "nettools__tls_info", "nettools__http_get"):
        check(f"outil exposé: {n}", n in names)
    check("ping 127.0.0.1 répond (ou ERREUR propre si ping absent du sandbox)",
          "127.0.0.1" in at(r, "nettools__ping") or "bytes" in at(r, "nettools__ping")
          or at(r, "nettools__ping").startswith("ERREUR"))
    check("http_get schéma non http -> ERREUR", "ERREUR" in at(r, "nettools__http_get"))
    check("tls_info port fermé -> ERREUR propre", "ERREUR" in at(r, "nettools__tls_info"))
    check("listening_ports non vide", at(r, "nettools__listening_ports").strip() != "")
    check("ip_addr non vide", at(r, "nettools__ip_addr").strip() != "")

    # --- by_server / descriptions (regroupement UI réglages) ---
    p = McpToolProvider({"echo": SERVERS["echo"], "netutils": SERVERS["netutils"],
                         "memory": SERVERS["memory"]})
    await p.start()
    bs = p.by_server()
    check("by_server regroupe par serveur",
          set(bs) == {"echo", "netutils", "memory"}
          and any(n.startswith("echo__") for n in bs["echo"])
          and all(v == sorted(v) for v in bs.values()))
    d = p.descriptions()
    check("descriptions couvrent tous les outils",
          set(d) == set(p.names()) and all(isinstance(v, str) for v in d.values()))
    check("description echo__ping non vide", bool(d.get("echo__ping", "").strip()))
    await p.stop()

    # --- echo toujours OK ---
    p = McpToolProvider({"echo": SERVERS["echo"]})
    await p.start()
    r = await call_all(p, [("echo__ping", {"target": "test"})])
    await p.stop()
    check("echo__ping", at(r, "echo__ping") == "pong vers test")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} tests OK")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
