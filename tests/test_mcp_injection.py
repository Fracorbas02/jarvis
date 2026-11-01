"""Régression sécurité des MCP servers : injection d'arguments et ReDoS.

Un LLM compromis (ou prompt-injecté via une page web lue par fetch) contrôle
les arguments des outils. Ces tests vérifient que les valeurs dangereuses
(flags déguisés en arguments) sont refusées avant d'atteindre le binaire :
- nettools: ping/dig/traceroute/whois/tls_info/bandwidth_test valident l'hôte ;
- docker: container_logs valide la fenêtre `since` ;
- git: git_diff/git_show valident la ref (jamais d'option --output/--ext-diff) ;
- textutils: regex_test est plafonnée en temps (anti-ReDoS) et en taille.

Usage: uv run pytest tests/test_mcp_injection.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"mcp_{name}", ROOT / "mcp_servers" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nettools = _load("nettools")
docker = _load("docker")
git = _load("git")
textutils = _load("textutils")


# ---------- nettools : hôte jamais pris pour un flag ----------

def test_ping_refuse_flag_comme_host():
    for evil in ("--help", "-f", "-c 99 evil", "host;id", "host|id", "$(id)", "`id`", "-o /tmp/x"):
        out = nettools.ping(evil)
        assert out.startswith("ERREUR"), (evil, out)


def test_dig_traceroute_whois_refusent_flags():
    for evil in ("--version", "-f", "a b", "a;id"):
        assert nettools.dig(evil).startswith("ERREUR"), evil
        assert nettools.traceroute(evil).startswith("ERREUR"), evil
        assert nettools.whois(evil).startswith("ERREUR"), evil
    # record type contrôlé aussi
    assert nettools.dig("example.com", record="A;id").startswith("ERREUR")
    assert nettools.dig("example.com", record="--help").startswith("ERREUR")
    # serveur DNS optionnel validé
    assert nettools.dig("example.com", server="8.8.8.8;id").startswith("ERREUR")


def test_tls_info_valide_host_et_port():
    assert nettools.tls_info("--connect=evil:1").startswith("ERREUR")
    assert nettools.tls_info("host", port=0).startswith("ERREUR")
    assert nettools.tls_info("host", port=70000).startswith("ERREUR")


def test_bandwidth_test_server_valide():
    assert nettools.bandwidth_test("").startswith("ERREUR")
    assert nettools.bandwidth_test("-t 999 evil").startswith("ERREUR")
    assert nettools.bandwidth_test("host;id").startswith("ERREUR")


def test_host_valide_passe_la_validation():
    assert nettools._check_host("example.com") is None
    assert nettools._check_host("192.168.1.10") is None
    assert nettools._check_host("host.local") is None
    assert nettools._check_host("2001:db8::1") is None


# ---------- docker : since jamais un flag ----------

def test_container_logs_since_valide():
    for ok in ("1h", "10m", "30s", "today", "yesterday", "all",
               "2026-01-01", "2026-01-01T10:00:00", "2026-01-01 10:00"):
        # on ne lance pas docker : on vérifie juste que la validation passe
        # (le binaire est absent ici -> ERREUR 'docker' introuvable, pas 'since invalide')
        out = docker.container_logs("web", since=ok)
        assert "since invalide" not in out, (ok, out)
    for evil in ("--timestamps=false", "1h --since=all", "-o /tmp/x", "1h;id", "", "1hh"):
        out = docker.container_logs("web", since=evil)
        assert out.startswith("ERREUR: since invalide"), (evil, out)


def test_container_logs_name_valide():
    assert docker.container_logs("--rm -it evil").startswith("ERREUR")
    assert docker.container_logs("web;id").startswith("ERREUR")


# ---------- git : ref jamais une option ----------

def test_git_diff_refuse_option_comme_what():
    for evil in ("--output=/tmp/pwn", "--ext-diff", "-S", "--", "main;id", "HEAD && id"):
        out = git.git_diff(what=evil)
        assert out.startswith("ERREUR: ref invalide"), (evil, out)


def test_git_show_refuse_option_comme_ref():
    for evil in ("--output=/tmp/pwn", "-Sfoo", "HEAD~1;id", "--ext-diff"):
        out = git.git_show(evil)
        assert out.startswith("ERREUR: ref invalide"), (evil, out)


# ---------- textutils : anti-ReDoS ----------

def test_regex_test_reule_pattern_catastrophique():
    """(a+)+$ sur 30 'a' + '!' : backtracking exponentiel -> timeout 2s."""
    t0 = time.monotonic()
    out = textutils.regex_test(r"(a+)+$", "a" * 30 + "!")
    dt = time.monotonic() - t0
    assert "interrompue" in out, out
    assert dt < 10, f"trop long: {dt:.1f}s"


def test_regex_test_fonctionne_normalement():
    out = textutils.regex_test(r"\d+", "abc 123 def 456")
    assert "2 match(s)" in out and "123" in out and "456" in out
    assert textutils.regex_test(r"\d+", "rien").startswith("0 match")
    assert textutils.regex_test("(invalide", "x").startswith("ERREUR")
    assert textutils.regex_test("x", "y", flags="z").startswith("ERREUR: flag")


def test_regex_test_texte_tronque():
    # 50 000 caractères -> ramené à 2 000, pas d'explosion
    out = textutils.regex_test(r"a", "a" * 50_000)
    assert "match(s)" in out
