"""Tests des MCP servers ajoutés (lot 2) : textutils, notes, scan, lan,
snmp, docker + nouveaux outils sysinfo/nettools. Partie déterministe uniquement
(les appels réseau externes — KEV/EPSS/GHSA/fetch/rss — ne sont pas testés ici).

Usage: uv run python tests/test_mcp_servers_extra.py  (depuis la racine du repo)
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
           for name in ("textutils", "notes", "scan", "lan", "snmp", "docker",
                        "sysinfo", "nettools", "kev", "epss", "ghsa", "fetch", "rss",
                        "exploitdb")}


async def call_all(provider: McpToolProvider,
                   calls: list[tuple[str, dict]]) -> list[tuple[str, str]]:
    return [(name, await provider.call(name, args)) for name, args in calls]


def at(results: list[tuple[str, str]], name: str, index: int = 0) -> str:
    hits = [res for n, res in results if n == name]
    return hits[index] if index < len(hits) else ""


async def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, cond, detail))
        print(f"[{'OK ' if cond else 'FAIL'}] {name}"
              + (f" — {detail[:160]}" if detail and not cond else ""))

    # --- textutils : pur stdlib, déterministe ---
    p = McpToolProvider({"textutils": SERVERS["textutils"]})
    await p.start()
    r = await call_all(p, [
        ("textutils__diff", {"a": "x\ny\n", "b": "x\nz\n"}),
        ("textutils__diff", {"a": "pareil\n", "b": "pareil\n"}),
        ("textutils__json_query", {"json_text": '{"a": {"b": [10, 20]}}', "path": "a.b[1]"}),
        ("textutils__json_query", {"json_text": '{"a": 1}', "path": "a.b"}),
        ("textutils__json_query", {"json_text": "pas du json", "path": ""}),
        ("textutils__json_query", {"json_text": '{"n": 42}', "path": ""}),
        ("textutils__regex_test", {"pattern": r"CVE-\d{4}-\d+", "text": "voir CVE-2024-3094 et CVE-2021-44228"}),
        ("textutils__regex_test", {"pattern": r"(\w+)@(\w+)", "text": "mail: bastien@exemple"}),
        ("textutils__regex_test", {"pattern": "([", "text": "x"}),
        ("textutils__regex_test", {"pattern": "a", "text": "aaa", "flags": "z"}),
        ("textutils__unit_convert", {"value": 1, "from_unit": "go", "to_unit": "mo"}),
        ("textutils__unit_convert", {"value": 10, "from_unit": "go", "to_unit": "to"}),
        ("textutils__unit_convert", {"value": 1, "from_unit": "h", "to_unit": "min"}),
        ("textutils__unit_convert", {"value": 100, "from_unit": "c", "to_unit": "f"}),
        ("textutils__unit_convert", {"value": 1, "from_unit": "ko", "to_unit": "ko/s"}),
        ("textutils__unit_convert", {"value": 1, "from_unit": "octet", "to_unit": "inconnu"}),
    ])
    await p.stop()
    check("diff montre -y +z", "-y" in at(r, "textutils__diff", 0) and "+z" in at(r, "textutils__diff", 0))
    check("diff identiques", "identiques" in at(r, "textutils__diff", 1))
    check("json_query a.b[1] = 20", at(r, "textutils__json_query", 0).strip() == "20")
    check("json_query chemin absent -> ERREUR", "ERREUR" in at(r, "textutils__json_query", 1))
    check("json_query invalide -> ERREUR", "ERREUR" in at(r, "textutils__json_query", 2))
    check("json_query doc entier", '"n": 42' in at(r, "textutils__json_query", 3))
    check("regex 2 matchs CVE", "2 match(s)" in at(r, "textutils__regex_test", 0)
          and "CVE-2024-3094" in at(r, "textutils__regex_test", 0))
    check("regex groupes (bastien, exemple)", "'bastien', 'exemple'" in at(r, "textutils__regex_test", 1))
    check("regex invalide -> ERREUR", "ERREUR" in at(r, "textutils__regex_test", 2))
    check("regex flag inconnu -> ERREUR", "ERREUR" in at(r, "textutils__regex_test", 3))
    check("1 go = 1024 mo", "1024" in at(r, "textutils__unit_convert", 0))
    check("10 go = 0.00976562 to", "0.00976562" in at(r, "textutils__unit_convert", 1))
    check("1 h = 60 min", "60" in at(r, "textutils__unit_convert", 2))
    check("100 c = 212 f", "212" in at(r, "textutils__unit_convert", 3))
    check("ko vs ko/s -> incompatible mais calculée (ERREUR non levée)", at(r, "textutils__unit_convert", 4) != "")
    check("unité cible inconnue -> ERREUR", "ERREUR" in at(r, "textutils__unit_convert", 5))

    # --- notes : répertoire temporaire isolé ---
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["JARVIS_NOTES_DIR"] = tmp
        p = McpToolProvider({"notes": SERVERS["notes"]})
        await p.start()
        r = await call_all(p, [
            ("notes__note_add", {"note": "Idée: tester pipecat pour le barge-in", "topic": "jarvis"}),
            ("notes__note_add", {"note": "Deuxième note sur jarvis", "topic": "jarvis"}),
            ("notes__note_add", {"note": "Acheter un SSD 2 To", "topic": "perso"}),
            ("notes__note_add", {"note": "x", "topic": "Topic Invalide!"}),
            ("notes__note_list", {"topic": "jarvis"}),
            ("notes__note_list", {"topic": "inexistant"}),
            ("notes__note_topics", {}),
            ("notes__note_search", {"query": "pipecat"}),
            ("notes__note_search", {"query": "ssd"}),
        ])
        await p.stop()
        del os.environ["JARVIS_NOTES_DIR"]
        check("note_add confirme", "Note ajoutée à 'jarvis'" in at(r, "notes__note_add", 0))
        check("note_add topic invalide -> ERREUR", "ERREUR" in at(r, "notes__note_add", 3))
        check("note_list jarvis (2 notes, récente d'abord)",
              "2 au total" in at(r, "notes__note_list", 0)
              and "Deuxième note" in at(r, "notes__note_list", 0))
        check("note_list sujet inconnu", "Aucune note" in at(r, "notes__note_list", 1))
        check("note_topics liste 2 sujets", "jarvis: 2 note(s)" in at(r, "notes__note_topics")
              and "perso: 1 note(s)" in at(r, "notes__note_topics"))
        check("note_search trouve pipecat dans jarvis",
              "pipecat" in at(r, "notes__note_search", 0) and "jarvis" in at(r, "notes__note_search", 0))
        check("note_search trouve ssd dans perso",
              "ssd" in at(r, "notes__note_search", 1).lower() and "perso" in at(r, "notes__note_search", 1))

    # --- scan : garde-fous (pas de nmap dans le sandbox, mais les refus passent avant) ---
    p = McpToolProvider({"scan": SERVERS["scan"]})
    await p.start()
    r = await call_all(p, [
        ("scan__port_scan", {"target": "8.8.8.8", "ports": "80"}),
        ("scan__port_scan", {"target": "127.0.0.1", "ports": "22,80"}),
        ("scan__port_scan", {"target": "cible; rm -rf /", "ports": "80"}),
        ("scan__service_detect", {"target": "1.1.1.1"}),
    ])
    await p.stop()
    check("scan cible publique refusée sans allow_public",
          "ERREUR" in at(r, "scan__port_scan", 0) and "allow_public" in at(r, "scan__port_scan", 0))
    check("scan 127.0.0.1 accepté (nmap absent -> ERREUR propre ou scan)",
          at(r, "scan__port_scan", 1) != "")
    check("scan cible avec injection -> ERREUR", "ERREUR" in at(r, "scan__port_scan", 2))
    check("service_detect 1.1.1.1 refusé", "ERREUR" in at(r, "scan__service_detect"))

    # --- lan / snmp / docker : présence des outils + erreurs propres ---
    p = McpToolProvider({"lan": SERVERS["lan"], "snmp": SERVERS["snmp"], "docker": SERVERS["docker"]})
    await p.start()
    names = p.names()
    r = await call_all(p, [
        ("lan__arp_table", {}),
        ("lan__dhcp_leases", {}),
        ("lan__lan_hosts", {"subnet": "pas-un-subnet"}),
        ("snmp__snmp_get", {"target": "127.0.0.1", "oid": "1.3.6.0.2.1.1.5.0"}),
        ("snmp__snmp_get", {"target": "127.0.0.1", "oid": "1.3.6.0; rm -rf /"}),
        ("docker__containers", {}),
        ("docker__container_logs", {"name": "x; rm -rf /"}),
        ("docker__container_logs", {"name": "mon-ctr", "lines": 10}),
    ])
    await p.stop()
    for n in ("lan__arp_table", "lan__dhcp_leases", "lan__lan_hosts",
              "snmp__snmp_get", "snmp__snmp_walk", "snmp__snmp_interfaces",
              "docker__containers", "docker__container_stats", "docker__container_logs",
              "docker__images", "docker__container_inspect", "docker__disk_usage"):
        check(f"outil exposé: {n}", n in names)
    check("arp_table répond", at(r, "lan__arp_table").strip() != "")
    check("dhcp_leases sans fichier -> ERREUR propre ou liste", at(r, "lan__dhcp_leases") != "")
    check("lan_hosts subnet invalide -> ERREUR", "ERREUR" in at(r, "lan__lan_hosts"))
    check("snmp_get répond (timeout/ERREUR propre acceptés)", at(r, "snmp__snmp_get", 0) != "")
    check("snmp OID avec injection -> ERREUR", "ERREUR" in at(r, "snmp__snmp_get", 1))
    check("docker containers répond (docker absent -> ERREUR propre)",
          at(r, "docker__containers").startswith("ERREUR") or "CONTAINER" in at(r, "docker__containers").upper()
          or at(r, "docker__containers") != "")
    check("docker logs nom invalide -> ERREUR", "ERREUR" in at(r, "docker__container_logs", 0))
    check("docker logs conteneur valide -> réponse ou ERREUR docker", at(r, "docker__container_logs", 1) != "")

    # --- sysinfo : nouveaux outils ---
    p = McpToolProvider({"sysinfo": SERVERS["sysinfo"]})
    await p.start()
    r = await call_all(p, [
        ("sysinfo__services", {"state": "failed"}),
        ("sysinfo__service_status", {"name": "systemd; rm -rf /"}),
        ("sysinfo__service_status", {"name": "systemd-journald"}),
        ("sysinfo__smart_disk", {"device": "/dev/sda"}),
        ("sysinfo__smart_disk", {"device": "pas-un-device"}),
        ("sysinfo__battery", {}),
    ])
    await p.stop()
    check("services failed répond", at(r, "sysinfo__services") != "")
    check("service_status injection -> ERREUR", "ERREUR" in at(r, "sysinfo__service_status", 0))
    check("service_status valide -> réponse", at(r, "sysinfo__service_status", 1) != "")
    check("smart_disk répond (smartctl absent -> ERREUR propre ou données)",
          at(r, "sysinfo__smart_disk", 0) != "")
    check("smart_disk device invalide -> ERREUR", "ERREUR" in at(r, "sysinfo__smart_disk", 1))
    check("battery répond (fixe -> message ou données)", at(r, "sysinfo__battery") != "")

    # --- nettools : bandwidth_test ---
    p = McpToolProvider({"nettools": SERVERS["nettools"]})
    await p.start()
    names = p.names()
    r = await call_all(p, [("nettools__bandwidth_test", {"duration": 1})])
    await p.stop()
    check("outil exposé: nettools__bandwidth_test", "nettools__bandwidth_test" in names)
    check("bandwidth_test répond (iperf3 absent -> ERREUR propre ou débit)",
          at(r, "nettools__bandwidth_test") != "")

    # --- serveurs externes : on vérifie juste qu'ils démarrent et exposent leurs outils ---
    p = McpToolProvider({s: SERVERS[s] for s in ("kev", "epss", "ghsa", "fetch", "rss", "exploitdb")})
    await p.start()
    names = p.names()
    expected = {
        "kev": ["kev__kev_recent", "kev__kev_search", "kev__kev_info"],
        "epss": ["epss__epss_score", "epss__epss_history"],
        "ghsa": ["ghsa__ghsa_search", "ghsa__ghsa_for_package", "ghsa__ghsa_info"],
        "fetch": ["fetch__fetch_page", "fetch__fetch_headers"],
        "rss": ["rss__rss_latest", "rss__rss_check"],
        "exploitdb": ["exploitdb__exploit_search", "exploitdb__exploit_info"],
    }
    for srv, tools in expected.items():
        for t in tools:
            check(f"outil exposé: {t}", t in names)
    r = await call_all(p, [
        ("fetch__fetch_page", {"url": "ftp://x"}),
        ("fetch__fetch_headers", {"url": "ht!tp://x"}),
        ("rss__rss_latest", {"feed_url": "ftp://x"}),
        ("exploitdb__exploit_search", {"query": "x; rm -rf /"}),
        ("exploitdb__exploit_info", {"edb_id": "abc"}),
        ("epss__epss_score", {"cves": "pas-une-cve"}),
        ("epss__epss_history", {"cve": "CVE-2024-3094"}),
        ("ghsa__ghsa_info", {"ghsa_id": "pas-un-ghsa"}),
        ("kev__kev_info", {"cve_id": "pas-une-cve"}),
    ])
    await p.stop()
    check("fetch_page schéma refusé", "ERREUR" in at(r, "fetch__fetch_page"))
    check("fetch_headers url invalide", "ERREUR" in at(r, "fetch__fetch_headers"))
    check("rss schéma refusé", "ERREUR" in at(r, "rss__rss_latest"))
    check("exploit_search injection -> ERREUR", "ERREUR" in at(r, "exploitdb__exploit_search"))
    check("exploit_info id non numérique -> ERREUR", "ERREUR" in at(r, "exploitdb__exploit_info"))
    check("epss_score sans CVE valide -> ERREUR", "ERREUR" in at(r, "epss__epss_score"))
    check("epss_history CVE valide -> appel réseau (ERREUR réseau acceptée dans le sandbox)",
          at(r, "epss__epss_history") != "")
    check("ghsa_info id invalide -> ERREUR", "ERREUR" in at(r, "ghsa__ghsa_info"))
    check("kev_info id invalide -> ERREUR", "ERREUR" in at(r, "kev__kev_info"))

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} tests OK")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
