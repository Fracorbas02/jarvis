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
        ("textutils__unit_convert", {"value": 50, "from_unit": "%", "to_unit": "1"}),
        ("textutils__unit_convert", {"value": 5, "from_unit": "s", "to_unit": "go"}),
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
    check("50 % = 0.5 (ratio)", "0.5" in at(r, "textutils__unit_convert", 4))
    check("5 s -> go refusé (dimensions différentes)", "ERREUR" in at(r, "textutils__unit_convert", 5)
          and "dimensions" in at(r, "textutils__unit_convert", 5))
    check("ko (taille) -> ko/s (débit) refusé", "ERREUR" in at(r, "textutils__unit_convert", 6))
    check("unité cible inconnue -> ERREUR", "ERREUR" in at(r, "textutils__unit_convert", 7))

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
        ("scan__port_scan", {"target": "localhost", "ports": "22,80"}),
        ("scan__port_scan", {"target": "printer.local", "ports": "80"}),
        ("scan__port_scan", {"target": "cible; rm -rf /", "ports": "80"}),
        ("scan__port_scan", {"target": "-sS", "ports": "80"}),
        ("scan__port_scan", {"target": "127.0.0.1", "ports": "1-65535"}),
        ("scan__service_detect", {"target": "1.1.1.1"}),
        ("scan__os_detect", {"target": "-O; rm -rf /"}),
        ("scan__os_detect", {"target": "127.0.0.1"}),
    ])
    await p.stop()
    check("scan cible publique refusée sans allow_public",
          "ERREUR" in at(r, "scan__port_scan", 0) and "allow_public" in at(r, "scan__port_scan", 0))
    check("scan 127.0.0.1 accepté (nmap absent -> ERREUR propre ou scan)",
          at(r, "scan__port_scan", 1) != "" and "hors LAN" not in at(r, "scan__port_scan", 1))
    check("scan localhost accepté (bug docstring corrigé)",
          "hors LAN" not in at(r, "scan__port_scan", 2))
    check("scan *.local accepté", "hors LAN" not in at(r, "scan__port_scan", 3))
    check("scan cible avec injection -> ERREUR", "ERREUR" in at(r, "scan__port_scan", 4))
    check("scan cible '-sS' (option) -> ERREUR", "ERREUR" in at(r, "scan__port_scan", 5))
    check("scan plage 1-65535 refusée (>4096 ports)",
          "ERREUR" in at(r, "scan__port_scan", 6) and "trop large" in at(r, "scan__port_scan", 6))
    check("service_detect 1.1.1.1 refusé", "ERREUR" in at(r, "scan__service_detect"))
    check("os_detect cible invalide -> ERREUR (regex ajouté)", "ERREUR" in at(r, "scan__os_detect", 0))
    check("os_detect 127.0.0.1 accepté", at(r, "scan__os_detect", 1) != ""
          and "hors LAN" not in at(r, "scan__os_detect", 1))

    # --- lan / snmp / docker : présence des outils + erreurs propres ---
    p = McpToolProvider({"lan": SERVERS["lan"], "snmp": SERVERS["snmp"], "docker": SERVERS["docker"]})
    await p.start()
    names = p.names()
    r = await call_all(p, [
        ("lan__arp_table", {}),
        ("lan__dhcp_leases", {}),
        ("lan__dhcp_leases", {"path": "/etc/hostname"}),
        ("lan__dhcp_leases", {"path": "/var/lib/dhcp/dhcpd.leases"}),
        ("lan__lan_hosts", {"subnet": "pas-un-subnet"}),
        ("lan__lan_hosts", {"subnet": "8.8.8.0/24"}),
        ("lan__lan_hosts", {"subnet": "192.168.1.0/24"}),
        ("snmp__snmp_get", {"target": "127.0.0.1", "oid": "1.3.6.0.2.1.1.5.0"}),
        ("snmp__snmp_get", {"target": "127.0.0.1", "oid": "1.3.6.0; rm -rf /"}),
        ("snmp__snmp_get", {"target": "--option", "oid": "1.3.6.0.2.1.1.5.0"}),
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
    check("dhcp_leases sans fichier -> ERREUR propre ou liste", at(r, "lan__dhcp_leases", 0) != "")
    check("dhcp_leases path hors /var/lib,/etc -> refusé",
          "ERREUR" in at(r, "lan__dhcp_leases", 1) and "autorisés" in at(r, "lan__dhcp_leases", 1))
    check("dhcp_leases path autorisé mais absent -> ERREUR propre",
          at(r, "lan__dhcp_leases", 2).startswith("ERREUR"))
    check("lan_hosts subnet invalide -> ERREUR", "ERREUR" in at(r, "lan__lan_hosts", 0))
    check("lan_hosts 8.8.8.0/24 (public) refusé",
          "ERREUR" in at(r, "lan__lan_hosts", 1) and "public" in at(r, "lan__lan_hosts", 1))
    check("lan_hosts 192.168.1.0/24 (privé) accepté",
          "public" not in at(r, "lan__lan_hosts", 2).lower() or "refusé" not in at(r, "lan__lan_hosts", 2))
    check("snmp_get répond (timeout/ERREUR propre acceptés)", at(r, "snmp__snmp_get", 0) != "")
    check("snmp OID avec injection -> ERREUR", "ERREUR" in at(r, "snmp__snmp_get", 1))
    check("snmp target '--option' -> ERREUR (anti-option)", "ERREUR" in at(r, "snmp__snmp_get", 2))
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
        ("sysinfo__service_status", {"name": "--no-pager"}),
        ("sysinfo__service_status", {"name": "systemd-journald"}),
        ("sysinfo__logs", {"service": "--no-pager"}),
        ("sysinfo__smart_disk", {"device": "/dev/sda"}),
        ("sysinfo__smart_disk", {"device": "pas-un-device"}),
        ("sysinfo__battery", {}),
    ])
    await p.stop()
    check("services failed répond", at(r, "sysinfo__services") != "")
    check("service_status injection -> ERREUR", "ERREUR" in at(r, "sysinfo__service_status", 0))
    check("service_status '--no-pager' (option) -> ERREUR", "ERREUR" in at(r, "sysinfo__service_status", 1))
    check("logs '--no-pager' (option) -> ERREUR", "ERREUR" in at(r, "sysinfo__logs"))
    check("service_status valide -> réponse", at(r, "sysinfo__service_status", 1) != "")
    check("smart_disk répond (smartctl absent -> ERREUR propre ou données)",
          at(r, "sysinfo__smart_disk", 0) != "")
    check("smart_disk device invalide -> ERREUR", "ERREUR" in at(r, "sysinfo__smart_disk", 1))
    check("battery répond (fixe -> message ou données)", at(r, "sysinfo__battery") != "")

    # --- nettools : bandwidth_test + SSRF http_get ---
    p = McpToolProvider({"nettools": SERVERS["nettools"]})
    await p.start()
    names = p.names()
    r = await call_all(p, [
        ("nettools__bandwidth_test", {"server": "", "duration": 1}),
        ("nettools__bandwidth_test", {"server": "192.168.1.10", "duration": 1}),
        ("nettools__bandwidth_test", {"server": "-c", "duration": 1}),
        ("nettools__http_get", {"url": "http://169.254.169.254/latest/meta-data"}),
        ("nettools__http_get", {"url": "http://127.0.0.1:9931/v1/models"}),
        ("nettools__http_get", {"url": "http://[::1]/"}),
    ])
    await p.stop()
    check("outil exposé: nettools__bandwidth_test", "nettools__bandwidth_test" in names)
    check("bandwidth_test sans server -> ERREUR (plus de serveur public par défaut)",
          "ERREUR" in at(r, "nettools__bandwidth_test", 0) and "server requis" in at(r, "nettools__bandwidth_test", 0))
    check("bandwidth_test avec server -> tentative (iperf3 absent -> ERREUR propre)",
          at(r, "nettools__bandwidth_test", 1) != "" and "server requis" not in at(r, "nettools__bandwidth_test", 1))
    check("bandwidth_test server '-c' (option) -> ERREUR", "ERREUR" in at(r, "nettools__bandwidth_test", 2))
    check("http_get 169.254.169.254 (métadonnées cloud) -> ERREUR SSRF",
          "SSRF" in at(r, "nettools__http_get", 0))
    check("http_get 127.0.0.1 -> ERREUR SSRF", "SSRF" in at(r, "nettools__http_get", 1))
    check("http_get [::1] -> ERREUR SSRF", "SSRF" in at(r, "nettools__http_get", 2))

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
        ("fetch__fetch_page", {"url": "http://169.254.169.254/latest/meta-data"}),
        ("fetch__fetch_page", {"url": "http://127.0.0.1:8080/"}),
        ("rss__rss_latest", {"feed_url": "ftp://x"}),
        ("rss__rss_latest", {"feed_url": "http://169.254.169.254/feed"}),
        ("exploitdb__exploit_search", {"query": "x; rm -rf /"}),
        ("exploitdb__exploit_info", {"edb_id": "abc"}),
        ("epss__epss_score", {"cves": "pas-une-cve"}),
        ("epss__epss_history", {"cve": "CVE-2024-3094"}),
        ("ghsa__ghsa_info", {"ghsa_id": "pas-un-ghsa"}),
        ("kev__kev_info", {"cve_id": "pas-une-cve"}),
    ])
    await p.stop()
    check("fetch_page schéma refusé", "ERREUR" in at(r, "fetch__fetch_page", 0))
    check("fetch_headers url invalide", "ERREUR" in at(r, "fetch__fetch_headers"))
    check("fetch_page métadonnées cloud -> ERREUR SSRF", "SSRF" in at(r, "fetch__fetch_page", 1))
    check("fetch_page 127.0.0.1 -> ERREUR SSRF", "SSRF" in at(r, "fetch__fetch_page", 2))
    check("rss schéma refusé", "ERREUR" in at(r, "rss__rss_latest", 0))
    check("rss métadonnées cloud -> ERREUR SSRF", "SSRF" in at(r, "rss__rss_latest", 1))
    check("exploit_search injection -> ERREUR", "ERREUR" in at(r, "exploitdb__exploit_search"))
    check("exploit_info id non numérique -> ERREUR", "ERREUR" in at(r, "exploitdb__exploit_info"))
    check("epss_score sans CVE valide -> ERREUR", "ERREUR" in at(r, "epss__epss_score"))
    check("epss_history CVE valide -> appel réseau (ERREUR réseau acceptée dans le sandbox)",
          at(r, "epss__epss_history") != "")
    check("ghsa_info id invalide -> ERREUR", "ERREUR" in at(r, "ghsa__ghsa_info"))
    check("kev_info id invalide -> ERREUR", "ERREUR" in at(r, "kev__kev_info"))

    # --- rss/fetch : parsing réel RSS 2.0 + Atom contre un serveur HTTP local ---
    import http.server
    import threading

    RSS20 = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Blog Test RSS</title><description>desc rss</description>
<item><title>Article RSS Un</title><link>https://exemple.test/a1</link><pubDate>Mon, 27 Oct 2026 10:00:00 GMT</pubDate><description>résumé un</description></item>
<item><title>Article RSS Deux</title><link>https://exemple.test/a2</link><pubDate>Tue, 28 Oct 2026 10:00:00 GMT</pubDate><description>résumé deux</description></item>
</channel></rss>"""
    ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Blog Test Atom</title><subtitle>desc atom</subtitle>
<entry><title>Article Atom Un</title><link rel="alternate" href="https://exemple.test/e1"/><updated>2026-10-28T09:00:00Z</updated><summary>résumé atom un</summary></entry>
<entry><title>Article Atom Deux</title><link rel="self" href="https://exemple.test/feed"/><link rel="alternate" href="https://exemple.test/e2"/><updated>2026-10-29T09:00:00Z</updated><content>contenu deux</content></entry>
</feed>"""
    PAGE = "<html><head><title>Doc</title><style>x{}</style></head><body><h1>Titre</h1><p>Bonjour <b>monde</b></p><script>evil()</script></body></html>"

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = {"/rss.xml": RSS20, "/atom.xml": ATOM, "/page": PAGE}.get(self.path, "")
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/xml" if self.path != "/page" else "text/html")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    os.environ["JARVIS_FETCH_ALLOW_PRIVATE"] = "1"  # override test local uniquement
    try:
        p = McpToolProvider({"rss": SERVERS["rss"], "fetch": SERVERS["fetch"]})
        await p.start()
        r = await call_all(p, [
            ("rss__rss_latest", {"feed_url": f"http://127.0.0.1:{port}/rss.xml", "n": 5}),
            ("rss__rss_latest", {"feed_url": f"http://127.0.0.1:{port}/atom.xml", "n": 5}),
            ("rss__rss_check", {"feed_url": f"http://127.0.0.1:{port}/atom.xml"}),
            ("fetch__fetch_page", {"url": f"http://127.0.0.1:{port}/page", "max_chars": 500}),
        ])
        await p.stop()
    finally:
        del os.environ["JARVIS_FETCH_ALLOW_PRIVATE"]
        httpd.shutdown()
    check("RSS 2.0: titre du site + 2 articles avec liens",
          "Blog Test RSS" in at(r, "rss__rss_latest", 0)
          and "Article RSS Un" in at(r, "rss__rss_latest", 0)
          and "https://exemple.test/a1" in at(r, "rss__rss_latest", 0))
    check("Atom: titres résolus via namespace (bug corrigé)",
          "Article Atom Un" in at(r, "rss__rss_latest", 1))
    check("Atom: lien rel=alternate choisi (pas le rel=self)",
          "https://exemple.test/e2" in at(r, "rss__rss_latest", 1)
          and "exemple.test/feed" not in at(r, "rss__rss_latest", 1).split("e2")[0][-40:])
    check("Atom: contenu en fallback de summary pour l'article 2",
          "contenu deux" in at(r, "rss__rss_latest", 1))
    check("rss_check Atom: titre du feed lu", "Blog Test Atom" in at(r, "rss__rss_check"))
    check("fetch_page local: texte extrait, script/style retirés",
          "Bonjour" in at(r, "fetch__fetch_page") and "monde" in at(r, "fetch__fetch_page")
          and "evil()" not in at(r, "fetch__fetch_page"))

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} tests OK")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
