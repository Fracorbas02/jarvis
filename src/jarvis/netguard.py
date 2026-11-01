"""Garde SSRF partagée (côté app, stdlib uniquement).

Même politique que les MCP servers fetch/rss/nettools : http(s) uniquement,
toutes les IP résolues sont vérifiées, loopback / link-local (169.254.169.254
métadonnées cloud) / non routable / multicast toujours refusés, IP privées
refusées sauf allow_private=True. Utilisée pour les images rendues dans le
markdown GTK, qu'un LLM peut peupler de n'importe quelle URL.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.parse


def check_url_ssrf(url: str, allow_private: bool = False) -> str | None:
    """Retourne un message d'erreur si l'URL est refusée, None si OK."""
    if not url.startswith(("http://", "https://")):
        return "ERREUR: url doit commencer par http:// ou https://"
    host = urllib.parse.urlsplit(url).hostname or ""
    if not host:
        return "ERREUR: url sans hôte."
    try:
        ipaddress.ip_address(host)
        addrs = [host]
    except ValueError:
        try:
            addrs = [info[4][0] for info in socket.getaddrinfo(host, None)]
        except socket.gaierror:
            return f"ERREUR: hôte '{host}' non résoluble."
    # override explicite (tests locaux) : désactive les contrôles d'IP,
    # jamais la vérification de schéma
    if os.environ.get("JARVIS_FETCH_ALLOW_PRIVATE") == "1":
        return None
    for a in addrs:
        addr = ipaddress.ip_address(a)
        if addr.is_loopback or addr.is_link_local or addr.is_unspecified or addr.is_multicast:
            return (f"ERREUR SSRF: '{host}' résout vers {addr} "
                    "(loopback/link-local/non routable interdit).")
        if not allow_private and addr.is_private:
            return f"ERREUR SSRF: '{host}' résout vers {addr} (privée)."
    return None
