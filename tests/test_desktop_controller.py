"""Test du controller desktop (sans fenêtre Tk) : boucle asyncio + queue + mock LLM.

Valide que les événements du backend remontent bien dans la queue du thread
Tk : backend_ready -> reasoning -> tool_call -> content -> done, et le statut
du serveur (live sur 9931).

Usage: uv run python tests/test_desktop_controller.py
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from jarvis.desktop.controller import DesktopController  # noqa: E402
from mock_llm import start_mock  # noqa: E402

CHATMOCK = ROOT / "tests" / "config.chatmock.toml"


def drain(ctrl: DesktopController, timeout: float = 30.0) -> list[dict]:
    events = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            events.append(ctrl.events.get(timeout=0.5))
        except queue.Empty:
            if events and events[-1].get("type") == "done":
                break
    return events


def main() -> int:
    srv, thread = start_mock(18082)
    try:
        ctrl = DesktopController(config_path=str(CHATMOCK))
        ctrl.start()
        events = drain(ctrl, timeout=60)
        assert any(e["type"] == "backend_ready" and e["tools"] == ["echo__ping"]
                   for e in events), events
        print("OK  backend_ready:", [e for e in events if e["type"] == "backend_ready"])

        ctrl.stream_turn("fais un ping vers 127.0.0.1")
        events = drain(ctrl, timeout=60)
        kinds = [e["type"] for e in events]
        assert "reasoning" in kinds and "tool_call" in kinds and "content" in kinds \
            and kinds[-1] == "done", kinds
        content = "".join(e["text"] for e in events if e["type"] == "content")
        assert "Résultat du ping" in content, content
        print("OK  stream_turn:", kinds)

        ctrl.refresh_status()
        events = drain(ctrl, timeout=15)
        # le mock écoute sur 18082, pas 9931 : le statut doit être offline ici
        st = [e for e in events if e["type"] == "server_status"]
        assert st and st[0]["port"] == 18082, events
        print("OK  server_status:", st[0]["state"], st[0]["port"])

        ctrl.reset()
        time.sleep(1.0)
        assert ctrl.backend.history == []
        ctrl.stop()
        print("OK  stop propre")
        return 0
    finally:
        srv.shutdown()
        thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
