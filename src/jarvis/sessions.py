"""Sessions de conversation persistantes, globales (pas par dossier).

Chaque session = un fichier JSON dans ~/.cache/jarvis/sessions/<id>.json
contenant l'historique complet (messages role/content/tool_calls). Au
démarrage, la dernière session est reprise automatiquement ; /sessions liste
les conversations passées, /load N en reprend une, /new en démarre une autre.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path


class SessionStore:
    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir) / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.dir / f"{sid}.json"

    @staticmethod
    def new_id() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def title_of(history: list[dict]) -> str:
        for m in history:
            if m.get("role") == "user" and m.get("content"):
                return str(m["content"])[:80]
        return "(vide)"

    def save(self, sid: str, history: list[dict]) -> None:
        data = {
            "id": sid,
            "title": self.title_of(history),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "messages": history,
        }
        tmp = self._path(sid).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(sid))  # écriture atomique

    def list(self, limit: int = 20) -> list[dict]:
        out = []
        for f in self.dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({
                "id": data.get("id", f.stem),
                "title": data.get("title", "?"),
                "updated": data.get("updated", ""),
                "count": len(data.get("messages", [])),
            })
        out.sort(key=lambda d: d["updated"], reverse=True)
        return out[:limit]

    def load(self, sid: str) -> list[dict]:
        data = json.loads(self._path(sid).read_text(encoding="utf-8"))
        return data.get("messages", [])

    def latest_id(self) -> str | None:
        sessions = self.list(limit=1)
        return sessions[0]["id"] if sessions else None

    def delete(self, sid: str) -> bool:
        """Supprime le fichier de session. Retourne True si supprimé."""
        path = self._path(sid)
        try:
            path.unlink()
            return True
        except OSError:
            return False
