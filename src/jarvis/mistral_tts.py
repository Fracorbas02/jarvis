"""TTS via l'API Mistral (Voxtral) : POST https://api.mistral.ai/v1/audio/speech.

La clé API n'est JAMAIS dans config.toml : elle est stockée localement dans
~/.config/jarvis/mistral.json (ou $XDG_CONFIG_HOME/jarvis/mistral.json), avec
chmod 600, et peut aussi venir de l'env MISTRAL_API_KEY (qui prime si défini).

Endpoint (docs Mistral, 2026) :
  POST {base_url}/v1/audio/speech   body: {"model", "input", "voice_id",
                                            "response_format": "mp3"}
  -> 200 application/json {"audio_data": "<base64>"} (non-stream)
  GET  {base_url}/v1/audio/voices   -> liste des voix (preset + workspace)

Modèles TTS : "voxtral-mini-tts-2603" (recommandé, le plus léger/rapide),
"voxtral-tts-2603" (qualité max). Voix : voice_id preset (ex. "voice_592"
Marie FR, "gb_jane_neutral"...) ou voix clonée du workspace.
"""

from __future__ import annotations

import base64
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import httpx

from .gpu import config_dir

DEFAULT_BASE_URL = "https://api.mistral.ai"
DEFAULT_MODEL = "voxtral-mini-tts-3-0"  # Voxtral TTS 3 (20/10/2026)
# Modeles TTS connus en fallback (combo UI si l'API est hors ligne).
# La liste reelle est rafraichie via GET /v1/models (voir list_tts_models) :
# voxtral-mini-tts-3-0 = Voxtral TTS 3 (20/10/2026, 35+ langues, public preview).
TTS_MODELS = ["voxtral-mini-tts-3-0", "voxtral-mini-tts-2603", "voxtral-tts-2603",
              "voxtral-mini-tts-latest", "voxtral-tts-latest"]


def is_tts_model(model_id: str) -> bool:
    return "tts" in model_id.lower()
# voix preset pratiques (doc Mistral "Voices") : nom -> (voice_id, langue)
PRESET_VOICES: dict[str, tuple[str, str]] = {
    "Marie (FR, femme)": ("voice_592", "fr"),
    "Manon (FR, femme)": ("voice_3137", "fr"),
    "Lea (FR, femme)": ("voice_589", "fr"),
    "Camille (FR, femme)": ("voice_3141", "fr"),
    "Jane (EN-GB, femme)": ("gb_jane_neutral", "en-gb"),
    "Sheila (EN-US, femme)": ("voice_478", "en-us"),
    "Paul (EN-US, homme)": ("en_paul_neutral", "en-us"),
    "Rob (EN-US, homme)": ("voice_485", "en-us"),
    "James (EN-US, homme)": ("voice_3150", "en-us"),
    "David (EN-GB, homme)": ("voice_1788", "en-gb"),
    "Heidi (DE, femme)": ("voice_481", "de"),
    "Lorenzo (IT, homme)": ("voice_482", "it"),
    "Elena (ES, femme)": ("voice_483", "es"),
    "Gabriela (PT, femme)": ("voice_480", "pt"),
}


class MistralTTSError(RuntimeError):
    pass


def api_key_path() -> Path:
    return config_dir() / "mistral.json"


def load_api_key() -> str:
    """Clé API : env MISTRAL_API_KEY (prime) puis fichier local mistral.json."""
    env_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        data = json.loads(api_key_path().read_text(encoding="utf-8"))
        return str(data.get("api_key", "")).strip()
    except Exception:  # noqa: BLE001
        return ""


def save_api_key(api_key: str) -> Path:
    """Enregistre la clé localement (fichier 0600, jamais loguée/affichée)."""
    p = api_key_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"api_key": api_key}, indent=2), encoding="utf-8")
    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    return p


def delete_api_key() -> None:
    try:
        api_key_path().unlink()
    except OSError:
        pass


def has_api_key() -> bool:
    return bool(load_api_key())


@dataclass
class MistralVoice:
    id: str          # voice_id à passer en requête ("voice_592", "gb_jane_neutral", id custom)
    name: str        # libellé lisible
    language: str = ""
    gender: str = ""
    kind: str = "preset"  # "preset" | "custom"


class MistralTTSClient:
    """Client minimal de l'API audio de Mistral (httpx, comme le reste du projet)."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, timeout: float = 60.0):
        if not api_key.strip():
            raise MistralTTSError(
                "clé API Mistral absente — la définir dans Réglages > Voix API "
                "(ou export MISTRAL_API_KEY)")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def synthesize(self, text: str, voice_id: str | None = None,
                         response_format: str = "mp3") -> bytes:
        """Synthétise `text` -> octets audio (mp3 par défaut).

        L'API renvoie {"audio_data": "<base64>"} en non-stream ; on décode.
        """
        if not text.strip():
            raise MistralTTSError("texte vide")
        payload: dict = {"model": self.model, "input": text,
                         "response_format": response_format}
        if voice_id:
            payload["voice_id"] = voice_id
        try:
            resp = await self._http.post("/v1/audio/speech", json=payload)
        except httpx.HTTPError as exc:
            raise MistralTTSError(f"API Mistral inaccessible : {exc}") from exc
        if resp.status_code == 401:
            raise MistralTTSError("clé API invalide ou expirée (401)")
        if resp.status_code == 403:
            raise MistralTTSError("403 : modération du contenu ou quota atteint")
        if resp.status_code == 404:
            raise MistralTTSError(f"404 : modèle {self.model!r} ou voix {voice_id!r} inconnu")
        if resp.status_code != 200:
            # SECURITE : on ne met JAMAIS le corps de la reponse dans le message
            # (un proxy/WAF en erreur peut echoer des en-tetes, dont Authorization).
            # Status + Content-Type + longueur seulement.
            ctype = resp.headers.get("Content-Type", "?")
            raise MistralTTSError(
                f"API Mistral HTTP {resp.status_code} ({ctype}, {len(resp.content)} o) "
                f"sur /v1/audio/speech — corps non journalisé")
        # non-stream -> JSON {"audio_data": base64} ; stream -> SSE (non utilisé ici)
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            try:
                data = resp.json()
            except ValueError as exc:
                raise MistralTTSError("réponse 200 non-JSON de /v1/audio/speech "
                                      f"({resp.headers.get('Content-Type', '?')})") from exc
            b64 = data.get("audio_data")
            if not b64:
                # on ne cite que les cles, pas les valeurs (audio_data peut être gros)
                raise MistralTTSError("réponse sans audio_data "
                                      f"(cles: {sorted(data)[:10]})")
            return base64.b64decode(b64)
        # réponse binaire directe (certains proxys) : on la prend telle quelle
        return resp.content

    async def list_tts_models(self) -> list[str]:
        """GET /v1/models -> ids des modeles TTS du compte (filtre 'tts').

        Permet de ne jamais rater un nouveau modele (ex. voxtral-mini-tts-3-0)
        sans mettre a jour le code. Leve MistralTTSError sur erreur HTTP."""
        try:
            resp = await self._http.get("/v1/models")
        except httpx.HTTPError as exc:
            raise MistralTTSError(f"GET /v1/models inaccessible : {exc}") from exc
        if resp.status_code == 401:
            raise MistralTTSError("cle API invalide ou expiree (401) sur /v1/models")
        if resp.status_code != 200:
            raise MistralTTSError(
                f"GET /v1/models -> HTTP {resp.status_code} "
                f"({resp.headers.get('Content-Type', '?')}, corps non journalise)")
        try:
            data = resp.json()
        except ValueError as exc:
            raise MistralTTSError("GET /v1/models -> reponse non-JSON") from exc
        items = data.get("data", data) if isinstance(data, dict) else data
        models = sorted({m.get("id", "") for m in items
                         if isinstance(m, dict) and m.get("id")
                         and is_tts_model(str(m["id"]))})
        return models

    async def list_voices(self, page_size: int = 100) -> list[MistralVoice]:
        """GET /v1/audio/voices (paginé) -> voix preset + custom du workspace."""
        voices: list[MistralVoice] = []
        cursor = None
        for _ in range(10):  # plafond de pages
            params: dict = {"page_size": page_size}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = await self._http.get("/v1/audio/voices", params=params)
            except httpx.HTTPError as exc:
                raise MistralTTSError(f"GET /v1/audio/voices inaccessible : {exc}") from exc
            if resp.status_code == 401:
                raise MistralTTSError("clé API invalide ou expirée (401) sur /v1/audio/voices")
            if resp.status_code != 200:
                raise MistralTTSError(
                    f"GET /v1/audio/voices -> HTTP {resp.status_code} "
                    f"({resp.headers.get('Content-Type', '?')}, corps non journalisé)")
            try:
                data = resp.json()
            except ValueError as exc:
                raise MistralTTSError("GET /v1/audio/voices -> réponse non-JSON") from exc
            page = data.get("result", data) if isinstance(data, dict) else data
            items = page.get("data", []) if isinstance(page, dict) else []
            for v in items:
                if not isinstance(v, dict):
                    continue
                vid = v.get("slug") or v.get("id") or ""
                if not vid:
                    continue
                langs = v.get("languages") or []
                voices.append(MistralVoice(
                    id=vid,
                    name=str(v.get("name") or vid),
                    language=",".join(str(l) for l in langs) if langs else str(v.get("language") or ""),
                    gender=str(v.get("gender") or ""),
                    kind=str(v.get("type") or "preset"),
                ))
            cursor = page.get("next") if isinstance(page, dict) else None
            if not cursor or not items:
                break
        return voices

    async def aclose(self) -> None:
        await self._http.aclose()


def preset_voice_options() -> list[tuple[str, str]]:
    """[(label, voice_id)] pour un sélecteur, trié par langue puis nom."""
    return [(label, vid) for label, (vid, _lang) in
            sorted(PRESET_VOICES.items(), key=lambda kv: (kv[1][1], kv[0]))]
