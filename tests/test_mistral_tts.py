"""Tests du TTS API Mistral (jarvis.mistral_tts) contre un mock HTTP local.

Valide sans r\u00e9seau ni cl\u00e9 r\u00e9elle :
- save/load de la cl\u00e9 API locale (mistral.json, chmod 0600, env prioritaire) ;
- POST /v1/audio/speech -> d\u00e9codage base64 de audio_data, payload attendu
  (model, input, voice_id, response_format) ;
- gestion des erreurs HTTP (401 cl\u00e9 invalide, 404 voix/mod\u00e8le inconnu, 403) ;
- GET /v1/audio/voices -> liste de voix preset/custom ;
- config : [voice.tts_api] pars\u00e9 depuis un TOML.

Lancement : uv run pytest tests/test_mistral_tts.py -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import stat
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jarvis.mistral_tts import (MistralTTSClient, MistralTTSError,
                                 delete_api_key, has_api_key, load_api_key,
                                 preset_voice_options, save_api_key)

FAKE_AUDIO = b"ID3-fake-mp3-bytes"  # simulacre de mp3


class Handler(BaseHTTPRequestHandler):
    """Mock de l'API Mistral : /v1/audio/speech + /v1/audio/voices."""

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        auth = self.headers.get("Authorization", "")
        if self.path.rstrip("/") == "/v1/audio/speech":
            if auth != "Bearer test-key-123":
                self._json({"message": "invalid api key"}, code=401)
                return
            if body.get("voice_id") == "voice_inconnue":
                self._json({"message": "voice not found"}, code=404)
                return
            if body.get("input") == "MODERE":
                self._json({"message": "content moderated"}, code=403)
                return
            # on v\u00e9rifie le contrat et on renvoie l'audio en base64
            assert body.get("model"), "model manquant"
            assert body.get("input"), "input manquant"
            Handler.last_payload = body
            self._json({"audio_data": base64.b64encode(FAKE_AUDIO).decode()})
            return
        self._json({"message": "not found"}, code=404)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer test-key-123":
            self._json({"message": "invalid api key"}, code=401)
            return
        if self.path.startswith("/v1/audio/voices"):
            self._json({"result": {"data": [
                {"id": "voice_592", "name": "Marie", "slug": "voice_592",
                 "languages": ["fr"], "gender": "female", "type": "preset"},
                {"id": "gb_jane_neutral", "name": "Jane", "slug": "gb_jane_neutral",
                 "languages": ["en-gb"], "gender": "female", "type": "preset"},
                {"id": "cl_abc123", "name": "Ma voix clon\u00e9e",
                 "languages": ["fr"], "gender": "", "type": "custom"},
            ], "next": None}})
            return
        self._json({"message": "not found"}, code=404)


def _start_mock() -> tuple[ThreadingHTTPServer, object]:
    import threading
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


def test_api_key_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert not has_api_key()
    p = save_api_key("cle-secrete-xyz")
    assert has_api_key() and load_api_key() == "cle-secrete-xyz"
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o600, f"mistral.json doit \u00eatre 0600, pas {oct(mode)}"
    # le contenu ne doit pas contenir la cl\u00e9 en clair dans un autre fichier
    delete_api_key()
    assert not has_api_key()
    # env prioritaire sur le fichier
    save_api_key("cle-fichier")
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-env")
    assert load_api_key() == "cle-env"


def test_preset_voices_sane():
    opts = preset_voice_options()
    assert any(vid == "voice_592" for _label, vid in opts)       # Marie FR
    assert any(vid == "gb_jane_neutral" for _label, vid in opts)  # EN-GB
    assert any("DE" in label for label, _ in opts)                # allemand existe


def test_synthesize_decodes_base64():
    srv, t = _start_mock()
    port = srv.server_address[1]
    try:
        async def go():
            c = MistralTTSClient("test-key-123",
                                 base_url=f"http://127.0.0.1:{port}",
                                 model="voxtral-mini-tts-2603")
            try:
                audio = await c.synthesize("Bonjour Jarvis", voice_id="voice_592",
                                           response_format="mp3")
                return audio
            finally:
                await c.aclose()
        audio = asyncio.run(go())
        assert audio == FAKE_AUDIO
        # le payload envoy\u00e9 respecte le contrat de l'API
        assert Handler.last_payload["model"] == "voxtral-mini-tts-2603"
        assert Handler.last_payload["input"] == "Bonjour Jarvis"
        assert Handler.last_payload["voice_id"] == "voice_592"
        assert Handler.last_payload["response_format"] == "mp3"
    finally:
        srv.shutdown()
        srv.server_close()


def test_synthesize_errors():
    srv, t = _start_mock()
    port = srv.server_address[1]
    try:
        async def go():
            out = {}
            # mauvaise cl\u00e9 -> 401
            c = MistralTTSClient("mauvaise", base_url=f"http://127.0.0.1:{port}")
            try:
                await c.synthesize("x")
            except MistralTTSError as e:
                out["401"] = "401" in str(e) or "invalide" in str(e)
            finally:
                await c.aclose()
            # voix inconnue -> 404
            c = MistralTTSClient("test-key-123", base_url=f"http://127.0.0.1:{port}")
            try:
                await c.synthesize("x", voice_id="voice_inconnue")
            except MistralTTSError as e:
                out["404"] = "404" in str(e)
            finally:
                await c.aclose()
            # texte vide -> erreur locale avant requ\u00eate
            c = MistralTTSClient("test-key-123", base_url=f"http://127.0.0.1:{port}")
            try:
                await c.synthesize("   ")
            except MistralTTSError as e:
                out["empty"] = "vide" in str(e)
            finally:
                await c.aclose()
            return out
        out = asyncio.run(go())
        assert out.get("401") and out.get("404") and out.get("empty"), out
    finally:
        srv.shutdown()
        srv.server_close()


def test_synthesize_no_key_raises():
    async def go():
        c = MistralTTSClient("", base_url="http://127.0.0.1:1")
        try:
            await c.synthesize("x")
        except MistralTTSError as e:
            return "cl\u00e9" in str(e)
        finally:
            await c.aclose()
        return False
    # le constructeur l\u00e8ve d\u00e9j\u00e0 sur cl\u00e9 vide
    try:
        MistralTTSClient("  ", base_url="http://127.0.0.1:1")
        raise AssertionError("devait lever sur cl\u00e9 vide")
    except MistralTTSError:
        pass


def test_list_voices():
    srv, t = _start_mock()
    port = srv.server_address[1]
    try:
        async def go():
            c = MistralTTSClient("test-key-123", base_url=f"http://127.0.0.1:{port}")
            try:
                return await c.list_voices()
            finally:
                await c.aclose()
        voices = asyncio.run(go())
        ids = [v.id for v in voices]
        assert "voice_592" in ids and "gb_jane_neutral" in ids and "cl_abc123" in ids
        custom = [v for v in voices if v.kind == "custom"]
        assert custom and custom[0].name == "Ma voix clon\u00e9e"
        fr = [v for v in voices if "fr" in v.language]
        assert fr
    finally:
        srv.shutdown()
        srv.server_close()


def test_config_tts_api_toml(tmp_path):
    from jarvis.config import load_config
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[llm]\nbase_url = \"http://127.0.0.1:9931/v1\"\n\n"
        "[voice.tts_api]\n"
        "enabled = true\n"
        "model = \"voxtral-tts-2603\"\n"
        "voice_id = \"gb_jane_neutral\"\n"
        "response_format = \"wav\"\n",
        encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.tts_api.enabled is True
    assert cfg.tts_api.model == "voxtral-tts-2603"
    assert cfg.tts_api.voice_id == "gb_jane_neutral"
    assert cfg.tts_api.response_format == "wav"
    assert cfg.tts_api.base_url == "https://api.mistral.ai"  # d\u00e9faut conserv\u00e9


def test_config_tts_api_default_disabled(tmp_path):
    from jarvis.config import load_config
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[llm]\nbase_url = \"http://127.0.0.1:9931/v1\"\n",
                         encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.tts_api.enabled is False  # off par d\u00e9faut -> TTS local


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        sig = inspect.signature(fn)
        kwargs = {}
        if "tmp_path" in sig.parameters or "monkeypatch" in sig.parameters:
            continue  # lance via pytest pour les fixtures
        try:
            fn()
            print("OK ", fn.__name__)
        except Exception as e:
            failed += 1
            print("FAIL", fn.__name__, e)
    print("done, failed:", failed)
