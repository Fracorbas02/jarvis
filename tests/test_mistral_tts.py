"""Tests du TTS API Mistral (jarvis.mistral_tts) contre un mock HTTP local.

Valide sans réseau ni clé réelle :
- save/load de la clé API locale (mistral.json, chmod 0600, env prioritaire) ;
- POST /v1/audio/speech -> décodage base64 de audio_data, payload attendu
  (model, input, voice_id, response_format) ;
- gestion des erreurs HTTP (401 clé invalide, 404 voix/modèle inconnu, 403) ;
- GET /v1/audio/voices -> liste de voix preset/custom ;
- config : [voice.tts_api] parsé depuis un TOML.

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
            # on vérifie le contrat et on renvoie l'audio en base64
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
        if self.path.startswith("/v1/models"):
            self._json({"object": "list", "data": [
                {"id": "mistral-large-latest", "object": "model"},
                {"id": "voxtral-mini-tts-3-0", "object": "model"},
                {"id": "voxtral-mini-tts-2603", "object": "model"},
                {"id": "voxtral-tts-2603", "object": "model"},
                {"id": "voxtral-mini-transcribe-3-0", "object": "model"},
            ]})
            return
        if self.path.startswith("/v1/audio/voices"):
            self._json({"result": {"data": [
                {"id": "voice_592", "name": "Marie", "slug": "voice_592",
                 "languages": ["fr"], "gender": "female", "type": "preset"},
                {"id": "gb_jane_neutral", "name": "Jane", "slug": "gb_jane_neutral",
                 "languages": ["en-gb"], "gender": "female", "type": "preset"},
                {"id": "cl_abc123", "name": "Ma voix clonée",
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
    assert mode == 0o600, f"mistral.json doit être 0600, pas {oct(mode)}"
    # le contenu ne doit pas contenir la clé en clair dans un autre fichier
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
        # le payload envoyé respecte le contrat de l'API
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
            # mauvaise clé -> 401
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
            # texte vide -> erreur locale avant requête
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
            return "clé" in str(e)
        finally:
            await c.aclose()
        return False
    # le constructeur lève déjà sur clé vide
    try:
        MistralTTSClient("  ", base_url="http://127.0.0.1:1")
        raise AssertionError("devait lever sur clé vide")
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
        assert custom and custom[0].name == "Ma voix clonée"
        fr = [v for v in voices if "fr" in v.language]
        assert fr
    finally:
        srv.shutdown()
        srv.server_close()


def test_error_messages_never_contain_body():
    """Securite : les messages d'erreur ne doivent contenir AUCUN extrait du
    corps de la reponse HTTP (un proxy/WAF peut y echoer l'en-tete Authorization)."""
    srv, t = _start_mock()
    port = srv.server_address[1]
    try:
        async def go():
            c = MistralTTSClient("test-key-123", base_url=f"http://127.0.0.1:{port}")
            try:
                await c.synthesize("x", voice_id="voice_inconnue")  # -> 404 avec corps JSON
            except MistralTTSError as e:
                msg = str(e)
                assert "voice not found" not in msg, f"corps fuité dans: {msg}"
                assert "404" in msg
                return msg
            finally:
                await c.aclose()
        msg = asyncio.run(go())
        # le 404 voix inconnue a un message metier sans corps HTTP ; on verifie
        # surtout qu'aucun extrait du corps n'y figure
        assert msg and "voice not found" not in msg
    finally:
        srv.shutdown()
        srv.server_close()


def test_list_voices_http_error_raises():
    """list_voices ne doit plus avaler une erreur HTTP : elle leve MistralTTSError."""
    srv, t = _start_mock()
    port = srv.server_address[1]
    try:
        async def go():
            # mauvaise cle -> 401 sur /v1/audio/voices
            c = MistralTTSClient("mauvaise", base_url=f"http://127.0.0.1:{port}")
            try:
                await c.list_voices()
                return None
            except MistralTTSError as e:
                return str(e)
            finally:
                await c.aclose()
        msg = asyncio.run(go())
        assert msg and "401" in msg, msg
    finally:
        srv.shutdown()
        srv.server_close()


def test_tts_models_contient_la_v3():
    """Le fallback statique inclut le dernier modele (Voxtral TTS 3, 20/10/2026)."""
    from jarvis.mistral_tts import TTS_MODELS
    assert "voxtral-mini-tts-3-0" in TTS_MODELS
    assert TTS_MODELS[0] == "voxtral-mini-tts-3-0"


def test_list_tts_models_filtre_dynamiquement():
    """GET /v1/models -> seuls les modeles contenant 'tts' sont gardes,
    y compris le nouveau voxtral-mini-tts-3-0."""
    srv, t = _start_mock()
    port = srv.server_address[1]
    try:
        async def go():
            c = MistralTTSClient("test-key-123", base_url=f"http://127.0.0.1:{port}")
            try:
                return await c.list_tts_models()
            finally:
                await c.aclose()
        models = asyncio.run(go())
        assert "voxtral-mini-tts-3-0" in models
        assert "voxtral-mini-tts-2603" in models
        assert "voxtral-tts-2603" in models
        # les modeles non-TTS sont exclus
        assert "mistral-large-latest" not in models
        assert "voxtral-mini-transcribe-3-0" not in models
    finally:
        srv.shutdown()
        srv.server_close()


def test_list_tts_models_401_raises():
    srv, t = _start_mock()
    port = srv.server_address[1]
    try:
        async def go():
            c = MistralTTSClient("mauvaise", base_url=f"http://127.0.0.1:{port}")
            try:
                await c.list_tts_models()
                return None
            except MistralTTSError as e:
                return str(e)
            finally:
                await c.aclose()
        msg = asyncio.run(go())
        assert msg and "401" in msg
    finally:
        srv.shutdown()
        srv.server_close()


def test_default_model_est_la_v3():
    from jarvis.mistral_tts import DEFAULT_MODEL
    assert DEFAULT_MODEL == "voxtral-mini-tts-3-0"


def test_tts_tags_prefixes_dans_input():
    """synthesize(..., tts_tags=['[emotion=calm]']) -> input commence par le tag."""
    srv, t = _start_mock()
    port = srv.server_address[1]
    try:
        async def go():
            c = MistralTTSClient("test-key-123", base_url=f"http://127.0.0.1:{port}")
            try:
                audio = await c.synthesize("Bonjour !", voice_id="voice_592",
                                           tts_tags=["[emotion=calm]", "[pace=slow]"])
                return audio
            finally:
                await c.aclose()
        audio = asyncio.run(go())
        assert audio == FAKE_AUDIO
        inp = Handler.last_payload["input"]
        assert inp.startswith("[emotion=calm][pace=slow] "), inp
        assert inp.endswith("Bonjour !"), inp
    finally:
        srv.shutdown()
        srv.server_close()


def test_tts_tags_module():
    """extract_lead_tags / strip_all_tags / apply_tags."""
    from jarvis.tts_tags import extract_lead_tags, strip_all_tags, apply_tags
    tags, clean = extract_lead_tags("[emotion=happy] Salut !")
    assert tags == ["[emotion=happy]"] and clean == "Salut !"
    tags2, clean2 = extract_lead_tags("Salut [emotion=happy] !")
    assert tags2 == [] and clean2 == "Salut [emotion=happy] !"
    assert strip_all_tags("a [emotion=sad] b") == "a  b"
    assert apply_tags("Yo", ["[pace=fast]"]) == "[pace=fast] Yo"


def test_agent_extrait_tag_et_nettoie():
    """Tour dont la reponse commence par [emotion=..] : tag dans trace.tts_tags,
    content retourne/historique nettoye."""
    import asyncio as _a
    from jarvis.agent import Agent

    class FakeLLM:
        async def chat(self, messages, tools=None, stream=False, **kw):
            return {"role": "assistant",
                    "content": "[emotion=calm] Voici la réponse."}

    class FakeTools:
        def schemas(self):
            return []
        async def call(self, name, args):
            return ""

    agent = Agent(FakeLLM(), FakeTools(), "sys", stream=False)
    reply, history, trace = _a.run(agent.turn([], "test"))
    assert reply == "Voici la réponse.", repr(reply)
    assert trace.tts_tags == ["[emotion=calm]"], trace.tts_tags
    assert history[-1]["content"] == "Voici la réponse."


def test_voice_tone_internal_tool():
    """L'outil interne jarvis__voice_tone fixe trace.tts_tags via un appel d'outil."""
    import asyncio as _a
    from jarvis.agent import Agent
    from jarvis.voice_tone import voice_tone_tool

    class FakeLLM:
        def __init__(self): self.calls = 0
        async def chat(self, messages, tools=None, stream=False, **kw):
            self.calls += 1
            if self.calls == 1:
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "jarvis__voice_tone",
                                 "arguments": '{"emotion": "excited", "pace": "fast"}'}}]}
            return {"role": "assistant", "content": "Bonne nouvelle !"}

    class FakeTools:
        def schemas(self): return []
        async def call(self, n, a): return ""

    agent = Agent(FakeLLM(), FakeTools(), "sys", stream=False,
                  internal_tools={"jarvis__voice_tone": voice_tone_tool})
    reply, _hist, trace = _a.run(agent.turn([], "test"))
    assert reply == "Bonne nouvelle !"
    assert trace.tts_tags == ["[emotion=excited]", "[pace=fast]"], trace.tts_tags

    # emotion invalide -> ERREUR remontee au modele, pas de tag
    agent2 = Agent(FakeLLM(), FakeTools(), "sys", stream=False,
                   internal_tools={"jarvis__voice_tone": voice_tone_tool})
    async def bad():
        return await voice_tone_tool.fn(agent2, [], {"emotion": " rage"})
    res, _ = _a.run(bad())
    assert res.startswith("ERREUR"), res


def test_streaming_tag_jamais_affiche():
    """En streaming, le tag [emotion=..] du 1er fragment ne doit JAMAIS passer
    par on_content (affichage), mais doit etre dans trace.tts_tags (TTS)."""
    import asyncio as _a
    from jarvis.agent import Agent

    class FakeLLMStream:
        async def chat(self, messages, tools=None, stream=False,
                       on_reasoning=None, on_content=None):
            assert stream
            on_content("[emotion=calm] Voici ")
            on_content("le resultat.")
            return {"role": "assistant",
                    "content": "[emotion=calm] Voici le resultat."}

    class FakeTools:
        def schemas(self): return []
        async def call(self, n, a): return ""

    shown = []
    agent = Agent(FakeLLMStream(), FakeTools(), "sys", stream=True)
    reply, hist, trace = _a.run(agent.turn([], "test",
                                           on_content=lambda c: shown.append(c)))
    assert shown == ["Voici ", "le resultat."], shown      # jamais le tag a l'ecran
    assert reply == "Voici le resultat."
    assert hist[-1]["content"] == "Voici le resultat."      # historique propre
    assert trace.tts_tags == ["[emotion=calm]"]             # mais present pour le TTS


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
    assert cfg.tts_api.base_url == "https://api.mistral.ai"  # défaut conservé


def test_config_tts_api_default_disabled(tmp_path):
    from jarvis.config import load_config
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[llm]\nbase_url = \"http://127.0.0.1:9931/v1\"\n",
                         encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.tts_api.enabled is False  # off par défaut -> TTS local


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
