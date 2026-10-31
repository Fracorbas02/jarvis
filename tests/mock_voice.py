"""Mock des serveurs voix (stdlib, threadé) pour tester sans modèles.

- mode "stt" (whisper-server) : GET /health -> {"status":"ok"},
  POST /inference (multipart avec un champ file) -> {"text": "..."}.
- mode "voxtral" (llama-server + modèle audio, ex. Voxtral Mini 4B Realtime) :
  GET /health -> ok, POST /v1/audio/transcriptions (multipart, champ file)
  -> {"text": "..."} ; /inference renvoie 404 (pour tester le fallback).
- mode "voxtral-501" : comme voxtral, mais /v1/audio/transcriptions renvoie
  501 {"error":{"message":"this server does not support chat/completions"}}
  et /v1/chat/completions accepte un bloc input_audio (base64) -> chat
  classique avec le texte transcrit (teste le fallback chat multimodal).
- mode "tts" (llama-server TTS) : GET /health -> ok, POST /v1/audio/speech
  (JSON avec "input") -> WAV 16 kHz valide ; si le JSON contient
  "_use_tts_fallback", /v1/audio/speech renvoie 404 (fallback POST /tts).
"""

from __future__ import annotations

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_wav(text_hint: str = "", rate: int = 16000, seconds: float = 0.2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(rate * seconds)
        w.writeframes(b"\x00\x01" * n)
    return buf.getvalue()


def _read_multipart_file(body: bytes, boundary: str) -> bytes:
    """Extrait le contenu du premier champ 'file' d'un multipart/form-data."""
    delim = b"--" + boundary.encode()
    for part in body.split(delim):
        if b'name="file"' in part:
            header, _, content = part.partition(b"\r\n\r\n")
            return content.rstrip(b"\r\n-")
    return b""


class Handler(BaseHTTPRequestHandler):
    mode = "stt"          # "stt" | "voxtral" | "tts" (remplacé via server attributes)
    tts_fallback = False  # si True, /v1/audio/speech -> 404 (test du fallback /tts)

    def log_message(self, *a):
        pass

    @property
    def _mode(self) -> str:
        return self.server.mode  # type: ignore[attr-defined]

    @property
    def _fallback(self) -> bool:
        return self.server.tts_fallback  # type: ignore[attr-defined]

    def _json(self, obj, code: int = 200):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _wav(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health",):
            self._json({"status": "ok"})
        elif self._mode == "tts" and self.path.startswith("/v1/audio/voices"):
            self._json({"voices": [{"id": "casual_female"}, {"id": "casual_male"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")

        if self._mode == "voxtral-501" and self.path.rstrip("/") == "/v1/audio/transcriptions":
            self._json({"error": {"code": 501,
                                  "message": "this server does not support chat/completions",
                                  "type": "not_supported_error"}}, code=501)
            return

        if self._mode in ("stt", "voxtral", "voxtral-501") and self.path.rstrip("/") == "/v1/chat/completions" and self._mode == "voxtral-501":
            payload = json.loads(body)
            msgs = payload.get("messages", [])
            has_audio = any(
                isinstance(m.get("content"), list)
                and any(isinstance(c, dict) and c.get("type") == "input_audio"
                        for c in m["content"])
                for m in msgs)
            if has_audio:
                self._json({"choices": [{"message": {"role": "assistant",
                                                     "content": "transcription chat multimodal"}}]})
                return
            self._json({"error": "no audio"}, code=400)
            return

        if self._mode in ("stt", "voxtral", "voxtral-501"):
            # whisper-server sert /inference, llama-server audio sert
            # /v1/audio/transcriptions ; l'autre endpoint renvoie 404
            path = self.path.rstrip("/")
            want = "/inference" if self._mode == "stt" else "/v1/audio/transcriptions"
            if path == want:
                if "multipart/form-data" not in ctype:
                    self._json({"error": "attendu multipart/form-data"}, code=400)
                    return
                boundary = ctype.split("boundary=")[-1].strip()
                audio = _read_multipart_file(body, boundary)
                if not audio:
                    self._json({"error": "pas de champ file"}, code=400)
                    return
                self._json({"text": f"transcription de {len(audio)} octets audio"})
                return
            self.send_error(404)
            return

        if self._mode == "tts":
            if self.path.startswith("/v1/audio/speech"):
                if self._fallback:
                    self._json({"error": "unknown route"}, code=404)
                    return
                payload = json.loads(body)
                self._wav(make_wav(payload.get("input", "")))
                return
            if self.path.rstrip("/") == "/tts":
                payload = json.loads(body)
                self._wav(make_wav(payload.get("input", "")))
                return

        self.send_error(404)


def start_mock_voice(port: int, mode: str, tts_fallback: bool = False
                     ) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.mode = mode  # type: ignore[attr-defined]
    server.tts_fallback = tts_fallback  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "stt"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else (9932 if mode == "tts" else 8178)
    srv, t = start_mock_voice(port, mode)
    print(f"mock {mode} sur http://127.0.0.1:{port}")
    try:
        t.join()
    except KeyboardInterrupt:
        srv.shutdown()
