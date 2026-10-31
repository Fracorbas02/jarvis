"""Mock minimal de llama-server /v1/chat/completions (stdlib, threadé).

Scénario scripté pour tester la boucle tool-calling sans LLM réel :
1. Premier appel avec message user contenant "ping" -> tool_call sur echo__ping.
2. Dès qu'un message role=tool est présent -> réponse finale "pong ...".

Supporte les deux modes de llama-server :
- non-streaming : un POST, un JSON complet ;
- streaming (stream=true) : deltas SSE, avec reasoning_content et tool_calls
  fragmentés (name/arguments en plusieurs morceaux) pour couvrir le réassemblage.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "echo__ping", "arguments": json.dumps({"target": "127.0.0.1"})},
}


def _assistant(message: dict) -> dict:
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def _scenario(messages: list[dict]) -> list[dict]:
    """Retourne la liste des deltas SSE pour la requête reçue."""
    has_tool_result = any(m.get("role") == "tool" for m in messages)
    last_user = next((m["content"] for m in messages
                      if m.get("role") == "user"), "") or ""

    def delta(d: dict, finish: str | None = None) -> dict:
        choice = {"delta": d}
        if finish:
            choice["finish_reason"] = finish
        return {"choices": [choice]}

    if has_tool_result:
        tool_msg = next(m["content"] for m in messages if m.get("role") == "tool")
        text = f"Résultat du ping : {tool_msg}"
        mid = len(text) // 2
        return [
            delta({"reasoning_content": "Je reformule le résultat de l'outil."}),
            delta({"content": text[:mid]}),
            delta({"content": text[mid:]}),
            delta({}, finish="stop"),
        ]
    if "ping" in last_user.lower():
        return [
            delta({"reasoning_content": "L'utilisateur demande un ping : j'appelle echo__ping."}),
            delta({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                   "function": {"name": "echo__"}}]}),
            delta({"tool_calls": [{"index": 0,
                                   "function": {"name": "ping", "arguments": "{\"target\": "}}]}),
            delta({"tool_calls": [{"index": 0,
                                   "function": {"arguments": "\"127.0.0.1\"}"}}]}),
            delta({}, finish="stop"),
        ]
    return [
        delta({"content": "réponse directe "}),
        delta({"content": "sans outil"}),
        delta({}, finish="stop"),
    ]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        if not self.path.rstrip("/").endswith("chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])

        last_user = next((m["content"] for m in messages
                          if m.get("role") == "user"), "") or ""
        # on exige que les tools aient bien été transmis pour l'appel d'outil
        if "ping" in last_user.lower() and not any(m.get("role") == "tool" for m in messages):
            assert body.get("tools"), "le harnais n'a pas transmis les tools au LLM"

        if body.get("stream"):
            self._handle_stream(body, messages)
        else:
            self._handle_json(messages)

    def _handle_json(self, messages: list[dict]) -> None:
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        last_user = next((m["content"] for m in messages
                          if m.get("role") == "user"), "") or ""
        if has_tool_result:
            tool_msg = next(m["content"] for m in messages if m.get("role") == "tool")
            msg = {"role": "assistant",
                   "content": f"Résultat du ping : {tool_msg}"}
        elif "ping" in last_user.lower():
            msg = {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
        else:
            msg = {"role": "assistant",
                   "content": "réponse directe sans outil"}
        out = json.dumps(_assistant(msg)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _handle_stream(self, body: dict, messages: list[dict]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in _scenario(messages):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def start_mock(port: int = 18080) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    srv, t = start_mock()
    print(f"mock llama-server sur http://127.0.0.1:18080/v1")
    try:
        t.join()
    except KeyboardInterrupt:
        srv.shutdown()
