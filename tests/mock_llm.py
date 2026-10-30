"""Mock minimal de llama-server /v1/chat/completions (stdlib, threadé).

Scénario scripté pour tester la boucle tool-calling sans LLM réel :
1. Premier appel avec message user contenant "ping" -> tool_call sur echo__ping.
2. Dès qu'un message role=tool est présent -> réponse finale "pong ...".
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
        # on exige que les tools aient bien été transmis pour l'appel d'outil
        if "ping" in last_user.lower() and not has_tool_result:
            assert body.get("tools"), "le harnais n'a pas transmis les tools au LLM"

        out = json.dumps(_assistant(msg)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


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
