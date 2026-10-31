"""Mock websocket /v1/realtime?intent=transcription (stdlib asyncio, thread d\u00e9di\u00e9).

Pour tester RealtimeSTTClient sans mod\u00e8le : handshake RFC 6455 (101),
puis lit les trames masqu\u00e9es du client, r\u00e9pond session.updated, et apr\u00e8s
input_audio_buffer.commit envoie conversation.item.input_audio_transcription.completed
avec un texte bidon + response.done.

Utilisation :
    from ws_mock import start_ws_realtime
    port, ready, stop = start_ws_realtime(text="bonjour jarvis")
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import struct
import threading

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_ws_realtime(text: str = "bonjour jarvis"):
    """D\u00e9marre le mock WS dans un thread. Retourne (port, ready_event, stop_fn)."""
    port = _free_port()
    ready = threading.Event()
    state = {"loop": None, "server": None}

    async def send_text(writer, obj):
        payload = json.dumps(obj).encode()
        header = bytearray([0x81])
        n = len(payload)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header += struct.pack(">H", n)
        else:
            header.append(127)
            header += struct.pack(">Q", n)
        writer.write(bytes(header) + payload)
        await writer.drain()

    async def handle(reader, writer):
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = await reader.read(4096)
            if not chunk:
                writer.close()
                return
            buf += chunk
        head, _, buf = buf.partition(b"\r\n\r\n")
        if b"/v1/realtime" not in head.split(b"\r\n")[0]:
            writer.write(b"HTTP/1.1 404 Not Found\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        key = next(l for l in head.split(b"\r\n")
                   if l.lower().startswith(b"sec-websocket-key"))
        key = key.split(b":", 1)[1].strip()
        accept = base64.b64encode(hashlib.sha1(key + GUID.encode()).digest()).decode()
        writer.write((f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n").encode())
        await writer.drain()

        def parse_frames():
            nonlocal buf
            out = []
            while True:
                if len(buf) < 2:
                    return out
                b1, b2 = buf[0], buf[1]
                op = b1 & 0x0F
                ln = b2 & 0x7F
                off = 2
                if ln == 126:
                    if len(buf) < 4:
                        return out
                    ln = struct.unpack(">H", buf[2:4])[0]
                    off = 4
                elif ln == 127:
                    if len(buf) < 10:
                        return out
                    ln = struct.unpack(">Q", buf[2:10])[0]
                    off = 10
                if len(buf) < off + 4 + ln:
                    return out
                mask = buf[off:off + 4]
                payload = bytes(b ^ mask[i % 4]
                                for i, b in enumerate(buf[off + 4:off + 4 + ln]))
                buf = buf[off + 4 + ln:]
                out.append((op, payload))

        try:
            while True:
                frames = parse_frames()
                if not frames:
                    chunk = await reader.read(65536)
                    if not chunk:
                        return
                    buf += chunk
                    continue
                for op, payload in frames:
                    if op != 0x1:
                        continue
                    ev = json.loads(payload.decode())
                    etype = ev.get("type")
                    if etype == "session.update":
                        await send_text(writer, {"type": "session.updated",
                                                 "session": {"id": "sess-mock"}})
                    elif etype == "input_audio_buffer.commit":
                        await send_text(writer, {"type": "input_audio_buffer.committed"})
                        await send_text(writer, {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "item_mock",
                            "content_index": 0,
                            "transcript": text})
                        await send_text(writer, {"type": "response.done",
                                                 "response": {"status": "completed"}})
                        return
        finally:
            writer.close()

    def serve():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        state["loop"] = loop

        async def main():
            srv = await asyncio.start_server(handle, "127.0.0.1", port)
            state["server"] = srv
            ready.set()
            async with srv:
                await srv.serve_forever()

        loop.run_until_complete(main())

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    assert ready.wait(timeout=5)

    def stop():
        loop = state["loop"]
        if loop is not None:
            loop.call_soon_threadsafe(lambda: state["server"].close() if state["server"] else None)

    return port, ready, stop
