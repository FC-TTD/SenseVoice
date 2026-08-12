#!/usr/bin/env python3
"""Minimal OpenAI-realtime-style WebSocket STT client for sensevoice-server.

Zero-dependency RFC6455 client over stdlib socket. Uses a dedicated reader
thread so the socket stream can never desync while the server computes slow
partial decodes. Streams a 16 kHz mono PCM16 WAV as `input_audio_buffer.append`
chunks and prints every `conversation.item.input_audio_transcription.completed`
transcript (one per line). Used by run_server_smoke.sh to validate the WS
endpoint against the golden sensevoice.txt transcript.

Usage:
  stream_client.py HOST PORT WAV [CHUNK_MS]
"""
import base64
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
import wave

OP_TEXT = 0x1
OP_BIN = 0x2
OP_CLOSE = 0x8


class WSClient:
    def __init__(self, host, port, path):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.sock.settimeout(30)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            "GET %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: %s\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n" % (path, host, port, key)
        )
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("connection closed during handshake")
            resp += chunk
        head, _, _ = resp.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise RuntimeError("handshake failed: %r" % head)

        self.events = queue.Queue()
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _send_frame(self, opcode, payload):
        mask = os.urandom(4)
        b0 = 0x80 | opcode
        n = len(payload)
        if n < 126:
            hdr = struct.pack("!BB", b0, 0x80 | n)
        elif n < 65536:
            hdr = struct.pack("!BBH", b0, 0x80 | 126, n)
        else:
            hdr = struct.pack("!BBQ", b0, 0x80 | 127, n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(hdr + mask + masked)

    def send_text(self, s):
        self._send_frame(OP_TEXT, s.encode())

    def send_binary(self, b):
        self._send_frame(OP_BIN, b)

    def _recvn(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("closed")
            buf += chunk
        return buf

    def _recv_frame(self):
        b0, b1 = self._recvn(2)
        op = b0 & 0x0F
        masked = bool(b1 & 0x80)
        ln = b1 & 0x7F
        if ln == 126:
            ln = struct.unpack("!H", self._recvn(2))[0]
        elif ln == 127:
            ln = struct.unpack("!Q", self._recvn(8))[0]
        mask_key = self._recvn(4) if masked else b""
        data = self._recvn(ln)
        if masked:
            data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
        return op, data

    def _read_loop(self):
        try:
            while not self._stop.is_set():
                op, data = self._recv_frame()
                if op == OP_CLOSE:
                    self.events.put(("close", data))
                    break
                self.events.put((op, data))
        except RuntimeError:
            self.events.put(("close", b""))
        except socket.error:
            self.events.put(("close", b""))

    def next_event(self, timeout):
        """Block up to `timeout`s for the next (opcode, payload) datum frame."""
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._stop.set()
        try:
            self._send_frame(OP_CLOSE, struct.pack("!H", 1000))
        except OSError:
            pass
        self.sock.close()


def transcribe(host, port, wav_path, chunk_ms):
    with wave.open(wav_path, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2
        frames = w.readframes(w.getnframes())

    ws = WSClient(host, port, "/v1/realtime?intent=transcription")
    ws.send_text('{"type":"session.update","session":{"turn_detection":{"type":"server_vad"}}}')
    time.sleep(0.05)  # let the server ack session.created before streaming
    ws.next_event(2.0)  # swallow session.created

    transcripts = []
    chunk = 16000 * chunk_ms // 1000 * 2  # bytes per 200 ms chunk
    for off in range(0, len(frames), chunk):
        blob = frames[off:off + chunk]
        ws.send_text('{"type":"input_audio_buffer.append","audio":"%s"}'
                     % base64.b64encode(blob).decode())
        # drain whatever the server emitted for this chunk (partials, endpointing)
        for _ in range(200):
            ev = ws.next_event(0.01)
            if ev is None:
                break
            op, msg = ev
            if op == OP_TEXT:
                s = msg.decode("utf-8", "replace")
                if '"conversation.item.input_audio_transcription.completed"' in s:
                    try:
                        transcripts.append(json.loads(s).get("transcript", ""))
                    except Exception:
                        transcripts.append(s)
        time.sleep(0.02)

    ws.send_text('{"type":"input_audio_buffer.commit"}')
    deadline = time.time() + 90
    while time.time() < deadline:
        ev = ws.next_event(1.0)
        if ev is None:
            continue
        op, msg = ev
        if op == OP_TEXT:
            s = msg.decode("utf-8", "replace")
            if '"conversation.item.input_audio_transcription.completed"' in s:
                try:
                    tr = json.loads(s).get("transcript", "")
                except Exception:
                    tr = s
                transcripts.append(tr)
                break  # final full transcript for the committed turn
        if op == OP_CLOSE:
            break
    ws.close()
    out = " ".join(transcripts).strip()
    if out:
        print(out)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    host, port, wav = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    chunk_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    transcribe(host, port, wav, chunk_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
