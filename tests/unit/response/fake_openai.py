"""A local fake of an OpenAI-compatible chat-completions server for tests.

It binds to 127.0.0.1 on an ephemeral port, replays scripted replies and records every
request it receives. Tests never call real model APIs.
"""

from __future__ import annotations

import json
import socketserver
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _LoopbackServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse DNS lookup in HTTPServer.server_bind.

    ``socket.getfqdn`` on a loopback address can block for tens of seconds on macOS.
    """

    daemon_threads = True
    block_on_close = False

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


@dataclass
class Reply:
    content: object = ""
    finish_reason: str = "stop"
    status: int = 200
    delay_s: float = 0.0
    headers: dict = field(default_factory=dict)
    raw_body: bytes | None = None
    usage: dict | None = None


class FakeOpenAIServer:
    def __init__(self, replies=()):
        self.replies = [reply if isinstance(reply, Reply) else Reply(reply) for reply in replies]
        self.requests = []
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                with fake._lock:
                    fake.requests.append(
                        {
                            "path": self.path,
                            "headers": {key.lower(): value for key, value in self.headers.items()},
                            "body": json.loads(body.decode("utf-8")),
                        }
                    )
                    reply = fake.replies.pop(0) if fake.replies else Reply(status=500)
                if reply.delay_s:
                    time.sleep(reply.delay_s)
                if reply.raw_body is not None:
                    payload = reply.raw_body
                elif reply.status != 200:
                    payload = json.dumps({"error": {"message": "scripted failure"}}).encode()
                else:
                    text = reply.content if isinstance(reply.content, str) else ""
                    payload = json.dumps(
                        {
                            "id": "chatcmpl-fake",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": reply.content},
                                    "finish_reason": reply.finish_reason,
                                }
                            ],
                            "usage": reply.usage
                            or {"prompt_tokens": 120, "completion_tokens": len(text.split())},
                        }
                    ).encode("utf-8")
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    for key, value in reply.headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = _LoopbackServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        return False
