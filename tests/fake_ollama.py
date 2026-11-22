"""A stand-in for the Ollama HTTP API so the whole test suite runs without Ollama or any model.

Embeddings are hashed bags of words (deterministic, and similar texts get similar vectors). The "chat model"
follows the same prompts Quillgrep sends: it plans searches, scores excerpts and writes a short cited answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 256
STOP = {"the", "and", "for", "what", "with", "that", "this", "are", "was", "from", "have", "about", "should"}


def words(text: str) -> list[str]:
    """Lower-cased words with a crude plural strip, so "conference" matches "conferences"."""
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in re.findall(r"\w+", text.lower())
            if w not in STOP]


def embed_text(text: str) -> list[float]:
    text = re.sub(r"^(search_document|search_query): ", "", text)
    vec = [0.0] * DIM
    for w in words(text):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    return vec


class FakeOllama:
    def __init__(self):
        self.installed = ["nomic-embed-text:latest", "qwen3.5:9b"]
        self.embed_calls: list[list[str]] = []
        self.chat_calls: list[dict] = []
        self.plan: dict = {
            "queries": ["seed swap date", "conferences cfp deadline"],
            "keywords": ["okr", "conferences"],
        }
        self.answer: Callable[[str], str] | None = None  # override the answer text, given the prompt
        self.rerank_reply: Callable[[str], str] | None = None  # override the re-ranker reply
        self.think_prefix = False  # emit <think>…</think> inline, as some models do
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakeOllama:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, body: dict | list[dict], ndjson: bool = False):
                if ndjson:
                    data = b"".join(json.dumps(b).encode() + b"\n" for b in body)  # type: ignore[union-attr]
                    ctype = "application/x-ndjson"
                else:
                    data, ctype = json.dumps(body).encode(), "application/json"
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": n} for n in fake.installed]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model = body.get("model", "")
                if not any(model == n or f"{model}:latest" == n for n in fake.installed):
                    return self._send(404, {"error": f'model "{model}" not found, try pulling it first'})
                if self.path == "/api/embed":
                    fake.embed_calls.append(body["input"])
                    self._send(200, {"embeddings": [embed_text(t) for t in body["input"]]})
                elif self.path == "/api/chat":
                    fake.chat_calls.append(body)
                    text = fake.reply(body)
                    if body.get("stream"):
                        pieces = re.findall(r"\S+\s*", text)
                        chunks = [{"message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
                        chunks.append({"message": {"role": "assistant", "content": ""}, "done": True})
                        self._send(200, chunks, ndjson=True)
                    else:
                        self._send(200, {"message": {"role": "assistant", "content": text}, "done": True})
                else:
                    self._send(404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    # -- the pretend model ----------------------------------------------------------------------

    def reply(self, body: dict) -> str:
        prompt = body["messages"][-1]["content"]
        if body.get("format") == "json":
            if "Break this question" in prompt:
                return json.dumps(self.plan)
            if self.rerank_reply:
                return self.rerank_reply(prompt)
            return json.dumps({"scores": self.score(prompt)})
        text = self.answer(prompt) if self.answer else self.default_answer(prompt)
        return ("<think>pondering…</think>" if self.think_prefix else "") + text

    @staticmethod
    def score(prompt: str) -> list[dict]:
        question = re.search(r"Question: (.*?)\n\n", prompt, flags=re.S).group(1)
        qwords = set(words(question))
        out = []
        for m in re.finditer(r"^\[(\d+)\] \(.*?\)\n(.*?)(?=\n\n\[\d+\] \(|\Z)", prompt, flags=re.S | re.M):
            overlap = len(qwords & set(words(m.group(2))))
            out.append({"id": int(m.group(1)), "score": min(10, overlap * 3)})
        return out

    @staticmethod
    def default_answer(prompt: str) -> str:
        first = re.search(r"^\[1\][^\n]*\n(.*?)(?=\n\n\[2\]|\n\nQuestion:)", prompt, flags=re.S | re.M)
        gist = " ".join((first.group(1) if first else "nothing").split()[:12])
        return f"According to your notes, {gist} [1]. Related details appear in [2]."
