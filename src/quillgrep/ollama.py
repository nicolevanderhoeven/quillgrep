"""A small client for a local Ollama server (https://ollama.com), using only the standard library."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np

INSTALL_URL = "https://ollama.com/download"


class OllamaError(Exception):
    """Something went wrong talking to Ollama. The message is written for the person at the keyboard."""


class OllamaUnavailable(OllamaError):
    pass


class ModelMissing(OllamaError):
    pass


def model_name_matches(wanted: str, installed: str) -> bool:
    """`nomic-embed-text` matches `nomic-embed-text:latest`; `qwen3.5:9b` only matches itself."""
    if wanted == installed:
        return True
    if ":" not in wanted:
        return installed == f"{wanted}:latest"
    return False


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 900):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # -- plumbing -------------------------------------------------------------------------------

    def _open(self, path: str, body: dict | None = None, timeout: float | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.host + path, data, {"Content-Type": "application/json"} if data else {}
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            if e.code == 404 and "not found" in detail.lower():
                model = (body or {}).get("model", "the model")
                raise ModelMissing(
                    f"Ollama does not have the model `{model}` yet. Download it with:\n    ollama pull {model}"
                ) from e
            raise OllamaError(f"Ollama returned an error ({e.code}): {detail}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self.host}.\n"
                f"  - Not installed? Get it from {INSTALL_URL}\n"
                "  - Installed? Make sure it is running (open the Ollama app, or run `ollama serve`)."
            ) from e

    def _json(self, path: str, body: dict | None = None, timeout: float | None = None) -> Any:
        with self._open(path, body, timeout) as r:
            return json.load(r)

    # -- API ------------------------------------------------------------------------------------

    def installed_models(self) -> list[str]:
        data = self._json("/api/tags", timeout=10)
        return [m.get("name", "") for m in data.get("models", [])]

    def embed(self, model: str, texts: list[str]) -> np.ndarray:
        """Embed `texts`; returns an (n, dim) float32 array of unit-length vectors."""
        data = self._json("/api/embed", {"model": model, "input": texts})
        vecs = np.asarray(data["embeddings"], dtype=np.float32)
        if vecs.ndim != 2 or len(vecs) != len(texts):
            raise OllamaError(f"Ollama returned an unexpected embedding response for model `{model}`.")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def _chat_body(self, model, messages, *, stream, fmt, num_ctx) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": False,  # we want the answer, not the model's scratch-pad (and it is much faster)
            "options": {"num_ctx": num_ctx},
        }
        if fmt:
            body["format"] = fmt
        return body

    def chat(self, model: str, messages: list[dict], *, fmt: str | None = None, num_ctx: int = 8192) -> str:
        data = self._json("/api/chat", self._chat_body(model, messages, stream=False, fmt=fmt, num_ctx=num_ctx))
        return strip_thinking(data.get("message", {}).get("content", "")).strip()

    def chat_stream(self, model: str, messages: list[dict], *, num_ctx: int = 8192) -> Iterator[str]:
        """Yield the answer text piece by piece as the model writes it."""
        body = self._chat_body(model, messages, stream=True, fmt=None, num_ctx=num_ctx)
        with self._open("/api/chat", body) as r:
            in_think = False
            for raw in r:
                raw = raw.strip()
                if not raw:
                    continue
                msg = json.loads(raw)
                if "error" in msg:
                    raise OllamaError(f"Ollama error: {msg['error']}")
                piece = msg.get("message", {}).get("content", "")
                # Some models still emit <think>…</think> inline; hide it from the reader.
                while piece:
                    if in_think:
                        end = piece.find("</think>")
                        if end < 0:
                            piece = ""
                        else:
                            piece, in_think = piece[end + len("</think>"):].lstrip("\n"), False
                    else:
                        start = piece.find("<think>")
                        if start < 0:
                            yield piece
                            piece = ""
                        else:
                            if start:
                                yield piece[:start]
                            piece, in_think = piece[start + len("<think>"):], True
                if msg.get("done"):
                    break


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def check_ready(client, models: list[str]) -> list[str]:
    """Return a list of human-readable problems (empty = ready). Never raises for the expected failures."""
    try:
        installed = client.installed_models()
    except OllamaUnavailable as e:
        return [str(e)]
    except OllamaError as e:
        return [str(e)]
    problems = []
    for model in dict.fromkeys(m for m in models if m):
        if not any(model_name_matches(model, name) for name in installed):
            problems.append(f"The model `{model}` is not installed. Download it with:\n    ollama pull {model}")
    return problems
