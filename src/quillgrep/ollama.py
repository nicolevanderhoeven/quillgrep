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


CHARS_PER_TOKEN = 3.0  # a deliberately low estimate (English is nearer 4), so the window is never too small
CTX_STEPS = (4096, 8192, 16384, 32768, 65536, 131072)


def fit_ctx(messages: list[dict], cap: int, reserve: int = 1536) -> int:
    """A context window sized to what is actually sent, plus `reserve` tokens for the reply.

    Ollama reloads a model when its window changes between calls, so the size snaps to a few coarse steps
    rather than following the prompt exactly. `cap` (the configured maximum) is never exceeded.
    """
    tokens = int(sum(len(m.get("content", "")) for m in messages) / CHARS_PER_TOKEN) + reserve
    for step in CTX_STEPS:
        if tokens <= step:
            return min(step, cap)
    return cap


def planner_ctx(models, messages: list[dict], reserve: int) -> int:
    """The window for planner and re-rank calls. When the planner is also the answering model, use that
    model's full window so Ollama does not reload it between stages."""
    if models.planner_model == models.deep_model:
        return models.deep_num_ctx
    if models.planner_model == models.ask_model:
        return models.num_ctx
    return fit_ctx(messages, models.num_ctx, reserve=reserve)


def keep_alive_value(setting: str | int | float) -> str | int | float:
    """Ollama wants a number of seconds (int) or a duration string like "1h"; "-1" and "0" are numbers."""
    text = str(setting).strip()
    try:
        return int(text)
    except ValueError:
        return text


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 900, keep_alive: str = "1h"):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive_value(keep_alive)

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
        data = self._json("/api/embed", {"model": model, "input": texts, "keep_alive": self.keep_alive})
        vecs = np.asarray(data["embeddings"], dtype=np.float32)
        if vecs.ndim != 2 or len(vecs) != len(texts):
            raise OllamaError(f"Ollama returned an unexpected embedding response for model `{model}`.")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def _chat_body(self, model, messages, *, stream, fmt, num_ctx, max_tokens=None) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": False,  # we want the answer, not the model's scratch-pad (and it is much faster)
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": num_ctx},
        }
        if max_tokens:
            body["options"]["num_predict"] = max_tokens
        if fmt:
            body["format"] = fmt
        return body

    def chat(self, model: str, messages: list[dict], *, fmt: str | None = None, num_ctx: int = 8192,
             max_tokens: int | None = None) -> str:
        body = self._chat_body(model, messages, stream=False, fmt=fmt, num_ctx=num_ctx, max_tokens=max_tokens)
        data = self._json("/api/chat", body)
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


def resolve_models(cfg, client):
    """Fall back to the main model when an optional one (`ask_chat`, `planner`) is not installed.

    Returns (config to use, notes for the person at the keyboard). The input config is not changed. If Ollama
    cannot be reached nothing is changed either: the normal readiness check reports that.
    """
    import dataclasses

    try:
        installed = client.installed_models()
    except OllamaError:
        return cfg, []
    models = dataclasses.replace(cfg.models)
    notes = []
    for key in ("ask_chat", "planner"):
        wanted = getattr(models, key)
        if wanted and not any(model_name_matches(wanted, name) for name in installed):
            setattr(models, key, "")
            notes.append(f"The {key.replace('_chat', '')} model `{wanted}` is not installed, so the main model is "
                         f"used instead. For faster answers: ollama pull {wanted}")
    return dataclasses.replace(cfg, models=models), notes


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
