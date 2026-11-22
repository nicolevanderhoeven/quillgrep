"""An MCP server (stdio) that lets other AI tools ask questions of your notes without ever reading them.

The question is answered by the LOCAL model. What goes back is only that model's synthesis, with the
privacy rules from the config applied, plus (optionally) note titles as citations. Raw note text, file
paths and excerpts are never returned. Every request and response is written to the audit log.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

from . import __version__
from . import answer as answer_mod
from .audit import AuditLog
from .config import Config
from .ollama import OllamaClient, check_ready
from .privacy import clean_for_export, redact
from .store import Store

SUPPORTED_PROTOCOLS = ["2025-06-18", "2025-03-26", "2024-11-05"]
MAX_QUESTION_CHARS = 4000

TOOLS = [
    {
        "name": "ask_notes",
        "title": "Ask the user's private notes",
        "description": (
            "Ask a question about the user's private Obsidian notes. A local AI model on the user's own computer "
            "reads the notes and returns a short synthesis with note-title citations. You never receive the raw "
            "notes. Use depth 'fast' for a direct lookup (tens of seconds) and 'deep' for questions that need "
            "several notes combined (a few minutes)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "What to find out, in plain language."},
                "depth": {"type": "string", "enum": ["fast", "deep"], "default": "fast"},
                "recent": {"type": "boolean", "default": False,
                           "description": "Favour recently edited notes (good for 'what should I do next')."},
            },
            "required": ["question"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "notes_status",
        "title": "Notes index status",
        "description": "How many notes are indexed and when. Returns counts only, no titles or content.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
]


class McpServer:
    def __init__(self, cfg: Config, client: OllamaClient | None = None):
        self.cfg = cfg
        self.client = client or OllamaClient(cfg.models.host, cfg.models.timeout_seconds)
        self.audit = AuditLog(cfg.audit_path)
        self._store: Store | None = None

    # -- helpers --------------------------------------------------------------------------------

    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self.cfg.index_path)
        return self._store

    def log(self, msg: str):
        print(f"[quillgrep mcp] {msg}", file=sys.stderr, flush=True)

    # -- JSON-RPC -------------------------------------------------------------------------------

    def handle(self, msg: Any) -> dict | None:
        """Handle one JSON-RPC message; returns the response, or None for notifications."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return _error(None, -32600, "Invalid request")
        method, params, mid = msg.get("method"), msg.get("params") or {}, msg.get("id")
        is_request = "id" in msg
        if not isinstance(method, str):
            return _error(mid, -32600, "Invalid request") if is_request else None
        if method == "initialize":
            wanted = params.get("protocolVersion")
            version = wanted if wanted in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
            return _result(mid, {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "quillgrep", "title": "Quillgrep", "version": __version__},
                "instructions": ("Quillgrep answers questions about the user's private notes using a local AI "
                                 "model. Use ask_notes. You will receive a synthesis, never the notes themselves."),
            })
        if method == "ping":
            return _result(mid, {})
        if method == "tools/list":
            return _result(mid, {"tools": TOOLS})
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in {t["name"] for t in TOOLS}:
                return _error(mid, -32602, f"Unknown tool: {name}")
            if not isinstance(args, dict):
                return _error(mid, -32602, "Tool arguments must be an object")
            return _result(mid, self.call_tool(name, args))
        if is_request:
            return _error(mid, -32601, f"Method not found: {method}")
        return None  # notifications (initialized, cancelled, ...) need no reply

    # -- tools ----------------------------------------------------------------------------------

    def call_tool(self, name: str, args: dict) -> dict:
        if name == "notes_status":
            return self._status(args)
        return self._ask(args)

    def _status(self, args: dict) -> dict:
        req = self.audit.record("request", tool="notes_status", arguments={})
        try:
            s = self.store().stats()
            text = (f"{s['notes']} notes indexed ({s['chunks']} searchable passages), "
                    f"last indexed {s['indexed_at'] or 'never'}.")
            self.audit.record("response", request=req, returned=text, citations=[], error=None)
            return _text(text)
        except Exception as e:  # noqa: BLE001 - report any failure to the caller, never crash the server
            msg = "Quillgrep could not read its index. Run `quillgrep index` on this computer."
            self.audit.record("response", request=req, returned=msg, citations=[], error=str(e))
            return _text(msg, error=True)

    def _ask(self, args: dict) -> dict:
        question = args.get("question")
        depth = args.get("depth", "fast")
        recent = bool(args.get("recent", False))
        req = self.audit.record("request", tool="ask_notes",
                                arguments={"question": question, "depth": depth, "recent": recent})

        def refuse(message: str, error: str) -> dict:
            self.audit.record("response", request=req, returned=message, citations=[], error=error)
            return _text(message, error=True)

        try:
            return self._answer(args, question, depth, recent, req, refuse)
        except Exception as e:  # noqa: BLE001 - every exit must be audited and reported to the caller
            first = (str(e).splitlines() or [type(e).__name__])[0]
            return refuse("The local AI hit an error while answering: " + first, str(e) or type(e).__name__)

    def _answer(self, args: dict, question, depth, recent: bool, req: str, refuse) -> dict:
        if not isinstance(question, str) or not question.strip():
            return refuse("Please provide a question.", "missing question")
        if len(question) > MAX_QUESTION_CHARS:
            return refuse(f"The question is too long (limit {MAX_QUESTION_CHARS} characters).", "question too long")
        if depth not in ("fast", "deep"):
            return refuse("depth must be 'fast' or 'deep'.", "bad depth")
        problems = check_ready(self.client, [self.cfg.models.embedding, self.cfg.models.chat,
                                             self.cfg.models.planner_model])
        if problems:
            return refuse("The local AI is not ready on the user's computer: " + problems[0].splitlines()[0],
                          problems[0])
        store = self.store()
        if store.stats()["chunks"] == 0:
            return refuse("No notes are indexed yet. The user needs to run `quillgrep index`.", "empty index")
        t0 = time.time()
        run = answer_mod.deep if depth == "deep" else answer_mod.ask
        result = run(self.cfg, self.client, store, question.strip(), recent=recent, synthesis=True,
                     excluded_folders=self.cfg.privacy.excluded_folders)
        text, citations = self._export(result)
        self.audit.record("response", request=req, returned=text, citations=citations, error=None,
                          excerpts_used=len(result.hits), depth=depth, seconds=round(time.time() - t0, 1))
        return _text(text)

    def _export(self, result: answer_mod.Answer) -> tuple[str, list[str]]:
        """Apply the privacy rules and build the only thing that leaves: synthesis + (optional) titles."""
        p = self.cfg.privacy
        body = clean_for_export(result.text, p, [h.chunk.text for h in result.hits])
        if not p.show_titles:
            return re.sub(r"\s*\[\d{1,3}\]", "", body).strip(), []
        by_title: dict[str, list[int]] = {}
        for n, hit in result.cited(fallback=False):
            title = redact(hit.chunk.title, p.redact, p.redact_patterns)
            by_title.setdefault(title, []).append(n)
        lines = ["".join(f"[{n}]" for n in nums) + f" {title}" for title, nums in by_title.items()]
        if not lines:
            return body, []
        return body + "\n\nSources (note titles):\n" + "\n".join(lines), list(by_title)


def _result(mid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _text(text: str, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": error}


def serve(cfg: Config, stdin=None, stdout=None) -> None:
    """Run the server on stdin/stdout (newline-delimited JSON-RPC) until stdin closes."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    server = McpServer(cfg)
    server.log(f"ready (audit log: {cfg.audit_path})")
    for raw in stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            reply = _error(None, -32700, "Parse error")
        else:
            try:
                reply = server.handle(msg)
            except Exception as e:  # noqa: BLE001 - a bug must not kill the connection
                server.log(f"internal error: {e!r}")
                reply = _error(msg.get("id") if isinstance(msg, dict) else None, -32603, "Internal error")
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False).encode("utf-8") + b"\n")
            stdout.flush()
