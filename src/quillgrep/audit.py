"""An append-only, tamper-evident local log of everything the MCP server was asked and returned."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

GENESIS = "0" * 64


class AuditLog:
    """JSON lines, one per event. Each line carries the SHA-256 of the previous line, so edits are detectable.

    The file is only ever opened for appending; Quillgrep never rewrites or truncates it.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
            try:
                os.chmod(self.path, 0o600)  # owner only (no effect on Windows, which uses folder permissions)
            except OSError:
                pass
        self._last = self._last_hash()

    def _last_hash(self) -> str:
        last = b""
        with open(self.path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            for line in f.read().splitlines():
                if line.strip():
                    last = line
        return hashlib.sha256(last).hexdigest() if last else GENESIS

    def record(self, event: str, **fields) -> str:
        """Append one event; returns its id. The line is flushed to disk before returning."""
        entry = {"id": uuid.uuid4().hex[:12], "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                 "event": event, **fields, "prev": self._last}
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with open(self.path, "ab") as f:
            f.write(line + b"\n")
            f.flush()
            os.fsync(f.fileno())
        self._last = hashlib.sha256(line).hexdigest()
        return entry["id"]


def read_entries(path: Path, tail: int | None = None) -> list[dict]:
    if not Path(path).exists():
        return []
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries[-tail:] if tail else entries


def verify(path: Path) -> tuple[bool, str]:
    """Check the hash chain. Returns (ok, message)."""
    prev = GENESIS
    count = 0
    if not Path(path).exists():
        return True, "No audit log yet."
    with open(path, "rb") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip(b"\n")
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                return False, f"Line {n} is not valid JSON."
            if entry.get("prev") != prev:
                return False, f"The chain is broken at line {n}: an earlier line was changed or removed."
            prev = hashlib.sha256(line).hexdigest()
            count += 1
    return True, f"{count} entries, chain intact."
