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

    @staticmethod
    def _lock(f) -> None:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock(f) -> None:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _last_hash(f) -> str:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        window = 65536
        while True:
            start = max(0, size - window)
            f.seek(start)
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
            if start == 0 or len(lines) > 1:
                break
            window *= 4
        return hashlib.sha256(lines[-1]).hexdigest() if lines else GENESIS

    def record(self, event: str, **fields) -> str:
        """Append one event; returns its id. The line is flushed to disk before returning.

        Several server processes may share one log, so the previous hash is re-read from the file
        under an exclusive lock for every append.
        """
        entry = {"id": uuid.uuid4().hex[:12], "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                 "event": event, **fields}
        with open(self.path, "a+b") as f:
            self._lock(f)
            try:
                entry["prev"] = self._last_hash(f)
                line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                f.seek(0, os.SEEK_END)
                f.write(line + b"\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                self._unlock(f)
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
