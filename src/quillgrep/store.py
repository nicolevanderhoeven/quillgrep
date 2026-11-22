"""The local index: one SQLite file holding notes, chunks, vectors and a full-text (FTS5) index."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1"


class StoreError(Exception):
    pass


@dataclass(frozen=True)
class Chunk:
    id: int
    note_id: int
    path: str
    title: str
    ord: int
    text: str
    mtime: float


def fts_available() -> bool:
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(a)")
        return True
    except sqlite3.Error:
        return False


def fts_query(text: str, max_terms: int = 24) -> str:
    """Turn free text into a safe FTS5 query: every word, quoted, OR-ed together."""
    words = re.findall(r"\w+", text.lower())
    seen: list[str] = []
    for w in words:
        if len(w) >= 2 and w not in seen:
            seen.append(w)
    return " OR ".join(f'"{w}"' for w in seen[:max_terms])


class Store:
    def __init__(self, path: Path):
        if not fts_available():
            raise StoreError(
                "This Python's SQLite was built without FTS5 full-text search, which Quillgrep needs.\n"
                "Install Python from https://www.python.org/downloads/ (or use your system's python3)."
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._create()
        self._matrix: np.ndarray | None = None
        self._matrix_ids: np.ndarray | None = None
        self._matrix_notes: np.ndarray | None = None
        self._matrix_version: int | None = None

    # -- schema ---------------------------------------------------------------------------------

    def _create(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
                kind TEXT NOT NULL, mtime REAL NOT NULL, size INTEGER NOT NULL, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                ord INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS chunks_note ON chunks(note_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                title, path, text, tokenize='unicode61 remove_diacritics 2');
            """
        )
        if self.get_meta("schema") is None:
            self.set_meta("schema", SCHEMA_VERSION)
        self.db.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str):
        self.db.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=?",
                        (key, value, value))

    def close(self):
        self.db.close()

    def reset(self):
        """Forget everything (used when the embedding model changes or on --rebuild)."""
        self.db.executescript("DELETE FROM chunks; DELETE FROM notes; DELETE FROM chunks_fts;")
        self.db.commit()
        self._matrix = None

    # -- writing --------------------------------------------------------------------------------

    def note_state(self) -> dict[str, sqlite3.Row]:
        return {r["path"]: r for r in self.db.execute("SELECT id, path, mtime, size, hash FROM notes")}

    def touch_note(self, note_id: int, mtime: float, size: int):
        self.db.execute("UPDATE notes SET mtime=?, size=? WHERE id=?", (mtime, size, note_id))

    def delete_note(self, path: str):
        row = self.db.execute("SELECT id FROM notes WHERE path=?", (path,)).fetchone()
        if not row:
            return
        self._delete_chunks(row["id"])
        self.db.execute("DELETE FROM notes WHERE id=?", (row["id"],))

    def _delete_chunks(self, note_id: int):
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE note_id=?", (note_id,))]
        self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        self.db.execute("DELETE FROM chunks WHERE note_id=?", (note_id,))

    def replace_note(self, *, path: str, title: str, kind: str, mtime: float, size: int, digest: str,
                     texts: list[str], vectors: np.ndarray):
        """Insert or replace one note and its chunks in a single transaction."""
        row = self.db.execute("SELECT id FROM notes WHERE path=?", (path,)).fetchone()
        if row:
            note_id = row["id"]
            self._delete_chunks(note_id)
            self.db.execute("UPDATE notes SET title=?, kind=?, mtime=?, size=?, hash=? WHERE id=?",
                            (title, kind, mtime, size, digest, note_id))
        else:
            cur = self.db.execute(
                "INSERT INTO notes(path, title, kind, mtime, size, hash) VALUES(?,?,?,?,?,?)",
                (path, title, kind, mtime, size, digest))
            note_id = cur.lastrowid
        for i, (text, vec) in enumerate(zip(texts, vectors, strict=True)):
            cur = self.db.execute("INSERT INTO chunks(note_id, ord, text, embedding) VALUES(?,?,?,?)",
                                  (note_id, i, text, np.asarray(vec, dtype=np.float16).tobytes()))
            self.db.execute("INSERT INTO chunks_fts(rowid, title, path, text) VALUES(?,?,?,?)",
                            (cur.lastrowid, title, path, text))

    def commit(self):
        self.db.commit()

    # -- reading --------------------------------------------------------------------------------

    def stats(self) -> dict:
        n_notes = self.db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        n_chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_kind = {r[0]: r[1] for r in self.db.execute("SELECT kind, COUNT(*) FROM notes GROUP BY kind")}
        return {"notes": n_notes, "chunks": n_chunks, "by_kind": by_kind,
                "embedding_model": self.get_meta("embedding_model"), "indexed_at": self.get_meta("indexed_at"),
                "size_mb": round(self.path.stat().st_size / 1e6, 1) if self.path.exists() else 0.0}

    def _data_version(self) -> int:
        return self.db.execute("PRAGMA data_version").fetchone()[0]

    def matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(vectors, chunk ids, note ids) for every chunk, as numpy arrays. Cached between queries."""
        version = self._data_version()
        if self._matrix is None or self._matrix_version != version:
            rows = self.db.execute("SELECT id, note_id, embedding FROM chunks ORDER BY id").fetchall()
            if rows:
                ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                notes = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
                mat = np.vstack([np.frombuffer(r[2], dtype=np.float16) for r in rows]).astype(np.float32)
            else:
                ids = notes = np.zeros(0, dtype=np.int64)
                mat = np.zeros((0, 1), dtype=np.float32)
            self._matrix, self._matrix_ids, self._matrix_notes = mat, ids, notes
            self._matrix_version = version
        return self._matrix, self._matrix_ids, self._matrix_notes

    def excluded_note_ids(self, folders: list[str]) -> set[int]:
        """Ids of notes under any of `folders` (used to keep private folders out of MCP answers)."""
        if not folders:
            return set()
        from .extract import in_folders

        return {r["id"] for r in self.db.execute("SELECT id, path FROM notes") if in_folders(r["path"], folders)}

    def chunks_by_ids(self, ids: list[int]) -> dict[int, Chunk]:
        out: dict[int, Chunk] = {}
        for start in range(0, len(ids), 500):
            part = ids[start : start + 500]
            marks = ",".join("?" * len(part))
            for r in self.db.execute(
                f"SELECT c.id, c.note_id, n.path, n.title, c.ord, c.text, n.mtime FROM chunks c "
                f"JOIN notes n ON n.id = c.note_id WHERE c.id IN ({marks})", part):
                out[r[0]] = Chunk(*r)
        return out

    def keyword_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """BM25 over title, path and text. Returns (chunk id, score) best first; score is positive."""
        q = fts_query(query)
        if not q:
            return []
        try:
            rows = self.db.execute(
                "SELECT rowid, bm25(chunks_fts, 4.0, 2.0, 1.0) AS s FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?", (q, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r[0], -float(r[1])) for r in rows]

    def title_search(self, words: list[str], limit: int) -> list[int]:
        """Chunk ids (the first chunk of each note) whose title or path contains one of `words`."""
        words = [w for w in (w.strip().lower() for w in words) if len(w) >= 3]
        if not words:
            return []
        clause = " OR ".join("lower(path) LIKE ? ESCAPE '\\'" for _ in words)
        args = ["%" + w.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for w in words]
        rows = self.db.execute(
            f"SELECT MIN(c.id) FROM notes n JOIN chunks c ON c.note_id = n.id WHERE {clause} "
            f"GROUP BY n.id ORDER BY n.mtime DESC LIMIT ?", [*args, limit]).fetchall()
        return [r[0] for r in rows]

    def note_chunk_ids(self, path: str) -> list[int]:
        return [r[0] for r in self.db.execute(
            "SELECT c.id FROM chunks c JOIN notes n ON n.id=c.note_id WHERE n.path=? ORDER BY c.ord", (path,))]

    def resolve_note(self, ref: str) -> list[str]:
        """Find notes named by an anchor reference: an exact path, a path without `.md`, or a title."""
        ref_l = ref.replace("\\", "/").strip("/").lower()
        out = []
        for r in self.db.execute("SELECT path, title FROM notes"):
            p = r["path"].lower()
            if p == ref_l or p.rsplit(".", 1)[0] == ref_l or r["title"].lower() == ref_l or p.startswith(ref_l + "/"):
                out.append(r["path"])
        return out
