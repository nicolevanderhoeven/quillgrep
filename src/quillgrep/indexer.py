"""Incremental indexing: only new or changed files are read and embedded again."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

from .config import Config
from .extract import chunk_text, read_text, scan_vault, title_of
from .ollama import OllamaClient
from .store import Store

EMBED_BATCH = 32
MIN_CHUNK_CHARS = 20


@dataclass
class IndexReport:
    seen: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    empty: int = 0
    chunks: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (f"{self.seen} files: {self.added} new, {self.updated} changed, {self.unchanged} unchanged, "
                f"{self.removed} removed; {self.chunks} chunks embedded in {self.seconds:.0f}s")


def embedding_input(title: str, text: str) -> str:
    """nomic-embed-text wants a task prefix; including the note title helps short chunks find their note."""
    return f"search_document: {title}\n{text}"


def index_vault(cfg: Config, client: OllamaClient, store: Store, *, rebuild: bool = False,
                progress: Callable[[str], None] | None = None) -> IndexReport:
    say = progress or (lambda _msg: None)
    t0 = time.time()
    report = IndexReport()
    model = cfg.models.embedding

    if rebuild or (store.get_meta("embedding_model") not in (None, model)):
        if not rebuild:
            say(f"Embedding model changed to {model}; rebuilding the index from scratch.")
        store.reset()
    store.set_meta("embedding_model", model)
    store.commit()

    known = store.note_state()
    seen_paths: set[str] = set()

    for src in scan_vault(cfg.vault_path, cfg.index):
        report.seen += 1
        seen_paths.add(src.rel)
        old = known.get(src.rel)
        if old and old["mtime"] == src.mtime and old["size"] == src.size:
            report.unchanged += 1
            continue
        try:
            text = read_text(src.path, src.kind, cfg.index)
        except OSError:
            continue  # unreadable right now (for example mid-sync); try again next run
        digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
        if old and old["hash"] == digest:  # touched, but the words did not change
            store.touch_note(old["id"], src.mtime, src.size)
            store.commit()
            report.unchanged += 1
            continue
        title = title_of(src.rel)
        chunks = [c for c in chunk_text(text, cfg.index.chunk_chars, cfg.index.chunk_overlap)
                  if len(c.strip()) >= MIN_CHUNK_CHARS]
        if not chunks:
            if old:
                store.delete_note(src.rel)
                store.commit()
            report.empty += 1
            continue
        vectors = _embed_all(client, model, [embedding_input(title, c) for c in chunks])
        store.replace_note(path=src.rel, title=title, kind=src.kind, mtime=src.mtime, size=src.size,
                           digest=digest, texts=chunks, vectors=vectors)
        store.commit()
        report.chunks += len(chunks)
        if old:
            report.updated += 1
        else:
            report.added += 1
        if (report.added + report.updated) % 25 == 0:
            say(f"  {report.seen} files seen, {report.added + report.updated} indexed, "
                f"{report.chunks} chunks, {time.time() - t0:.0f}s")

    for path in set(known) - seen_paths:
        store.delete_note(path)
        report.removed += 1
    store.set_meta("indexed_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    store.commit()
    report.seconds = time.time() - t0
    return report


def _embed_all(client: OllamaClient, model: str, texts: list[str]):
    import numpy as np

    parts = [client.embed(model, texts[i : i + EMBED_BATCH]) for i in range(0, len(texts), EMBED_BATCH)]
    return np.vstack(parts)
