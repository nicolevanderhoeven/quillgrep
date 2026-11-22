"""Hybrid retrieval: semantic + keyword search, fusion, recency boost, anchors, per-note cap, re-ranking."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .ollama import OllamaClient, OllamaError
from .store import Chunk, Store

RRF_K = 60  # reciprocal-rank-fusion constant; the usual default


@dataclass
class Hit:
    chunk: Chunk
    score: float = 0.0
    via: set[str] = field(default_factory=set)  # which stages found it: semantic, keyword, title, anchor
    rerank: float | None = None


class Retriever:
    def __init__(self, cfg: Config, client: OllamaClient, store: Store, *, excluded_folders: list[str] | None = None):
        self.cfg = cfg
        self.client = client
        self.store = store
        self.excluded = store.excluded_note_ids(excluded_folders or [])
        self._qcache: dict[str, np.ndarray] = {}

    # -- single searches ------------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            self._qcache[query] = self.client.embed(self.cfg.models.embedding, ["search_query: " + query])[0]
        return self._qcache[query]

    def _allowed(self, note_ids: np.ndarray) -> np.ndarray:
        if not self.excluded:
            return np.ones(len(note_ids), dtype=bool)
        return ~np.isin(note_ids, np.fromiter(self.excluded, dtype=np.int64))

    def semantic(self, query: str, limit: int) -> list[tuple[int, float]]:
        mat, ids, notes = self.store.matrix()
        if len(ids) == 0:
            return []
        sims = mat @ self.embed_query(query)
        sims = np.where(self._allowed(notes), sims, -np.inf)
        k = min(limit, len(ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(ids[i]), float(sims[i])) for i in top if np.isfinite(sims[i])]

    def keyword(self, query: str, limit: int) -> list[tuple[int, float]]:
        fetch = limit * 4 if self.excluded else limit
        found = self.store.keyword_search(query, fetch)
        if self.excluded:
            owner = self._owners([i for i, _ in found])
            found = [(i, s) for i, s in found if owner.get(i) not in self.excluded]
        return found[:limit]

    def _owners(self, chunk_ids: list[int]) -> dict[int, int]:
        return {i: c.note_id for i, c in self.store.chunks_by_ids(chunk_ids).items()}

    def search(self, query: str, *, limit: int | None = None, boost: float = 0.0) -> list[Hit]:
        """Hybrid search for one query. Returns chunks best first (not yet capped per note)."""
        limit = limit or self.cfg.retrieval.candidates
        sem = self.semantic(query, limit)
        kw = self.keyword(query, limit)
        scores: dict[int, float] = {}
        via: dict[int, set[str]] = {}
        for name, ranked in (("semantic", sem), ("keyword", kw)):
            for rank, (cid, _) in enumerate(ranked):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                via.setdefault(cid, set()).add(name)
        chunks = self.store.chunks_by_ids(list(scores))
        now = time.time()
        hits = []
        for cid, score in scores.items():
            chunk = chunks.get(cid)
            if chunk is None:
                continue
            if boost > 0:
                score *= recency_factor(chunk.mtime, now, boost, self.cfg.retrieval.recency_half_life_days)
            hits.append(Hit(chunk, score, via[cid]))
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    # -- pooling several queries ----------------------------------------------------------------

    def pool(self, queries: list[str], *, keywords: list[str] | None = None, boost: float = 0.0,
             per_note: int | None = None, limit: int | None = None) -> list[Hit]:
        """Run several searches and merge them fairly: round-robin by rank, then the per-note cap."""
        per_note = per_note or self.cfg.retrieval.per_note
        limit = limit or self.cfg.retrieval.rerank_pool
        lists = [self.search(q, boost=boost) for q in queries if q.strip()]
        merged: list[Hit] = []
        seen: set[int] = set()
        for rank in range(max((len(lst) for lst in lists), default=0)):
            for lst in lists:
                if rank < len(lst) and lst[rank].chunk.id not in seen:
                    seen.add(lst[rank].chunk.id)
                    merged.append(lst[rank])
        # keyword safety net: notes whose title or path contains a planner keyword (e.g. "OKR")
        for cid in self.store.title_search(keywords or [], 6):
            if cid not in seen:
                chunk = self.store.chunks_by_ids([cid]).get(cid)
                if chunk and chunk.note_id not in self.excluded:
                    seen.add(cid)
                    merged.append(Hit(chunk, 0.0, {"title"}))
        return cap_per_note(merged, per_note, limit)

    # -- anchors --------------------------------------------------------------------------------

    def anchors(self, question: str, exclude_ids: set[int]) -> list[Hit]:
        """Always-considered notes from the config: their excerpts most relevant to the question."""
        refs = self.cfg.retrieval.anchor_notes
        if not refs:
            return []
        mat, ids, notes = self.store.matrix()
        if len(ids) == 0:
            return []
        q = self.embed_query(question)
        out: list[Hit] = []
        done_paths: set[str] = set()
        for ref in refs:
            for path in self.store.resolve_note(ref):
                if path in done_paths:
                    continue
                done_paths.add(path)
                cids = [c for c in self.store.note_chunk_ids(path) if c not in exclude_ids]
                if not cids:
                    continue
                rows = np.searchsorted(ids, cids)
                if notes[rows[0]] in self.excluded:
                    continue
                order = np.argsort(-(mat[rows] @ q))[: self.cfg.retrieval.anchor_excerpts]
                chunks = self.store.chunks_by_ids([cids[i] for i in order])
                out += [Hit(chunks[cids[i]], 1.0, {"anchor"}) for i in order if cids[i] in chunks]
        return out

    # -- the whole pipeline ---------------------------------------------------------------------

    def retrieve(self, question: str, *, queries: list[str] | None = None, keywords: list[str] | None = None,
                 keep: int, recent: bool = False, rerank: bool | None = None,
                 trace: list[str] | None = None) -> list[Hit]:
        """Hybrid search -> per-note cap -> re-rank -> add anchors. Returns the excerpts to read, in order."""
        r = self.cfg.retrieval
        boost = max(r.recency_boost, 0.5) if recent else r.recency_boost
        candidates = self.pool([question, *(queries or [])], keywords=keywords, boost=boost)
        if trace is not None:
            trace.append(f"{len(candidates)} candidate excerpts from hybrid search")
        use_rerank = r.rerank if rerank is None else rerank
        if use_rerank and len(candidates) > 1:
            candidates = self.rerank(question, candidates, keep, trace, topics=queries)
        else:
            candidates = candidates[:keep]
        have = {h.chunk.id for h in candidates}
        extra = [h for h in self.anchors(question, have)]
        if extra and trace is not None:
            trace.append(f"{len(extra)} excerpts added from anchor notes")
        return candidates + extra

    def rerank(self, question: str, hits: list[Hit], keep: int, trace: list[str] | None = None,
               min_score: float = 2.0, min_keep: int = 3, topics: list[str] | None = None) -> list[Hit]:
        """Ask the small planner model to score each candidate, so the chat model reads fewer, better excerpts."""
        prompt = (
            "You are helping pick which excerpts from a personal notes vault are useful for answering a question.\n"
            "Rate every excerpt from 0 (unrelated) to 10 (directly helps answer the question).\n"
            'Reply only with JSON: {"scores": [{"id": 1, "score": 7}, ...]} with one entry per excerpt.\n\n'
            f"Question: {question}\n"
            + (f"Also relevant (the question was split into these searches): {' | '.join(topics)}\n" if topics else "")
            + "\n"
            + "\n\n".join(f"[{i}] ({h.chunk.title})\n{h.chunk.text[:400]}" for i, h in enumerate(hits, 1))
        )
        try:
            raw = self.client.chat(self.cfg.models.planner_model, [{"role": "user", "content": prompt}],
                                   fmt="json", num_ctx=self.cfg.models.num_ctx)
            scores = parse_scores(raw, len(hits))
        except (OllamaError, ValueError) as e:
            if trace is not None:
                trace.append(f"re-ranking skipped ({str(e).splitlines()[0]})")
            return hits[:keep]
        for i, h in enumerate(hits, 1):
            h.rerank = scores.get(i)
        ranked = sorted(enumerate(hits), key=lambda t: (-(t[1].rerank if t[1].rerank is not None else 5.0), t[0]))
        ordered = [h for _, h in ranked]
        good = [h for h in ordered if (h.rerank if h.rerank is not None else 5.0) >= min_score]
        chosen = (good if len(good) >= min_keep else ordered[:min_keep])[:keep]
        if trace is not None:
            trace.append(f"re-ranked {len(hits)} candidates, keeping {len(chosen)}")
        return chosen


def recency_factor(mtime: float, now: float, boost: float, half_life_days: float) -> float:
    age_days = max(0.0, (now - mtime) / 86400.0)
    return 1.0 + boost * math.exp(-math.log(2) * age_days / max(half_life_days, 1.0))


def cap_per_note(hits: list[Hit], per_note: int, limit: int) -> list[Hit]:
    """Keep at most `per_note` excerpts from any one note, so related notes get a look in."""
    counts: dict[int, int] = {}
    out = []
    for h in hits:
        if counts.get(h.chunk.note_id, 0) >= per_note:
            continue
        counts[h.chunk.note_id] = counts.get(h.chunk.note_id, 0) + 1
        out.append(h)
        if len(out) >= limit:
            break
    return out


def parse_scores(raw: str, n: int) -> dict[int, float]:
    """Read the re-ranker's JSON reply. Tolerant of surrounding prose and of a bare list."""
    m = re.search(r"[\[{].*[\]}]", raw, flags=re.S)
    if not m:
        raise ValueError("no JSON in re-ranker reply")
    data = json.loads(m.group(0))
    items = data.get("scores", []) if isinstance(data, dict) else data
    out: dict[int, float] = {}
    for item in items:
        try:
            i, s = int(item["id"]), float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= i <= n:
            out[i] = s
    if not out:
        raise ValueError("re-ranker returned no usable scores")
    return out
