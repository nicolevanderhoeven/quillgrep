"""Answering: the fast `ask` path and the multi-search `deep` path."""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import Config
from .ollama import OllamaClient, OllamaError, stage_ctx, strip_thinking
from .retrieval import Hit, Retriever
from .store import Store

SYNTHESIS_RULES = (
    "Your answer will be handed to another AI assistant, not read by the owner of the notes, so write a synthesis "
    "in your own words. Refer to excerpts only by their number, never by file name. "
    "Do not copy passages from the notes; quote at most a short phrase. "
    "Never include email addresses, phone numbers or account numbers."
)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    mode: str = "ask"
    queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    seconds: float = 0.0
    timing: dict[str, float] = field(default_factory=dict)  # seconds: plan (deep), retrieve, first_token, total

    def cited(self, fallback: bool = True) -> list[tuple[int, Hit]]:
        """The excerpts the answer actually cites ([n] markers); every excerpt if it cites none and `fallback`."""
        nums = sorted({int(n) for n in re.findall(r"\[(\d{1,3})\]", self.text) if 1 <= int(n) <= len(self.hits)})
        cited = [(n, self.hits[n - 1]) for n in nums]
        return cited or (list(enumerate(self.hits, 1)) if fallback else [])


def excerpts_block(hits: list[Hit], with_paths: bool = True) -> str:
    """Numbered excerpts. For the MCP server the model sees numbers only, so it cannot leak paths or titles."""
    return "\n\n".join(
        f"[{i}] ({h.chunk.path})\n{h.chunk.text}" if with_paths else f"[{i}]\n{h.chunk.text}"
        for i, h in enumerate(hits, 1))


def build_messages(question: str, hits: list[Hit], *, deep: bool, synthesis: bool = False) -> list[dict]:
    today = dt.date.today().isoformat()
    if deep:
        rules = ("Answer the question using these excerpts from my notes. Combine information across them; it is "
                 "fine to reason and suggest, but say which excerpts each point draws on, citing them by number in "
                 "square brackets like [1] or [2][3]. Say clearly if something important is missing from the notes.")
    else:
        rules = ("Answer the question using only these excerpts from my notes. Cite the excerpts you used by number "
                 "in square brackets, like [1] or [2][3]. If the notes do not contain the answer, say so plainly "
                 "instead of guessing.")
    if synthesis:
        rules += " " + SYNTHESIS_RULES
    block = excerpts_block(hits, with_paths=not synthesis)
    prompt = f"Today's date is {today}.\n{rules}\n\n{block}\n\nQuestion: {question}"
    return [{"role": "user", "content": prompt}]


def _strings(value) -> list[str]:
    """The non-empty strings of a model-supplied list (models sometimes reply with null or a bare string)."""
    return [v.strip() for v in value if isinstance(v, str) and v.strip()] if isinstance(value, list) else []


def plan_queries(
    cfg: Config, client: OllamaClient, question: str, answer_model: str, answer_cap: int
) -> tuple[list[str], list[str]]:
    """The planner step: split a question into several searches plus keywords for note titles."""
    prompt = (
        "You help search a personal Obsidian vault. Break this question into 3 to 5 short, separate search queries "
        "that together find everything needed to answer it (for example the person's goals, their past work, their "
        "recent interests). Also give 2 to 4 single keywords likely to appear in relevant note titles. "
        'Reply as JSON: {"queries": [...], "keywords": [...]}.\n\nQuestion: ' + question)
    try:
        messages = [{"role": "user", "content": prompt}]
        raw = client.chat(cfg.models.planner_model, messages, fmt="json",
                          num_ctx=stage_ctx(cfg.models.planner_model, answer_model, messages, answer_cap, 768))
        plan = json.loads(raw)
        queries = _strings(plan.get("queries"))[:5]
        keywords = _strings(plan.get("keywords"))[:4]
    except (OllamaError, ValueError, AttributeError):
        queries, keywords = [], []
    return queries, keywords


def _generate(client: OllamaClient, model: str, messages: list[dict], cap: int,
              on_text: Callable[[str], None] | None, planner_model: str) -> str:
    num_ctx = stage_ctx(model, planner_model, messages, cap)
    if on_text is None:
        return client.chat(model, messages, num_ctx=num_ctx)
    parts = []
    for piece in client.chat_stream(model, messages, num_ctx=num_ctx):
        parts.append(piece)
        on_text(piece)
    return strip_thinking("".join(parts)).strip()


class _FirstToken:
    """Wraps `on_text` to note when the first piece of the answer arrives."""

    def __init__(self, t0: float, on_text: Callable[[str], None] | None):
        self.t0, self.on_text, self.seconds = t0, on_text, None

    def __call__(self, piece: str) -> None:
        if self.seconds is None and piece:
            self.seconds = time.perf_counter() - self.t0
        self.on_text(piece)


def _timing(t0: float, first: _FirstToken | None, **stages: float) -> dict[str, float]:
    out = dict(stages)
    if first is not None and first.seconds is not None:
        out["first_token"] = first.seconds
    out["total"] = time.perf_counter() - t0
    return out


def ask(cfg: Config, client: OllamaClient, store: Store, question: str, *, recent: bool = False,
        rerank: bool | None = None, on_text: Callable[[str], None] | None = None,
        excluded_folders: list[str] | None = None, synthesis: bool = False) -> Answer:
    t0 = time.perf_counter()
    trace: list[str] = []
    retriever = Retriever(cfg, client, store, excluded_folders=excluded_folders,
                          answer_model=cfg.models.ask_model, answer_cap=cfg.models.num_ctx)
    hits = retriever.retrieve(question, keep=cfg.retrieval.excerpts, recent=recent, rerank=rerank, trace=trace,
                              pool_size=cfg.retrieval.rerank_pool)
    retrieved = time.perf_counter() - t0
    if not hits:
        return Answer("I could not find anything in the index. Run `quillgrep index` first.", [], "ask", trace=trace,
                      seconds=time.perf_counter() - t0, timing=_timing(t0, None, retrieve=retrieved))
    messages = build_messages(question, hits, deep=False, synthesis=synthesis)
    first = _FirstToken(t0, on_text) if on_text else None
    text = _generate(client, cfg.models.ask_model, messages, cfg.models.num_ctx, first, cfg.models.planner_model)
    return Answer(text, hits, "ask", trace=trace, seconds=time.perf_counter() - t0,
                  timing=_timing(t0, first, retrieve=retrieved))


def deep(cfg: Config, client: OllamaClient, store: Store, question: str, *, recent: bool = False,
         rerank: bool | None = None, on_text: Callable[[str], None] | None = None,
         excluded_folders: list[str] | None = None, synthesis: bool = False) -> Answer:
    t0 = time.perf_counter()
    trace: list[str] = []
    retriever = Retriever(cfg, client, store, excluded_folders=excluded_folders,
                          answer_model=cfg.models.deep_model, answer_cap=cfg.models.deep_num_ctx)
    with ThreadPoolExecutor(max_workers=max(1, cfg.retrieval.concurrency)) as executor:
        # The question's own search does not depend on the plan, so embed it while the planner is thinking.
        retriever.prefetch([question], executor)
        queries, keywords = plan_queries(cfg, client, question, cfg.models.deep_model, cfg.models.deep_num_ctx)
        retriever.prefetch(queries, executor)  # ...and the planned searches all at once
    planned = time.perf_counter() - t0
    trace.append(f"planner split the question into {len(queries)} searches")
    hits = retriever.retrieve(question, queries=queries, keywords=keywords, keep=cfg.retrieval.deep_excerpts,
                              recent=recent, rerank=rerank, trace=trace, pool_size=cfg.retrieval.deep_rerank_pool)
    retrieved = time.perf_counter() - t0 - planned
    if not hits:
        return Answer("I could not find anything in the index. Run `quillgrep index` first.", [], "deep",
                      queries, keywords, trace, time.perf_counter() - t0,
                      timing=_timing(t0, None, plan=planned, retrieve=retrieved))
    messages = build_messages(question, hits, deep=True, synthesis=synthesis)
    first = _FirstToken(t0, on_text) if on_text else None
    text = _generate(client, cfg.models.deep_model, messages, cfg.models.deep_num_ctx, first, cfg.models.planner_model)
    return Answer(text, hits, "deep", queries, keywords, trace, time.perf_counter() - t0,
                  timing=_timing(t0, first, plan=planned, retrieve=retrieved))


def source_entries(answer: Answer) -> list[tuple[list[int], str]]:
    """The notes the answer cites, as ([excerpt numbers], note path), in order of first citation."""
    by_path: dict[str, list[int]] = {}
    for n, hit in answer.cited():
        by_path.setdefault(hit.chunk.path, []).append(n)
    return [(nums, path) for path, nums in by_path.items()]


def format_sources(answer: Answer) -> str:
    """`[1][4] Projects/Talk.md` lines for the notes the answer cites."""
    return "\n".join("".join(f"[{n}]" for n in nums) + f" {path}" for nums, path in source_entries(answer))


def format_timing(answer: Answer) -> str:
    """One line for `--timing`: when the first word appeared, the total, and where the time went."""
    t = answer.timing
    parts = []
    if "first_token" in t:
        parts.append(f"first token {t['first_token']:.1f}s")
    parts.append(f"total {t['total']:.1f}s")
    stages = [f"{name} {t[name]:.1f}s" for name in ("plan", "retrieve") if name in t]
    if "first_token" in t:
        stages.append(f"answer {t['total'] - t.get('plan', 0) - t.get('retrieve', 0):.1f}s")
    return "timing: " + ", ".join(parts) + (" (" + ", ".join(stages) + ")" if stages else "")
