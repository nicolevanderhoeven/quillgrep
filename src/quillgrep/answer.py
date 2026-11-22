"""Answering: the fast `ask` path and the multi-search `deep` path."""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import Config
from .ollama import OllamaClient, OllamaError, strip_thinking
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

    def cited(self) -> list[tuple[int, Hit]]:
        """The excerpts the answer actually cites ([n] markers); every excerpt if it cites none."""
        nums = sorted({int(n) for n in re.findall(r"\[(\d{1,3})\]", self.text) if 1 <= int(n) <= len(self.hits)})
        return [(n, self.hits[n - 1]) for n in nums] or list(enumerate(self.hits, 1))


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


def plan_queries(cfg: Config, client: OllamaClient, question: str) -> tuple[list[str], list[str]]:
    """The planner step: split a question into several searches plus keywords for note titles."""
    prompt = (
        "You help search a personal Obsidian vault. Break this question into 3 to 5 short, separate search queries "
        "that together find everything needed to answer it (for example the person's goals, their past work, their "
        "recent interests). Also give 2 to 4 single keywords likely to appear in relevant note titles. "
        'Reply as JSON: {"queries": [...], "keywords": [...]}.\n\nQuestion: ' + question)
    try:
        raw = client.chat(cfg.models.planner_model, [{"role": "user", "content": prompt}], fmt="json",
                          num_ctx=cfg.models.num_ctx)
        plan = json.loads(raw)
        queries = _strings(plan.get("queries"))[:5]
        keywords = _strings(plan.get("keywords"))[:4]
    except (OllamaError, ValueError, AttributeError):
        queries, keywords = [], []
    return queries, keywords


def _generate(cfg: Config, client: OllamaClient, messages: list[dict], num_ctx: int,
              on_text: Callable[[str], None] | None) -> str:
    if on_text is None:
        return client.chat(cfg.models.chat, messages, num_ctx=num_ctx)
    parts = []
    for piece in client.chat_stream(cfg.models.chat, messages, num_ctx=num_ctx):
        parts.append(piece)
        on_text(piece)
    return strip_thinking("".join(parts)).strip()


def ask(cfg: Config, client: OllamaClient, store: Store, question: str, *, recent: bool = False,
        rerank: bool | None = None, on_text: Callable[[str], None] | None = None,
        excluded_folders: list[str] | None = None, synthesis: bool = False) -> Answer:
    t0 = time.time()
    trace: list[str] = []
    retriever = Retriever(cfg, client, store, excluded_folders=excluded_folders)
    hits = retriever.retrieve(question, keep=cfg.retrieval.excerpts, recent=recent, rerank=rerank, trace=trace)
    if not hits:
        return Answer("I could not find anything in the index. Run `quillgrep index` first.", [], "ask", trace=trace,
                      seconds=time.time() - t0)
    messages = build_messages(question, hits, deep=False, synthesis=synthesis)
    text = _generate(cfg, client, messages, cfg.models.num_ctx, on_text)
    return Answer(text, hits, "ask", trace=trace, seconds=time.time() - t0)


def deep(cfg: Config, client: OllamaClient, store: Store, question: str, *, recent: bool = False,
         rerank: bool | None = None, on_text: Callable[[str], None] | None = None,
         excluded_folders: list[str] | None = None, synthesis: bool = False) -> Answer:
    t0 = time.time()
    trace: list[str] = []
    queries, keywords = plan_queries(cfg, client, question)
    trace.append(f"planner split the question into {len(queries)} searches")
    retriever = Retriever(cfg, client, store, excluded_folders=excluded_folders)
    hits = retriever.retrieve(question, queries=queries, keywords=keywords, keep=cfg.retrieval.deep_excerpts,
                              recent=recent, rerank=rerank, trace=trace)
    if not hits:
        return Answer("I could not find anything in the index. Run `quillgrep index` first.", [], "deep",
                      queries, keywords, trace, time.time() - t0)
    messages = build_messages(question, hits, deep=True, synthesis=synthesis)
    text = _generate(cfg, client, messages, cfg.models.deep_num_ctx, on_text)
    return Answer(text, hits, "deep", queries, keywords, trace, time.time() - t0)


def format_sources(answer: Answer) -> str:
    """`[1][4] Projects/Talk.md` lines for the notes the answer cites."""
    by_path: dict[str, list[int]] = {}
    for n, hit in answer.cited():
        by_path.setdefault(hit.chunk.path, []).append(n)
    return "\n".join("".join(f"[{n}]" for n in nums) + f" {path}" for path, nums in by_path.items())
