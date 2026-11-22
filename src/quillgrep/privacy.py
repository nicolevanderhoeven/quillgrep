"""Privacy filters for anything that leaves Quillgrep through the MCP server."""

from __future__ import annotations

import re

from .config import Privacy

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# Digits with the usual phone separators. Counted in code so ordinary dates (2026-11-22) are left alone.
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d ()./-]{6,}\d(?![\w])")
_NUMBER = re.compile(r"\d{8,}")  # long runs of digits: account numbers, IDs, card numbers


def _redact_phone(m: re.Match) -> str:
    digits = sum(c.isdigit() for c in m.group(0))
    return "[phone]" if digits >= 9 else m.group(0)


def redact(text: str, kinds: list[str], extra_patterns: list[str] | None = None) -> str:
    """Replace emails, phone numbers and long digit strings (and any custom patterns) with a marker."""
    if "email" in kinds:
        text = _EMAIL.sub("[email]", text)
    if "phone" in kinds:
        text = _PHONE.sub(_redact_phone, text)
    if "number" in kinds:
        text = _NUMBER.sub("[number]", text)
    for pattern in extra_patterns or []:
        try:
            text = re.sub(pattern, "[redacted]", text)
        except re.error:
            continue
    return text


def _norm(word: str) -> str:
    return re.sub(r"\W+", "", word).lower()


def limit_quotes(answer: str, source_texts: list[str], max_words: int) -> str:
    """Cut out any run of more than `max_words` words that is copied verbatim from a note."""
    if max_words <= 0 or not answer:
        return answer
    window = max_words + 1
    grams: set[tuple[str, ...]] = set()
    for text in source_texts:
        words = [w for w in (_norm(w) for w in text.split()) if w]
        for i in range(len(words) - window + 1):
            grams.add(tuple(words[i : i + window]))
    if not grams:
        return answer
    parts = re.split(r"(\s+)", answer)  # words at even positions, whitespace at odd ones
    norm = {i: _norm(parts[i]) for i in range(0, len(parts), 2)}
    idx = [i for i, n in norm.items() if n]  # positions of real words
    drop: set[int] = set()
    for a in range(len(idx) - window + 1):
        if tuple(norm[idx[a + j]] for j in range(window)) in grams:
            drop.update(idx[a : a + window])
    if not drop:
        return answer
    out = "".join("[…]" if i in drop else p for i, p in enumerate(parts))
    return re.sub(r"\[…\](?:\s+\[…\])+", "[…]", out)


def clean_for_export(text: str, privacy: Privacy, source_texts: list[str]) -> str:
    """Everything the MCP server applies to a synthesis before it leaves: quote limit, then redaction."""
    text = limit_quotes(text, source_texts, privacy.max_quote_words)
    return redact(text, privacy.redact, privacy.redact_patterns)
