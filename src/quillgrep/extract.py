"""Turn vault files into clean text: Markdown notes, Excalidraw drawings and PDFs."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import IndexRules

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_COMPRESSED = re.compile(r"```compressed-json.*?```", re.S)
_DATA_URI = re.compile(r"data:[a-zA-Z]+/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
_LONG_BLOB = re.compile(r"[A-Za-z0-9+/=]{200,}")
_EXCALIDRAW_TEXT = re.compile(r"## Text Elements\n(.*?)(?:\n## |\n%%|\Z)", re.S)
_ELEMENT_ID = re.compile(r"\s*\^[A-Za-z0-9]{6,}")


def is_excalidraw(name: str, text: str) -> bool:
    return name.endswith(".excalidraw.md") or "excalidraw-plugin:" in text[:2000]


def clean_markdown(text: str, name: str = "", excalidraw: bool = True) -> str:
    """Strip front matter and embedded binary data; keep only the words of an Excalidraw drawing."""
    text = text.replace("\r\n", "\n")
    if is_excalidraw(name, text):
        if not excalidraw:
            return ""
        m = _EXCALIDRAW_TEXT.search(text)
        text = _ELEMENT_ID.sub("", m.group(1)) if m else ""
    text = _FRONTMATTER.sub("", text)
    text = _COMPRESSED.sub("", text)
    text = _DATA_URI.sub("", text)
    text = _LONG_BLOB.sub("", text)
    return text.strip()


def pdf_text(path: Path, max_mb: int = 50) -> str:
    """Extract the text of a PDF. Unreadable, encrypted or huge files yield an empty string."""
    try:
        if path.stat().st_size > max_mb * 1024 * 1024:
            return ""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return ""
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:  # a single broken page should not lose the whole document
                continue
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def read_text(path: Path, kind: str, rules: IndexRules) -> str:
    if kind == "pdf":
        return pdf_text(path, rules.max_pdf_mb)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return clean_markdown(raw, path.name, rules.excalidraw)


def _matches(rel: str, patterns: list[str]) -> bool:
    """A rule matches a folder (or file) by prefix, or as a glob such as `Archive/*/old`."""
    rel = rel.strip("/")
    for pat in patterns:
        pat = pat.replace("\\", "/").strip("/")
        if not pat:
            continue
        if rel == pat or rel.startswith(pat + "/") or fnmatch.fnmatchcase(rel, pat):
            return True
        # allow a glob to name a folder: `Archive/*` excludes everything beneath it
        parts = rel.split("/")
        if any(fnmatch.fnmatchcase("/".join(parts[: i + 1]), pat) for i in range(len(parts))):
            return True
    return False


def in_folders(rel: str, folders: list[str]) -> bool:
    return _matches(rel, folders)


@dataclass(frozen=True)
class Source:
    rel: str  # vault-relative path with forward slashes, e.g. "Projects/Talk.md"
    path: Path
    kind: str  # "md" | "pdf"
    mtime: float
    size: int


def scan_vault(vault: Path, rules: IndexRules) -> Iterator[Source]:
    """Yield every indexable file, applying the include/exclude rules from the config."""
    exclude = rules.exclude
    stack = [vault]
    while stack:
        folder = stack.pop()
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for p in entries:
            rel = p.relative_to(vault).as_posix()
            if p.is_symlink() and p.is_dir():
                continue  # avoid loops
            if p.is_dir():
                if not _matches(rel, exclude):
                    stack.append(p)
                continue
            if _matches(rel, exclude):
                continue
            suffix = p.suffix.lower()
            kind = "md" if suffix == ".md" else "pdf" if suffix == ".pdf" and rules.pdf else None
            if not kind:
                continue
            if rules.include and not _matches(rel, rules.include):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            yield Source(rel, p, kind, st.st_mtime, st.st_size)


def title_of(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    for suffix in (".excalidraw.md", ".md", ".pdf"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into chunks of about `size` characters, preferring paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) > size:  # a very long paragraph: cut it into overlapping windows
            flush()
            step = max(1, size - overlap)
            for i in range(0, len(block), step):
                chunks.append(block[i : i + size])
                if i + size >= len(block):
                    break
            continue
        if current and len(current) + len(block) + 2 > size:
            flush()
        current = f"{current}\n\n{block}" if current else block
    flush()
    return chunks
