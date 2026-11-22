"""Configuration: a single TOML file, with defaults for everything except the vault path."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

APP = "quillgrep"


class ConfigError(Exception):
    """The configuration is missing or invalid. The message tells the user how to fix it."""


def config_dir() -> Path:
    if env := os.environ.get("QUILLGREP_HOME"):
        return Path(env) / "config"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / APP
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / APP


def data_dir() -> Path:
    if env := os.environ.get("QUILLGREP_HOME"):
        return Path(env) / "data"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / APP


def default_config_path() -> Path:
    return config_dir() / "config.toml"


@dataclass
class Models:
    host: str = "http://127.0.0.1:11434"
    embedding: str = "nomic-embed-text"
    chat: str = "qwen3.5:9b"
    planner: str = ""  # small/fast model for query planning and re-ranking; empty = use `chat`
    num_ctx: int = 8192  # context window for `ask`
    deep_num_ctx: int = 16384  # context window for `deep` and MCP deep mode
    timeout_seconds: int = 900

    @property
    def planner_model(self) -> str:
        return self.planner or self.chat


@dataclass
class IndexRules:
    include: list[str] = field(default_factory=list)  # empty = the whole vault
    exclude: list[str] = field(default_factory=lambda: [".trash", ".obsidian"])
    pdf: bool = True
    excalidraw: bool = True
    chunk_chars: int = 1200
    chunk_overlap: int = 150
    max_pdf_mb: int = 50


@dataclass
class Retrieval:
    candidates: int = 40  # how many chunks each search stage proposes before fusion
    per_note: int = 3  # per-note excerpt cap: at most this many excerpts from one note
    excerpts: int = 8  # excerpts the chat model reads in `ask`
    deep_excerpts: int = 18  # excerpts the chat model reads in `deep`
    rerank: bool = True  # let the planner model re-rank candidates before answering
    rerank_pool: int = 30  # candidates sent to the re-ranker
    recency_boost: float = 0.0  # 0 = off. 0.5 = a note edited today scores up to 50% higher
    recency_half_life_days: float = 180.0
    anchor_notes: list[str] = field(default_factory=list)
    anchor_excerpts: int = 2  # excerpts taken from each anchor note per question


@dataclass
class Privacy:
    """Controls for the MCP server, which hands answers to other (usually cloud) AI models."""

    excluded_folders: list[str] = field(default_factory=list)
    show_titles: bool = True
    redact: list[str] = field(default_factory=lambda: ["email", "phone", "number"])
    redact_patterns: list[str] = field(default_factory=list)
    max_quote_words: int = 20  # longer verbatim runs from a note are cut from the synthesis; 0 = off
    audit_log: str = ""  # empty = <data dir>/audit.log


@dataclass
class Config:
    vault: str = ""
    models: Models = field(default_factory=Models)
    index: IndexRules = field(default_factory=IndexRules)
    retrieval: Retrieval = field(default_factory=Retrieval)
    privacy: Privacy = field(default_factory=Privacy)
    path: Path | None = None  # where this config was loaded from (None = defaults)

    @property
    def vault_path(self) -> Path:
        if not self.vault:
            raise ConfigError(
                "No vault is set. Run `quillgrep init /path/to/your/vault`, "
                "or add `vault = \"/path/to/your/vault\"` to your config file "
                f"({default_config_path()})."
            )
        p = Path(self.vault).expanduser()
        if not p.is_dir():
            raise ConfigError(f"The vault folder does not exist: {p}\nFix `vault` in {self.path or 'your config'}.")
        return p

    @property
    def index_path(self) -> Path:
        """One index file per vault, so several vaults can share a data directory."""
        key = hashlib.sha256(str(self.vault_path.resolve()).encode()).hexdigest()[:10]
        return data_dir() / f"index-{key}.db"

    @property
    def audit_path(self) -> Path:
        return Path(self.privacy.audit_log).expanduser() if self.privacy.audit_log else data_dir() / "audit.log"


_SECTIONS = {"models": Models, "index": IndexRules, "retrieval": Retrieval, "privacy": Privacy}


def _fill(obj, table: dict, where: str):
    known = {f for f in obj.__dataclass_fields__}
    for key, value in table.items():
        if key not in known:
            raise ConfigError(f"Unknown setting `{key}` in [{where}]. Known settings: {', '.join(sorted(known))}.")
        current = getattr(obj, key)
        if isinstance(current, bool) and not isinstance(value, bool):
            raise ConfigError(f"`{key}` in [{where}] must be true or false.")
        if isinstance(current, list) and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ConfigError(f"`{key}` in [{where}] must be a list of strings.")
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"`{key}` in [{where}] must be a number.")
        if isinstance(current, str) and not isinstance(value, str):
            raise ConfigError(f"`{key}` in [{where}] must be text.")
        setattr(obj, key, value)


def load_config(path: str | os.PathLike | None = None, vault: str | None = None) -> Config:
    """Load the config file (if any), then apply a `--vault` override."""
    path = Path(path or os.environ.get("QUILLGREP_CONFIG") or default_config_path()).expanduser()
    cfg = Config()
    if path.is_file():
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(
                f"Could not read {path}: {e}\n"
                "Tip: on Windows write paths with forward slashes (C:/Users/me/Vault) or in single quotes."
            ) from e
        cfg.path = path
        for key, value in raw.items():
            if key == "vault":
                if not isinstance(value, str):
                    raise ConfigError("`vault` must be text, for example vault = \"/path/to/vault\".")
                cfg.vault = value
            elif key in _SECTIONS:
                if not isinstance(value, dict):
                    raise ConfigError(f"[{key}] must be a table.")
                _fill(getattr(cfg, key), value, key)
            else:
                raise ConfigError(f"Unknown top-level setting `{key}` in {path}.")
    elif os.environ.get("QUILLGREP_CONFIG") or path != default_config_path():
        raise ConfigError(f"Config file not found: {path}")
    if vault:
        cfg.vault = vault
    return cfg


def starter_config(vault: str) -> str:
    """The text `quillgrep init` writes. Comments explain every setting a beginner might want."""
    return f"""\
# Quillgrep configuration. Everything except `vault` is optional.
# Docs: https://github.com/nicolevanderhoeven/quillgrep#configuration

# The folder that holds your Obsidian vault. On Windows use forward slashes (C:/Users/me/Vault).
vault = {json.dumps(vault)}

[models]
# Quillgrep talks to a local Ollama install (https://ollama.com). Nothing here ever leaves your computer.
host = "http://127.0.0.1:11434"
embedding = "nomic-embed-text"
chat = "qwen3.5:9b"
# A smaller, faster model for planning searches and re-ranking. Empty = use `chat`.
planner = ""

[index]
# Only index these folders (relative to the vault). Empty = everything.
include = []
# Never index these folders. Obsidian's own settings and the trash are skipped by default.
exclude = [".trash", ".obsidian"]
pdf = true
excalidraw = true

[retrieval]
per_note = 3             # at most this many excerpts from a single note per answer
rerank = true            # have the planner model pick the best excerpts before answering
recency_boost = 0.0      # 0 = off; try 0.5 to favour recently edited notes (or use --recent)
anchor_notes = []        # notes that are always considered, e.g. ["Goals/OKRs.md", "Conferences"]

[privacy]
# These apply to the MCP server (`quillgrep mcp`), which lets other AI tools ask questions
# of your notes. They only ever receive a synthesis written by your local model, never note text.
excluded_folders = []    # folders the MCP server may never read from, e.g. ["Journal", "Private"]
show_titles = true       # false = do not even reveal note titles in citations
redact = ["email", "phone", "number"]   # scrubbed from every answer the MCP server returns
max_quote_words = 20     # cut verbatim quotes from notes longer than this; 0 = off
"""
