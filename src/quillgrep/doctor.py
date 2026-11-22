"""The first-run check: is everything Quillgrep needs in place, and if not, exactly what to do."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .config import Config, ConfigError
from .ollama import INSTALL_URL, OllamaClient, OllamaError, check_ready, model_name_matches
from .store import Store, fts_available


@dataclass
class Check:
    ok: bool
    label: str
    fix: str = ""
    blocking: bool = True


def run_checks(cfg: Config, client: OllamaClient) -> list[Check]:
    checks: list[Check] = []
    checks.append(Check(sys.version_info >= (3, 11), f"Python {sys.version.split()[0]}",
                        "Quillgrep needs Python 3.11 or newer: https://www.python.org/downloads/"))
    checks.append(Check(fts_available(), "SQLite full-text search (FTS5)",
                        "Install Python from https://www.python.org/downloads/ (your Python's SQLite lacks FTS5)."))
    try:
        vault = cfg.vault_path
        checks.append(Check(True, f"Vault: {vault}"))
    except ConfigError as e:
        checks.append(Check(False, "Vault", str(e)))
        vault = None

    required = {"embedding": cfg.models.embedding, "chat": cfg.models.deep_model}
    # Optional: when one is missing Quillgrep uses the main model instead, so this never blocks.
    optional = {"ask": cfg.models.ask_chat, "planner": cfg.models.planner}
    try:
        installed = client.installed_models()
        checks.append(Check(True, f"Ollama is running at {cfg.models.host}"))
        for role, model in required.items():
            present = any(model_name_matches(model, name) for name in installed)
            checks.append(Check(present, f"Model for {role}: {model}", f"Download it with:  ollama pull {model}"))
        for role, model in optional.items():
            if model:
                present = any(model_name_matches(model, name) for name in installed)
                checks.append(Check(present, f"Model for {role}: {model}",
                                    f"Not installed, so the main model is used instead (slower). "
                                    f"For faster answers:  ollama pull {model}", blocking=False))
    except OllamaError as e:
        checks.append(Check(False, f"Ollama at {cfg.models.host}", str(e)))

    if vault is not None:
        try:
            stats = Store(cfg.index_path).stats() if cfg.index_path.exists() else {"notes": 0}
            if stats["notes"]:
                checks.append(Check(True, f"Index: {stats['notes']} notes, {stats['chunks']} passages "
                                          f"(last indexed {stats['indexed_at']})", blocking=False))
            else:
                checks.append(Check(False, "Index is empty", "Run:  quillgrep index", blocking=False))
        except Exception as e:  # noqa: BLE001
            checks.append(Check(False, "Index", f"Could not open the index: {e}", blocking=False))
    return checks


def report(checks: list[Check], out=None) -> bool:
    """Print the checks; returns True when nothing blocking failed."""
    out = out or sys.stdout
    ok = True
    for c in checks:
        mark = "✓" if c.ok else ("✗" if c.blocking else "•")
        print(f"  {mark} {c.label}", file=out)
        if not c.ok:
            for line in c.fix.splitlines():
                print(f"      {line}", file=out)
            ok = ok and not c.blocking
    return ok


def ensure_ready(cfg: Config, client: OllamaClient, models: list[str], err=None) -> bool:
    """A quick check before indexing or asking. Prints what to install or pull when something is missing."""
    err = err or sys.stderr
    problems = check_ready(client, models)
    for p in problems:
        print(p, file=err)
    if problems:
        print("\nRun `quillgrep doctor` for a full check.", file=err)
    return not problems


__all__ = ["Check", "run_checks", "report", "ensure_ready", "INSTALL_URL"]
