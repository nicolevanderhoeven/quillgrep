"""The `quillgrep` command line."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, default_config_path, load_config, starter_config
from .doctor import ensure_ready, report, run_checks
from .ollama import OllamaClient, OllamaError, resolve_models

SKIP_DIRS = {"Library", "AppData", "node_modules", ".git", ".cache", "Applications", "Windows", "Program Files"}


def _client(cfg: Config) -> OllamaClient:
    return OllamaClient(cfg.models.host, cfg.models.timeout_seconds, cfg.models.keep_alive)


def _open_store(cfg: Config):
    from .store import Store

    return Store(cfg.index_path)


def find_vaults(root: Path, depth: int = 3, limit: int = 8) -> list[Path]:
    """Look for Obsidian vaults (folders containing `.obsidian`) near `root`."""
    found: list[Path] = []

    def walk(folder: Path, level: int):
        if len(found) >= limit or level > depth:
            return
        try:
            entries = [e for e in os.scandir(folder) if e.is_dir(follow_symlinks=False)]
        except OSError:
            return
        if any(e.name == ".obsidian" for e in entries):
            found.append(folder)
            return
        for e in sorted(entries, key=lambda e: e.name):
            if e.name not in SKIP_DIRS and not e.name.startswith("."):
                walk(Path(e.path), level + 1)

    walk(root, 0)
    return found


# -- commands -----------------------------------------------------------------------------------


def cmd_init(args, cfg: Config) -> int:
    path = Path(args.config or os.environ.get("QUILLGREP_CONFIG") or default_config_path()).expanduser()
    if path.exists() and not args.force:
        print(f"A config file already exists: {path}\nEdit it, or run `quillgrep init --force` to replace it.",
              file=sys.stderr)
        return 1
    vault = args.vault_arg or args.vault
    if not vault:
        candidates = find_vaults(Path.home())
        if candidates and sys.stdin.isatty():
            print("Obsidian vaults found on this computer:")
            for i, c in enumerate(candidates, 1):
                print(f"  {i}. {c}")
            choice = input("Which one? Enter a number, or paste a different folder path: ").strip()
            picked = choice.isdigit() and 0 < int(choice) <= len(candidates)
            vault = str(candidates[int(choice) - 1]) if picked else choice
        elif sys.stdin.isatty():
            vault = input("Path to your Obsidian vault folder: ").strip()
    vault = os.path.expanduser(vault.strip().strip("'\"")) if vault else ""
    if not vault or not Path(vault).is_dir():
        print("That folder does not exist. Usage: quillgrep init /path/to/your/vault", file=sys.stderr)
        return 1
    vault = str(Path(vault).resolve()).replace("\\", "/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(starter_config(vault), encoding="utf-8")
    print(f"Wrote {path}\n")
    cfg = load_config(path)
    ok = report(run_checks(cfg, _client(cfg)))
    print("\nNext: run `quillgrep index` to read your notes (the first run takes a while), then "
          "`quillgrep ask \"your question\"`." if ok else
          "\nFix the ✗ items above, then run `quillgrep doctor` again.")
    return 0 if ok else 2


def cmd_doctor(args, cfg: Config) -> int:
    print(f"Quillgrep {__version__}  (config: {cfg.path or 'none, using defaults'})")
    return 0 if report(run_checks(cfg, _client(cfg))) else 2


def cmd_status(args, cfg: Config) -> int:
    cfg.vault_path  # noqa: B018 - raises a helpful ConfigError when unset
    s = _open_store(cfg).stats()
    print(json.dumps(s, indent=2) if args.json else
          f"{s['notes']} notes ({s['chunks']} passages) in {s['size_mb']} MB, embedding model "
          f"{s['embedding_model']}, last indexed {s['indexed_at'] or 'never'}.\nIndex file: {cfg.index_path}")
    return 0


def cmd_index(args, cfg: Config) -> int:
    from .indexer import index_vault

    cfg.vault_path  # noqa: B018
    client = _client(cfg)
    if not ensure_ready(cfg, client, [cfg.models.embedding]):
        return 2
    store = _open_store(cfg)
    print(f"Indexing {cfg.vault_path} ...", file=sys.stderr)
    report_ = index_vault(cfg, client, store, rebuild=args.rebuild, progress=lambda m: print(m, file=sys.stderr))
    print("Done: " + report_.summary())
    return 0


def _run_answer(args, cfg: Config, deep: bool) -> int:
    from . import answer as answer_mod
    from .render import make_output

    cfg.vault_path  # noqa: B018
    client = _client(cfg)
    cfg, notes = resolve_models(cfg, client)  # an optional model that is not installed falls back to the main one
    models = [cfg.models.embedding, cfg.models.deep_model if deep else cfg.models.ask_model]
    if deep or cfg.retrieval.rerank:
        models.append(cfg.models.planner_model)
    if not ensure_ready(cfg, client, models):
        return 2
    for note in notes:
        print("Note: " + note, file=sys.stderr)
    store = _open_store(cfg)
    if store.stats()["chunks"] == 0:
        print("Nothing is indexed yet. Run `quillgrep index` first.", file=sys.stderr)
        return 1
    question = " ".join(args.question).strip()
    if not question:
        print("Please give a question, for example: quillgrep ask \"What did I plan for the talk?\"", file=sys.stderr)
        return 1
    run = answer_mod.deep if deep else answer_mod.ask
    out = make_output(args.plain, args.markdown, cfg.output.links)
    if out.mode == "markdown":
        out.start("Planning searches, then reading the best excerpts..." if deep else "Searching your notes...")
    elif deep:
        print("Planning searches, then reading the best excerpts. This can take a few minutes...", file=sys.stderr)
    result = run(cfg, client, store, question, recent=args.recent,
                 rerank=False if args.no_rerank else None, on_text=out.write)
    out.finish(result, cfg.obsidian_vault_name)
    if args.verbose or deep or args.timing:
        notes = [f"{result.mode}, {cfg.models.deep_model if deep else cfg.models.ask_model}, {result.seconds:.0f}s"]
        if result.queries:
            notes.append("searches: " + " | ".join(result.queries))
        if result.keywords:
            notes.append("title keywords: " + ", ".join(result.keywords))
        if args.verbose:
            notes += result.trace
        print("\n[" + "; ".join(notes) + "]", file=sys.stderr)
    if args.timing:
        print(answer_mod.format_timing(result), file=sys.stderr)
    return 0


def cmd_ask(args, cfg: Config) -> int:
    return _run_answer(args, cfg, deep=False)


def cmd_deep(args, cfg: Config) -> int:
    return _run_answer(args, cfg, deep=True)


def cmd_mcp(args, cfg: Config) -> int:
    if args.print_setup:
        return _print_mcp_setup(args)
    from .mcp_server import serve

    cfg.vault_path  # noqa: B018
    serve(cfg)
    return 0


def _print_mcp_setup(args) -> int:
    exe = shutil.which("quillgrep") or sys.argv[0]
    cmd_args = (["--config", str(Path(args.config).expanduser().resolve())] if args.config else []) + ["mcp"]
    desktop = {"mcpServers": {"quillgrep": {"command": exe, "args": cmd_args}}}
    code = "claude mcp add quillgrep -- " + " ".join([json.dumps(exe) if " " in exe else exe,
                                                        *[json.dumps(a) if " " in a else a for a in cmd_args]])
    print("Claude Code (run this once in a terminal):\n\n  " + code + "\n")
    print("Claude Desktop (add to claude_desktop_config.json, then restart Claude Desktop):\n")
    print(json.dumps(desktop, indent=2))
    return 0


def cmd_audit(args, cfg: Config) -> int:
    from .audit import read_entries, verify

    if args.verify:
        ok, msg = verify(cfg.audit_path)
        print(msg)
        return 0 if ok else 1
    entries = read_entries(cfg.audit_path, args.tail)
    if not entries:
        print(f"Nothing logged yet. The audit log lives at {cfg.audit_path}")
        return 0
    for e in entries:
        if e["event"] == "request":
            print(f"{e['ts']}  REQUEST  {e.get('tool')}  {json.dumps(e.get('arguments'), ensure_ascii=False)}")
        else:
            cites = ", ".join(e.get("citations") or []) or "none"
            print(f"{e['ts']}  RETURNED (request {e.get('request')}; citations: {cites}"
                  + (f"; error: {e['error']}" if e.get("error") else "") + ")\n    "
                  + str(e.get("returned", "")).replace("\n", "\n    "))
    print(f"\n(log file: {cfg.audit_path})")
    return 0


def cmd_config(args, cfg: Config) -> int:
    note = "" if cfg.path else "  (not created yet; run `quillgrep init`)"
    print(f"Config file: {cfg.path or default_config_path()}{note}")
    if cfg.vault:
        print(f"Vault: {cfg.vault}\nIndex: {cfg.index_path}\nAudit log: {cfg.audit_path}")
    return 0


# -- parser -------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", metavar="FILE", default=argparse.SUPPRESS, help="config file to use")
    common.add_argument("--vault", metavar="DIR", default=argparse.SUPPRESS, help="vault folder (overrides the config)")

    p = argparse.ArgumentParser(prog="quillgrep", parents=[common],
                                description="Private, local AI search and answers over your Obsidian vault.")
    p.add_argument("--version", action="version", version=f"quillgrep {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    def add(name, fn, help_, **kw):
        sp = sub.add_parser(name, parents=[common], help=help_, description=help_, **kw)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("init", cmd_init, "create the config file for your vault and run the first-run check")
    sp.add_argument("vault_arg", nargs="?", metavar="VAULT", help="path to your Obsidian vault folder")
    sp.add_argument("--force", action="store_true", help="replace an existing config file")
    add("doctor", cmd_doctor, "check Ollama, the models and the config; says exactly what to install")
    sp = add("status", cmd_status, "show what is in the index")
    sp.add_argument("--json", action="store_true")
    sp = add("index", cmd_index, "read your notes and (re)build the search index; only changed files are redone")
    sp.add_argument("--rebuild", action="store_true", help="throw the index away and start again")
    for name, fn, help_ in (("ask", cmd_ask, "ask a question (fast: one search, streamed answer)"),
                            ("deep", cmd_deep, "ask a bigger question (splits it into several searches; slower)")):
        sp = add(name, fn, help_)
        sp.add_argument("question", nargs="+")
        sp.add_argument("--recent", action="store_true", help="favour recently edited notes")
        sp.add_argument("--no-rerank", action="store_true", help="skip the re-ranking step (faster, less precise)")
        sp.add_argument("-v", "--verbose", action="store_true", help="show what the search did")
        sp.add_argument("--timing", action="store_true", help="show the time to the first word and the total time")
        fmt = sp.add_mutually_exclusive_group()
        fmt.add_argument("--plain", action="store_true", help="print the answer as plain text (the default when piped)")
        fmt.add_argument("--markdown", action="store_true",
                         help="render the answer as formatted Markdown (the default in a terminal)")
    sp = add("mcp", cmd_mcp, "run the MCP server so other AI tools can ask your notes (answers only, never note text)")
    sp.add_argument("--print-setup", action="store_true", help="print how to add Quillgrep to Claude Code / Desktop")
    sp = add("audit", cmd_audit, "show what the MCP server was asked and what it returned")
    sp.add_argument("--tail", type=int, default=20, metavar="N", help="show the last N entries (default 20)")
    sp.add_argument("--verify", action="store_true", help="check that the log has not been edited")
    add("config", cmd_config, "show where the config, index and audit log live")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to a legacy codepage
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 0
    args.config = getattr(args, "config", None)
    args.vault = getattr(args, "vault", None)
    try:
        cfg = Config() if args.command == "init" else load_config(args.config, args.vault)
        return args.fn(args, cfg)
    except ConfigError as e:
        print(f"Config problem: {e}", file=sys.stderr)
        return 1
    except OllamaError as e:
        print(str(e), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
