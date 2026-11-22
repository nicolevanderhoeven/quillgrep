# Contributing to Quillgrep

Thanks for helping. Quillgrep is a private, local AI for Obsidian vaults, so a few rules matter more than usual.

## The one hard rule

**Never commit real notes, vault paths, exported chats or secrets.** This repository is public.

- Tests and docs use the synthetic vault in `examples/sample-vault/` and notes built inside the tests. Add to
  those; do not copy from your own vault, not even "just a few lines".
- `.gitignore` is an allowlist: everything is ignored unless it is listed on purpose. If you add a new kind of
  file to the project, add it to the allowlist in the same change. Check `git status` before you commit.
- Never put API keys, tokens, `.env` files, your config file or your index or audit log in a commit.
- Output you paste into an issue (for example from `quillgrep doctor`) should not contain note text.

## Set up

You need Python 3.11 or newer. You do **not** need Ollama or any model to work on Quillgrep.

```bash
git clone https://github.com/nicolevanderhoeven/quillgrep && cd quillgrep
python3 -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Check your work

```bash
pytest          # the whole suite, with a fake Ollama server (tests/fake_ollama.py)
ruff check .    # lint
```

CI runs both on macOS, Linux and Windows (Python 3.11 and 3.13) for every pull request, so avoid
platform-specific assumptions: use `pathlib`, store note paths with forward slashes, open files with an explicit
encoding, and keep stdout free of anything but the intended output in the MCP server (stdout is the protocol).

## How the tests work

`tests/fake_ollama.py` is a small HTTP server that speaks the parts of the Ollama API Quillgrep uses. Embeddings
are hashed bags of words, and the pretend chat model plans searches, scores excerpts and writes a short cited
answer. Real client code runs against it, so the tests exercise the actual HTTP paths. The `isolated_home`
fixture points all config, index and audit files at a temporary folder, so running tests never touches your own.

If you change a prompt, retrieval step or privacy rule, add or update a test. For anything that changes answer
quality, also try it against a real model on the sample vault (`quillgrep init examples/sample-vault`) and
describe what you saw in the pull request.

## Pull requests

- Keep changes focused, and explain the *why* in the description.
- New settings need a default, validation (see `config.py`), an entry in `quillgrep.example.toml`/`starter_config`
  if beginners need it, and a line in the README configuration section.
- Anything that changes what the MCP server can return needs extra care: it is the privacy boundary. Describe
  the effect on the data that leaves the machine, and add a test that proves the limit holds.
- By contributing you agree that your contribution is licensed under the Apache License 2.0.

## Reporting security problems

If you find a way for note content, paths or secrets to leak (for example past the MCP privacy controls),
please do not open a public issue. Use GitHub's private vulnerability reporting on this repository
(Security tab → *Report a vulnerability*).
