"""Showing answers in a terminal: rendered Markdown that streams in, or plain text; clickable note links."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from urllib.parse import quote

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

from .answer import Answer, format_sources, source_entries

MARKDOWN, PLAIN = "markdown", "plain"

# Terminals known to open OSC 8 hyperlinks (matched against $TERM_PROGRAM / $TERM, lower-cased).
_LINK_TERM_PROGRAMS = {"iterm.app", "wezterm", "vscode", "ghostty", "hyper", "rio", "tabby", "alacritty"}
_LINK_TERMS = {"xterm-kitty", "xterm-ghostty", "alacritty", "foot", "foot-extra"}


def choose_mode(plain: bool, markdown: bool, is_tty: bool) -> str:
    """Rendered Markdown for a terminal, plain text for a pipe or file; `--plain` / `--markdown` override."""
    if plain and markdown:
        raise ValueError("--plain and --markdown cannot be combined")
    if plain:
        return PLAIN
    if markdown:
        return MARKDOWN
    return MARKDOWN if is_tty else PLAIN


def supports_hyperlinks(is_tty: bool, env: Mapping[str, str] | None = None, setting: str = "auto") -> bool:
    """Will this terminal turn OSC 8 links into clickable text? `setting` is the config's `links` value."""
    if setting == "never":
        return False
    if setting == "always":
        return True
    env = os.environ if env is None else env
    if env.get("FORCE_HYPERLINK") not in (None, "", "0"):
        return True
    if not is_tty or env.get("NO_HYPERLINKS") or env.get("TERM", "").lower() == "dumb":
        return False
    if env.get("TERM_PROGRAM", "").lower() in _LINK_TERM_PROGRAMS or env.get("TERM", "").lower() in _LINK_TERMS:
        return True
    if any(env.get(k) for k in ("WT_SESSION", "KITTY_WINDOW_ID", "WEZTERM_EXECUTABLE", "KONSOLE_VERSION")):
        return True  # Windows Terminal, kitty, WezTerm, Konsole
    vte = env.get("VTE_VERSION", "")
    return vte.isdigit() and int(vte) >= 5000  # GNOME Terminal and friends (VTE 0.50+)


def obsidian_url(vault_name: str, path: str) -> str:
    """`obsidian://open?vault=...&file=...` for a note path relative to the vault (the .md extension is dropped)."""
    file = path.replace("\\", "/")
    if file.lower().endswith(".md"):
        file = file[:-3]
    return f"obsidian://open?vault={quote(vault_name, safe='')}&file={quote(file, safe='')}"


class _Live:
    """What the live display shows: the Markdown of the text so far, parsed only when a frame is drawn."""

    def __init__(self):
        self.text = ""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Never draw more rows than fit on screen: Live cannot erase rows that have scrolled off, and
        # Live.stop() forces a final "visible" refresh, so the frame itself must stay short enough to clear.
        lines = console.render_lines(Markdown(self.text), options.update(height=None), pad=False)
        room = max(1, (options.max_height or console.height) - 1)
        for i, line in enumerate(lines[-room:]):
            if i:
                yield Segment.line()
            yield from line


class PlainOutput:
    """Raw text straight to stdout as it arrives, then a plain Sources list. What pipes and files get."""

    mode = PLAIN

    def __init__(self, stdout=None):
        self.out = stdout or sys.stdout

    def start(self, waiting: str = "") -> None:
        pass

    def write(self, piece: str) -> None:
        self.out.write(piece)
        self.out.flush()

    def abort(self) -> None:
        pass

    def finish(self, answer: Answer, vault_name: str = "") -> None:
        print(file=self.out)
        sources = format_sources(answer)
        if sources:
            print("\nSources:\n" + sources, file=self.out)


class MarkdownOutput:
    """Rendered Markdown (real bold, headings, lists), redrawn as the answer streams in."""

    mode = MARKDOWN

    def __init__(self, stdout=None, stderr=None, links: str = "auto", force_terminal: bool | None = None):
        stdout = stdout or sys.stdout
        self.console = Console(file=stdout, highlight=False, force_terminal=force_terminal)
        self.status_console = Console(file=stderr or sys.stderr, highlight=False)
        self.links = supports_hyperlinks(self.console.is_terminal, setting=links)
        self._body = _Live()
        self._live: Live | None = None
        self._status = None

    def start(self, waiting: str = "Searching your notes...") -> None:
        if self.status_console.is_terminal:
            self._status = self.status_console.status(waiting)
            self._status.start()

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def write(self, piece: str) -> None:
        if self._live is None:
            self._stop_status()
            self._live = Live(self._body, console=self.console, refresh_per_second=10,
                              vertical_overflow="crop", transient=True)
            self._live.start()
        self._body.text += piece
        self._live.update(self._body)

    def abort(self) -> None:
        self._stop_status()
        self._stop_live()

    def finish(self, answer: Answer, vault_name: str = "") -> None:
        self._stop_status()
        self._stop_live()
        self._body.text = answer.text or self._body.text  # the cleaned-up final text replaces the streamed pieces
        self.console.print(Markdown(self._body.text))
        entries = source_entries(answer)
        if entries:
            self.console.print()
            self.console.print(Text("Sources", style="bold"))
            for nums, path in entries:
                self.console.print(self.source_line(nums, path, vault_name))

    def source_line(self, nums: list[int], path: str, vault_name: str) -> Text:
        line = Text("  " + "".join(f"[{n}]" for n in nums) + " ", style="dim", no_wrap=True)
        style = Style(link=obsidian_url(vault_name, path)) if self.links and vault_name else Style()
        line.append(path, style=style)
        return line


def make_output(plain: bool, markdown: bool, links: str = "auto", stdout=None, stderr=None):
    stdout = stdout or sys.stdout
    mode = choose_mode(plain, markdown, stdout.isatty())
    return MarkdownOutput(stdout, stderr, links) if mode == MARKDOWN else PlainOutput(stdout)
