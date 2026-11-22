import io

import pytest

from quillgrep import answer as answer_mod
from quillgrep.answer import Answer
from quillgrep.cli import main
from quillgrep.render import (
    MARKDOWN,
    PLAIN,
    MarkdownOutput,
    PlainOutput,
    choose_mode,
    make_output,
    obsidian_url,
    supports_hyperlinks,
)

MARKDOWN_REPLY = "**Bold claim** about the swap.\n\n### Next steps\n\n- bring seeds [1]\n- confirm the date [2]"


class FakeTty(io.StringIO):
    def isatty(self):
        return True


# -- which output mode ---------------------------------------------------------------------------


@pytest.mark.parametrize("plain,markdown,tty,expected", [
    (False, False, True, MARKDOWN),   # a terminal gets rendered Markdown
    (False, False, False, PLAIN),     # a pipe or file gets plain text
    (True, False, True, PLAIN),       # --plain wins in a terminal
    (False, True, False, MARKDOWN),   # --markdown wins when piped
])
def test_choose_mode(plain, markdown, tty, expected):
    assert choose_mode(plain, markdown, tty) == expected


def test_plain_and_markdown_cannot_be_combined():
    with pytest.raises(ValueError):
        choose_mode(True, True, True)


def test_make_output_follows_the_stream():
    assert isinstance(make_output(False, False, stdout=io.StringIO()), PlainOutput)
    assert isinstance(make_output(False, False, stdout=FakeTty(), stderr=io.StringIO()), MarkdownOutput)
    assert isinstance(make_output(True, False, stdout=FakeTty()), PlainOutput)
    assert isinstance(make_output(False, True, stdout=io.StringIO(), stderr=io.StringIO()), MarkdownOutput)


# -- links ---------------------------------------------------------------------------------------


def test_obsidian_url_encodes_vault_and_drops_md_extension():
    assert obsidian_url("My Vault", "Projects/Garden Talk.md") == \
        "obsidian://open?vault=My%20Vault&file=Projects%2FGarden%20Talk"
    assert obsidian_url("V", "Reference\\Catalogue.pdf").endswith("file=Reference%2FCatalogue.pdf")  # keeps non-md
    assert obsidian_url("V", "Notes/Q&A #1.md").endswith("file=Notes%2FQ%26A%20%231")


@pytest.mark.parametrize("env,tty,expected", [
    ({"TERM_PROGRAM": "iTerm.app"}, True, True),
    ({"WT_SESSION": "abc"}, True, True),
    ({"VTE_VERSION": "7200"}, True, True),
    ({"VTE_VERSION": "4000"}, True, False),
    ({"TERM": "xterm-kitty"}, True, True),
    ({"TERM_PROGRAM": "Apple_Terminal"}, True, False),
    ({"TERM": "dumb"}, True, False),
    ({"TERM_PROGRAM": "iTerm.app"}, False, False),            # never in a pipe
    ({"TERM_PROGRAM": "iTerm.app", "NO_HYPERLINKS": "1"}, True, False),
    ({"FORCE_HYPERLINK": "1"}, False, True),
])
def test_supports_hyperlinks_by_environment(env, tty, expected):
    assert supports_hyperlinks(tty, env) is expected


def test_links_setting_overrides_detection():
    assert supports_hyperlinks(True, {"TERM_PROGRAM": "iTerm.app"}, "never") is False
    assert supports_hyperlinks(False, {}, "always") is True


# -- what gets printed ---------------------------------------------------------------------------


def make_answer(cfg, indexed, text=MARKDOWN_REPLY):
    result = answer_mod.ask(cfg, answer_mod.OllamaClient(cfg.models.host, 30), indexed, "seed swap")
    return Answer(text, result.hits, "ask")


def test_markdown_output_renders_formatting_and_links_sources(cfg, indexed):
    out_stream = io.StringIO()
    out = MarkdownOutput(out_stream, io.StringIO(), links="always", force_terminal=True)
    result = make_answer(cfg, indexed)
    for piece in MARKDOWN_REPLY.split(" "):
        out.write(piece + " ")
    out.finish(result, "Garden Vault")
    shown = out_stream.getvalue()
    assert "**" not in shown and "###" not in shown       # the Markdown syntax is gone...
    assert "Bold claim" in shown and "Next steps" in shown  # ...but the words are not
    assert "\x1b[1m" in shown                               # real bold
    assert "\x1b]8;" in shown and "obsidian://open?vault=Garden%20Vault&file=" in shown
    assert "Sources" in shown and "[1]" in shown


def test_markdown_output_without_link_support_prints_plain_paths(cfg, indexed):
    out_stream = io.StringIO()
    out = MarkdownOutput(out_stream, io.StringIO(), links="never", force_terminal=True)
    out.write(MARKDOWN_REPLY)
    out.finish(make_answer(cfg, indexed), "Garden Vault")
    shown = out_stream.getvalue()
    assert "\x1b]8;" not in shown and "obsidian://" not in shown
    assert ".md" in shown  # the paths are still listed


def test_plain_output_is_the_raw_text_then_sources(cfg, indexed):
    stream = io.StringIO()
    out = PlainOutput(stream)
    out.write("**Bold** ")
    out.write("[1]")
    out.finish(make_answer(cfg, indexed), "Garden Vault")
    shown = stream.getvalue()
    assert shown.startswith("**Bold** [1]\n") and "\nSources:\n[1]" in shown and "\x1b" not in shown


# -- through the command line --------------------------------------------------------------------


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_prints_plain_text_when_piped(capsys, config_file, fake):
    run(capsys, "--config", str(config_file), "index")
    fake.answer = lambda prompt: MARKDOWN_REPLY
    code, out, _ = run(capsys, "--config", str(config_file), "ask", "seed swap")  # capsys is not a terminal
    assert code == 0 and "**Bold claim**" in out and "### Next steps" in out and "\nSources:\n[1]" in out


def test_cli_markdown_flag_renders_even_when_piped(capsys, config_file, fake):
    run(capsys, "--config", str(config_file), "index")
    fake.answer = lambda prompt: MARKDOWN_REPLY
    code, out, _ = run(capsys, "--config", str(config_file), "ask", "--markdown", "seed swap")
    assert code == 0 and "Bold claim" in out and "**" not in out and "###" not in out and "Sources" in out


def test_cli_plain_flag_and_markdown_flag_conflict(capsys, config_file):
    with pytest.raises(SystemExit):
        main(["--config", str(config_file), "ask", "--plain", "--markdown", "x"])
    assert "not allowed with" in capsys.readouterr().err
