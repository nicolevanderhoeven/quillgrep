import json

from quillgrep.cli import find_vaults, main
from quillgrep.config import load_config


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_no_command_prints_help(capsys):
    code, out, _ = run(capsys)
    assert code == 0 and "ask" in out and "deep" in out and "mcp" in out


def test_init_writes_a_config_that_works(capsys, tmp_path, vault, fake):
    cfg_path = tmp_path / "new" / "config.toml"
    code, out, _ = run(capsys, "--config", str(cfg_path), "init", str(vault))
    assert cfg_path.exists() and "Wrote" in out
    assert load_config(cfg_path).vault_path.resolve() == vault.resolve()
    assert code in (0, 2)  # 2 only because the real Ollama may be absent on the test machine
    code, _, err = run(capsys, "--config", str(cfg_path), "init", str(vault))
    assert code == 1 and "already exists" in err


def test_init_rejects_a_missing_folder(capsys, tmp_path):
    code, _, err = run(capsys, "--config", str(tmp_path / "c.toml"), "init", str(tmp_path / "nope"))
    assert code == 1 and "does not exist" in err


def test_doctor_all_green(capsys, config_file):
    code, out, _ = run(capsys, "--config", str(config_file), "doctor")
    assert code == 0 and "✗" not in out
    assert "Ollama is running" in out and "Model for chat: qwen3.5:9b" in out


def test_doctor_tells_you_exactly_what_to_pull(capsys, config_file, fake):
    fake.installed = ["nomic-embed-text:latest"]
    code, out, _ = run(capsys, "--config", str(config_file), "doctor")
    assert code == 2 and "✗ Model for chat: qwen3.5:9b" in out and "ollama pull qwen3.5:9b" in out


def test_doctor_when_ollama_is_not_running(capsys, config_file, fake):
    fake.stop()
    code, out, _ = run(capsys, "--config", str(config_file), "doctor")
    assert code == 2 and "https://ollama.com/download" in out and "ollama serve" in out


def test_first_run_check_blocks_index_and_ask(capsys, config_file, fake):
    fake.installed = []
    for cmd in (["index"], ["ask", "hello"], ["deep", "hello"]):
        code, _, err = run(capsys, "--config", str(config_file), *cmd)
        assert code == 2 and "ollama pull nomic-embed-text" in err and "quillgrep doctor" in err


def test_index_then_ask_then_deep(capsys, config_file):
    code, _, err = run(capsys, "--config", str(config_file), "ask", "seed swap")
    assert code == 1 and "quillgrep index" in err  # nothing indexed yet

    code, out, _ = run(capsys, "--config", str(config_file), "index")
    assert code == 0 and "Done:" in out and "new" in out
    code, out, _ = run(capsys, "--config", str(config_file), "index")
    assert "0 new, 0 changed" in out  # incremental

    code, out, err = run(capsys, "--config", str(config_file), "ask", "when", "is", "the", "seed", "swap?")
    assert code == 0 and "According to your notes" in out and "\nSources:\n[1]" in out

    code, out, err = run(capsys, "--config", str(config_file), "deep", "what should I focus on next")
    assert code == 0 and "Sources:" in out
    assert "searches: seed swap date | conferences cfp deadline" in err


def test_status_json(capsys, config_file):
    run(capsys, "--config", str(config_file), "index")
    code, out, _ = run(capsys, "--config", str(config_file), "status", "--json")
    stats = json.loads(out)
    assert code == 0 and stats["notes"] >= 8 and stats["by_kind"]["pdf"] == 1


def test_vault_flag_overrides_config_and_unknown_vault_is_explained(capsys, config_file, tmp_path):
    code, _, err = run(capsys, "--config", str(config_file), "--vault", str(tmp_path / "gone"), "index")
    assert code == 1 and "does not exist" in err


def test_bad_config_is_reported_not_a_traceback(capsys, tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[models]\nnope = 1\n")
    code, _, err = run(capsys, "--config", str(bad), "doctor")
    assert code == 1 and "Unknown setting `nope`" in err


def test_mcp_print_setup_gives_both_snippets(capsys, config_file):
    code, out, _ = run(capsys, "--config", str(config_file), "mcp", "--print-setup")
    assert code == 0 and "claude mcp add quillgrep --" in out and '"mcpServers"' in out
    assert "mcp" in json.loads(out[out.index("{"):])["mcpServers"]["quillgrep"]["args"]


def test_audit_command_lists_and_verifies(capsys, config_file, cfg, indexed):
    from quillgrep.mcp_server import McpServer

    McpServer(cfg).call_tool("ask_notes", {"question": "what are my goals"})
    # the audit log lives under QUILLGREP_HOME for both, so the CLI sees what the server wrote
    code, out, _ = run(capsys, "--config", str(config_file), "audit")
    assert code == 0 and "REQUEST" in out and "what are my goals" in out and "RETURNED" in out
    code, out, _ = run(capsys, "--config", str(config_file), "audit", "--verify")
    assert code == 0 and "chain intact" in out


def test_find_vaults_locates_obsidian_folders(tmp_path):
    (tmp_path / "Documents" / "Notes" / ".obsidian").mkdir(parents=True)
    (tmp_path / "Documents" / "Other").mkdir()
    (tmp_path / "Library" / "Hidden" / ".obsidian").mkdir(parents=True)
    assert find_vaults(tmp_path) == [tmp_path / "Documents" / "Notes"]


def test_example_config_in_the_repo_is_valid():
    from pathlib import Path

    example = Path(__file__).parent.parent / "quillgrep.example.toml"
    assert load_config(example).models.chat == "qwen3.5:9b"
