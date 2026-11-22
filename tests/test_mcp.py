import json
import os
import subprocess
import sys

import pytest

from quillgrep.audit import AuditLog, read_entries, verify
from quillgrep.mcp_server import McpServer


def rpc(server, method, params=None, mid=1):
    msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if mid is not None:
        msg["id"] = mid
    return server.handle(msg)


def call(server, name, **arguments):
    reply = rpc(server, "tools/call", {"name": name, "arguments": arguments})
    result = reply["result"]
    return result["content"][0]["text"], result["isError"]


@pytest.fixture
def server(cfg, indexed):
    return McpServer(cfg)


def test_handshake_and_tool_list(server):
    init = rpc(server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "quillgrep" and "tools" in init["result"]["capabilities"]
    assert rpc(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == "2025-06-18"
    assert rpc(server, "notifications/initialized", mid=None) is None
    assert rpc(server, "ping")["result"] == {}
    names = [t["name"] for t in rpc(server, "tools/list")["result"]["tools"]]
    assert names == ["ask_notes", "notes_status"]
    assert rpc(server, "nope")["error"]["code"] == -32601
    assert rpc(server, "tools/call", {"name": "delete_everything"})["error"]["code"] == -32602


def test_answer_is_a_synthesis_with_titles_and_never_raw_text_or_paths(server, indexed):
    text, is_error = call(server, "ask_notes", question="seed swap attendees venue announcement action")
    assert not is_error and "Sources (note titles):" in text
    assert "Seed Swap Planning" in text
    assert "/" not in text.split("Sources")[1]  # titles only, no folder paths
    assert ".md" not in text
    # nothing longer than a short phrase may be copied from the notes
    chunks = [r[0] for r in indexed.db.execute("SELECT text FROM chunks")]
    words = text.split()
    for i in range(len(words) - 21):
        window = " ".join(words[i : i + 21])
        assert not any(window in c for c in chunks)


def test_model_never_sees_paths_in_synthesis_mode(server, fake):
    call(server, "ask_notes", question="seed swap")
    prompt = fake.chat_calls[-1]["messages"][0]["content"]
    assert ".md" not in prompt and "[1]\n" in prompt and "never by file name" in prompt


def test_verbatim_quotes_and_pii_are_removed_from_the_synthesis(server, fake):
    def leaky(prompt):
        excerpt = prompt.split("[1]\n", 1)[1].split("\n\n[2]", 1)[0]
        return "Priya's email is priya.nair@example.com and her phone +1 555 010 0199. " + excerpt + " [1]"

    fake.answer = leaky
    text, _ = call(server, "ask_notes", question="who is Priya, the seed swap contact?")
    assert "priya.nair@example.com" not in text and "555 010 0199" not in text
    assert "[email]" in text and "[phone]" in text and "[…]" in text


def test_excluded_folders_are_invisible_to_the_mcp_server(make_cfg, indexed):
    cfg = make_cfg(privacy__excluded_folders=["Daily", "Reference", "Meetings"])
    server = McpServer(cfg)
    text, _ = call(server, "ask_notes", question="seed swap Priya heirloom tomatoes Black Krim", depth="deep")
    for hidden in ("2026-11-18", "2026-11-20", "Seed Catalogue", "Seed Swap Planning"):
        assert hidden not in text


def test_excluded_folder_names_match_regardless_of_case(make_cfg, indexed):
    server = McpServer(make_cfg(privacy__excluded_folders=["daily", "REFERENCE", "meetings"]))
    text, _ = call(server, "ask_notes", question="seed swap Priya heirloom tomatoes Black Krim", depth="deep")
    for hidden in ("2026-11-18", "2026-11-20", "Seed Catalogue", "Seed Swap Planning"):
        assert hidden not in text


def test_no_sources_are_listed_when_the_answer_cites_none(server, fake):
    fake.answer = lambda prompt: "Your notes do not say."
    text, is_error = call(server, "ask_notes", question="what are my goals")
    assert not is_error and "Sources" not in text
    assert read_entries(server.cfg.audit_path)[-1]["citations"] == []


def test_every_failure_is_audited_and_reported_as_an_error(server, monkeypatch):
    def boom(*a, **k):
        raise ValueError("malformed reply")

    monkeypatch.setattr("quillgrep.mcp_server.answer_mod.ask", boom)
    text, is_error = call(server, "ask_notes", question="what are my goals")
    assert is_error and text
    monkeypatch.setattr(McpServer, "store", boom)
    assert call(server, "ask_notes", question="what are my goals")[1]
    assert call(server, "notes_status")[1]
    entries = read_entries(server.cfg.audit_path)
    assert [e["event"] for e in entries] == ["request", "response"] * 3
    assert all(e["error"] for e in entries[1::2])


def test_audit_chain_survives_two_writers_alternating(tmp_path):
    path = tmp_path / "audit.log"
    a, b = AuditLog(path), AuditLog(path)
    for i in range(3):
        a.record("request", n=i)
        b.record("response", n=i)
    assert verify(path) == (True, "6 entries, chain intact.")


def test_titles_can_be_hidden_entirely(make_cfg, indexed):
    server = McpServer(make_cfg(privacy__show_titles=False))
    text, _ = call(server, "ask_notes", question="seed swap")
    assert "Sources" not in text and "[1]" not in text and text.strip()
    entry = read_entries(server.cfg.audit_path)[-1]
    assert entry["citations"] == []


def test_titles_are_redacted_too(make_cfg, indexed, vault):
    server = McpServer(make_cfg())
    indexed.db.execute(
        "UPDATE notes SET title = 'Call priya@example.com' WHERE path = 'Meetings/Seed Swap Planning.md'")
    indexed.db.commit()
    text, _ = call(server, "ask_notes", question="seed swap planning meeting attendees action")
    assert "priya@example.com" not in text


def test_every_request_and_response_is_audited(server, cfg):
    call(server, "ask_notes", question="what are my goals", depth="fast")
    call(server, "notes_status")
    call(server, "ask_notes", question="")
    entries = read_entries(cfg.audit_path)
    assert [e["event"] for e in entries] == ["request", "response"] * 3
    req, resp = entries[0], entries[1]
    assert req["arguments"]["question"] == "what are my goals" and resp["request"] == req["id"]
    assert resp["returned"].startswith(("According", "The")) or resp["returned"]
    assert resp["citations"] and resp["error"] is None and resp["excerpts_used"] >= 1
    assert entries[5]["error"] == "missing question"
    assert verify(cfg.audit_path) == (True, "6 entries, chain intact.")


def test_the_audit_log_detects_tampering_and_is_append_only(tmp_path):
    path = tmp_path / "audit.log"
    log = AuditLog(path)
    for i in range(3):
        log.record("request", n=i)
    assert verify(path)[0]
    AuditLog(path).record("request", n=3)  # a second writer continues the same chain
    assert verify(path) == (True, "4 entries, chain intact.")
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[1].replace('"n":1', '"n":9'), *lines[2:]]) + "\n")
    ok, msg = verify(path)
    assert not ok and "line 3" in msg
    if os.name == "posix":
        assert oct(path.stat().st_mode & 0o777) in ("0o600", "0o644")  # created owner-only; rewrite kept the mode


def test_bad_arguments_and_unready_ollama_are_refused_politely(server, fake):
    assert call(server, "ask_notes", question="x" * 5000)[1]
    assert call(server, "ask_notes", question="ok", depth="sideways")[1]
    fake.installed = []
    text, is_error = call(server, "ask_notes", question="what are my goals")
    assert is_error and "not ready" in text and "ollama pull" not in text  # no setup chatter sent to a cloud model
    entry = read_entries(server.cfg.audit_path)[-1]
    assert "ollama pull" in entry["error"]  # but the user can find it in the log


def test_status_reports_counts_only(server):
    text, is_error = call(server, "notes_status")
    assert not is_error and "notes indexed" in text and "Goals" not in text


def test_empty_index_is_reported(make_cfg, client):
    text, is_error = call(McpServer(make_cfg()), "ask_notes", question="anything")
    assert is_error and "quillgrep index" in text


def test_stdio_end_to_end(config_file, cfg, indexed, tmp_path):
    """Run the real server as a subprocess, the way Claude Desktop or Claude Code would."""
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "ask_notes", "arguments": {"question": "what are my goals this quarter?"}}},
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "quillgrep", "--config", str(config_file), "mcp"],
        input="\n".join(json.dumps(m) for m in messages) + "\n", capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    replies = [json.loads(line) for line in proc.stdout.splitlines()]  # stdout must be pure JSON-RPC
    assert [r["id"] for r in replies] == [1, 2, 3]
    assert replies[1]["result"]["tools"][0]["name"] == "ask_notes"
    assert "Sources (note titles)" in replies[2]["result"]["content"][0]["text"]
    assert "ready" in proc.stderr
