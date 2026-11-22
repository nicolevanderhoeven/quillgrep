from quillgrep import answer as answer_mod
from quillgrep.answer import format_sources


def test_ask_streams_pieces_and_cites_notes(cfg, client, indexed):
    pieces = []
    result = answer_mod.ask(cfg, client, indexed, "when is the seed swap planned", on_text=pieces.append)
    assert len(pieces) > 3  # streamed word by word, not one blob
    assert "".join(pieces).strip() == result.text
    assert "[1]" in result.text and result.hits
    sources = format_sources(result)
    assert sources.startswith("[1]") and ".md" in sources


def test_ask_prompt_contains_numbered_excerpts_paths_and_question(cfg, client, indexed, fake):
    answer_mod.ask(cfg, client, indexed, "what are my goals")
    prompt = fake.chat_calls[-1]["messages"][0]["content"]
    assert "Question: what are my goals" in prompt
    assert "[1] (" in prompt and ".md)" in prompt
    assert "say so plainly" in prompt  # told not to guess
    assert fake.chat_calls[-1]["options"]["num_ctx"] == cfg.models.num_ctx


def test_ask_reads_fewer_excerpts_than_it_retrieved(cfg, client, indexed, fake):
    result = answer_mod.ask(cfg, client, indexed, "seed swap")
    assert 1 <= len(result.hits) <= cfg.retrieval.excerpts


def test_per_note_cap_holds_in_answers(make_cfg, client, indexed):
    cfg = make_cfg(retrieval__per_note=1, retrieval__rerank=False)
    result = answer_mod.ask(cfg, client, indexed, "garden volunteers seed swap tomatoes")
    paths = [h.chunk.path for h in result.hits]
    assert len(paths) == len(set(paths))


def test_deep_plans_searches_pools_results_and_uses_bigger_context(cfg, client, indexed, fake):
    result = answer_mod.deep(cfg, client, indexed, "what should I focus on next")
    assert result.queries == fake.plan["queries"] and result.keywords == fake.plan["keywords"]
    assert any("Break this question" in c["messages"][0]["content"] for c in fake.chat_calls)
    final = fake.chat_calls[-1]
    assert final["options"]["num_ctx"] == cfg.models.deep_num_ctx and final["stream"] is False
    titles = {h.chunk.title for h in result.hits}
    assert {"Seed Swap Planning", "Conferences", "OKRs"} <= titles  # pooled from several searches + title keywords
    assert "Combine information across them" in final["messages"][0]["content"]


def test_deep_survives_a_bad_plan(cfg, client, indexed, fake):
    fake.plan = {"queries": "oops", "keywords": None}
    result = answer_mod.deep(cfg, client, indexed, "seed swap")
    assert result.queries == [] and result.hits and result.text


def test_planner_model_is_used_for_planning_and_reranking(make_cfg, client, indexed, fake):
    fake.installed.append("tiny:1b")
    cfg = make_cfg(models__planner="tiny:1b")
    answer_mod.deep(cfg, client, indexed, "seed swap")
    json_models = {c["model"] for c in fake.chat_calls if c.get("format") == "json"}
    answer_models = {c["model"] for c in fake.chat_calls if c.get("format") is None}
    assert json_models == {"tiny:1b"} and answer_models == {"qwen3.5:9b"}


def test_empty_index_gives_a_clear_message(cfg, client, store):
    result = answer_mod.ask(cfg, client, store, "anything")
    assert "quillgrep index" in result.text and result.hits == []
