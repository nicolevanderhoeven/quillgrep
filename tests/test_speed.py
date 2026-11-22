"""The speed features: model choice per command, the planner model, concurrent searches, keep-alive,
right-sized context windows, a tighter re-ranker and --timing. All against the fake Ollama."""

import time

from quillgrep import answer as answer_mod
from quillgrep.cli import main
from quillgrep.ollama import OllamaClient, fit_ctx, keep_alive_value, resolve_models
from quillgrep.retrieval import Retriever


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def answer_calls(fake):
    return [c for c in fake.chat_calls if c.get("format") != "json"]


def json_calls(fake):
    return [c for c in fake.chat_calls if c.get("format") == "json"]


# -- which model does what -----------------------------------------------------------------------


def test_ask_uses_the_ask_model_and_deep_uses_the_main_model(cfg, client, indexed, fake):
    answer_mod.ask(cfg, client, indexed, "seed swap")
    assert {c["model"] for c in answer_calls(fake)} == {"qwen3.5:4b"}
    fake.chat_calls.clear()
    answer_mod.deep(cfg, client, indexed, "seed swap")
    assert {c["model"] for c in answer_calls(fake)} == {"qwen3.5:9b"}


def test_planner_model_does_planning_and_reranking_only(cfg, client, indexed, fake):
    answer_mod.deep(cfg, client, indexed, "what should I focus on next")
    assert json_calls(fake) and {c["model"] for c in json_calls(fake)} == {"qwen3.5:2b"}
    assert any("Break this question" in c["messages"][0]["content"] for c in json_calls(fake))
    assert any("Rate every excerpt" in c["messages"][0]["content"] for c in json_calls(fake))


def test_deep_model_can_differ_from_chat(make_cfg, client, indexed, fake):
    fake.installed.append("big:30b")
    answer_mod.deep(make_cfg(models__deep_chat="big:30b"), client, indexed, "seed swap")
    assert {c["model"] for c in answer_calls(fake)} == {"big:30b"}


def test_missing_optional_models_fall_back_to_the_main_model(cfg, client, fake):
    fake.installed = ["nomic-embed-text:latest", "qwen3.5:9b"]
    resolved, notes = resolve_models(cfg, client)
    assert resolved.models.ask_model == resolved.models.planner_model == "qwen3.5:9b"
    assert len(notes) == 2 and "ollama pull qwen3.5:4b" in notes[0] and "ollama pull qwen3.5:2b" in notes[1]
    assert cfg.models.ask_chat == "qwen3.5:4b"  # the original config is untouched
    assert resolve_models(cfg, client)[0].models.ask_chat == ""  # and installing a model later is noticed
    fake.installed += ["qwen3.5:4b", "qwen3.5:2b"]
    assert resolve_models(cfg, client)[1] == []


def test_cli_ask_falls_back_when_the_small_models_are_not_pulled(capsys, config_file, fake):
    run(capsys, "--config", str(config_file), "index")
    fake.installed = ["nomic-embed-text:latest", "qwen3.5:9b"]
    fake.chat_calls.clear()
    code, out, err = run(capsys, "--config", str(config_file), "ask", "seed swap")
    assert code == 0 and "According to your notes" in out
    assert "ollama pull qwen3.5:4b" in err
    assert {c["model"] for c in fake.chat_calls} == {"qwen3.5:9b"}


def test_cli_ask_and_deep_pick_their_models(capsys, config_file, fake):
    run(capsys, "--config", str(config_file), "index")
    fake.chat_calls.clear()
    run(capsys, "--config", str(config_file), "ask", "seed swap")
    assert {c["model"] for c in answer_calls(fake)} == {"qwen3.5:4b"}
    fake.chat_calls.clear()
    run(capsys, "--config", str(config_file), "deep", "seed swap")
    assert {c["model"] for c in answer_calls(fake)} == {"qwen3.5:9b"}
    assert {c["model"] for c in json_calls(fake)} == {"qwen3.5:2b"}


def test_doctor_does_not_block_on_missing_optional_models(capsys, config_file, fake):
    fake.installed = ["nomic-embed-text:latest", "qwen3.5:9b"]
    code, out, _ = run(capsys, "--config", str(config_file), "doctor")
    assert code == 0 and "• Model for ask: qwen3.5:4b" in out and "ollama pull qwen3.5:2b" in out


# -- concurrency ---------------------------------------------------------------------------------


def test_deep_embeds_its_searches_concurrently(cfg, client, indexed, fake):
    fake.embed_delay = 0.2
    fake.max_inflight = 0
    t0 = time.time()
    answer_mod.deep(cfg, client, indexed, "what should I focus on next")
    elapsed = time.time() - t0
    assert fake.max_inflight >= 3  # the question plus the planned searches were in flight together
    assert elapsed < 0.2 * 3  # three sequential embeddings alone would have taken 0.6s


def test_concurrency_setting_limits_the_overlap(make_cfg, client, indexed, fake):
    fake.embed_delay = 0.1
    fake.max_inflight = 0
    answer_mod.deep(make_cfg(retrieval__concurrency=1), client, indexed, "what should I focus on next")
    assert fake.max_inflight == 1


def test_question_is_embedded_while_the_planner_thinks(cfg, client, indexed, fake):
    fake.chat_delay = 0.4
    fake.embed_calls.clear()
    t0 = time.time()
    result = answer_mod.deep(cfg, client, indexed, "seed swap")
    assert result.hits
    first_embed = fake.embed_calls[0]
    assert first_embed == ["search_query: seed swap"]  # embedded first, not after the plan
    assert time.time() - t0 > 0.4


def test_parallel_and_sequential_retrieval_agree(make_cfg, client, indexed):
    queries = ["seed swap date", "conferences cfp deadline", "okr goals"]
    one = Retriever(make_cfg(retrieval__concurrency=1), client, indexed).pool(["seed swap", *queries])
    many = Retriever(make_cfg(retrieval__concurrency=4), client, indexed).pool(["seed swap", *queries])
    assert [h.chunk.id for h in one] == [h.chunk.id for h in many]


# -- keep-alive and context size -----------------------------------------------------------------


def test_keep_alive_is_sent_with_every_request(make_cfg, indexed, fake):
    cfg = make_cfg(models__keep_alive="2h")
    client = OllamaClient(cfg.models.host, 30, cfg.models.keep_alive)
    fake.embed_bodies.clear()  # forget the indexing requests, made with the default
    answer_mod.deep(cfg, client, indexed, "seed swap")
    assert fake.chat_calls and all(c["keep_alive"] == "2h" for c in fake.chat_calls)
    assert fake.embed_bodies and all(b["keep_alive"] == "2h" for b in fake.embed_bodies)


def test_keep_alive_default_is_an_hour_and_numbers_become_seconds(cfg, client, indexed, fake):
    answer_mod.ask(cfg, client, indexed, "seed swap")
    assert fake.chat_calls[-1]["keep_alive"] == "1h"
    assert keep_alive_value("-1") == -1 and keep_alive_value("0") == 0 and keep_alive_value("30m") == "30m"


def test_context_window_fits_the_prompt():
    short = [{"role": "user", "content": "x" * 3000}]
    long = [{"role": "user", "content": "x" * 30000}]
    assert fit_ctx(short, 8192) == 4096                     # a small prompt does not get the big window
    assert fit_ctx(long, 16384) == 16384                    # a big prompt does
    assert fit_ctx(long, 8192) == 8192                      # but never more than the configured maximum
    assert fit_ctx(short, 2048) == 2048
    sizes = {fit_ctx([{"content": "x" * n}], 65536) for n in range(0, 40000, 500)}
    assert sizes <= {4096, 8192, 16384, 32768, 65536}       # coarse steps, so Ollama does not reload the model


def test_every_call_uses_a_right_sized_window(cfg, client, indexed, fake):
    answer_mod.ask(cfg, client, indexed, "seed swap")
    answer_mod.deep(cfg, client, indexed, "seed swap")
    for call in fake.chat_calls:
        prompt_tokens = len(call["messages"][0]["content"]) / 3
        assert prompt_tokens < call["options"]["num_ctx"] <= cfg.models.deep_num_ctx
    assert min(c["options"]["num_ctx"] for c in fake.chat_calls) < cfg.models.num_ctx  # smaller than the maximum


# -- the re-ranker -------------------------------------------------------------------------------


def test_ask_reads_about_six_excerpts_and_reranks_a_smaller_pool(cfg, client, indexed, fake):
    assert cfg.retrieval.excerpts == 6 and cfg.retrieval.rerank_pool < cfg.retrieval.deep_rerank_pool
    result = answer_mod.ask(cfg, client, indexed, "garden volunteers seed swap tomatoes conferences")
    assert len(result.hits) <= cfg.retrieval.excerpts
    prompt = json_calls(fake)[-1]["messages"][0]["content"]
    assert prompt.count("\n[") <= cfg.retrieval.rerank_pool + 1


def test_reranker_only_reads_the_start_of_each_excerpt(cfg, client, indexed, fake):
    from quillgrep.retrieval import RERANK_CHARS

    answer_mod.ask(cfg, client, indexed, "seed swap")
    prompt = json_calls(fake)[-1]["messages"][0]["content"]
    excerpts = prompt.split("\n\n[")[1:]
    assert excerpts and all(len(e) <= RERANK_CHARS + 80 for e in excerpts)


# -- --timing ------------------------------------------------------------------------------------


def test_answers_record_where_the_time_went(cfg, client, indexed, fake):
    pieces = []
    result = answer_mod.ask(cfg, client, indexed, "seed swap", on_text=pieces.append)
    t = result.timing
    assert 0 < t["retrieve"] <= t["first_token"] <= t["total"]
    deep = answer_mod.deep(cfg, client, indexed, "seed swap", on_text=pieces.append)
    assert 0 < deep.timing["plan"] and deep.timing["first_token"] <= deep.timing["total"]
    quiet = answer_mod.ask(cfg, client, indexed, "seed swap")  # not streamed: there is no first token to time
    assert "first_token" not in quiet.timing and quiet.timing["total"] > 0


def test_timing_flag_prints_first_token_and_total(capsys, config_file):
    run(capsys, "--config", str(config_file), "index")
    code, _, err = run(capsys, "--config", str(config_file), "ask", "--timing", "seed swap")
    assert code == 0 and err.count("timing:") == 1
    assert "first token" in err and "total" in err and "retrieve" in err
    code, out, err = run(capsys, "--config", str(config_file), "deep", "--timing", "seed swap")
    assert "plan " in err and "timing:" in err and "timing:" not in out
    code, _, err = run(capsys, "--config", str(config_file), "ask", "seed swap")
    assert "timing:" not in err  # off unless asked for


def test_reranker_asks_for_a_short_compact_reply(cfg, client, indexed, fake):
    answer_mod.ask(cfg, client, indexed, "seed swap")
    call = json_calls(fake)[-1]
    assert "single line" in call["messages"][0]["content"]
    assert 0 < call["options"]["num_predict"] < 400  # the scores and nothing else


def test_reranker_reply_may_be_pairs_or_objects():
    from quillgrep.retrieval import parse_scores

    assert parse_scores('{"scores": [[1, 7], [2, 0], [9, 5]]}', 3) == {1: 7.0, 2: 0.0}
    assert parse_scores('{"scores": [{"id": 1, "score": 4}]}', 3) == {1: 4.0}
