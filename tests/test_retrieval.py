import json
import time

import numpy as np

from quillgrep.retrieval import Retriever, cap_per_note, parse_scores, recency_factor


def titles(hits):
    return [h.chunk.title for h in hits]


def test_keyword_search_finds_exact_terms(cfg, client, indexed):
    r = Retriever(cfg, client, indexed)
    hits = r.keyword("Waltham Butternut", 5)
    assert hits and indexed.chunks_by_ids([hits[0][0]])[hits[0][0]].title == "Seed Catalogue"


def test_semantic_search_ranks_related_notes_first(cfg, client, indexed):
    r = Retriever(cfg, client, indexed)
    top = r.semantic("when is the seed swap date", 3)
    assert "Seed Swap Planning" in titles_of(indexed, top)


def titles_of(store, scored):
    return [c.title for c in store.chunks_by_ids([i for i, _ in scored]).values()]


def test_hybrid_search_fuses_both_signals(cfg, client, indexed):
    hits = Retriever(cfg, client, indexed).search("seed swap library room")
    assert hits[0].chunk.title in {"Seed Swap Planning", "Seed Catalogue", "2026-11-20"}
    assert any({"semantic", "keyword"} <= h.via for h in hits)


def test_per_note_cap_limits_excerpts_from_one_note(cfg, client, indexed):
    hits = Retriever(cfg, client, indexed).search("seed swap")
    capped = cap_per_note(hits * 3, 1, 50)
    assert len({h.chunk.note_id for h in capped}) == len(capped)
    assert len(cap_per_note(hits, 2, 3)) == min(3, len(hits))


def test_title_keywords_pull_in_notes_by_name(make_cfg, client, indexed):
    cfg = make_cfg(retrieval__candidates=2)  # too few for the OKR note to turn up on its own
    r = Retriever(cfg, client, indexed)
    pooled = r.pool(["tomato soup recipe"], keywords=["okrs"])
    assert next(h for h in pooled if h.chunk.title == "OKRs").via == {"title"}
    assert indexed.title_search(["okrs"], 5) and indexed.title_search(["ab"], 5) == []  # too short to be useful


def test_recency_factor_halves_the_boost_every_half_life():
    now = time.time()
    assert recency_factor(now, now, 0.5, 100) == 1.5
    assert abs(recency_factor(now - 100 * 86400, now, 0.5, 100) - 1.25) < 1e-9
    assert recency_factor(now - 10_000 * 86400, now, 0.5, 100) < 1.001


def test_recent_flag_moves_fresh_notes_up(cfg, client, indexed):
    old = time.time() - 3 * 365 * 86400
    indexed.db.execute("UPDATE notes SET mtime = ?", (old,))
    indexed.db.execute("UPDATE notes SET mtime = ? WHERE path = 'Meetings/Seed Swap Planning.md'", (time.time(),))
    indexed.db.commit()
    r = Retriever(cfg, client, indexed)
    plain = r.search("volunteers garden", boost=0.0)
    boosted = r.search("volunteers garden", boost=5.0)
    assert boosted[0].chunk.title == "Seed Swap Planning"
    assert plain[0].score != boosted[0].score


def test_anchor_notes_are_always_included(make_cfg, client, indexed, fake):
    cfg = make_cfg(retrieval__anchor_notes=["Goals/OKRs.md", "conferences"], retrieval__rerank=False)
    hits = Retriever(cfg, client, indexed).retrieve("tomato soup recipe", keep=2)
    found = titles(hits)
    assert "OKRs" in found and "Conferences" in found
    assert {"anchor"} == {next(iter(h.via)) for h in hits if h.chunk.title in ("OKRs", "Conferences")}


def test_anchor_folder_reference_and_missing_anchor(make_cfg, client, indexed):
    cfg = make_cfg(retrieval__anchor_notes=["Goals", "No Such Note"], retrieval__rerank=False)
    assert "OKRs" in titles(Retriever(cfg, client, indexed).retrieve("tomato soup", keep=1))


def test_excluded_folders_never_surface(cfg, client, indexed):
    r = Retriever(cfg, client, indexed, excluded_folders=["Daily", "Reference"])
    for query in ("Priya phone", "heirloom tomatoes Black Krim", "volunteers"):
        hits = r.search(query)
        assert hits and not any(h.chunk.path.startswith(("Daily", "Reference")) for h in hits)
    assert not r.anchors("x", set())


def test_excluded_folder_anchor_is_skipped(make_cfg, client, indexed):
    cfg = make_cfg(retrieval__anchor_notes=["Daily/2026-11-18.md"])
    r = Retriever(cfg, client, indexed, excluded_folders=["Daily"])
    assert r.anchors("steps", set()) == []


def test_rerank_puts_the_best_excerpt_first_and_keeps_fewer(cfg, client, indexed, fake):
    r = Retriever(cfg, client, indexed)
    trace = []
    hits = r.retrieve("which heirloom tomatoes Black Krim", keep=3, trace=trace)
    assert len(hits) <= 3 and hits[0].chunk.title == "Seed Catalogue"
    assert any("re-ranked" in t for t in trace)
    assert any(c["format"] == "json" for c in fake.chat_calls)


def test_rerank_failure_falls_back_to_search_order(cfg, client, indexed, fake):
    fake.rerank_reply = lambda prompt: "this is not json"
    trace = []
    hits = Retriever(cfg, client, indexed).retrieve("seed swap", keep=4, trace=trace)
    assert len(hits) == 4 and any("skipped" in t for t in trace)


def test_rerank_can_be_turned_off(cfg, client, indexed, fake):
    Retriever(cfg, client, indexed).retrieve("seed swap", keep=4, rerank=False)
    assert fake.chat_calls == []


def test_parse_scores_is_tolerant():
    reply = 'Sure! {"scores": [{"id": 1, "score": 7}, {"id": "2", "score": "3"}, {"id": 9, "score": 1}]}'
    assert parse_scores(reply, 3) == {1: 7.0, 2: 3.0}
    assert parse_scores(json.dumps([{"id": 1, "score": 4}]), 2) == {1: 4.0}
    for bad in ("nothing", '{"scores": []}'):
        try:
            parse_scores(bad, 2)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_empty_index_returns_nothing(cfg, client, store):
    assert Retriever(cfg, client, store).search("anything") == []
    assert np.asarray(store.matrix()[1]).size == 0
