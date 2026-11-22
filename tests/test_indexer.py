import os

from quillgrep.indexer import index_vault
from quillgrep.store import Store


def paths_in(store):
    return {r["path"] for r in store.db.execute("SELECT path FROM notes")}


def all_text(store):
    return "\n".join(r[0] for r in store.db.execute("SELECT text FROM chunks"))


def test_indexes_notes_pdfs_and_excalidraw_but_not_trash_or_obsidian(cfg, client, store):
    report = index_vault(cfg, client, store)
    found = paths_in(store)
    assert "Goals/OKRs.md" in found
    assert "Reference/Seed Catalogue.pdf" in found
    assert "Drawings/Garden Layout.excalidraw.md" in found
    assert not any(p.startswith((".trash", ".obsidian")) for p in found)
    assert report.added == len(found) and report.updated == report.removed == 0


def test_drawing_data_is_stripped_but_drawing_text_is_kept(indexed):
    text = all_text(indexed)
    assert "Compost bin (3 bays)" in text and "Water butt + drip line" in text
    assert "compressed-json" not in text and "filler" not in text and "^aB3dE9fG" not in text
    assert "Black Krim" in text  # from the PDF


def test_second_run_embeds_nothing(cfg, client, store, fake):
    index_vault(cfg, client, store)
    calls = len(fake.embed_calls)
    report = index_vault(cfg, client, store)
    assert len(fake.embed_calls) == calls
    assert report.unchanged == report.seen and report.added == 0


def test_changed_file_is_reindexed_and_only_that_file(cfg, client, store, fake, vault):
    index_vault(cfg, client, store)
    calls = len(fake.embed_calls)
    note = vault / "Recipes" / "Tomato Soup.md"
    note.write_text("# Tomato soup\n\nNow with a spoonful of smoked paprika for extra warmth.\n", encoding="utf-8")
    report = index_vault(cfg, client, store)
    assert (report.updated, report.added, report.removed) == (1, 0, 0)
    assert len(fake.embed_calls) == calls + 1
    assert "smoked paprika" in all_text(store) and "finish\nwith a spoon of cream" not in all_text(store)


def test_touching_a_file_without_changing_it_costs_nothing(cfg, client, store, fake, vault):
    index_vault(cfg, client, store)
    calls = len(fake.embed_calls)
    note = vault / "Recipes" / "Tomato Soup.md"
    stat = note.stat()
    os.utime(note, (stat.st_atime, stat.st_mtime + 100))
    report = index_vault(cfg, client, store)
    assert len(fake.embed_calls) == calls and report.updated == 0


def test_deleted_files_leave_the_index_including_search(cfg, client, store, vault):
    index_vault(cfg, client, store)
    (vault / "Recipes" / "Tomato Soup.md").unlink()
    report = index_vault(cfg, client, store)
    assert report.removed == 1 and "Recipes/Tomato Soup.md" not in paths_in(store)
    hits = store.keyword_search("garlic tomatoes roast", 10)
    assert all(store.chunks_by_ids([i])[i].title != "Tomato Soup" for i, _ in hits)


def test_exclude_rules_from_config(make_cfg, client):
    cfg = make_cfg(index__exclude=[".trash", ".obsidian", "Daily", "Reference"])
    store = Store(cfg.index_path)
    index_vault(cfg, client, store)
    found = paths_in(store)
    assert not any(p.startswith(("Daily", "Reference")) for p in found) and "Goals/OKRs.md" in found


def test_changing_the_embedding_model_rebuilds(make_cfg, client, fake):
    fake.installed.append("other-embed:latest")
    cfg = make_cfg()
    store = Store(cfg.index_path)
    index_vault(cfg, client, store)
    n = store.stats()["notes"]
    cfg.models.embedding = "other-embed"
    messages = []
    report = index_vault(cfg, client, store, progress=messages.append)
    assert report.added == n and any("rebuilding" in m for m in messages)
    assert store.get_meta("embedding_model") == "other-embed"


def test_concurrent_reader_sees_a_consistent_index_while_indexing(cfg, client, store):
    index_vault(cfg, client, store)
    reader = Store(cfg.index_path)
    assert reader.stats()["notes"] == store.stats()["notes"]
