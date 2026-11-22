from pdfmaker import write_pdf
from quillgrep.config import IndexRules
from quillgrep.extract import chunk_text, clean_markdown, pdf_text, scan_vault, title_of


def test_front_matter_is_stripped_and_text_kept():
    assert clean_markdown("---\ntags: [a]\n---\n# Hello\nworld") == "# Hello\nworld"


def test_base64_images_and_long_blobs_are_removed():
    blob = "A" * 400
    text = f"before ![x](data:image/png;base64,{'QUJD' * 50}) after {blob} end"
    out = clean_markdown(text)
    assert "base64" not in out and blob not in out
    assert out.startswith("before") and out.endswith("end")


def test_excalidraw_keeps_only_text_elements_without_ids():
    text = ("---\nexcalidraw-plugin: parsed\n---\n# Excalidraw Data\n\n## Text Elements\n"
            "Compost bin ^aB3dE9fG\n\nWater butt ^hJ4kL8mN\n\n%%\n## Drawing\n```compressed-json\nXXXX\n```\n%%")
    out = clean_markdown(text, "Layout.excalidraw.md")
    assert out == "Compost bin\n\nWater butt"
    assert clean_markdown(text, "Layout.excalidraw.md", excalidraw=False) == ""


def test_pdf_text_is_extracted(tmp_path):
    path = tmp_path / "doc.pdf"
    write_pdf(path, ["Heirloom tomatoes ripen late", "Second line"])
    out = pdf_text(path)
    assert "Heirloom tomatoes ripen late" in out and "Second line" in out


def test_broken_pdf_yields_empty_text(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"this is not a pdf")
    assert pdf_text(path) == ""


def paths(vault, **rules):
    return sorted(s.rel for s in scan_vault(vault, IndexRules(**rules)))


def test_scan_skips_trash_and_obsidian_by_default(vault):
    found = paths(vault)
    assert "Goals/OKRs.md" in found and "Reference/Seed Catalogue.pdf" in found
    assert not any(p.startswith((".trash", ".obsidian")) for p in found)


def test_include_and_exclude_rules(vault):
    assert paths(vault, include=["Goals"]) == ["Goals/OKRs.md"]
    assert "Daily/2026-11-18.md" not in paths(vault, exclude=[".trash", ".obsidian", "Daily"])
    assert not any(p.startswith("Daily") for p in paths(vault, exclude=["Dai*"]))
    assert not any(p.endswith(".pdf") for p in paths(vault, pdf=False))
    assert ".trash/Old idea.md" in paths(vault, exclude=[".obsidian"])  # rules are the user's to change


def test_title_of():
    assert title_of("Projects/Garden Talk.md") == "Garden Talk"
    assert title_of("Drawings/Layout.excalidraw.md") == "Layout"
    assert title_of("Reference/Seed Catalogue.pdf") == "Seed Catalogue"


def test_chunking_respects_size_and_keeps_everything():
    paragraphs = [f"Paragraph {i} " + "word " * 40 for i in range(30)]
    chunks = chunk_text("\n\n".join(paragraphs), size=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) > 5
    assert all(f"Paragraph {i} " in "\n".join(chunks) for i in range(30))


def test_a_huge_paragraph_is_split_with_overlap():
    chunks = chunk_text("x" * 2500, size=1000, overlap=100)
    assert [len(c) for c in chunks] == [1000, 1000, 700]


def test_short_text_is_one_chunk_and_empty_is_none():
    assert chunk_text("hello") == ["hello"]
    assert chunk_text("  \n ") == []


def test_folder_rules_ignore_case(vault):
    assert not any(p.startswith("Daily") for p in paths(vault, exclude=["daily"]))
    assert paths(vault, include=["goals"]) == ["Goals/OKRs.md"]
