"""Build a tiny, valid, text-only PDF without any dependency (used for fixtures and the sample vault)."""

from __future__ import annotations

from pathlib import Path


def make_pdf(lines: list[str]) -> bytes:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = "BT /F1 12 Tf 14 TL 72 720 Td " + " T* ".join(f"({esc(line)}) Tj" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def write_pdf(path: Path, lines: list[str]) -> None:
    path.write_bytes(make_pdf(lines))


if __name__ == "__main__":  # regenerate the sample vault's PDF:  python tests/pdfmaker.py <output.pdf>
    import sys

    write_pdf(
        Path(sys.argv[1]),
        [
            "Seed Catalogue 2027 - Elm Street Community Garden",
            "Heirloom tomatoes: Black Krim, Green Zebra, Amish Paste. Sow indoors in February.",
            "Beans: Scarlet Runner and Cherokee Trail of Tears. Sow outdoors after the last frost.",
            "Squash: Waltham Butternut. Needs a large bed and plenty of compost.",
            "Seed swap rule: bring at least one packet to take one packet.",
        ],
    )
