"""
Docling extraction.

`blocks_from_docling` is exercised against a stand-in for a DoclingDocument,
so the mapping from Docling's item labels to EduPilot's blocks is checked in
CI without installing the docling extra or downloading its layout models.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from edupilot.ingestion import build_sections, extract_document
from edupilot.ingestion.docling_parser import blocks_from_docling
from edupilot.ingestion.models import BlockKind, ExtractionStats


def item(label: str, text: str = "", page: int = 1, **extra):
    bbox = SimpleNamespace(l=10.0, t=20.0, r=300.0, b=40.0)
    return SimpleNamespace(
        label=SimpleNamespace(value=label),
        text=text,
        prov=[SimpleNamespace(page_no=page, bbox=bbox)],
        **extra,
    )


class FakeDocument:
    def __init__(self, items) -> None:
        self.items = items

    def iterate_items(self):
        return [(i, 0) for i in self.items]


def table(markdown: str, page: int = 2):
    t = item("table", page=page)
    t.export_to_markdown = lambda doc: markdown
    return t


DOC = FakeDocument(
    [
        item("page_header", "STAT 501 — Fall"),
        item("title", "Lecture 4: Hypothesis Testing"),
        item("section_header", "The null hypothesis", level=1),
        item("text", "The p-value is the probability of a result at least as extreme."),
        item("list_item", "Reject H0 when p < alpha."),
        item("section_header", "Power", level=2, page=2),
        item("formula", "1 - \\beta", page=2),
        table("| Decision | H0 true |\n|---|---|\n| Reject | Type I |"),
        item("picture", ""),
        item("caption", "Figure 1: rejection regions", page=2),
        item("page_footer", "page 2", page=2),
        item("text", "   "),
    ]
)


def test_items_become_typed_blocks_in_reading_order():
    stats = ExtractionStats()
    blocks = blocks_from_docling(DOC, stats)

    assert [(b.kind, b.level) for b in blocks] == [
        (BlockKind.HEADING, 1),
        (BlockKind.HEADING, 2),
        (BlockKind.PARAGRAPH, None),
        (BlockKind.LIST_ITEM, None),
        (BlockKind.HEADING, 3),
        (BlockKind.FORMULA, None),
        (BlockKind.TABLE, None),
        (BlockKind.CAPTION, None),
    ]
    assert blocks[6].text.startswith("| Decision |"), "tables arrive as markdown"
    assert blocks[4].page == 2 and blocks[0].bbox == (10.0, 20.0, 300.0, 40.0)
    assert (stats.table_count, stats.formula_count, stats.heading_count) == (1, 1, 3)
    assert stats.stripped_running_heads == 2, "page headers and footers are dropped"


def test_docling_headings_nest_into_the_same_breadcrumbs():
    sections = build_sections(blocks_from_docling(DOC))
    assert "Lecture 4: Hypothesis Testing > The null hypothesis > Power" in [
        s.breadcrumb for s in sections
    ]


def test_an_unknown_parser_is_refused(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nVariance measures spread.")
    with pytest.raises(ValueError, match="Unknown PDF parser"):
        extract_document(path, parser="ocr-magic")
    assert extract_document(path, parser="docling").blocks, "non-PDFs ignore the parser"


def test_the_rebuild_falls_back_to_the_fast_parser_for_one_bad_pdf(monkeypatch, tmp_path):
    from edupilot.cli import reindex
    from edupilot.ingestion import pipeline

    calls = []

    def fake_extract(path, *, parser="pymupdf", **kwargs):
        calls.append(parser)
        if parser == "docling":
            raise RuntimeError("Docling cannot read it")
        return SimpleNamespace(parser=parser)

    monkeypatch.setattr(pipeline, "extract_document", fake_extract)
    monkeypatch.setattr("edupilot.ingestion.extract_document", fake_extract)

    doc = reindex._extract(tmp_path / "broken.pdf", "docling")
    assert doc.parser == "pymupdf" and calls == ["docling", "pymupdf"]


def test_numbered_headings_nest_even_when_docling_flattens_them():
    flat = FakeDocument(
        [
            item("section_header", "Lecture 4: Hypothesis Testing", level=1),
            item("section_header", "2. Error types", level=1),
            item("text", "A Type I error rejects a true null."),
            item("section_header", "2.1 Power", level=1),
            item("text", "Power is one minus the Type II error rate."),
        ]
    )
    blocks = blocks_from_docling(flat)
    assert [b.level for b in blocks if b.kind is BlockKind.HEADING] == [1, 2, 3]
    assert "Lecture 4: Hypothesis Testing > 2. Error types > 2.1 Power" in [
        s.breadcrumb for s in build_sections(blocks)
    ]
