"""
Docling PDF extraction
======================
Layout-model extraction for PDFs, as an alternative to the rule-based
`extract_pdf`.

`pdf.py` recovers structure with heuristics: columns from block positions,
headings from font size and weight, tables from ruling lines via pdfplumber,
formulas from symbol density. Docling runs a document-layout model and a
table-structure model instead, so reading order, headings, tables and formulas
come from what the page looks like rather than from font metadata — which is
where lecture slides and two-column papers defeat the heuristics.

It produces the same `ParsedDocument`, so sections, chunking and citations
are unchanged. It is slower — roughly a second per page on CPU, far more with
OCR — and pulls in PyTorch, so it is opt-in: the `docling` extra, used by
`edupilot-reindex --rebuild --parser docling` for the course corpus. Student
uploads keep the fast parser, since a student waits on those.

OCR is off by default. Docling OCRs every bitmap region rather than only pages
without a text layer, which on image-heavy slides is slow and adds diagram
noise; set DOCLING_OCR=true when the corpus has scanned pages.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from .models import Block, BlockKind, ExtractionStats, ParsedDocument, file_content_hash
from .pdf import _infer_title

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "docling/1.0"

#: Docling item label -> block kind. Labels not listed are skipped: page
#: headers and footers (running heads), pictures and charts (their captions
#: arrive as separate caption items), and the table of contents.
_KINDS: dict[str, BlockKind] = {
    "title": BlockKind.HEADING,
    "section_header": BlockKind.HEADING,
    "text": BlockKind.PARAGRAPH,
    "paragraph": BlockKind.PARAGRAPH,
    "reference": BlockKind.PARAGRAPH,
    "footnote": BlockKind.PARAGRAPH,
    "handwritten_text": BlockKind.PARAGRAPH,
    "list_item": BlockKind.LIST_ITEM,
    "table": BlockKind.TABLE,
    "formula": BlockKind.FORMULA,
    "code": BlockKind.CODE,
    "caption": BlockKind.CAPTION,
}
_RUNNING_HEADS = frozenset({"page_header", "page_footer"})


def available() -> bool:
    """True when the docling extra is installed."""
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def _label(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


#: A heading's leading section number: "2.", "2.1", "3.4.1".
_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")


def _heading_level(label: str, text: str, item: Any, first: bool) -> int:
    """
    Depth for a heading block, 1 being the document title.

    Docling's layout model finds headings but often reports every section
    header at the same level, which would flatten "2.1 Power" into a sibling
    of "2. Error types". A section number states its depth outright, so it
    wins; an unnumbered first heading is the title; otherwise Docling's own
    level is used.
    """
    if label == "title":
        return 1
    numbered = _SECTION_NUMBER.match(text)
    if numbered:
        return min(1 + len(numbered.group(1).split(".")), 6)
    if first:
        return 1
    return min(1 + int(getattr(item, "level", 1) or 1), 6)


def blocks_from_docling(document: Any, stats: ExtractionStats | None = None) -> list[Block]:
    """
    Convert a DoclingDocument's items, in reading order, into `Block`s.

    Heading depths come from `_heading_level`, so `build_sections` nests them
    the same way it nests headings from the rule-based parser.
    """
    stats = stats if stats is not None else ExtractionStats()
    blocks: list[Block] = []
    for item, _depth in document.iterate_items():
        label = _label(item)
        if label in _RUNNING_HEADS:
            stats.stripped_running_heads += 1
            continue
        kind = _KINDS.get(label)
        if kind is None:
            continue

        if kind is BlockKind.TABLE:
            text = item.export_to_markdown(doc=document).strip()
        else:
            text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue

        level = None
        if kind is BlockKind.HEADING:
            level = _heading_level(label, text, item, first=stats.heading_count == 0)

        prov = (getattr(item, "prov", None) or [None])[0]
        page = int(getattr(prov, "page_no", 0) or 0)
        bbox = getattr(prov, "bbox", None)
        box = (bbox.l, bbox.t, bbox.r, bbox.b) if bbox is not None else (0.0, 0.0, 0.0, 0.0)

        blocks.append(Block(text=text, kind=kind, page=page, bbox=box, level=level))
        if kind is BlockKind.TABLE:
            stats.table_count += 1
        elif kind is BlockKind.FORMULA:
            stats.formula_count += 1
        elif kind is BlockKind.HEADING:
            stats.heading_count += 1
    return blocks


_converters: dict[tuple[bool, bool], Any] = {}
_converters_lock = threading.Lock()


def _converter(enable_ocr: bool, enable_tables: bool):
    """One converter per option set; building one loads the layout models."""
    key = (enable_ocr, enable_tables)
    with _converters_lock:
        if key not in _converters:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = enable_ocr
            options.do_table_structure = enable_tables
            _converters[key] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return _converters[key]


def _metadata_title(path) -> str:
    """The PDF's metadata title, read with PyMuPDF; Docling does not expose it."""
    try:
        import fitz

        with fitz.open(path) as pdf:
            return (pdf.metadata or {}).get("title") or ""
    except Exception:
        return ""


def extract_pdf_docling(
    path: str, *, enable_ocr: bool | None = None, enable_tables: bool = True
) -> ParsedDocument:
    """
    Extract a PDF with Docling into a `ParsedDocument` (sections not yet built).

    Raises:
        RuntimeError: the docling extra is missing, or Docling cannot read the file.
    """
    if not available():
        raise RuntimeError(
            "the docling parser needs the docling extra: `uv sync --extra docling`"
        )
    if enable_ocr is None:
        enable_ocr = os.getenv("DOCLING_OCR", "false").lower() == "true"

    started = time.perf_counter()
    try:
        result = _converter(enable_ocr, enable_tables).convert(str(path))
    except Exception as exc:
        raise RuntimeError(f"Docling cannot read {path!r}: {exc}") from exc
    document = result.document

    stats = ExtractionStats(page_count=len(getattr(document, "pages", {}) or {}))
    blocks = blocks_from_docling(document, stats)
    stats.block_count = len(blocks)
    pages_with_text = {b.page for b in blocks}
    stats.empty_pages = max(0, stats.page_count - len(pages_with_text))
    stats.duration_ms = int((time.perf_counter() - started) * 1000)

    p = Path(path)
    return ParsedDocument(
        source_path=str(p),
        filename=p.name,
        title=_infer_title(blocks, p.name, metadata_title=_metadata_title(p)),
        content_hash=file_content_hash(p),
        blocks=blocks,
        stats=stats,
        extractor=EXTRACTOR_VERSION,
    )
