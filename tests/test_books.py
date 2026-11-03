"""
Textbooks next to lectures: how they are found, tagged, cited, and kept from
crowding the course's own material out of an answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edupilot.agents.contracts import citation_label
from edupilot.core.config import DOMAINS
from edupilot.retrieval import course_files
from edupilot.retrieval.hybrid import compose_evidence
from edupilot.retrieval.rerank import RerankedChunk


def excerpt(cid: str, relevance: float, *, book: bool = False, filename: str | None = None) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=cid,
        text=f"text of {cid}",
        metadata={
            "filename": filename or ("book.pdf" if book else f"{cid}.pdf"),
            "source_type": "book" if book else "lecture",
        },
        relevance=relevance,
    )


# ---------------------------------------------------------------------------
# Finding and tagging
# ---------------------------------------------------------------------------


def test_files_in_a_book_folder_are_books_and_subfolders_are_read(tmp_path):
    (tmp_path / "book").mkdir()
    (tmp_path / "week1").mkdir()
    (tmp_path / "lec1.pdf").write_bytes(b"%PDF")
    (tmp_path / "book" / "textbook.pdf").write_bytes(b"%PDF")
    (tmp_path / "week1" / "lec2.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.xlsx").write_bytes(b"x")  # unsupported
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "draft.pdf").write_bytes(b"%PDF")

    found = {path.name: kind for path, kind in course_files(tmp_path)}
    assert found == {"lec1.pdf": "lecture", "textbook.pdf": "book", "lec2.pdf": "lecture"}


def test_a_second_file_with_the_same_name_is_skipped_not_overwritten(tmp_path):
    (tmp_path / "book").mkdir()
    (tmp_path / "intro.pdf").write_bytes(b"%PDF")
    (tmp_path / "book" / "intro.pdf").write_bytes(b"%PDF")
    assert [p.name for p, _ in course_files(tmp_path)] == ["intro.pdf"]


def test_book_citations_say_so():
    assert citation_label({"filename": "deep_learning.pdf", "page_number": 12, "source_type": "book"}) == (
        "Deep Learning (book), p.12"
    )
    assert citation_label({"filename": "lec3_svm.pdf", "page_number": 4}) == "Lec3 Svm, p.4"


# ---------------------------------------------------------------------------
# Choosing the excerpts
# ---------------------------------------------------------------------------


def test_lectures_come_first_and_the_book_adds_depth():
    ranked = [
        excerpt("b1", 0.99, book=True), excerpt("b2", 0.98, book=True),
        excerpt("b3", 0.97, book=True), excerpt("b4", 0.96, book=True),
        excerpt("l1", 0.90), excerpt("l2", 0.80), excerpt("l3", 0.70),
    ]
    chosen, counts = compose_evidence(ranked, 5, max_book=2)
    # The lectures lead, so [Source 1] is the course's own account.
    assert [c.chunk_id for c in chosen] == ["l1", "l2", "l3", "b1", "b2"]
    assert counts == {"book_excerpts": 2, "held_back": 2, "filled_from_held_back": 0}


def test_the_book_fills_in_where_the_lectures_say_nothing():
    ranked = [excerpt("b1", 0.9, book=True), excerpt("b2", 0.8, book=True),
              excerpt("b3", 0.7, book=True), excerpt("l1", 0.6)]
    chosen, counts = compose_evidence(ranked, 4, max_book=1)
    assert [c.chunk_id for c in chosen] == ["l1", "b1", "b2", "b3"]
    assert counts["filled_from_held_back"] == 2


def test_no_limit_takes_the_most_relevant():
    ranked = [excerpt("b1", 0.9, book=True), excerpt("b2", 0.8, book=True), excerpt("l1", 0.7)]
    chosen, _ = compose_evidence(ranked, 2, max_book=None)
    assert {c.chunk_id for c in chosen} == {"b1", "b2"}


def test_a_per_file_limit_spreads_the_excerpts():
    ranked = [excerpt(f"a{i}", 1 - i / 10, filename="a.pdf") for i in range(4)] + [excerpt("b0", 0.5)]
    chosen, _ = compose_evidence(ranked, 3, max_per_document=2)
    assert [c.chunk_id for c in chosen] == ["a0", "a1", "b0"]


def test_study_uploads_are_never_held_back():
    """Self Study files carry no book tag; one uploaded file can fill every slot."""
    ranked = [excerpt(f"u{i}", 1 - i / 10, filename="notes.pdf") for i in range(6)]
    chosen, _ = compose_evidence(ranked, 6, max_book=2)
    assert len(chosen) == 6


# ---------------------------------------------------------------------------
# Serving course PDFs: lectures to read, never textbooks
# ---------------------------------------------------------------------------


@pytest.fixture
def algo_files():
    root = Path(DOMAINS["ALGO"]["knowledge_base_path"])
    (root / "book").mkdir(parents=True, exist_ok=True)
    paths = [root / "book" / "clrs-test.pdf", root / "lec-test.pdf"]
    for path in paths:
        path.write_bytes(b"%PDF-1.4 test")
    yield paths
    for path in paths:
        path.unlink(missing_ok=True)


def test_a_lecture_opens_in_the_browser_and_a_textbook_is_not_served(algo_files):
    from edupilot.api.app import create_app

    with TestClient(create_app()) as client:
        lecture = client.get("/api/documents/ALGO/lec-test.pdf")
        book = client.get("/api/documents/ALGO/clrs-test.pdf")
    assert lecture.status_code == 200 and lecture.content.startswith(b"%PDF")
    assert lecture.headers["content-disposition"].startswith("inline")
    assert book.status_code == 403 and b"%PDF" not in book.content


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, title",
    [
        ("Regularization and\npolynomial regression", "Regularization and polynomial regression"),
        ("DPO and GRPO\nDirect Preference Optimization (DPO)", "DPO and GRPO"),
        ("TOP RAG CHUNKING METHODS YOU SHOULD\nKNOW", "TOP RAG CHUNKING METHODS YOU SHOULD KNOW"),
        ("Reinforcement Learning\n(Optional lecture)", "Reinforcement Learning (Optional lecture)"),
    ],
)
def test_a_heading_keeps_the_lines_that_continue_it(text, title):
    from edupilot.ingestion.titles import heading_title

    assert heading_title(text) == title


@pytest.mark.parametrize(
    "filename, title",
    [
        ("Machine.Learning.with.PyTorch.and.Scikit-Learn.Sebastian.Raschka.Packt.9781801819312.EBooksWorld.ir.pdf",
         "Machine Learning with PyTorch and Scikit-Learn Sebastian Raschka Packt"),
        ("Algorithms2022-[www.konkur.in].pdf", "Algorithms2022"),
        ("1-2 Algorithm Analysis.pdf.pdf", "1-2 Algorithm Analysis"),
        ("2_1_Algorithm_Design_FA26 (2).pdf", "2-1 Algorithm Design"),
        ("B505_2_4_Greedy_FA26.pdf", "2-4 Greedy"),
        ("LLMs-07-prompting-SP26.pdf", "LLMs 07 Prompting"),
        ("data_structures_review (1).pdf", "Data Structures Review"),
    ],
)
def test_file_names_lose_what_downloads_add(filename, title):
    from edupilot.ingestion.titles import title_from_filename

    assert title_from_filename(filename) == title


def test_a_good_metadata_title_wins_and_banners_do_not():
    from edupilot.ingestion.titles import choose_title

    assert choose_title(metadata="Introduction to Algorithms, Fourth Edition", heading="www.konkur.in",
                        filename="Algorithms2022.pdf") == "Introduction to Algorithms, Fourth Edition"
    assert choose_title(metadata="LLMs-01-intro", heading="Overview and Introduction",
                        filename="LLMs-01-intro-SP26.pdf") == "Overview and Introduction"
    assert choose_title(metadata="", heading="INFO-I590 Fundamentals and Applications of LLMs",
                        filename="LLMs-08-RLHF-SP26.pdf") == "LLMs 08 RLHF"
    assert choose_title(metadata="Photo", heading="arXiv:1810.04805v2 [cs.CL]", filename="BERT.pdf") == "BERT"


# ---------------------------------------------------------------------------
# Course Materials: lectures in the course's order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, number",
    [
        ("lec6_Regularization_polynomial_regression.pdf", (6,)),
        ("lec10_Decision_Trees.pdf", (10,)),
        ("LLMs-07-prompting-SP26.pdf", (7,)),
        ("1-2 Algorithm Analysis.pdf.pdf", (1, 2)),
        ("2_1_Algorithm_Design_FA26 (2).pdf", (2, 1)),
        ("B505_2_4_Greedy_FA26.pdf", (2, 4)),
        ("Lecture 3 - Trees.pdf", (3,)),
        ("2022 notes.pdf", None),
        ("LLM_FineTuning.pdf", None),
        ("BERT.pdf", None),
    ],
)
def test_lecture_numbers_come_from_the_file_name(filename, number):
    from edupilot.ingestion.titles import lecture_number

    assert lecture_number(filename) == number


def test_lectures_are_listed_in_course_order_under_their_titles():
    from edupilot.api.routes.knowledge_base import course_materials

    documents = [
        {"filename": name, "source_type": kind}
        for name, kind in [
            ("lec10_Decision_Trees.pdf", "lecture"),
            ("Notes.pdf", "lecture"),
            ("lec2_intro_to_ML.pdf", "lecture"),
            ("1-2 Algorithm Analysis.pdf", "lecture"),
            ("textbook.pdf", "book"),
        ]
    ]
    titles = {"lec2_intro_to_ML.pdf": "Introduction to machine learning", "textbook.pdf": "Deep Learning"}
    materials = course_materials(documents, titles)

    assert [(d["number"], d["title"]) for d in materials["lectures"]] == [
        ("1-2", "Algorithm Analysis"),
        ("2", "Introduction to machine learning"),
        ("10", "Decision Trees"),
        (None, "Notes"),
    ]
    assert materials["books"] == [{"filename": "textbook.pdf", "title": "Deep Learning"}]
