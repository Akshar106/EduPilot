"""
Document titles
===============
The name a document is cited by ("Regularization and polynomial regression,
pp.12-14"), chosen from, in order:

  1. the PDF's own metadata title, when it reads like one — books usually
     carry a proper title ("Introduction to Algorithms, Fourth Edition"),
     slides a file stem ("LLMs-01-intro") or a placeholder ("Photo");
  2. the first heading on page one, every line of it — a slide title set
     over two lines is one title, not "Regularization and";
  3. the file name, cleaned of what downloads and copies add to it: site
     tags ("[www.example.in]"), ISBNs and whatever trails them, copy markers
     ("(1)"), semester tags ("SP26"), course-code prefixes ("B505_") and
     doubled extensions (".pdf.pdf").

A heading that names the course rather than the document ("INFO-I590
Fundamentals and Applications of LLMs", on four decks alike) or an arXiv
stamp is skipped in favour of the next choice.
"""

from __future__ import annotations

import re

_ACRONYM = re.compile(r"^[A-Z][A-Z0-9]*s?$")
_SMALL_WORDS = frozenset("a an and as at by for from in of on or the to with vs".split())

#: Text that is page furniture, not a title.
_NOT_A_TITLE = re.compile(r"^(?:arxiv:|www\.|https?:|telegram|untitled|microsoft word)|^photo$", re.I)
#: A course code at the start: "INFO-I590 ...", "CSCI-B505 ...". Shared by
#: every deck in the course, so it identifies none of them.
_COURSE_BANNER = re.compile(r"^[A-Z]{2,6}[- ]?[A-Z]?\d{3,4}\b")

_EXTENSIONS = (".pdf", ".docx", ".md", ".txt")


def is_furniture(text: str) -> bool:
    """Page furniture that heads a page but titles nothing: an arXiv stamp, a URL, a course banner."""
    text = text.strip()
    return bool(_NOT_A_TITLE.search(text) or _COURSE_BANNER.match(text))


def _usable(title: str, *, min_words: int) -> bool:
    title = title.strip()
    return (
        3 <= len(title) <= 160
        and len(title.split()) >= min_words
        and not _NOT_A_TITLE.search(title)
        and not _COURSE_BANNER.match(title)
    )


def _looks_like_file_stem(title: str) -> bool:
    return " " not in title.strip() and bool(re.search(r"[-_.]", title))


_CONTINUES = re.compile(
    r"(?:\b(?:and|or|of|to|for|with|in|on|the|a|an|vs|via|from|by)|[:,&-])\s*$", re.I
)


#: A title line this long most likely wrapped at the slide's edge.
WRAP_CHARS = 30


def heading_title(text: str, *, max_chars: int = 120) -> str:
    """
    A heading's title: its first line, plus the lines that continue it.

    A line continues the title when the one before it is unfinished
    ("Regularization and"), long enough to have wrapped ("How to Improve RAG
    Accuracy: A Comprehensive"), or when it opens with lowercase or a bracket
    ("(optional lecture)"). A capitalized line after a short, finished one is
    a subtitle ("DPO and GRPO" / "Direct Preference Optimization") or a name,
    and stays out.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    title = previous = lines[0]
    for line in lines[1:]:
        wrapped = len(previous) >= WRAP_CHARS
        continues = _CONTINUES.search(previous) or wrapped or line[:1].islower() or line[:1] in "(["
        if not continues or len(title) + 1 + len(line) > max_chars:
            break
        title = f"{title} {line}"
        previous = line
    return re.sub(r"\s+", " ", title)


def title_from_filename(name: str) -> str:
    """A readable title from a file name, without what downloads tack on."""
    stem = name.strip()
    while stem.lower().endswith(_EXTENSIONS):
        stem = stem.rsplit(".", 1)[0]
    stem = re.sub(r"\[[^\]]*\]|\((?:\d+|copy)\)", " ", stem, flags=re.I)
    if stem.count(".") >= 2 and " " not in stem:
        stem = stem.replace(".", " ")  # Words.Joined.By.Dots
    stem = re.sub(r"\b(?:97[89])?\d{9}[\dX]\b.*$", "", stem)  # an ISBN, and the publisher/site after it
    stem = re.sub(r"(?:^|[-_ ])(?:SP|FA|SU|WI)\d{2}\b", " ", stem, flags=re.I)
    stem = re.sub(r"^[A-Z]{1,6}\d{3,4}[-_ ]+(?=\S)", "", stem)  # a course code prefix: B505_
    stem = re.sub(r"(?<=\d)[-_](?=\d)", "\x00", stem)  # 1-2, 2_1: section numbers
    stem = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "\x01", stem)  # Scikit-Learn, Hands-On
    stem = re.sub(r"[-_]+", " ", stem).replace("\x00", "-").replace("\x01", "-")
    words = stem.split()
    out = []
    for i, word in enumerate(words):
        if _ACRONYM.match(word) or any(c.isupper() for c in word[1:]):
            out.append(word)  # PyTorch, LLMs, BERT
        elif i and word.lower() in _SMALL_WORDS:
            out.append(word.lower())
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out) or name


#: How a course numbers its lecture files, once any course-code prefix is off:
#: "lec6_...", "Lecture 6 ...", "LLMs-07-...", "1-2 Algorithm Analysis",
#: "2_1_Algorithm_Design". At most two parts of at most two digits each, so a
#: year ("2022 notes") is not a lecture number.
_NUMBERED = re.compile(r"^(?:lec(?:ture)?|llms?)?[-_ ]?(\d{1,2})(?:[-_.](\d{1,2}))?(?=[-_ ]|$)", re.I)
#: The same number at the start of a title, where the course list shows it apart.
_LEADING_NUMBER = re.compile(r"^(?:(?:lec(?:ture)?|llms?)\s*)?\d{1,2}(?:[-.]\d{1,2})?(?:\s*[:.-]\s*|\s+)(?=\S)", re.I)


def lecture_number(filename: str) -> tuple[int, ...] | None:
    """The number a course gives a lecture file — lec6 is (6,), 1-2 is (1, 2) — or None."""
    stem = filename.strip()
    while stem.lower().endswith(_EXTENSIONS):
        stem = stem.rsplit(".", 1)[0]
    stem = re.sub(r"^[A-Z]{1,6}\d{3,4}[-_ ]+(?=\S)", "", stem)  # B505_2_4_Greedy
    match = _NUMBERED.match(stem)
    return tuple(int(g) for g in match.groups() if g is not None) if match else None


def without_numbering(title: str) -> str:
    """A numbered lecture's title without its number: "1-2 Algorithm Analysis" -> "Algorithm Analysis"."""
    return _LEADING_NUMBER.sub("", title.strip()) or title


def choose_title(*, metadata: str | None, heading: str | None, filename: str) -> str:
    """The best of the three candidates, per the order in the module docstring."""
    meta = (metadata or "").strip()
    if meta and _usable(meta, min_words=2) and not _looks_like_file_stem(meta):
        return re.sub(r"\s+", " ", meta)
    if heading and _usable(heading, min_words=1):
        return heading
    return title_from_filename(filename)
