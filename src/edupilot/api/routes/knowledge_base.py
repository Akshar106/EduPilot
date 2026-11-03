"""
Course knowledge base
=====================
The shared corpus every student's answers are grounded in.

Reads are open to any authenticated user; mutation is admin-only. Both were
previously unauthenticated, so any visitor could inject documents that every
student's answers would then cite.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from edupilot.api.deps import AdminUser, CurrentUser, run_blocking
from edupilot.core.config import DOMAINS
from edupilot.core.services import services
from edupilot.ingestion.titles import lecture_number, title_from_filename, without_numbering
from edupilot.security import AppError, ErrorCode, rate_limited, resolve_within, validate_batch
from edupilot.security.uploads import UploadRejected

router = APIRouter(prefix="/api", tags=["knowledge-base"])


def _course_file(kb_path: Path, filename: str) -> Path:
    """
    Where a course document is on disk: at the top of the folder, or in a
    subfolder such as book/. `resolve_within` validates the name either way.
    """
    path = resolve_within(kb_path, filename)
    if path.is_file():
        return path
    from edupilot.retrieval import course_files

    for candidate, _ in course_files(kb_path):
        if candidate.name == path.name:
            return candidate
    return path


def _require_domain(domain: str) -> dict:
    if domain not in DOMAINS:
        raise AppError(
            code=ErrorCode.VALIDATION_FAILED, message=f"Unknown domain '{domain}'."
        )
    return DOMAINS[domain]


@router.get("/kb/status", dependencies=[Depends(rate_limited("read"))])
async def kb_status(user: CurrentUser):
    stats = await run_blocking(services.store.stats)
    out = {}
    for domain, cfg in DOMAINS.items():
        namespace = cfg["namespace"]
        out[domain] = {
            "name": cfg["name"],
            "color": cfg["color"],
            "chunk_count": stats.get("namespaces", {}).get(namespace, 0),
            "documents": await run_blocking(services.registry.list_documents, namespace),
        }
    return out


def course_materials(documents: list[dict], titles: dict[str, str]) -> dict:
    """
    One course's documents as a student browses them: lectures in the
    course's own order under their titles, then textbooks.

    `documents` are the course's registry rows; `titles` the titles stored
    with the chunks, by filename. A lecture's number comes from its file
    name and is shown apart, so it leaves the title ("1-2 Algorithm Analysis"
    is lecture 1-2, "Algorithm Analysis"). Unnumbered lectures follow the
    numbered ones, alphabetically.
    """
    lectures, books = [], []
    for doc in documents:
        filename = doc["filename"]
        title = titles.get(filename) or title_from_filename(filename)
        if doc.get("source_type") == "book":
            books.append({"filename": filename, "title": title})
            continue
        number = lecture_number(filename)
        lectures.append((
            (number is None, number or (), title.lower()),
            {
                "filename": filename,
                "title": without_numbering(title) if number else title,
                "number": "-".join(map(str, number)) if number else None,
            },
        ))
    return {
        "lectures": [item for _, item in sorted(lectures, key=lambda pair: pair[0])],
        "books": sorted(books, key=lambda b: b["title"].lower()),
    }


@router.get("/kb/materials", dependencies=[Depends(rate_limited("read"))])
async def list_course_materials(user: CurrentUser):
    """The Course Materials page: each course's lectures and textbooks."""
    documents = await run_blocking(services.registry.list_documents)
    out = {}
    for domain, cfg in DOMAINS.items():
        namespace = cfg["namespace"]
        titles = await run_blocking(services.store.document_titles, namespace)
        out[domain] = {
            "name": cfg["name"],
            "color": cfg["color"],
            **course_materials([d for d in documents if d["namespace"] == namespace], titles),
        }
    return out


@router.get("/kb/documents", dependencies=[Depends(rate_limited("read"))])
async def list_kb_documents(user: CurrentUser):
    documents = await run_blocking(services.registry.list_documents)
    return {
        domain: {
            "name": cfg["name"],
            "color": cfg["color"],
            "documents": [d for d in documents if d["namespace"] == cfg["namespace"]],
        }
        for domain, cfg in DOMAINS.items()
    }


@router.post("/kb/upload", status_code=202, dependencies=[Depends(rate_limited("upload"))])
async def upload_to_kb(
    admin: AdminUser,
    domain: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    Add documents to the shared course knowledge base. Admin only.

    The file is kept in the course folder, where rebuilds and source links read
    it, and queued for indexing; each job reports at `GET /api/jobs/{job_id}`.
    """
    from edupilot.ingestion import jobs

    cfg = _require_domain(domain)

    raw = [(f.filename or "", await f.read()) for f in files]
    try:
        validated = validate_batch(raw)
    except UploadRejected as exc:
        raise AppError(code=ErrorCode.UPLOAD_REJECTED, message=str(exc)) from exc

    kb_path = Path(cfg["knowledge_base_path"])
    kb_path.mkdir(parents=True, exist_ok=True)

    queued = []
    for item in validated:
        # resolve_within re-checks containment after sanitization. The old
        # code did `kb_path / uf.filename` with no check at all.
        dest = resolve_within(kb_path, item.safe_name)
        await run_blocking(dest.write_bytes, item.content)
        ids = await run_blocking(
            jobs.enqueue,
            kind="course",
            user_id=admin.user_id,
            namespace=cfg["namespace"],
            domain=domain,
            filename=item.safe_name,
            content=item.content,
            file_type=item.extension,
        )
        queued.append({
            "filename": item.safe_name,
            "original_name": item.original_name,
            "job_id": ids["job_id"],
            "status": "queued",
            "warnings": item.warnings,
        })

    return {"queued": queued}


@router.get("/documents/{domain}/{filename}", dependencies=[Depends(rate_limited("read"))])
async def serve_document(domain: str, filename: str, user: CurrentUser):
    """
    Serve a lecture's PDF for viewing in the browser, from a citation or the
    Course Materials page.

    Textbooks are not served. They are copyrighted books, often whole ones,
    and the deployed app is open to every student; an answer cites a book by
    page, and the student reads it in their own copy.
    """
    if domain not in DOMAINS:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"unknown domain {domain}")
    kb_path = Path(DOMAINS[domain]["knowledge_base_path"])
    try:
        path = await run_blocking(_course_file, kb_path, filename)
    except UploadRejected as exc:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=str(exc)) from exc
    if not path.is_file():
        # A deployment can serve a pre-built vector index without shipping the
        # source PDFs — the corpus is ~191 MB of copyrighted course material.
        # Answers and citations still work; only opening the original file
        # does not. Say so, rather than returning a bare "not found" that
        # reads as a bug.
        if not kb_path.is_dir():
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message=(
                    "Source documents are not available in this deployment. "
                    "The answer and its citations are unaffected."
                ),
                internal=f"knowledge base directory absent: {kb_path}",
            )
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"missing {path}")
    from edupilot.retrieval import source_type_of

    if source_type_of(path, kb_path) == "book":
        raise AppError(
            code=ErrorCode.FORBIDDEN,
            message="Textbooks aren't available to open here. The citation gives the page to look up.",
            internal=f"refused to serve a textbook: {path.name}",
        )
    return FileResponse(str(path), filename=path.name, content_disposition_type="inline")


@router.delete("/kb/{domain}/{filename}")
async def delete_kb_document(domain: str, filename: str, admin: AdminUser):
    if domain not in DOMAINS:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"unknown domain {domain}")
    namespace = DOMAINS[domain]["namespace"]
    safe = await run_blocking(_course_file, Path(DOMAINS[domain]["knowledge_base_path"]), filename)

    removed = await run_blocking(services.indexer.remove_document, namespace, safe.name)
    if safe.is_file():
        safe.unlink()
    return {"deleted": safe.name, "was_indexed": removed}
