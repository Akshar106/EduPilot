"""
Index management CLI
====================
Blue/green rebuild of the vector index, plus a few migration helpers.

    edupilot-reindex --status              show the active index and counts
    edupilot-reindex --rebuild             full rebuild into a new version
    edupilot-reindex --rebuild --dry-run   extract and chunk only, no writes
    edupilot-reindex --rebuild --parser docling   parse PDFs with Docling layout models
    edupilot-reindex --promote edupilot-v2 point traffic at a built version
    edupilot-reindex --rollback            revert to the previous version
    edupilot-reindex --drop edupilot-v1    delete a version that is not live

Equivalently, `python -m edupilot.cli.reindex --status`.

A rebuild writes into a **new** index (`edupilot-v1`, `-v2`, ...) and promotes
it only after every domain has been indexed successfully. The live index keeps
serving throughout, and a failed rebuild changes nothing. Running API
processes pick up a promotion within a few seconds (core/services.py).

Students' Self Study uploads are not in the course folders, so a rebuild
copies them from the live version into the new one rather than losing them.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Importing config loads .env — it walks up from the package to the repo root.
from edupilot.core.config import DOMAINS
from edupilot.core.observability import configure_logging

configure_logging()
logger = logging.getLogger("edupilot.reindex")


def cmd_status() -> int:
    from edupilot import db
    from edupilot.core.config import VECTOR_BACKEND
    from edupilot.core.services import services

    db.init_db()
    pointer = services.index_pointer
    print(f"backend      : {VECTOR_BACKEND}")
    print(f"active index : {pointer.active or '(none — run --rebuild)'}")
    print(f"version      : v{pointer.version}")
    print(f"embed model  : {services.embedder.model_name} ({services.embedder.dimension}d, "
          f"{services.embedder.max_tokens} tokens)")

    encoder = services.sparse_encoder
    print(f"bm25 encoder : {'fitted, ' + str(len(encoder.idf)) + ' terms' if encoder else 'missing'}")

    versions = pointer.records()
    if versions:
        print("\nversions:")
        for v in versions:
            print(f"  {v['name']:20s} {v['status']:9s} {v['embed_model'] or '':28s} {v['note']}")

    if not pointer.active:
        return 1

    try:
        stats = services.store.stats()
        print(f"\nvectors      : {stats['total_vectors']}")
        for namespace, count in sorted(stats.get("namespaces", {}).items()):
            print(f"  {namespace:24s} {count:6d}")
    except Exception as exc:
        print(f"\ncould not read index stats: {exc}")
        return 1

    docs = services.registry.list_documents()
    print(f"\nindexed documents: {len(docs)}")
    return 0


def _extract(path: Path, parser: str):
    """Extract with the chosen parser; a PDF Docling cannot read gets the fast one."""
    from edupilot.ingestion import extract_document

    try:
        return extract_document(path, parser=parser)
    except RuntimeError as exc:
        if parser == "pymupdf":
            raise
        logger.warning("docling could not read %s (%s) — using the fast parser", path.name, exc)
        return extract_document(path)


def cmd_rebuild(dry_run: bool, force_promote: bool, parser: str = "pymupdf") -> int:
    from edupilot import db
    from edupilot.db.engine import LOCK_REBUILD, advisory_lock

    if parser == "docling":
        from edupilot.ingestion.docling_parser import available

        if not available():
            logger.error("--parser docling needs the docling extra: uv sync --extra docling")
            return 1

    if dry_run:
        return _rebuild(dry_run=True, force_promote=force_promote, parser=parser)

    db.init_db()
    # One rebuild at a time across every machine sharing the database: two
    # would pick the same next version and write into one table.
    with db.get_engine().connect() as conn, advisory_lock(conn, LOCK_REBUILD, wait=False) as got:
        if not got:
            logger.error("another rebuild is running against this database — wait for it to finish")
            return 1
        return _rebuild(dry_run=False, force_promote=force_promote, parser=parser)


def _rebuild(*, dry_run: bool, force_promote: bool, parser: str) -> int:
    from edupilot.core.services import services
    from edupilot.ingestion import chunk_document
    from edupilot.retrieval import BM25Encoder, Indexer, course_files

    started = time.perf_counter()
    # config already matched each course to its folder on disk.
    folders = {domain: Path(cfg["knowledge_base_path"]) for domain, cfg in DOMAINS.items()}
    chunking = services.chunking

    # The corpus is fixed here, once: BM25 is fitted to exactly these files and
    # exactly these are indexed. A file copied in while the rebuild runs waits
    # for the next one. Subfolders are included; files under book/ are books.
    corpus = [
        (p, domain, kind)
        for domain, folder in folders.items()
        for p, kind in course_files(folder)
    ]
    books = sum(1 for _, _, kind in corpus if kind == "book")
    if not corpus:
        logger.error("no documents found under any knowledge_base_path")
        return 1

    if dry_run:
        version, index_name = 0, "none (dry run)"
    else:
        pointer = services.index_pointer
        previous = pointer.active
        version, index_name = pointer.next_version_name()

    print(f"target index : {index_name}" + (f" (v{version})" if version else ""))
    print(f"documents    : {len(corpus)} ({books} book{'s' if books != 1 else ''})")
    print(f"embed model  : {services.embedder.model_name}")
    print(f"pdf parser   : {parser}")
    print()

    # --- fit BM25 over the whole corpus before indexing anything ------------
    # IDF must be stable across every document or term weights are not
    # comparable between them, so this cannot be done incrementally.
    print("extracting and chunking the corpus...")
    texts: list[str] = []
    failures: list[str] = []
    # Chunks are kept so the indexing pass does not extract everything a
    # second time. Holding the whole corpus is a few MB of text and saves
    # roughly four minutes on this knowledge base.
    prepared: dict[str, list] = {}
    pages = empty_pages = tables = 0
    for i, (path, domain, _) in enumerate(corpus, 1):
        try:
            doc = _extract(path, parser)
            pages += doc.stats.page_count
            empty_pages += doc.stats.empty_pages
            tables += doc.stats.table_count
            namespace = DOMAINS[domain]["namespace"]
            chunks = chunk_document(doc, domain, config=chunking, scope=namespace)
            prepared[str(path)] = chunks
            texts.extend(c.text for c in chunks if not c.is_parent)
            print(f"  [{i}/{len(corpus)}] {path.name[:52]}", flush=True)
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            logger.warning("skipping %s during extraction: %s", path.name, exc)

    if not texts:
        logger.error("extraction produced no text — aborting")
        return 1

    encoder = BM25Encoder().fit(texts)
    print(f"\n  {len(texts)} chunks, {len(encoder.idf)} distinct terms\n")

    if dry_run:
        tokens = sorted(len(t.split()) for t in texts)
        print("DRY RUN — nothing written")
        print(f"  chunks       : {len(texts)}")
        print(f"  median words : {tokens[len(tokens) // 2]}")
        # Compare parsers with two dry runs: pages left empty are text a
        # parser lost (scanned pages, if OCR is off), tables are structure it kept.
        print(f"  pages        : {pages} ({empty_pages} with no text)")
        print(f"  tables       : {tables}")
        print(f"  failures     : {len(failures)}")
        for f in failures:
            print(f"    {f}")
        return 0

    pointer.begin(
        index_name,
        version,
        embed_model=services.embedder.model_name,
        dimension=services.embedder.dimension,
    )
    pointer.save_sparse_encoder(index_name, encoder.to_dict())

    # --- index into the new version -----------------------------------------
    print("embedding and upserting...")
    store = services.store_for_index(index_name)
    indexer = Indexer(
        store, services.embedder, services.registry, sparse_encoder=encoder, chunking=chunking
    )

    total = 0
    per_domain: dict[str, dict] = {}
    built: set[tuple[str, str]] = set()
    for domain, cfg in DOMAINS.items():
        namespace = cfg["namespace"]
        # Only what was extracted above; a file that failed extraction is
        # already reported, and retrying it here would fail the same way.
        results = [
            indexer.index_document(
                path,
                namespace=namespace,
                domain=domain,
                force=True,
                prepared_chunks=prepared[str(path)],
                source_type=kind,
            )
            for path, d, kind in corpus
            if d == domain and str(path) in prepared
        ]
        built |= {(namespace, r.filename) for r in results if not r.error}
        indexed = sum(r.chunks_indexed for r in results)
        errored = [f"{r.filename}: {r.error}" for r in results if r.error]
        total += indexed
        per_domain[domain] = {"documents": len(results), "chunks": indexed, "errors": errored}
        print(f"  {domain:6s} {len(results):3d} docs -> {indexed:5d} chunks"
              + (f"  ({len(errored)} failed)" if errored else ""))
        for e in errored:
            print(f"         {e}")

    elapsed = time.perf_counter() - started
    print(f"\nindexed {total} chunks in {elapsed:.0f}s")

    if total == 0:
        logger.error("rebuild produced no chunks — not promoting")
        return 1

    all_errors = [e for d in per_domain.values() for e in d["errors"]]
    if all_errors and not force_promote:
        print(f"\n{len(all_errors)} document(s) failed. Not promoting.")
        print(f"Built index {index_name} is complete but inactive. Review the errors, then:")
        print(f"  edupilot-reindex --promote {index_name}   # accept as-is")
        print("  edupilot-reindex --rebuild --force        # promote despite failures")
        return 1

    _carry_over_study_uploads(store, previous)
    pointer.promote(index_name, version, note=f"{total} chunks, {len(all_errors)} failures")
    forgotten = services.registry.prune(built)
    if forgotten:
        print(f"forgot {forgotten} document(s) no longer in the corpus")
    services.reload_sparse_encoder()
    print(f"\npromoted {index_name} — now serving traffic")
    return 0


def _carry_over_study_uploads(store, previous: str | None) -> None:
    """Copy students' Self Study vectors from the live version into the new one."""
    if not previous or not hasattr(store, "copy_namespaces"):
        return
    copied = store.copy_namespaces(previous, prefix="ss_")
    if copied:
        print(f"carried over {copied} Self Study vectors from {previous}")


def cmd_promote(index_name: str) -> int:
    from edupilot.core.services import services

    pointer = services.index_pointer
    try:
        version = int(index_name.rsplit("-v", 1)[1])
    except (IndexError, ValueError):
        logger.error("index name must end in -v<N>, got %r", index_name)
        return 1

    store = services.store_for_index(index_name)
    try:
        stats = store.stats()
    except Exception as exc:
        logger.error("cannot read %s — refusing to promote an unreachable index: %s", index_name, exc)
        return 1

    if not stats.get("total_vectors"):
        logger.error("%s is empty — refusing to promote", index_name)
        return 1

    _carry_over_study_uploads(store, pointer.active)
    pointer.promote(index_name, version, note="manual promotion")
    print(f"promoted {index_name} ({stats['total_vectors']} vectors)")
    return 0


def cmd_rollback() -> int:
    from edupilot.core.services import services

    pointer = services.index_pointer
    history = pointer.history()
    if not history:
        logger.error("no previous version recorded")
        return 1

    previous = history[0]
    print(f"rolling back from {pointer.active} to {previous['name']}")
    return cmd_promote(previous["name"])


def cmd_drop(index_name: str) -> int:
    """Delete an index version's vectors and record. Never the live one."""
    from edupilot.core.services import services

    pointer = services.index_pointer
    if index_name == pointer.active:
        logger.error("%s is serving traffic — promote another version first", index_name)
        return 1
    store = services.store_for_index(index_name)
    if not hasattr(store, "drop_index"):
        logger.error("dropping is only supported for the pgvector backend")
        return 1
    store.drop_index(index_name)
    pointer.forget(index_name)
    print(f"dropped {index_name}")
    return 0


def main() -> int:
    from edupilot.ingestion import PDF_PARSERS

    parser = argparse.ArgumentParser(description="EduPilot index management")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="show index state")
    group.add_argument("--rebuild", action="store_true", help="full blue/green rebuild")
    group.add_argument("--promote", metavar="INDEX", help="promote a built index")
    group.add_argument("--rollback", action="store_true", help="revert to the previous version")
    group.add_argument("--drop", metavar="INDEX", help="delete a version that is not live")
    parser.add_argument("--dry-run", action="store_true", help="with --rebuild: no writes")
    parser.add_argument("--force", action="store_true", help="with --rebuild: promote despite failures")
    parser.add_argument(
        "--parser",
        choices=PDF_PARSERS,
        default="pymupdf",
        help="with --rebuild: PDF extractor (docling needs the docling extra)",
    )
    args = parser.parse_args()

    if args.status:
        return cmd_status()
    if args.rebuild:
        return cmd_rebuild(args.dry_run, args.force, args.parser)
    if args.promote:
        return cmd_promote(args.promote)
    if args.rollback:
        return cmd_rollback()
    if args.drop:
        return cmd_drop(args.drop)
    return 1


if __name__ == "__main__":
    sys.exit(main())
