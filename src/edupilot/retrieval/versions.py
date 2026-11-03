"""
Index versions (blue/green)
===========================
Which vector index serves traffic, kept in the `index_versions` table.

Indexes are named `<base>-v1`, `<base>-v2`, ... A rebuild writes into a new
version and calls `promote()` only after it finishes, so readers never see a
partially built index, and rollback is promoting the previous version again.

This used to be a JSON file under the data directory. In the database it is
shared by every API process, a promotion is one transaction, and the fitted
BM25 table travels with the version whose sparse vectors it produced instead
of sitting in a second file that could fall out of step.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import Engine, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from edupilot.db.rows import row_dict
from edupilot.db.tables import index_versions as versions

logger = logging.getLogger(__name__)


class IndexPointer:
    """
    The registry of index versions and the pointer to the live one.

    Args:
        base_name: Index name prefix; versions are `<base>-v<N>`.
        backend: Vector store backend that builds the versions (recorded).
        engine: The SQLAlchemy engine. Defaults to the app's shared one.
        cache_seconds: How long `active` may be served from memory. Every
            search asks which index is live, so the app caches the answer
            briefly; a promotion made by another process (the reindex CLI,
            another API instance) reaches this one within that window.
    """

    def __init__(
        self,
        base_name: str,
        *,
        backend: str,
        engine: Engine | None = None,
        cache_seconds: float = 0.0,
    ) -> None:
        from edupilot.db import get_engine

        self.base_name = base_name
        self.backend = backend
        self.engine = engine or get_engine()
        self.cache_seconds = cache_seconds
        self._cached: tuple[float, Any] | None = None

    # -- reads -------------------------------------------------------------

    def _active_row(self):
        if self._cached and time.monotonic() - self._cached[0] < self.cache_seconds:
            return self._cached[1]
        with self.engine.connect() as conn:
            row = conn.execute(
                select(versions.c.name, versions.c.version).where(versions.c.status == "active")
            ).first()
        self._cached = (time.monotonic(), row)
        return row

    def refresh(self) -> None:
        """Forget the cached active version; the next read asks the database."""
        self._cached = None

    @property
    def active(self) -> str | None:
        """Name of the index serving traffic, or None before the first promotion."""
        row = self._active_row()
        return row.name if row else None

    @property
    def version(self) -> int:
        """Version number of the active index, 0 when there is none."""
        row = self._active_row()
        return int(row.version) if row else 0

    def name_for_version(self, version: int) -> str:
        return f"{self.base_name}-v{version}"

    def next_version_name(self) -> tuple[int, str]:
        """
        The next unused version.

        Counts every recorded version, not just the active one, so a rebuild
        that finished but was never promoted is not overwritten by the next.
        """
        with self.engine.connect() as conn:
            highest = conn.execute(select(func.max(versions.c.version))).scalar() or 0
        version = int(highest) + 1
        return version, self.name_for_version(version)

    def get(self, name: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(versions).where(versions.c.name == name)
            ).first()
        if not row:
            return None
        out = row_dict(row)
        out.pop("sparse_encoder", None)
        return out

    def records(self) -> list[dict[str, Any]]:
        """Every recorded version, newest first, without the BM25 tables."""
        columns = [c for c in versions.c if c.name != "sparse_encoder"]
        with self.engine.connect() as conn:
            rows = conn.execute(select(*columns).order_by(versions.c.version.desc()))
            return [row_dict(r) for r in rows]

    def history(self) -> list[dict[str, Any]]:
        """Previously active versions, most recently retired first."""
        return sorted(
            (v for v in self.records() if v["status"] == "retired"),
            key=lambda v: v["retired_at"] or "",
            reverse=True,
        )

    # -- writes ------------------------------------------------------------

    def begin(self, name: str, version: int, *, embed_model: str, dimension: int) -> None:
        """Record a version being built. It serves nothing until promoted."""
        statement = pg_insert(versions).values(
            version=version,
            name=name,
            backend=self.backend,
            embed_model=embed_model,
            dimension=dimension,
            status="building",
        )
        with self.engine.begin() as conn:
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=["version"],
                    set_={"embed_model": embed_model, "dimension": dimension},
                    where=versions.c.status == "building",
                )
            )

    def promote(self, name: str, version: int, *, note: str = "") -> None:
        """
        Point live traffic at `name`. Called only after a rebuild succeeds.

        Retiring the old version and activating the new one is a single
        transaction, and a partial unique index allows only one active row, so
        two promotions racing cannot leave two versions live.
        """
        with self.engine.begin() as conn:
            conn.execute(
                update(versions)
                .where(versions.c.status == "active", versions.c.name != name)
                .values(status="retired", retired_at=func.now())
            )
            statement = pg_insert(versions).values(
                version=version,
                name=name,
                backend=self.backend,
                status="active",
                note=note,
                promoted_at=func.now(),
            )
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=["version"],
                    set_={
                        "status": "active",
                        "note": note,
                        "promoted_at": func.now(),
                        "retired_at": None,
                    },
                )
            )
        self.refresh()
        logger.info("promoted index %s (v%d) to active", name, version)

    def forget(self, name: str) -> None:
        """Remove a version's record. Refuses the active one."""
        with self.engine.begin() as conn:
            conn.execute(
                versions.delete().where(versions.c.name == name, versions.c.status != "active")
            )

    # -- the BM25 table that belongs to a version ---------------------------

    def save_sparse_encoder(self, name: str, encoder: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(versions).where(versions.c.name == name).values(sparse_encoder=encoder)
            )

    def sparse_encoder(self, name: str | None = None) -> dict[str, Any] | None:
        """The fitted BM25 table of `name`, or of the active version by default."""
        condition = versions.c.name == name if name else versions.c.status == "active"
        with self.engine.connect() as conn:
            return conn.execute(select(versions.c.sparse_encoder).where(condition)).scalar()
