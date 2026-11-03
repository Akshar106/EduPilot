"""Row conversion shared by the query modules."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Row


def iso(value: datetime | None) -> str | None:
    """A timestamp as ISO 8601 in UTC, to the second: what the API returns."""
    if value is None:
        return None
    return value.astimezone(UTC).isoformat(timespec="seconds")


def row_dict(row: Row) -> dict[str, Any]:
    """A result row as a plain dict, with timestamps as ISO strings."""
    return {k: iso(v) if isinstance(v, datetime) else v for k, v in row._mapping.items()}
