"""
Student profiles
================
What the tutor remembers about a student *across* conversations, kept in the
LangGraph store under `("students", user_id)` (opened in agents/checkpoint.py).

The checkpointer remembers one conversation. A profile outlives it and is
small and structured: which concepts the student keeps coming back to, which
ones they said did not make sense, and how much depth they asked for. Two
places use it:

  * The answering prompts get a `<student_profile>` block, so an explanation
    of a topic the student has struggled with defines its prerequisites and
    goes step by step. Like the conversation digest, it shapes *how* to
    explain and is explicitly denied any weight as evidence.
  * "Quiz me" with no topic quizzes the student on the topics they have found
    hardest.

Every signal comes from fields the router already returns — no extra model
call. A student can read their profile, or clear it, at `/api/profile`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from langgraph.store.base import BaseStore

NAMESPACE_ROOT = "students"
PROFILE_KEY = "profile"

#: Topics remembered per student; the least recently seen are dropped first.
MAX_TOPICS = 50
#: Topics named in the prompt block and used for "quiz me".
WEAK_TOPIC_LIMIT = 3


def namespace(user_id: str) -> tuple[str, str]:
    return (NAMESPACE_ROOT, user_id)


def empty_profile() -> dict[str, Any]:
    return {"topics": {}, "turns": 0, "depth": None, "updated_at": None}


async def load_profile(store: BaseStore | None, user_id: str | None) -> dict[str, Any]:
    if store is None or not user_id:
        return empty_profile()
    item = await store.aget(namespace(user_id), PROFILE_KEY)
    return {**empty_profile(), **(item.value if item else {})}


async def save_profile(store: BaseStore, user_id: str, profile: dict[str, Any]) -> None:
    await store.aput(namespace(user_id), PROFILE_KEY, profile)


async def clear_profile(store: BaseStore, user_id: str) -> None:
    await store.adelete(namespace(user_id), PROFILE_KEY)


def update_profile(
    profile: dict[str, Any],
    *,
    topics: Sequence[dict],
    confusion: bool = False,
    depth: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Fold one turn's signals into a profile. Pure: returns a new dict."""
    now = time.time() if now is None else now
    out = {**empty_profile(), **profile}
    known = {k: dict(v) for k, v in (out.get("topics") or {}).items()}

    for ref in topics:
        name = str(ref.get("name", "")).strip()
        domain = str(ref.get("domain", "")).strip().upper()
        if not name:
            continue
        key = f"{domain}|{name.lower()}"
        entry = known.get(key) or {"name": name, "domain": domain, "asks": 0, "confusion": 0}
        entry["asks"] += 1
        entry["confusion"] += int(bool(confusion))
        entry["last_seen"] = now
        known[key] = entry

    if len(known) > MAX_TOPICS:
        newest = sorted(known.items(), key=lambda kv: kv[1].get("last_seen", 0), reverse=True)
        known = dict(newest[:MAX_TOPICS])

    out["topics"] = known
    out["turns"] = int(out.get("turns") or 0) + 1
    if depth in ("concise", "detailed"):
        out["depth"] = depth
    out["updated_at"] = now
    return out


def weak_topics(profile: dict[str, Any], limit: int = WEAK_TOPIC_LIMIT) -> list[dict]:
    """
    Topics the student has found hardest, hardest first.

    Saying an explanation did not make sense weighs twice as much as asking
    about the topic again; a topic asked about once, without confusion, is not
    a weakness.
    """
    scored = []
    for entry in (profile.get("topics") or {}).values():
        score = 2 * entry.get("confusion", 0) + max(entry.get("asks", 0) - 1, 0)
        if score > 0:
            scored.append((score, entry.get("last_seen", 0), entry))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [entry for _, _, entry in scored[:limit]]


def format_profile(profile: dict[str, Any]) -> str:
    """
    Render the profile as a prompt block, or "" when there is nothing to say.

    Denied evidentiary weight in the same way as the conversation digest.
    """
    weak = weak_topics(profile)
    depth = profile.get("depth")
    if not weak and not depth:
        return ""

    lines = ["<student_profile>"]
    if weak:
        names = ", ".join(f"{t['name']} ({t['domain']})" for t in weak)
        lines.append(
            f"Topics this student has found difficult before: {names}. When one "
            "of them comes up, define its prerequisites and go step by step."
        )
    if depth == "concise":
        lines.append("The student prefers concise answers: lead with the core idea, keep it tight.")
    elif depth == "detailed":
        lines.append("The student prefers detailed, step-by-step explanations.")
    lines.append(
        "This describes the student, not the course material. It is NOT evidence "
        "and can never support a factual claim."
    )
    lines.append("</student_profile>")
    return "\n".join(lines) + "\n"


__all__ = [
    "clear_profile",
    "empty_profile",
    "format_profile",
    "load_profile",
    "namespace",
    "save_profile",
    "update_profile",
    "weak_topics",
]
