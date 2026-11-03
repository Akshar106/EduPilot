"""
The caller's learning profile: what the tutor has learned about them.

Students can see exactly what is kept — topics they have asked about, which of
those they found hard, their preferred depth — and wipe it. The profile only
ever shapes how explanations are pitched and which topics "quiz me" picks.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from edupilot.api.deps import CurrentUser
from edupilot.core.services import services
from edupilot.security import rate_limited

router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("", dependencies=[Depends(rate_limited("read"))])
async def get_profile(user: CurrentUser):
    from edupilot.agents.profile import load_profile, weak_topics

    profile = await load_profile(services.agent.store, user.user_id)
    topics = sorted(
        profile["topics"].values(), key=lambda t: t.get("last_seen", 0), reverse=True
    )
    return {
        "topics": topics,
        "weak_topics": weak_topics(profile),
        "depth": profile["depth"],
        "turns": profile["turns"],
        "updated_at": profile["updated_at"],
    }


@router.delete("")
async def clear_profile(user: CurrentUser):
    from edupilot.agents.profile import clear_profile as clear

    if services.agent.store is not None:
        await clear(services.agent.store, user.user_id)
    return {"cleared": True}
