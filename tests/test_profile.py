"""
Student learning profiles.

The update rules are pure functions; persistence is tested against LangGraph's
in-memory store, which implements the same interface as the SQLite one.
"""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from edupilot.agents.profile import (
    MAX_TOPICS,
    clear_profile,
    empty_profile,
    format_profile,
    load_profile,
    save_profile,
    update_profile,
    weak_topics,
)

PVALUES = {"name": "p-values", "domain": "ALGO"}
ATTENTION = {"name": "attention", "domain": "LLM"}


def test_a_topic_asked_once_is_not_a_weakness():
    profile = update_profile(empty_profile(), topics=[PVALUES])
    assert profile["topics"]["ALGO|p-values"]["asks"] == 1
    assert weak_topics(profile) == []
    assert format_profile(profile) == ""


def test_returning_to_a_topic_and_confusion_both_count():
    profile = empty_profile()
    profile = update_profile(profile, topics=[PVALUES], now=1)
    profile = update_profile(profile, topics=[PVALUES], now=2)
    profile = update_profile(profile, topics=[ATTENTION], confusion=True, now=3)

    weak = weak_topics(profile)
    # Confusion (weight 2) outranks a single repeat (weight 1).
    assert [t["name"] for t in weak] == ["attention", "p-values"]


def test_topic_keys_ignore_case():
    profile = update_profile(empty_profile(), topics=[{"name": "P-Values", "domain": "algo"}])
    profile = update_profile(profile, topics=[PVALUES])
    assert list(profile["topics"]) == ["ALGO|p-values"]
    assert profile["topics"]["ALGO|p-values"]["asks"] == 2


def test_a_depth_preference_sticks_until_changed():
    profile = update_profile(empty_profile(), topics=[], depth="concise")
    profile = update_profile(profile, topics=[], depth=None)
    assert profile["depth"] == "concise"
    profile = update_profile(profile, topics=[], depth="detailed")
    assert profile["depth"] == "detailed"
    profile = update_profile(profile, topics=[], depth="sideways")
    assert profile["depth"] == "detailed", "unknown depths are ignored"


def test_the_profile_is_bounded():
    profile = empty_profile()
    for i in range(MAX_TOPICS + 10):
        profile = update_profile(profile, topics=[{"name": f"topic {i}", "domain": "AML"}], now=i)
    assert len(profile["topics"]) == MAX_TOPICS
    assert "AML|topic 0" not in profile["topics"], "the least recently seen go first"


def test_the_prompt_block_names_weak_topics_and_denies_itself_evidentiary_weight():
    profile = update_profile(empty_profile(), topics=[ATTENTION], confusion=True, depth="concise")
    block = format_profile(profile)
    assert "<student_profile>" in block
    assert "attention (LLM)" in block
    assert "concise" in block
    assert "NOT evidence" in block


def test_update_does_not_mutate_its_input():
    original = empty_profile()
    update_profile(original, topics=[PVALUES])
    assert original == empty_profile()


async def test_profiles_round_trip_through_the_store_per_student():
    store = InMemoryStore()
    profile = update_profile(empty_profile(), topics=[PVALUES])
    await save_profile(store, "student-a", profile)

    assert (await load_profile(store, "student-a"))["topics"] == profile["topics"]
    assert (await load_profile(store, "student-b"))["topics"] == {}, "profiles are per student"

    await clear_profile(store, "student-a")
    assert (await load_profile(store, "student-a"))["topics"] == {}


async def test_no_store_or_no_user_means_an_empty_profile():
    assert await load_profile(None, "student-a") == empty_profile()
    assert await load_profile(InMemoryStore(), None) == empty_profile()
