"""
EduPilot load test
==================
Simulated students against a running stack.

    LLM_SIMULATED=true docker compose up -d --build
    uv run --group load locust -f loadtest/locustfile.py --host http://localhost:8088

or headless, for a fixed number of students:

    uv run --group load locust -f loadtest/locustfile.py --host http://localhost:8088 \\
        --headless -u 20 -r 2 -t 3m --csv results/u20

Run the stack with LLM_SIMULATED=true: against the real providers this
measures their rate limits, not EduPilot. Retrieval, reranking, guardrails,
checkpoints and every database write still run for real.

Each student gets their own anonymous identity (a cookie, as a browser
would), so per-user rate limits apply per student. What is reported besides
Locust's usual request stats:

  SSE  first token      question sent -> first answer token streamed
  SSE  answer complete  question sent -> final result event
  SSE  overloaded       turns refused with 503 (the system shedding load)

UNIQUE_QUESTIONS=1 makes every question unique, so the answer cache never
hits and every turn runs the full pipeline. Without it, questions repeat the
way a class's do, and the cache serves the popular ones. NEW_CONVERSATION
(default 0.5) is how often a question starts a new conversation.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid

from locust import HttpUser, between, events, task

QUESTIONS = [
    # Applied Algorithms
    "Why is merge sort O(n log n)?",
    "What is the difference between Big-O and Theta notation?",
    "How does quicksort partition an array around a pivot?",
    "What are the best and worst cases of insertion sort?",
    "How does the master method solve recurrences?",
    "How does dynamic programming solve the rod cutting problem?",
    "What is the difference between memoization and bottom-up dynamic programming?",
    "How does the greedy activity selection algorithm work?",
    "How are Huffman codes constructed?",
    "How does heapsort use a max-heap?",
    "How do counting sort and radix sort run in linear time?",
    "How is a queue implemented with a linked list?",
    # Large Language Models
    "How does self-attention work in a transformer?",
    "What is the difference between RLHF and DPO?",
    "How does retrieval-augmented generation reduce hallucination?",
    "What is LoRA and why is it parameter efficient?",
    "What is chain-of-thought prompting?",
    "How are LLM agents able to use tools?",
    "What chunking strategies work best for RAG?",
    "What happens during pretraining of a language model?",
    # Applied Machine Learning
    "What is the bias-variance tradeoff?",
    "How does logistic regression make a classification?",
    "How does a random forest reduce overfitting?",
    "What is BERT's masked language modelling objective?",
    "Why does the Transformer use multi-head attention?",
]

UNIQUE = os.getenv("UNIQUE_QUESTIONS", "") == "1"

#: Chance a question opens a new conversation rather than following up. Only a
#: conversation's first question can be served from the answer cache, since a
#: follow-up ("explain that more simply") depends on what came before.
NEW_CONVERSATION = float(os.getenv("NEW_CONVERSATION", "0.5"))


def _fire(name: str, started: float, *, error: str | None = None) -> None:
    events.request.fire(
        request_type="SSE",
        name=name,
        response_time=(time.perf_counter() - started) * 1000,
        response_length=0,
        exception=Exception(error) if error else None,
        context={},
    )


class Student(HttpUser):
    """Asks a question, reads the answer, sometimes looks around or rates it."""

    # Reading an answer and typing the next question.
    wait_time = between(3, 8)

    def on_start(self) -> None:
        self.session_id: str | None = None
        self.answers: list[int] = []
        self.client.get("/api/config", name="/api/config")

    @task(6)
    def ask(self) -> None:
        question = random.choice(QUESTIONS)
        if UNIQUE:
            question = f"{question} ({uuid.uuid4().hex[:6]})"
        if random.random() < NEW_CONVERSATION:
            self.session_id = None
        body = {"query": question, "session_id": self.session_id}

        started = time.perf_counter()
        first_token = False
        with self.client.post(
            "/api/chat/stream", json=body, stream=True, catch_response=True, name="/api/chat/stream"
        ) as response:
            if response.status_code == 503:
                response.success()  # shedding load is the designed behaviour, not an error
                _fire("overloaded", started)
                # As the frontend does: wait as long as the server asked before
                # trying again, rather than adding to the queue at once.
                time.sleep(float(response.headers.get("Retry-After", 5)))
                return
            if response.status_code == 429:
                response.success()
                _fire("rate limited", started)
                return
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            event = None
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("event:"):
                    event = raw[6:].strip()
                    if event == "token" and not first_token:
                        first_token = True
                        _fire("first token", started)
                    continue
                if not raw.startswith("data:"):
                    continue
                if event == "result":
                    result = json.loads(raw[5:])
                    self.session_id = result.get("session_id") or self.session_id
                    if result.get("assistant_message_id"):
                        self.answers.append(result["assistant_message_id"])
                    _fire("answer complete", started)
                    response.success()
                    return
                if event == "error":
                    response.failure(f"stream error: {raw[5:][:120]}")
                    _fire("answer complete", started, error="stream error")
                    return
            response.failure("stream ended without a result")

    @task(2)
    def browse_history(self) -> None:
        self.client.get("/api/sessions", name="/api/sessions")
        if self.session_id:
            self.client.get(f"/api/sessions/{self.session_id}", name="/api/sessions/{session_id}")

    @task(1)
    def knowledge_base(self) -> None:
        self.client.get("/api/kb/status", name="/api/kb/status")

    @task(1)
    def rate_an_answer(self) -> None:
        if not self.answers:
            return
        self.client.post(
            "/api/feedback",
            json={"kind": "chat", "message_id": random.choice(self.answers), "rating": random.choice([1, 1, 1, -1])},
            name="/api/feedback",
        )
