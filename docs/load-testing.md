# Load testing

How EduPilot behaves with many students at once: where the time goes, what
breaks, and what the overload controls buy. Measured on 2026-11-02 with
`loadtest/locustfile.py`.

## Setup

| | |
|---|---|
| Machine | one Apple Silicon laptop, 10 cores, 24 GB |
| API | 2 processes behind nginx (round robin), local `bge-small-en-v1.5` embedder and `bge-reranker-base` cross-encoder |
| Ingestion | 1 `edupilot-worker` process |
| Data | Postgres 17 + pgvector and Redis 7 in Docker; the real index (`edupilot-v2`, 1,951 vectors, 31 documents) |
| Models | `LLM_SIMULATED=true`: every model call takes 400 ms ± 25% to its first token and streams at 150 tokens/s, and the answers quote the retrieved excerpts, so the guardrails pass them |
| Students | Locust users with 3–8 s of think time. 6 of 10 actions are a streamed question; the rest list conversations, open the Knowledge Base page, or rate an answer. On a 503 they wait the `Retry-After` the server sent |
| Questions | unique per ask (`UNIQUE_QUESTIONS=1`), so nothing is served from the answer cache, except in the last scenario |
| Duration | 90 s per run, all students starting together |

The API ran on the host rather than in containers: torch's arm64 Linux
build, running in Docker's VM on this Mac, reranks about 10× slower than
natively (13 s against ~1.2 s for 48 passages), which would measure the VM
rather than the design. The compose stack (`docker compose up`) was checked
separately: it boots, migrates, load-balances and streams.

Simulated models keep the provider out of the measurement. With Groq's free
tier, a few students would hit its requests-per-minute limit long before
EduPilot's own limits.

## Where a turn's time goes

One student, no contention: about 6 s to a complete answer. Of that:

| Stage | Time |
|---|---|
| pgvector search (dense + BM25, fused) | ~50 ms |
| Cross-encoder rerank, 48 passages | ~1–1.5 s |
| Claim-level grounding check (same cross-encoder) | ~1 s |
| Simulated model calls (router, grader, answer, verifier, ...) | ~2.5 s |
| Everything else (database writes, checkpoints, guardrails) | < 200 ms |

The CPU work is the two cross-encoder passes. Everything that waits on the
network overlaps across students; that work does not.

## Results

Times are seconds. TTFT is time to the first answer token; "answers" are
completed turns in the 90 s window.

**Baseline: no inference gate, no turn limit.** Every question is admitted at
once and every cross-encoder call runs with all 10 cores' threads.

| Students | Answers | TTFT p50 | TTFT p95 | Answer p50 | Answer p95 | Refused |
|---|---|---|---|---|---|---|
| 5 | 14 | 13.0 | 18.0 | 19.0 | 31.0 | 0 |
| 10 | 13 | 28.0 | 36.0 | 44.0 | 58.0 | 0 |
| 20 | 17 | 59.0 | 74.0 | 63.0 | 84.0 | 0 |

Throughput is flat from 5 students on and latency grows with every student
added: concurrent cross-encoder calls do not share the CPU, they fight over
it. Nobody is refused, and everyone waits longer.

**Overload control, 4 turns per process, 8 overall, 15 s admission wait.**

| Students | Answers | TTFT p50 | TTFT p95 | Answer p50 | Answer p95 | Refused |
|---|---|---|---|---|---|---|
| 10 | 30 | 6.8 | 15.0 | 12.0 | 23.0 | 0 |
| 20 | 20 | 27.0 | 35.0 | 39.0 | 50.0 | 18 |
| 40 | 22 | 24.0 | 33.0 | 35.0 | 42.0 | 77 |

**Overload control, 2 turns per process, 4 overall, 8 s wait** (the compose
defaults).

| Students | Answers | TTFT p50 | TTFT p95 | Answer p50 | Answer p95 | Refused |
|---|---|---|---|---|---|---|
| 5 | 21 | 6.0 | 12.0 | 11.0 | 17.0 | 0 |
| 10 | 28 | 9.4 | 13.0 | 15.0 | 19.0 | 9 |
| 20 | 18 | 15.0 | 20.0 | 23.0 | 35.0 | 52 |
| 40 | 20 | 14.0 | 19.0 | 20.0 | 28.0 | 134 |

Every run: 0 failed requests. Session lists, the Knowledge Base page and
ratings stayed under 90 ms at p95 throughout, overload included.

**Repeated questions** (`UNIQUE_QUESTIONS=0`, half of all questions open a new
conversation), 20 students for 150 s, tight limits: 36 answers, TTFT p50 14 s
and p95 21 s, 85 refused, and 5 of 113 cache lookups hit. The cache stores a
question only on its third ask and serves only a conversation's first
question, so it warms slowly; over a semester's worth of repeats it carries
more.

## What the numbers say

- **Capacity is the CPU, and it is about 0.25 answers per second here** (20–30
  per 90 s). Past that, the choice is between everyone waiting longer
  (baseline) and some students being told to retry (overload control).
  Throughput cannot go up without more CPU; tail latency can come down.
- **Serializing cross-encoder calls doubled throughput at 10 students** (13 →
  30 answers) and cut TTFT p50 from 28 s to 7 s. One call at a time with a
  process's share of the cores beats several calls preempting each other.
- **The turn limit sets the tail.** With 4 per process, admitted students at
  20–40 concurrent saw TTFT p95 of 33–35 s; with 2, 19–20 s, at the cost of
  more 503s. Students over capacity get a retry message in under 8 s instead
  of a minute of nothing.
- **Horizontal scaling works the obvious way:** replicas share no memory, so
  a third replica on another machine adds its CPU to the total; the global
  turn limit in Redis keeps the sum within what the model provider allows.

## Problems found, and fixed

1. **CPU contention between cross-encoder calls.** Five students took 94 s to
   a first token. Fixed by the inference gate (`retrieval/inference.py`): one
   call at a time per process, `INFERENCE_THREADS` cores each.
2. **Refused turns still did work.** A turn refused after its admission wait
   had already saved the question (leaving an unanswered message in the
   conversation) and embedded it for the cache lookup. The slot is now taken
   first.
3. **Leaked slots on cancellation.** A request cancelled while waiting for a
   global slot (the client gave up) never returned its local slot and could
   leave its Redis lease behind, so capacity shrank run after run. Both are
   now released whatever the cancellation path, and
   `test_a_turn_cancelled_while_waiting_gives_every_slot_back` pins it.

## Next levers

- Fewer passages to the cross-encoder: 24 instead of 48 costs 0.8 points of
  keyword recall on the evaluation cases for ~22% faster retrieval. A
  measured trade, not a free one.
- A hosted reranker (`RERANKER_MODEL=bge-reranker-v2-m3`) takes the CPU out of
  the turn, after which the provider's rate limit is the ceiling.
- An int8-quantized or ONNX cross-encoder.

## Running it

```bash
LLM_SIMULATED=true docker compose --profile metrics up -d --build
uv run --group load locust -f loadtest/locustfile.py --host http://localhost:8088
```

Locust's UI is at <http://localhost:8089>, Grafana's EduPilot dashboard at
<http://localhost:3001>. Headless, with results as CSV:

```bash
UNIQUE_QUESTIONS=1 uv run --group load locust -f loadtest/locustfile.py \
  --host http://localhost:8088 --headless -u 20 -r 20 -t 90s --csv results/u20
```
