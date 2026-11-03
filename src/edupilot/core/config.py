"""
EduPilot Configuration
======================
Central configuration for all modules. Loads secrets from .env.
Edit DOMAINS here to add new course domains — no other code changes needed.
"""

import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Must be set before the HuggingFace `tokenizers` Rust extension initializes.
#
# This pipeline touches the tokenizer from two places in one process: the
# chunker counts tokens to size chunks, and the embedder tokenizes to encode.
# `tokenizers` parallelizes with a rayon thread pool, and re-entering that pool
# from a second call site deadlocks — the process parks in `__psynch_cvwait`
# with zero CPU and never recovers. Observed here as a rebuild that hung
# indefinitely after loading the embedding model.
#
# The lost parallelism is irrelevant: encoding is already batched, and the
# batch loop is where the real throughput comes from.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Load .env (python-dotenv — safe no-op if file is absent)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    # Walk up from this file to the repo root looking for .env. The extra
    # levels versus the flat layout are src/ and edupilot/.
    _here = Path(__file__).resolve().parent
    for _candidate in [_here, *_here.parents[:4]]:
        if (_candidate / ".env").exists():
            load_dotenv(_candidate / ".env")
            break
except ImportError:
    pass  # dotenv not installed; rely on shell environment

# ---------------------------------------------------------------------------
# Project paths
#
# Code and data are separated: PACKAGE_DIR holds importable Python and the
# bundled web assets, DATA_DIR holds everything the application reads or
# writes at runtime. Only DATA_DIR needs to be writable, and it is the only
# path that needs mounting when this runs in a container.
# ---------------------------------------------------------------------------

#: The installed package — src/edupilot. Never written to.
PACKAGE_DIR: Path = Path(__file__).resolve().parent.parent

#: Repo root. Correct for an editable/source checkout; irrelevant once
#: EDUPILOT_DATA_DIR is set, which is how a real deployment should run.
PROJECT_ROOT: Path = Path(os.getenv("EDUPILOT_PROJECT_ROOT", str(PACKAGE_DIR.parent.parent)))

#: Files the app reads: the course corpus, for rebuilds and for opening a cited
#: PDF. Everything else, student uploads included, lives in Postgres.
DATA_DIR: Path = Path(os.getenv("EDUPILOT_DATA_DIR", str(PROJECT_ROOT / "data")))

KNOWLEDGE_BASE_DIR: Path = DATA_DIR / "knowledge_base"

#: Bundled single-page frontend, served at /static.
STATIC_DIR: Path = PACKAGE_DIR / "web" / "static"

# ---------------------------------------------------------------------------
# API keys (read from environment / .env)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str      = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str    = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY: str  = os.getenv("PINECONE_API_KEY", "")

# ---------------------------------------------------------------------------
# Tracing (LangSmith)
# ---------------------------------------------------------------------------
# Off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set. The LangChain
# and LangGraph runtimes read those variables themselves; this module only
# names the project. Traces contain students' questions and the course
# excerpts — set LANGSMITH_HIDE_INPUTS / LANGSMITH_HIDE_OUTPUTS=true to keep
# the run structure, timings and token counts without the text.
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "edupilot")
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

# ---------------------------------------------------------------------------
# Database
#
# One Postgres holds everything the app keeps: accounts and conversations, the
# document registry, the vectors (pgvector), LangGraph checkpoints and student
# profiles, and the embedding and answer caches. Nothing that matters lives on
# local disk, so any number of API processes can serve from the same database.
#
# Any libpq URL works (`postgres://`, `postgresql://`, or SQLAlchemy's
# `postgresql+psycopg://`); db/engine.py adapts it for each client. The default
# is the database `docker compose up -d db` starts, published on port 5434.
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://edupilot:edupilot@localhost:5434/edupilot"
)

#: Connections each API process keeps open for request work, plus how many more
#: it may open under a burst. Managed Postgres plans cap total connections
#: (often 25-100), so size these against processes x (pool + overflow) plus the
#: LangGraph pool below.
DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
#: Seconds a request waits for a free connection before failing.
DB_POOL_TIMEOUT: float = float(os.getenv("DB_POOL_TIMEOUT", "10"))
#: Async connections for LangGraph checkpoints and profiles, per process.
CHECKPOINT_POOL_SIZE: int = int(os.getenv("CHECKPOINT_POOL_SIZE", "10"))

# ---------------------------------------------------------------------------
# Vector store
#
# `pgvector` keeps vectors in the same Postgres as everything else; `pinecone`
# uses a Pinecone serverless index instead. Either way an index is versioned:
# concrete indexes are "<base>-v1", "<base>-v2", ..., the index_versions table
# records which one serves traffic, and a rebuild writes a new version that is
# promoted only on success (blue/green), so a failed re-index can never leave
# the live index half-populated.
# ---------------------------------------------------------------------------
VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "pgvector").lower()
INDEX_BASE: str = os.getenv("INDEX_BASE", os.getenv("PINECONE_INDEX_BASE", "edupilot"))

# Pinecone backend only. Native sparse-dense hybrid search requires the
# dotproduct metric; with L2-normalized embeddings the dot product equals
# cosine similarity.
PINECONE_CLOUD: str  = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "dotproduct")

# ---------------------------------------------------------------------------
# Scale-out: shared state, backpressure, background work, metrics
# ---------------------------------------------------------------------------

#: Redis, shared by every API process for rate limits and the turn limiter.
#: Unset, each process keeps those in memory — right for one process, and the
#: limits multiply by the process count otherwise.
REDIS_URL: str = os.getenv("REDIS_URL", "")

#: Turns (one question answered) a process runs at once. Each turn makes
#: several model calls and some CPU-bound retrieval, so this is what bounds
#: provider concurrency and CPU contention per process. With the local
#: reranker, a process gets through about one turn's inference at a time, and
#: load tests (docs/load-testing.md) found 2-4 the range where latency stays
#: bounded; with hosted models the CPU is idle and 8 or more is fine.
MAX_CONCURRENT_TURNS: int = int(os.getenv("MAX_CONCURRENT_TURNS", "4"))
#: Turns allowed at once across every process sharing REDIS_URL — the budget
#: the model provider's rate limit can actually sustain. 0 disables it.
GLOBAL_MAX_TURNS: int = int(os.getenv("GLOBAL_MAX_TURNS", "0"))
#: Seconds a turn may wait for a slot before the API answers 503 with
#: Retry-After, rather than letting a backlog grow without bound.
ADMISSION_TIMEOUT_SECONDS: float = float(os.getenv("ADMISSION_TIMEOUT_SECONDS", "10"))

#: Where uploads are indexed: "embedded" runs the ingestion worker inside each
#: API process (one process, or local development); "external" leaves it to
#: separate `edupilot-worker` processes.
INGEST_WORKER: str = os.getenv("INGEST_WORKER", "embedded").lower()

#: Cross-encoder calls (reranking, the grounding check) a process runs at once,
#: and torch threads per call. CPU inference contends rather than shares, so
#: one call at a time using every core beats several fighting over them; see
#: retrieval/inference.py. 0 threads keeps torch's default of one per core.
INFERENCE_CONCURRENCY: int = int(os.getenv("INFERENCE_CONCURRENCY", "1"))
INFERENCE_THREADS: int = int(os.getenv("INFERENCE_THREADS", "0"))

#: When set, /metrics requires `Authorization: Bearer <token>`.
METRICS_TOKEN: str = os.getenv("METRICS_TOKEN", "")

#: Replace every model call with a simulated one (fixed latency, streamed
#: tokens). For load tests, which should measure this system rather than a
#: provider's rate limit. Refused in production.
LLM_SIMULATED: bool = os.getenv("LLM_SIMULATED", "false").lower() == "true"
SIMULATED_LLM_LATENCY_MS: int = int(os.getenv("SIMULATED_LLM_LATENCY_MS", "400"))
SIMULATED_LLM_TOKENS_PER_SECOND: int = int(os.getenv("SIMULATED_LLM_TOKENS_PER_SECOND", "150"))

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
ENV: str = os.getenv("EDUPILOT_ENV", "development").lower()
IS_PRODUCTION: bool = ENV == "production"

# ---------------------------------------------------------------------------
# Open mode
#
# With auth disabled there is no sign-in. Access control is not removed: each
# browser is issued its own anonymous identity (security/deps.anonymous_user),
# sessions are still written with an owner, and every ownership check still
# runs. What goes away is only the login step.
#
# This was previously refused outright in production, because "auth off" then
# meant every caller shared one *admin* identity — which let any visitor upload
# into the shared knowledge base and read everyone else's conversations. Both
# are now addressed directly: identities are per-browser, and anonymous callers
# are granted the admin role only outside production. So the mode is allowed in
# production, and the role boundary carries the safety instead of a blanket ban.
#
# Requiring sign-in is still the stronger posture and remains the production
# default; turn it off deliberately when the deployment has no login UI.
# ---------------------------------------------------------------------------
AUTH_REQUIRED: bool = (
    os.getenv("EDUPILOT_AUTH_REQUIRED", "true" if IS_PRODUCTION else "false").lower()
    == "true"
)

# Signing key for JWTs. Required in production; security/auth.py raises at
# startup rather than falling back to a default, because a predictable signing
# key lets anyone mint an admin token.
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")

# Allowed browser origins. The previous config used allow_origins=["*"], which
# lets any website call the API with the user's cookies attached.
_origins_raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
CORS_ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _origins_raw.split(",") if o.strip()]
    if _origins_raw
    else ["http://localhost:8000", "http://127.0.0.1:8000"]
)

# Only honour X-Forwarded-For when actually behind a trusted reverse proxy.
# The header is client-controlled, so trusting it otherwise lets a caller
# forge a new identity per request and bypass rate limiting entirely.
TRUST_PROXY_HEADERS: bool = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"

# Longest accepted student question. Unbounded input is unbounded prompt cost.
MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "4000"))

# Bootstrap admin, created on first startup when no users exist. Without this
# there is no way to reach the admin-only knowledge-base routes.
BOOTSTRAP_ADMIN_EMAIL: str = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "")
BOOTSTRAP_ADMIN_PASSWORD: str = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")

# ---------------------------------------------------------------------------
# Domain registry
# Each domain has its own isolated RAG pipeline (its own vector-store namespace).
# To add a new domain, add an entry here — no other core code changes needed.
# ---------------------------------------------------------------------------
DOMAINS: dict[str, dict] = {
    "AML": {
        "name": "Applied Machine Learning",
        "abbr": "AML",
        "color": "#4CAF50",
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "aml"),
        "namespace": "aml",
        "description": (
            "Supervised and unsupervised learning, bias-variance tradeoff, overfitting, "
            "regularization, neural networks and CNNs, decision trees, SVMs, clustering, "
            "dimensionality reduction, model evaluation, feature engineering, "
            "autoencoders, GANs and diffusion models."
        ),
        "keywords": [
            "machine learning", "ML", "neural network", "deep learning",
            "bias", "variance", "overfitting", "underfitting", "regularization",
            "gradient descent", "classification", "regression", "clustering",
            "random forest", "SVM", "cross-validation", "feature engineering",
        ],
        "example_question": "Why does L2 regularization shrink weights?",
        "example_topic": "the bias-variance tradeoff",
    },
    "ALGO": {
        "name": "Applied Algorithms",
        "abbr": "ALGO",
        "color": "#2196F3",
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "applied_algorithms"),
        "namespace": "algo",
        "description": (
            "Algorithm analysis and asymptotic notation (Big-O, Theta), recurrences "
            "and the master method, divide and conquer, sorting (insertion, merge, "
            "quick, heap, counting, radix, bucket), randomized algorithms, dynamic "
            "programming, greedy algorithms, and basic data structures."
        ),
        "keywords": [
            "algorithm", "Big-O", "asymptotic", "time complexity", "running time",
            "recurrence", "master method", "divide and conquer", "merge sort",
            "quicksort", "heapsort", "heap", "insertion sort", "counting sort",
            "radix sort", "bucket sort", "dynamic programming", "memoization",
            "rod cutting", "matrix-chain", "greedy", "activity selection", "Huffman",
            "randomized", "hiring problem", "linked list", "stack", "queue",
        ],
        "example_question": "Why is merge sort O(n log n)?",
        "example_topic": "dynamic programming",
    },
    "LLM": {
        "name": "Large Language Models",
        "abbr": "LLM",
        "color": "#9C27B0",
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "llm"),
        "namespace": "llm",
        "description": (
            "Transformer architecture, attention and positional encoding, tokenization "
            "and embeddings, pretraining, instruction tuning, RLHF, DPO, LoRA, prompting "
            "techniques, RAG pipelines, LLM agents, hallucination, quantization, and LLM "
            "evaluation."
        ),
        "keywords": [
            "LLM", "large language model", "transformer", "attention", "GPT", "BERT",
            "LLaMA", "Mistral", "fine-tuning", "LoRA", "QLoRA", "RLHF", "DPO",
            "prompt", "chain of thought", "RAG", "retrieval augmented generation",
            "agent", "hallucination", "tokenization", "embedding", "pretraining",
            "instruction tuning", "in-context learning", "few-shot", "zero-shot",
            "quantization", "KV cache", "context window", "perplexity",
        ],
        "example_question": "How does multi-head attention differ from single-head?",
        "example_topic": "attention in transformers",
    },
}


def _course_folder(path: str) -> str:
    """
    A course folder as it exists on disk.

    Folders copied from a course site arrive as `AML/` or `Applied Algorithms/`
    rather than `aml/` or `applied_algorithms/`. macOS would find the first
    either way; Linux, where the container runs, finds neither. So a folder
    whose name matches ignoring case, spaces, underscores and hyphens counts.
    """
    folder = Path(path)
    if folder.is_dir() or not folder.parent.is_dir():
        return path
    key = re.sub(r"[^a-z0-9]", "", folder.name.lower())
    matches = [
        p for p in folder.parent.iterdir()
        if p.is_dir() and re.sub(r"[^a-z0-9]", "", p.name.lower()) == key
    ]
    return str(matches[0]) if len(matches) == 1 else path


for _domain in DOMAINS.values():
    _domain["knowledge_base_path"] = _course_folder(_domain["knowledge_base_path"])

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
# Evidence budget. DEFAULT_RERANK_TOP_K is the single strongest lever on
# answer depth: it is how many chunks the generator actually sees. At 3-5 the
# model has too little to be detailed from, and the grounding rules correctly
# stop it inventing the difference — producing short, paraphrase-like answers.
DEFAULT_TOP_K: int = 12          # candidates retrieved before reranking
DEFAULT_RERANK_TOP_K: int = 8    # chunks handed to the generator


def _optional_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    return None if raw.strip().lower() in ("", "none", "off", "-1") else int(raw)


#: Of those chunks, how many may come from a textbook (a file under a course's
#: book/ subfolder) while the course's own material can supply the rest. A
#: book is used beyond this only to fill slots the lectures cannot. "none"
#: turns the limit off. See retrieval/hybrid.py:compose_evidence.
MAX_BOOK_EXCERPTS: int | None = _optional_int("MAX_BOOK_EXCERPTS", 2)
#: The same limit for any one file; off by default.
MAX_EXCERPTS_PER_DOCUMENT: int | None = _optional_int("MAX_EXCERPTS_PER_DOCUMENT", None)

# Chunking is now measured in embedding-model TOKENS, not words. The previous
# 800-*word* setting was applied per PDF page, so it almost never bound: 46% of
# chunks still overflowed the 256-token window of the old embedder and were
# silently truncated, discarding 47.9% of the indexed corpus.
CHUNK_MAX_TOKENS: int = 448      # hard ceiling, under the 512-token window
CHUNK_TARGET_TOKENS: int = 320   # preferred size
CHUNK_MIN_TOKENS: int = 64       # below this, merge into a neighbour
CHUNK_OVERLAP_SENTENCES: int = 1
PARENT_MAX_TOKENS: int = 1400    # wider window for small-to-big retrieval

# Hybrid search blend lives with the fusion code that applies it — see
# PROBE_WEIGHTS and RRF_K in retrieval/hybrid.py. Reciprocal Rank Fusion
# weights each *probe* (dense, sparse, HyDE, multi-query), not two fixed
# modalities, so a pair of scalars here could not express it.

# Minimum cross-encoder relevance for a chunk to count as evidence. Below this
# the retriever returns nothing and the agent refuses — answering off
# low-relevance context is how ungrounded answers happen.
#
# The scale is model-specific, so this is calibrated per reranker in
# retrieval/rerank.py rather than fixed here. Leave as None to use the
# calibrated default for the configured RERANKER_MODEL.
DEFAULT_MIN_RELEVANCE: float | None = None

# Query transformation. Acronym expansion is deterministic and always on; the
# two LLM-backed steps each cost a round-trip.
ENABLE_MULTI_QUERY: bool = os.getenv("ENABLE_MULTI_QUERY", "false").lower() == "true"
ENABLE_HYDE: bool = os.getenv("ENABLE_HYDE", "false").lower() == "true"
ENABLE_PARENT_EXPANSION: bool = True

# ---------------------------------------------------------------------------
# Agentic behaviour
# ---------------------------------------------------------------------------
# Each of these trades latency and tokens for answer quality, so each can be
# turned down per deployment.

# Corrective retrieval: grade each domain's excerpts with an LLM, and when none
# actually address the question, rephrase the search in the course's own
# terminology and retry — up to MAX_QUERY_REWRITES times — before refusing.
CORRECTIVE_RETRIEVAL: bool = os.getenv("CORRECTIVE_RETRIEVAL", "true").lower() == "true"
MAX_QUERY_REWRITES: int = int(os.getenv("MAX_QUERY_REWRITES", "1"))

# Verify -> revise: when the verifier finds unsupported or miscited claims,
# rewrite the answer against the same evidence and verify again, up to this
# many times. The better-scoring draft is kept. 0 disables revision.
MAX_REVISIONS: int = int(os.getenv("MAX_REVISIONS", "1"))

# Tutor tools: the answering agent may call the restricted calculator
# (agents/tools.py) for arithmetic and statistics on numbers the student gives.
ENABLE_TUTOR_TOOLS: bool = os.getenv("ENABLE_TUTOR_TOOLS", "true").lower() == "true"

#: Practice items generated when the student does not say how many.
DEFAULT_QUIZ_SIZE: int = 5
MAX_QUIZ_SIZE: int = 10

# Accepted upload types are owned by the ingestion package, which is what
# actually dispatches on them — see ingestion.SUPPORTED_EXTENSIONS.

# ---------------------------------------------------------------------------
# LLM / Embedding models
# ---------------------------------------------------------------------------
# Model IDs: https://console.groq.com/docs/models and
# https://ai.google.dev/gemini-api/docs/models. Providers retire IDs without
# notice (Groq's Llama 3 models and Gemini 2.0 went in 2026), and a retired ID
# fails every call with model_not_found, so check both lists when answers stop.
DEFAULT_MODEL: str = "openai/gpt-oss-120b"   # Groq — free, fast, reliable structured output
VERIFY_MODEL: str  = "openai/gpt-oss-120b"   # Groq — same large model; small models anchor scores

# bge-small-en-v1.5: 384-dim (matches the previous index dimension) with a
# 512-token window — double MiniLM's 256 — and materially stronger retrieval on
# technical text. It is asymmetric: queries get an instruction prefix,
# passages do not. See retrieval/embeddings.py.
#
# Local models need the local-models extra. `llama-text-embed-v2` is hosted on
# Pinecone Inference instead (1024-dim, no torch); an index must be built with
# whichever model serves queries, so changing this means a rebuild.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Reranker for retrieval candidates. Now active on every path; previously the
# Chat pipeline silently used keyword overlap instead. A local cross-encoder,
# or a Pinecone-hosted one (`bge-reranker-v2-m3`) — see retrieval/rerank.py.
# The claim-level grounding check always uses a local cross-encoder, so with
# hosted models and no local-models extra it is skipped and reported as not
# measured; the LLM verifier still audits every answer.
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# Groq counts reserved output tokens against TPM even if the response is shorter.
# Keep these tight — they are upper bounds, not targets.
LLM_MAX_TOKENS_CLASSIFY: int = 512
LLM_MAX_TOKENS_GENERATE: int = 3500   # domain chat answers — high for quality
LLM_MAX_TOKENS_SS: int       = 4096   # self-study: user wants long, detailed answers
LLM_MAX_TOKENS_SYNTH: int    = 3500   # cross-domain synthesis — high for quality
LLM_MAX_TOKENS_VERIFY: int   = 768    # JSON blob with scores

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
ENABLE_VERIFICATION_DEFAULT: bool = True

# ---------------------------------------------------------------------------
# Available models (for sidebar dropdown)
# Groq models are prefixed with "groq:" to distinguish them from Gemini.
# ---------------------------------------------------------------------------
# A retired model stays out: every call to it fails, and it silently spends a
# slot in the fallback chain (gemma2-9b-it, then the Llama 3 models).
AVAILABLE_MODELS: list[str] = [
    # ── Groq (free tier, fast) ──────────────────────────────────────────────
    "openai/gpt-oss-120b",        # best quality
    "qwen/qwen3.8-27b",           # fast, strong at structured output
    "openai/gpt-oss-20b",         # fastest; weaker at routing
    # ── Gemini (Google AI Studio) ───────────────────────────────────────────
    "gemini-3.5-flash",           # best Gemini quality
    "gemini-3.5-flash-lite",      # fastest Gemini, highest free quota
]

# Groq models — used to detect provider in call_llm
GROQ_MODELS: list[str] = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]
