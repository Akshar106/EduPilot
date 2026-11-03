# Deploying EduPilot

The container targets Hugging Face Spaces (Docker SDK) but is plain Docker, so
it runs unchanged on Fly, Render, or any VPS.

```
Dockerfile              image: CPU torch, models baked in, non-root, port 7860
.dockerignore           keeps data/, .env and the venv out of the build context
deploy/entrypoint.sh    checks DATABASE_URL, then execs uvicorn as PID 1
deploy/nginx.conf       the load balancer in front of the compose stack's replicas
deploy/prometheus.yml   scrapes every replica and the worker
deploy/grafana/         the provisioned EduPilot dashboard
deploy/huggingface/     the Space's own README (frontmatter config)
```

## What the deployment needs, and why

**Postgres 15+ with the pgvector extension (0.8 or newer).** It holds
everything the app keeps: accounts and conversations, the vector index and
which version of it is live, the BM25 table, LangGraph checkpoints and student
profiles, and the embedding and answer caches. Set `DATABASE_URL` to it; the
app runs its migrations at startup (several instances starting together
migrate once, under an advisory lock). Managed options with pgvector: Neon,
Supabase, Fly Postgres, AWS RDS, Google Cloud SQL.

Size connections against the plan's limit: each process opens up to
`DB_POOL_SIZE + DB_MAX_OVERFLOW` (10 + 10) synchronous connections plus
`CHECKPOINT_POOL_SIZE` (10) async ones.

**Redis, once there is more than one API process.** Rate limits and the
global turn limit (`GLOBAL_MAX_TURNS`) are shared through it, so a student's
limit is the same on every instance. Set `REDIS_URL` (Upstash and Fly's Redis
work). Without it each process limits on its own, which is right for one
process. If Redis goes away, requests are let through rather than refused.

**Somewhere to index uploads.** By default each API process runs the
ingestion worker in a thread (`INGEST_WORKER=embedded`). On a larger
deployment set `INGEST_WORKER=external` and run `edupilot-worker` as its own
process (a Fly process group, a second container) so indexing never shares a
CPU with answering. Uploads travel through Postgres, so the worker needs no
shared disk.

**Overload settings sized for the hardware.** `MAX_CONCURRENT_TURNS`,
`GLOBAL_MAX_TURNS` and `INFERENCE_THREADS` decide how many questions run at
once before the API starts answering 503 with `Retry-After`. The defaults suit
the in-process reranker; [docs/load-testing.md](../docs/load-testing.md)
shows how they were chosen and how to measure your own. Prometheus can scrape
`/metrics` (set `METRICS_TOKEN` to require a bearer token).

**~1 GB of RAM, minimum 2 GB.** Measured resident set with both models loaded
and serving:

```
baseline python              15 MB
+ edupilot imported          91 MB
+ embedder (bge-small)      551 MB
+ reranker (bge-reranker)  1065 MB
```

This rules out 512 MB free tiers. Hugging Face's free CPU tier (16 GB) is
comfortable.

**The models are baked into the image.** Together they are ~1.2 GB. Downloading
them on first request means the first visitor waits minutes and every restart
re-downloads. `HF_HOME` is set identically at build and run time; changing one
without the other silently re-downloads everything.

**The corpus is not shipped.** `data/knowledge_base` is ~91 MB of copyrighted
lecture material. The vectors and chunk text live in Postgres, so answers and
citations work normally; only downloading an original PDF from a citation does
not, and that route says so rather than returning a bare 404.

## Access model

The bundled frontend has no login screen, so the image sets
`EDUPILOT_AUTH_REQUIRED=false`. That does **not** mean access control is off:

- Each browser is issued its own anonymous identity in a long-lived cookie.
  One visitor cannot list or open another's conversations.
- Anonymous callers are **students** in production, so uploading to or deleting
  from the shared knowledge base is refused (403). Manage the corpus with
  `edupilot-reindex`.
- The cookie authorizes nothing beyond "these are my chats", so it is not
  signed. Forging one yields a different, empty account.

Set `EDUPILOT_AUTH_REQUIRED=true` once a login UI exists; the backend already
has registration, login, refresh, logout and per-user ownership.

In development the anonymous identity is granted admin, so the operator can use
the Knowledge Base tab locally. That is why production must not grant it.

## CI/CD — GitHub Actions to Fly.io

```
push / PR ──► CI ──► lint · 172 tests · docker build + container smoke test
                 │
     main only   └──► Deploy ──► push image to GHCR ──► flyctl deploy ──► verify /api/health
```

`CI` runs on every push and pull request. `Deploy` triggers on `workflow_run`,
so it waits for CI to conclude and then checks it actually *succeeded* — a
plain `push` trigger would race the test job and could ship a red build.

Fly is deployed by digest (`:${{ github.sha }}`), not `:latest`, so a rollback
is `flyctl deploy --image ghcr.io/<owner>/<repo>:<old-sha>` and a re-run cannot
silently pick up a newer image.

### One-time setup

1. **Create the app, its volume and a database.** From the repo root, with
   [flyctl](https://fly.io/docs/flyctl/install/) installed:

   ```bash
   flyctl auth login
   flyctl apps create edupilot              # must match `app` in fly.toml
   flyctl volumes create edupilot_data --size 1 --region ord
   ```

   The volume holds only the course PDFs that citations link to. For the database, create a Postgres
   with pgvector (Fly Postgres, or Neon/Supabase) and build the index into it
   once from your machine — see "Updating the index" below.

2. **Set the runtime secrets on Fly** (these are for the *running app*):

   ```bash
   flyctl secrets set \
     DATABASE_URL=postgresql://... \
     GROQ_API_KEY=... \
     JWT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
     GEMINI_API_KEY=... \
     CORS_ALLOWED_ORIGINS=https://edupilot.fly.dev
   ```

3. **Set one secret on GitHub** (this is for the *pipeline*):

   Settings → Secrets and variables → Actions → New repository secret

   | Secret | Value |
   |---|---|
   | `FLY_API_TOKEN` | `flyctl tokens create deploy -x 999999h` |

   GHCR needs no secret — `GITHUB_TOKEN` is provided automatically.

4. **Push to main.** CI runs, then Deploy publishes and rolls out.

### What CI proves

The Docker job does not merely build the image — it runs the container and
curls it, because a build that succeeds and a container that serves are
different claims. It boots against a throwaway pgvector container with no
provider keys and no index, so health reports `degraded`; that is expected.
What is asserted is that the container starts, migrates the database, and
answers on the HTTP port.

The test job needs no secrets at all: `conftest.py` pins `EDUPILOT_DATA_DIR` at
a temp directory and the database at a fresh one on the job's Postgres service,
and nothing in the suite calls a provider. If a test ever needs an API key,
that test is reaching the network and should be rewritten.

### Cost and cold starts

`fly.toml` sets `min_machines_running = 0`, so the machine sleeps when idle and
an idle demo costs almost nothing. The tradeoff is that the first request after
a sleep pays roughly 20 seconds loading 1.1 GB of models. Before a live demo,
either warm it with one request or set `min_machines_running = 1`.

## Hugging Face Spaces (alternative)

1. Create a Space — **SDK: Docker**, hardware **CPU basic** (free).

2. Copy the Space README so Hugging Face can configure the container:

   ```bash
   cp deploy/huggingface/README.md README.md   # in the Space clone only
   ```

   It is kept separate so the GitHub README is not prefixed with a YAML block.

3. Add these under **Settings → Secrets**:

   | Secret | Required | Notes |
   |---|---|---|
   | `DATABASE_URL` | yes | Postgres with pgvector; holds the index and all app data |
   | `GROQ_API_KEY` | yes | default model is Groq |
   | `JWT_SECRET_KEY` | yes | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
   | `GEMINI_API_KEY` | no | fallback when Groq is rate limited |
   | `BOOTSTRAP_ADMIN_EMAIL` | no | creates an admin account on first boot |
   | `BOOTSTRAP_ADMIN_PASSWORD` | no | needed with the above |
   | `CORS_ALLOWED_ORIGINS` | no | set to the Space URL |

4. Push:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space main
   ```

5. **Persistent storage** (Settings → Persistent storage) mounts at `/data`,
   which is already `EDUPILOT_DATA_DIR`. It holds only the course PDFs that
   citations link to; accounts, chats, uploads and the index are in the
   database either way.

## Running the image directly

```bash
docker build -t edupilot .
docker run --rm -p 7860:7860 \
  -e DATABASE_URL=postgresql://... -e GROQ_API_KEY=... -e JWT_SECRET_KEY=... \
  -v edupilot-data:/data \
  edupilot
```

Or `docker compose up --build` from the repo root, which starts Postgres with
pgvector next to the app.

## Rehearsing the boot path without Docker

The entrypoint honours `PYTHON`, so the exact production startup can be
exercised against a virtualenv and the compose database — useful for checking
migrations before a build:

```bash
docker compose up -d db
DATABASE_URL=postgresql://edupilot:edupilot@localhost:5434/edupilot \
EDUPILOT_DATA_DIR=/tmp/vol EDUPILOT_ENV=production EDUPILOT_AUTH_REQUIRED=false \
PORT=8000 PYTHONPATH="$PWD/src" PYTHON="$PWD/.venv/bin/python" ./deploy/entrypoint.sh
```

## Updating the index

The deployment serves the index in its database; it does not index. To change
the corpus, run a rebuild where the source documents live, pointed at the
deployment's database:

```bash
DATABASE_URL=postgresql://... edupilot-reindex --rebuild
```

It builds a new index version next to the live one and promotes it when every
document has been indexed. Running instances notice the promotion within a few
seconds; no redeploy is needed. `edupilot-reindex --rollback` reverts, and
`edupilot-reindex --drop <old version>` frees its space once you are sure.
