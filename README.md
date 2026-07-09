# Campaign Copilot

An LLM agent that turns a marketer's plain-English goal into a ready-to-launch campaign:

> "Win back users who were active last month but haven't opened the app in the last 14 days. Send them a push notification with an image and a discount offer to bring them back."

Given a goal like that, the agent plans its own steps — it decides which tools to call and when, not a hardcoded script — to:

1. **Understand the goal** and figure out what data it needs.
2. **Query the users/events dataset** (via a constrained filter DSL, never raw SQL) to build and size a target segment.
3. **Ground itself** in Plotline's messaging guidelines via RAG, so its channel/copy choices follow real best practices, not invented ones.
4. **Draft and idempotently create the campaign** (segment, channel, copy, and — where they fit the goal — richer elements like an image prompt or an offer/incentive, plus cited guidelines).

Every claim it makes — a segment size, a guideline citation — is checked against what actually happened in that run, not just what the LLM said. See [§ Grounding](#grounding-not-just-instructed--verified) below.

---

## Quick start

```bash
git clone <this-repo>
cd campaign-copilot
make setup                    # venv + deps, creates .env from .env.example
# edit .env: set GOOGLE_API_KEY (https://aistudio.google.com/apikey)
make bootstrap                # derived DB tables + RAG index (needs the key from the step above,
                               #  so this must run *after* you edit .env, not instead of `make setup`)
make run                      # starts FastAPI on :8000
```

```bash
# POST returns immediately (202) with a run_id -- the agent loop runs in the background
curl -X POST localhost:8000/copilot/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Win back users active last month who have not opened in 14 days, push with a discount."}'
# {"run_id": "run_...", "status": "pending", "result": null, ...}

# poll for the result (typically a few seconds to ~1 minute for a real run)
curl localhost:8000/copilot/run/run_<the-run_id-from-above>
# {"run_id": "run_...", "status": "completed", "result": {"campaign_id": "camp_...", ...}}
```

```bash
make test                      # 87 tests, no API key needed, ~1.5s
```

That's the whole path from clone to a running Copilot with only this README and your own Gemini key. No Docker, no external services — SQLite and FAISS are both embedded/in-process.

---

## Table of contents

- [Architecture](#architecture)
- [Tech stack & why](#tech-stack--why)
- [Design decisions & tradeoffs](#design-decisions--tradeoffs)
- [Idempotency](#idempotency)
- [Data modeling](#data-modeling)
- [Richer campaign content (image, offer)](#richer-campaign-content-image-offer)
- [RAG: chunking & retrieval](#rag-chunking--retrieval)
- [Grounding: not just instructed — verified](#grounding-not-just-instructed--verified)
- [Resilience](#resilience)
- [Async execution: not holding the connection open](#async-execution-not-holding-the-connection-open)
- [Latency & cost awareness](#latency--cost-awareness)
- [Observability](#observability)
- [Testing](#testing)
- [Eval approach](#eval-approach)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [What's deliberately out of scope](#whats-deliberately-out-of-scope)
- [What I'd do with more time](#what-id-do-with-more-time)

---

## Architecture

```
   Marketer goal
        │
        ▼
POST /copilot/run  (FastAPI, src/main.py) ──► 202 {run_id, status:"pending"}  (returns in ms)
        │                                            ▲
        │ enqueues                                   │ GET /copilot/run/{run_id}
        ▼                                             │ (poll for status/result)
BackgroundTasks: _execute_run(run_id, goal)  ──► `runs` table (pending→running→completed|failed)
        │
        ▼
LangGraph StateGraph — plan → act → observe  (src/agent/graph.py)
        │
   ┌────┴────┬─────────────┬──────────────┐
   ▼         ▼             ▼              ▼
 agent      tools       fallback     (loops until
 node       node        node          agent says
(1 LLM      (dispatches (circuit       "done" or
 turn)      tool_calls,  breaker /     step budget
            grounding    step-budget   hit)
            check)       exceeded)
   │         │
   │    ┌────┼────────────┬─────────────────┐
   │    ▼    ▼             ▼                 ▼
   │ query_ search_    create_
   │ segment guidelines campaign
   │    │       │            │
   │    ▼       ▼            ▼
   │ SQLite   FAISS        SQLite
   │ (derived (Gemini-     (campaigns +
   │  tables) embedded     campaign_segment_members +
   │          guideline    unique idempotency_key)
   │          chunks)
   └─────────────────────────────────────────┘
        Full trace + structured logs emitted throughout; final result written into `runs.result`
```

**The endpoint doesn't hold the HTTP connection open for the agent loop.** A real run is a multi-turn LLM round-trip chain that can take anywhere from a few seconds to well over a minute (observed live during development — see [Latency & cost awareness](#latency--cost-awareness)). `POST /copilot/run` only does two fast things — a fail-fast dependency check, and inserting a `runs` row — then hands the actual work to a background task and returns `202` in milliseconds. `GET /copilot/run/{run_id}` polls for the result. See [Async execution](#async-execution-not-holding-the-connection-open) below.

The agent decides tool order itself — the graph only enforces a step *budget*, not a step *sequence*. Every LLM turn and tool call/result is appended to a plain-Python `trace` list, returned in full once the run completes (the primary debugging artifact — see [Observability](#observability)).

## Tech stack & why

| Layer | Choice | Why |
|---|---|---|
| Language/runtime | Python 3 + FastAPI | Best ecosystem for LLM tool-calling + embeddings; async, typed models, free OpenAPI docs. |
| LLM + embeddings | **Gemini** (`gemini-2.5-flash` + `gemini-embedding-001`) via `langchain-google-genai` | Single API key (`GOOGLE_API_KEY`) covers both the planning LLM and RAG embeddings — one fewer credential for a reviewer to manage. `MockLLMClient` behind the same interface for deterministic tests/eval (see [Testing](#testing)). |
| Orchestration | **LangGraph** `StateGraph`, not the legacy `AgentExecutor` | Keeps the plan→act→observe loop as explicit, named nodes with visible edges — closer to "real orchestration" than a single opaque `.run()` call, and lets resilience/grounding/compliance logic hook in at precise points (see below). |
| Structured data | **SQLite** (provided `data.sqlite`, extended with derived + campaigns tables) | Zero external services; real SQL joins/aggregates for recency/frequency logic; trivially portable. |
| Vector index | **FAISS**, in-memory, persisted to disk after `make ingest` | No vector DB to provision. Header-aware chunking + MMR retrieval (see [RAG](#rag-chunking--retrieval)). |
| Idempotency store | SQLite `campaigns.idempotency_key` (unique index) | Same datastore as everything else — no Redis needed at this scale. |

**Tradeoff called out explicitly**: LangGraph trades a little abstraction for (a) less tool-calling boilerplate, (b) an explicit graph instead of a hidden loop, (c) natural hook points for the resilience/grounding/compliance logic below. Tool implementations themselves are plain, framework-agnostic Python functions with pydantic schemas — `@tool`/`StructuredTool` wrapping happens only at the boundary (`src/tools/registry.py`), so the tool logic itself is unit-tested with zero LangChain/LangGraph in the loop.

## Design decisions & tradeoffs

### Idempotency

**Problem**: a retried "create campaign" call (client timeout, agent retry, network blip) must not double-create.

**Mechanism** ([src/tools/create_campaign.py](src/tools/create_campaign.py)):
1. The caller (or the agent, deterministically, from `hash(goal_text, segment_def, channel)`) derives an `idempotency_key` — or a client supplies one via the API request, which always overrides whatever the agent would have derived (`POST /copilot/run {"goal": ..., "idempotency_key": "..."}`).
2. `create_campaign` **INSERTs first**, catches the `sqlite3.IntegrityError` from the unique constraint on `campaigns.idempotency_key`, and **only then reads back** the existing row. Never "SELECT-then-decide-to-insert" — that ordering has a race window between the check and the write; the insert-first ordering makes the unique constraint itself the source of truth.
3. Verified with a **real concurrency test** ([tests/test_idempotency.py](tests/test_idempotency.py)): N threads fire the same idempotency key at once against a real file-backed SQLite DB; exactly one row lands.

Deliberately simple (single-node SQLite unique constraint), not a distributed idempotency service — appropriate at this scale; a real deployment would swap this for Redis `SETNX`+TTL or a dedicated idempotency table with a TTL sweep, without changing the tool's interface.

### Data modeling

Per the provided schema, `users` holds *only* profile attributes (signup_date, country, platform, plan) — every behavioral question (recency, frequency, feature adoption, spend, push responsiveness) must be derived from the append-only `events` log and joined back.

Rather than recomputing this per-query against a potentially large `events` table, two **materialized derived tables** are rebuilt from raw events (`make db-rebuild` / `python -m src.data_access.db`, using a fixed `DATASET_AS_OF_DATE` so segment sizes are reproducible regardless of when the job runs):

- `user_activity_summary` — one row per user: `days_since_last_open`, `opens_last_30d`, `lifetime_spend`, `push_open_rate_30d` (derived from the `notification_received`/`notification_opened` pair — informs *channel choice*, not just targeting), etc.
- `user_feature_adoption` — one row per `(user, feature)`, since a user can have zero-to-many adopted features; kept narrow rather than a sparse wide table.

`query_segment` ([src/tools/query_segment.py](src/tools/query_segment.py)) **never lets the LLM write raw SQL** — it exposes a small pydantic-validated filter DSL (`recency_days_max`, `inactive_days_min`, `plan`, `country`, `feature_adopted`, `push_open_rate_max`, ...), mapped to a single parameterized query against the derived tables. This is what keeps a prompt-injected or just-wrong LLM output from ever touching the database with arbitrary SQL, and keeps queries fast/predictable. It returns a **count + small sample of user_ids**, never a full PII dump, to control response size and token cost.

**Segment membership is snapshotted, not just counted.** `campaigns` stores `segment_def` (the filter) and `segment_size` (a count) — but a filter alone only tells you the *rule*, not who it actually matched, and re-running it later can drift as `user_activity_summary` gets rebuilt from the ever-growing `events` log. So `create_campaign` ([src/tools/create_campaign.py](src/tools/create_campaign.py)) additionally resolves the segment_def into the **full** matching `user_id` list (via `resolve_segment_user_ids` — never `query_segment`'s LLM-facing, 20-row-capped sample, which stays small specifically to control token cost) and writes it to a `campaign_segment_members(campaign_id, user_id)` table, in the **same transaction** as the campaign row itself — either both land or neither does. An idempotent replay reuses the original snapshot rather than re-resolving against whatever the data looks like *now* (tested directly: seed data changes between two calls with the same idempotency key, and the second call still reports the original count, not a re-computed one). The count is also surfaced as `segment_member_count` in the `POST /copilot/run` response, alongside `segment_size`, for visibility without dumping potentially thousands of raw user_ids over HTTP — the full list lives in the DB, directly queryable (`SELECT user_id FROM campaign_segment_members WHERE campaign_id = ...`).

### Richer campaign content (image, offer)

`create_campaign`'s payload isn't just segment + channel + copy — `offer` (a dict, e.g. `{"type": "discount", "value": "20%"}`) and `image_prompt` (a short description of an accompanying image) are both optional fields the agent decides to populate based on the goal, per the assignment's own framing ("richer elements like an image or an offer/incentive, where the channel supports it"). Neither is forced by channel type — push, email, and in-app messages all support an accompanying image in this design — the system prompt instructs the agent to include either only when it genuinely fits the goal, not by default. Both round-trip through the full pipeline: persisted to `campaigns` (`schema.sql`), surfaced in the `POST /copilot/run` response, and exercised in the golden-set eval (the fixture matching the assignment's own headline example asserts a non-empty `image_prompt`, not just `offer`).

### RAG: chunking & retrieval

The `/guidelines` corpus (17 markdown docs, deliberately overlapping — re-engagement, winback, and frequency-capping all touch "how often to message") is:

- **Header-aware split** (`MarkdownHeaderTextSplitter` on `#`/`##`/`###`) — a chunk is "the winback timing recommendation," not an arbitrary 500-token slice that cuts a bullet list in half. A secondary character-based splitter only kicks in for sections still too long after that (rarely, given how short these docs are). **17 docs → 90 chunks**, each tagged with `{source_doc, doc_title, section_header, topic_slug, chunk_id}`.
- Retrieved via **MMR (max marginal relevance), not pure top-k similarity** — because the corpus intentionally overlaps, naive top-k tends to return 4 near-duplicate chunks saying the same thing from slightly different docs. MMR trades a little relevance for diversity, which matters more here. `topic_slug` is available as an optional metadata filter, but retrieval stays hybrid (semantic + optional filter), never filter-only.
- **One rule sits outside pure semantic retrieval, on purpose**: for any campaign targeting an external channel (push/email — not in-app), the graph *always* additionally retrieves the top chunk from `15-consent-compliance-and-opt-outs.md`, regardless of its semantic score against the marketer's goal (`_ensure_compliance_citation` in [src/agent/graph.py](src/agent/graph.py)). Compliance guidance is exactly the kind of thing that's rarely the top semantic match for "win back lapsed users" but should never be silently skipped — enforced mechanically, not left to the system prompt. Verified live against the real Gemini API and the real embedding index, not just mocked.

### Grounding: not just instructed — verified

Any agent can be *told* "don't invent a segment size or a citation" in its system prompt. The system prompt here does say that — but nothing stops an LLM from writing plausible-sounding prose regardless. So there's a mechanical check, right before `create_campaign` is actually dispatched (`_check_grounding` in [src/agent/graph.py](src/agent/graph.py)):

1. `segment_size` must equal the `size` field of the **last real `query_segment` result** in that conversation. No prior `query_segment` call at all is also a violation.
2. Every `chunk_id` in `guideline_citations` must have actually been returned by a real `search_guidelines` call earlier in the conversation (including the forced compliance retrieval above).

A violation doesn't 500 or silently pass — it turns into a structured `{"error": "..."}` tool result, exactly like any other tool failure, so the agent sees it and can self-correct on its next turn (verified directly: a test scripts a wrong `segment_size`, gets rejected, then the agent retries with the correct one and the campaign is created).

This checks the **structured tool arguments**, not the free-text final answer — those structured args are what actually get persisted to the `campaigns` table, so validating them is both exact (no regex over prose) and more consequential than validating what the LLM merely *says* in its summary.

### Resilience

- **Timeouts**: every tool call and LLM call runs inside a hard timeout (`ThreadPoolExecutor` + `future.result(timeout=...)`) — 5s for SQLite tools, 3s for the in-memory vector search, 20s for the LLM call.
- **Retries with exponential backoff**, two deliberately different policies ([src/agent/resilience.py](src/agent/resilience.py)):
  - **Tool calls** retry *only* on timeout. A validation error or a real DB error will fail identically on retry — retrying it just adds latency for no benefit.
  - **LLM calls** retry on *any* exception. Classifying "retryable" would mean importing a specific provider's exception types, which this module deliberately avoids to stay provider-agnostic.
- **Circuit breaker**: if the LLM still fails after exhausting retries, the graph doesn't crash — it routes to a `fallback` node (the same one used when the step budget is exceeded) that produces a **small, deterministic templated response** — *not* a second LLM call — built from the last successfully-resolved segment, if any, per "use the raw filters the last successful tool call resolved."
- **Step budget** (`MAX_AGENT_STEPS`, default 6) is the primary cost/latency backstop against a runaway loop, independent of the above.

This isn't hypothetical: it was **verified against the real Gemini API's free-tier rate limit** mid-development. A live run hit a genuine `429 ResourceExhausted`, retried, exhausted retries, and degraded gracefully — HTTP 200 throughout, never a 500 — even correctly reusing an already-completed `create_campaign` result from earlier in that same run via the idempotency mechanism, instead of a bare apology.

### Async execution: not holding the connection open

**Problem**: a real agent run is a multi-turn LLM round-trip chain — verified live to take anywhere from a few seconds to over a minute. A naive synchronous `POST /copilot/run` holds the HTTP connection (and, since FastAPI thread-pools sync endpoints, a worker thread) open for that entire duration. Under concurrent load, that caps your server's throughput at "how many simultaneous LLM round-trips can this process hold open," not "how much of its own compute capacity does it actually have" — a classic scaling anti-pattern for anything that talks to a slow external dependency.

**Mechanism**:
1. `POST /copilot/run` does two fast, synchronous things only: a fail-fast check that the LLM/guidelines index are actually usable (so a misconfigured server rejects immediately, not after silently failing a job in the background later), and an `INSERT` into a new `runs` table with `status='pending'`. Returns **`202 Accepted`** with `{run_id, status: "pending"}` — verified live to return in ~0.3s, regardless of how long the underlying run eventually takes.
2. The actual agent loop is handed to FastAPI's `BackgroundTasks`, which runs *after* the response is already sent to the client. It's fully self-contained (opens its own DB connection rather than reusing anything request-scoped, since it executes with no HTTP request left to borrow resources from) and updates the same `runs` row through `pending → running → completed | failed` as it progresses.
3. `GET /copilot/run/{run_id}` polls for the current status and, once `completed`, the full result (nested under `result`, same shape the old synchronous endpoint used to return directly).

**Why `BackgroundTasks` and not Celery/RQ + Redis**: proportional to this project's "in-memory/embedded, no external services" scope (see [Tech stack](#tech-stack--why)) — Starlette already runs a sync callable added via `BackgroundTasks` in a worker thread automatically, so this needed **zero changes** to `graph.py`'s or `resilience.py`'s existing synchronous execution model. A real distributed deployment would swap this for an actual task queue without changing the tool/graph layer at all — the same kind of "simple now, swappable later" tradeoff already made for idempotency and the datastore.

**Verified live**, not just via tests: fired a real request, confirmed the `POST` returned `202` in 0.3s, then polled `GET /copilot/run/{run_id}` every 2s and watched the status genuinely progress `pending → running` (for ~17 real seconds, while a live Gemini agent loop executed against push/win-back reasoning) `→ completed`, with a real `campaign_id` and its segment correctly snapshotted (996 members, matching `segment_size` exactly) — all while the client's own connection was free the entire time. Also tested via `TestClient` is a subtlety worth knowing: it runs `BackgroundTasks` **synchronously** as part of the same request/response cycle, so the test suite doesn't need to sleep/poll to be deterministic — confirmed directly before relying on it.

## Latency & cost awareness

- Hard step cap (`MAX_AGENT_STEPS=6`) bounds worst-case cost/latency; real runs so far have completed in 3-4 steps.
- `query_segment` returns a count + a 20-row sample, never a full dump — keeps tool-result tokens small regardless of segment size.
- `search_guidelines` caps `k` (default 4) and truncates trace log entries to 300 chars — full untruncated content still reaches the LLM/response, only the *log line* is truncated.
- Embeddings for the guideline corpus are computed **once at ingest time** (`make ingest`), not per-request; only the short query text is embedded per `search_guidelines` call.
- **Approximate cost per run**: a real run is ~4 LLM turns (one per tool call, plus the final answer), with growing context as tool results accumulate — roughly 5,000 input + 400 output tokens total. At published Gemini 2.5 Flash pricing (`$0.30`/1M input, `$2.50`/1M output tokens) that's **≈$0.002–0.005 per run**, plus a negligible per-query embedding cost (`gemini-embedding-001` at `$0.15`/1M tokens, only the short query text, ~1-2 calls/run). Well under a cent per run; pricing current as of writing, verify at [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing).

## Observability

- Every run gets a `run_id`. The **full `trace`** (every LLM turn + tool call/result, as plain dicts) is returned in the API response itself — the single most useful artifact for "why did the agent do X," and it's built from our own state, not a LangChain-internal callback someone has to dig for.
- Structured **JSON-lines logging to stdout** ([src/observability/logging.py](src/observability/logging.py)): `run_start` / one `trace_step` line per trace entry / `run_end` (with latency_ms), all tagged with `run_id`. Deliberately complements rather than duplicates the response trace: the response trace is the rich artifact when the caller has it in hand; these lines are the server-side complement for when they don't (crashed client, aggregate monitoring).
- **Optional LangSmith hosted tracing**: set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` in `.env` (see `.env.example`) for a visual trace UI at [smith.langchain.com](https://smith.langchain.com) covering every LLM call and graph step. Zero code branches needed — `langchain-core` auto-detects these standard env vars. Each run's trace is named after our own `run_id` ([src/agent/graph.py](src/agent/graph.py)'s `run()`), so it correlates with the same id in the API response and stdout logs. Entirely inert if unset (the default) — verified that even an *invalid* key just logs a background warning and fails silently, never affecting the actual run.
- Not built: OpenTelemetry spans — stdlib logging + the response trace + optional LangSmith already cover "debug a bad run" without needing another integration.

## Testing

```bash
make test          # 87 tests (+ 8 opt-in live ones skipped by default), ~1.5s, zero network/API dependency
```

| File | Covers |
|---|---|
| `test_query_segment.py` | Filter DSL → parameterized SQL, against the real derived tables |
| `test_idempotency.py` | Insert-then-read-on-conflict, including a real multi-thread concurrency race |
| `test_search_guidelines.py` | Chunking, MMR, topic filtering, citation metadata |
| `test_llm_client.py` | Real/mock provider factory, `MockLLMClient` scripted replay |
| `test_orchestration.py` | The full graph: happy path, step-budget fallback, tool-error containment, compliance override, grounding rejection + self-correction, resilience retry/circuit-breaker |
| `test_resilience.py` | Timeout + retry-with-backoff, standalone, no LangGraph in the loop |
| `test_main.py` | Full HTTP request/response cycle via FastAPI's `TestClient` |
| `test_observability.py` | Structured log output |
| `test_eval_goldenset.py` | The eval harness — see below |

Everything runs against `MockLLMClient` (a caller-scripted list of responses, replayed deterministically — can also script an `Exception` instead of a message, to test retry/circuit-breaker paths without a real flaky provider) and a hashing-based local embedder for RAG tests, rather than the real Gemini API — deterministic, free, and independent of network access. The **real** path (`ChatGoogleGenerativeAI` + `gemini-embedding-001`) has also been verified live multiple times against the actual API during development (see [Resilience](#resilience) and [Grounding](#grounding-not-just-instructed--verified) above) — both paths share the identical `.bind_tools()`/`.invoke()` interface, so the graph genuinely cannot tell them apart.

## Eval approach

A small golden-set regression suite ([tests/fixtures/golden_set.py](tests/fixtures/golden_set.py) + [tests/test_eval_goldenset.py](tests/test_eval_goldenset.py)), not a full eval framework, per the assignment's own framing ("the instinct to measure quality in a non-deterministic system, not a full eval framework"):

- **8 realistic marketer goals** (win-back, onboarding, feature adoption, plan upsell, localization, push-fatigue rerouting, broad announcement, long-dormant win-back), each checked against **expected properties**, not exact string matches — e.g. for the assignment's own headline example ("win back users active last month, no open in 14 days, push with an image and a discount") → assert the resolved filters include `inactive_days_min`, `channel == "push"`, both an `offer` and an `image_prompt` are present, and (since push is external) a real consent-compliance citation exists.
- **Two modes over the same fixtures**:
  - `make test` runs them **deterministically** — each goal replayed through a scripted `MockLLMClient` trajectory. This is a regression test of the orchestration/tools/grounding/compliance pipeline, not of LLM reasoning (the trajectory is scripted, not produced) — property checks are strict since every input is controlled.
  - `make eval-live` (opt-in, `RUN_LIVE_EVAL=1`, **not** run in CI/`make test` — costs real API quota and isn't deterministic) runs the **same goal texts** through the real Gemini client end-to-end. This is what actually evaluates reasoning quality; property checks are relaxed to structural invariants (a segment was resolved, a campaign was created, a compliance citation exists when the channel is external) rather than exact filter/channel matches, since a real agent may reasonably resolve a goal differently than the scripted guess.
- A **retrieval spot-check**: 6 hand-labeled queries (e.g. "winning back dormant, churned users" → expect a `winback`/`re-engagement` chunk in the top-k) as a cheap sanity check on chunking/MMR, given how deliberately the guideline corpus overlaps.

## Configuration

All via `.env` (copy from `.env.example`; loaded automatically via `python-dotenv`):

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | *(required for real traffic)* | Single key for both the LLM and embeddings. Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). |
| `LLM_PROVIDER` | `mock` | `"gemini"` for real traffic; `"mock"` only works for `pytest` — the API refuses to serve live requests in mock mode (503). |
| `GEMINI_MODEL` | `gemini-2.5-flash` | |
| `EMBEDDING_PROVIDER` | `gemini` | `"gemini"` \| `"openai"` (needs `OPENAI_API_KEY`) \| `"local"` (no key at all, `sentence-transformers`, downloads a model on first run). |
| `MAX_AGENT_STEPS` | `6` | Hard step-budget cap. |
| `LLM_TIMEOUT_SECONDS` | `20` | Per-attempt LLM call timeout. |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | |

## Project structure

```
campaign-copilot/
├── prompts.yaml               # system prompt + tool descriptions (kept out of Python)
├── Makefile                   # setup / db-rebuild / ingest / bootstrap / run / test / eval-live / clean-*
├── .env.example
├── DESIGN.md                  # full architecture/decisions doc (this README's deeper reference)
├── data/                      # provided bundle: data.sqlite (+ derived/campaigns tables), DATA_README.md
├── guidelines/                # provided messaging best-practices corpus (RAG source)
├── src/
│   ├── main.py                 # FastAPI app: POST /copilot/run (202, async job), GET /copilot/run/{id}, GET /health
│   ├── config.py                # single source of truth for env vars + tunables
│   ├── agent/
│   │   ├── graph.py             # LangGraph StateGraph: agent/tools/fallback nodes, grounding, compliance override
│   │   ├── resilience.py        # timeout + retry-with-backoff wrapping
│   │   ├── llm_client.py        # real ChatGoogleGenerativeAI + MockLLMClient, shared interface
│   │   └── prompts.py           # loads prompts.yaml
│   ├── tools/
│   │   ├── query_segment.py     # plain function + pydantic filter DSL
│   │   ├── search_guidelines.py # plain function, wraps the FAISS retriever
│   │   ├── create_campaign.py   # plain function, idempotent write
│   │   └── registry.py          # wraps the three as LangChain StructuredTools (only file that imports LangChain into tools/)
│   ├── rag/
│   │   ├── ingest.py            # chunk → embed → index, `python -m src.rag.ingest`
│   │   ├── index.py             # FAISS load/query (MMR) wrapper
│   │   └── embeddings.py        # pluggable embedding backend (gemini/openai/local)
│   ├── data_access/
│   │   ├── schema.sql
│   │   └── db.py                 # connection + derived-table rebuild, `python -m src.data_access.db`
│   └── observability/
│       └── logging.py            # structured JSON-lines logging
└── tests/                       # 87 tests + 8 opt-in live ones — see Testing/Eval approach above
    └── fixtures/golden_set.py    # the eval harness's golden-set goals + expected properties
```

## What's deliberately out of scope

- **Multi-turn conversational refinement** — single-shot goal → campaign only, per the assignment's own framing.
- **Auth / multi-tenancy** — not asked for.
- **Distributed idempotency** — fine on single-node SQLite at this scale; noted above as the real-deployment tradeoff.
- **A UI** — API/CLI only.
- **A hosted/managed vector DB** — in-memory FAISS is sufficient and avoids provisioning, per the assignment's own preference.
- **A real scheduler for refreshing the derived tables** — `make db-rebuild` is a manual/cron-able job today; documented, not built, as a scheduled job for a real deployment.
- **A distributed task queue (Celery/RQ + Redis)** for the async job pattern — FastAPI's built-in `BackgroundTasks` is proportional to this project's scope and needed zero changes to the agent/graph layer; see [Async execution](#async-execution-not-holding-the-connection-open). A real multi-instance deployment would swap this in without touching the tool/graph code at all.
- **OpenTelemetry** — stdlib JSON-lines logging + the response trace + optional LangSmith (see Observability above) already cover "debug a bad run" without another integration.
- **Docker Compose** — a Makefile + venv is simpler for a pure-Python, no-external-services project; nothing here needs containerizing to run.

## What I'd do with more time

- An actual `send_campaign` step, with a `draft → pending_approval → sent` status lifecycle instead of `create_campaign` implicitly being the terminal state. Right now this agent produces a *ready-to-launch* campaign (per the assignment's own phrasing) and a real snapshot of who it would target (`campaign_segment_members`) — it does not autonomously dispatch anything to real users. A human-approval gate before an irreversible send is standard practice for any agent that can target real users, and worth making an explicit, enforced step rather than an implicit scope boundary.
- Grow the golden-set beyond 8 fixtures and add a scoring/reporting layer (pass-rate over time, not just pass/fail) once there's a real corpus of production goals to draw from.
- Streaming the agent's trace to the client incrementally as it reasons (SSE/websocket), rather than polling `GET /copilot/run/{run_id}` for the final result once it's done.
- A durable task queue (Celery/RQ + Redis, or a DB-polling worker) instead of `BackgroundTasks`. Worth naming the actual gap honestly: `BackgroundTasks` is in-process, in-memory scheduling — if the server process crashes or restarts mid-run, that run is simply lost, stuck at whatever `status` it last reached, with nothing to retry it. Fine for this scope (matches its "no external services" rationale), but the first thing to fix before this pattern is trusted for anything that must survive a restart.
- A provider-level circuit breaker that stays "open" across requests after repeated failures, rather than each call getting a fresh retry budget — fine at this scale, would matter more under sustained real-provider outages.
- Real distributed idempotency (Redis `SETNX`+TTL) if this moved beyond single-node.
- Confidence scoring on segment size (flag if a segment is suspiciously tiny/huge relative to the total user base) as an extra grounding safeguard, beyond the exact-match check that exists today.
- A/B-style comparison of chunking strategies (fixed-size vs. the header-aware approach here) on retrieval precision, to actually measure the chunking decision rather than just reason about it.

---

For the full, deeper design rationale (including sections not repeated here to keep this README a reasonable length), see [DESIGN.md](DESIGN.md).