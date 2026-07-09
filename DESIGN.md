# Campaign Copilot — Solution Design Doc

## 1. Goal Recap

Build a backend service exposing an LLM agent that turns a plain-English marketing goal into a ready-to-launch campaign, via:

1. Understanding the goal
2. Querying a users/events dataset to build a segment
3. Grounding itself in messaging guidelines via RAG
4. Drafting campaign content and creating it idempotently

Optimized for: **agent design quality**, **backend rigor** (idempotency, data modeling, resilience, observability), and **clear reasoning**, not feature count.

---

## 2. High-Level Architecture

```
                    ┌─────────────────────────┐
   Marketer prompt  │                         │
   ───────────────► │   POST /copilot/run     │
                     │   (FastAPI endpoint)   │
                     └───────────┬─────────────┘
                                 │
                                 ▼
                     ┌─────────────────────────┐
                     │   Agent Orchestrator    │
                     │  (plan → act → observe  │
                     │       loop)             │
                     └───────────┬─────────────┘
                                 │  tool calls
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
   ┌───────────────────┐ ┌───────────────┐  ┌────────────────────┐
   │ query_segment tool│ │search_guide-  │  │ create_campaign     │
   │ (SQLite: users +  │ │lines tool     │  │ tool (idempotent    │
   │  events join)     │ │(in-memory     │  │ write to campaigns  │
   │                    │ │vector index)  │  │ table)              │
   └───────────────────┘ └───────────────┘  └────────────────────┘
             │                   │                   │
             ▼                   ▼                   ▼
      SQLite (provided     FAISS/in-memory      SQLite (campaigns +
      data + a materialized cosine index over    idempotency_keys
      `user_activity`       chunked guideline     table)
      summary view)         docs

                                 │
                                 ▼
                     ┌─────────────────────────┐
                     │  Response + full trace  │
                     │  (segment, copy, offer, │
                     │   campaign_id, steps)   │
                     └─────────────────────────┘
```

Single-shot request/response (not multi-turn chat). The "conversation" is the agent's internal tool-calling loop, logged in a trace returned to the caller.

---

## 3. Tech Stack & Justification

| Layer | Choice | Why |
|---|---|---|
| Language/runtime | **Python 3.11 + FastAPI** | Best ecosystem for LLM tool-calling, embeddings, and data work (pandas/sqlite3). FastAPI gives async, typed request/response models, and OpenAPI docs for free — good "operability" signal. |
| LLM provider | **Anthropic Claude (claude-sonnet-4-6)** via `anthropic` SDK, native tool use | Configured via `ANTHROPIC_API_KEY` env var. A `MockLLMClient` behind the same interface is included for deterministic test runs (used in the eval harness and CI). |
| Structured data | **SQLite** (the provided prebuilt file, extended) | Already prebuilt in the bundle; zero external services; supports real SQL joins/aggregates for recency/frequency logic; trivially portable for reviewers. |
| Vector index (RAG) | **In-memory FAISS (flat, cosine)**, persisted to disk as a `.pkl`/`.npy` on first ingest | No external service to provision; ingestion is a one-time script (`make ingest`) so repeated runs are fast and reviewer setup is `clone → make setup → run`. |
| Embeddings | **`text-embedding-3-small` (OpenAI)** or **`voyage-3-lite` (Voyage AI)** — pluggable | Both are cheap and small-dimension. If the reviewer doesn't want a second provider key, a local `sentence-transformers/all-MiniLM-L6-v2` fallback is wired in (no API key needed at all — pure local RAG path). |
| Orchestration | **LangChain** — `create_tool_calling_agent` + `AgentExecutor`, or `LangGraph` `StateGraph` for explicit control | Gives native tool-calling scaffolding and built-in callback/tracing hooks. LangGraph specifically (over the older `AgentExecutor`) is preferred so the plan→act→observe loop stays visible as named nodes with an explicit step cap and per-node retry/fallback edges — closer to "real orchestration" than a single opaque `.run()` call. |
| Idempotency store | SQLite table `idempotency_keys` | Same datastore as everything else — no Redis needed for this scale. |

**Tradeoff called out explicitly:** using LangChain trades a bit of abstraction for (a) less tool-calling boilerplate, (b) built-in tracing/callback hooks for observability, and (c) a shape the interviewers likely already know, which should help the live pairing session. Mitigation: use LangGraph (explicit graph, not the more opaque legacy `AgentExecutor`) so each step is a visibly named node, and keep tool implementations as plain, framework-agnostic Python functions wrapped with `@tool` — easy to unit-test without spinning up LangChain at all.

---

## 4. Data Modeling

### 4.1 Provided tables (per `DATA_README.md`)

**`users`** — profile attributes only, one row per user:
`user_id, signup_date (YYYY-MM-DD), country, platform (Android/iOS/Web), app_version, plan (free/pro/enterprise)`

**`events`** — append-only behavioral log:
`event_id, user_id, event_name, timestamp (ISO 8601), properties (JSON)`

`event_name` values: `app_open`, `session_start`, `feature_used`, `purchase`, `notification_received`, `notification_opened`. `properties` is event-specific JSON, e.g. `{"feature_name": "voice_agent"}` for `feature_used`, `{"amount": 4900, "currency": "INR", "item": "pro_monthly"}` for `purchase`.

**`features`** — flat catalog of feature names referenced in `feature_used.properties.feature_name`, so "hasn't tried feature X" is well-defined against a known vocabulary rather than free text.

**Critical constraint:** `users` holds *only* profile attributes — every behavioral question (recency, frequency, feature adoption, spend) must be derived from `events`, joined back to `users`. There is no shortcut column on `users` for "last active" etc.

**As-of date:** the dataset is fixed to **2026-06-24** as "today." All recency math (`last N days`, `since signup`, etc.) is computed against this constant, not wall-clock time — this is what makes eval fixtures reproducible regardless of when the harness actually runs. Defined once as a config constant (`DATASET_AS_OF_DATE`) rather than scattered `date('now', ...)` calls, so it's a single documented override point.

### 4.2 Derived layer

A materialized summary table, rebuilt at ingest time (`make ingest`) rather than recomputed per-query, since `events` is append-only and can grow large:

```sql
CREATE TABLE user_activity_summary AS
SELECT
  u.user_id,
  u.signup_date,
  u.country,
  u.platform,
  u.plan,
  MAX(CASE WHEN e.event_name = 'app_open' THEN e.timestamp END)        AS last_open_at,
  COUNT(CASE WHEN e.event_name = 'app_open'
             AND e.timestamp >= datetime(:as_of, '-30 days')
        THEN 1 END)                                                     AS opens_last_30d,
  COUNT(CASE WHEN e.event_name = 'session_start'
             AND e.timestamp >= datetime(:as_of, '-30 days')
        THEN 1 END)                                                     AS sessions_last_30d,
  MAX(CASE WHEN e.event_name = 'purchase' THEN e.timestamp END)         AS last_purchase_at,
  SUM(CASE WHEN e.event_name = 'purchase'
           THEN json_extract(e.properties, '$.amount') END)             AS lifetime_spend,
  COUNT(CASE WHEN e.event_name = 'notification_opened'
             AND e.timestamp >= datetime(:as_of, '-30 days')
        THEN 1 END) * 1.0 /
    NULLIF(COUNT(CASE WHEN e.event_name = 'notification_received'
                       AND e.timestamp >= datetime(:as_of, '-30 days')
                  THEN 1 END), 0)                                        AS push_open_rate_30d,
  julianday(:as_of) - julianday(MAX(CASE WHEN e.event_name = 'app_open'
                                          THEN e.timestamp END))         AS days_since_last_open
FROM users u
LEFT JOIN events e ON e.user_id = u.user_id
GROUP BY u.user_id;

CREATE TABLE user_feature_adoption AS
SELECT DISTINCT
  user_id,
  json_extract(properties, '$.feature_name') AS feature_name,
  MIN(timestamp) AS first_used_at
FROM events
WHERE event_name = 'feature_used'
GROUP BY user_id, feature_name;
```

Two derived tables rather than one wide one:
- `user_activity_summary` — recency/frequency/spend/engagement, one row per user.
- `user_feature_adoption` — one row per `(user, feature)`, since a user can have zero-to-many adopted features; keeping it narrow avoids a sparse wide table as the `features` catalog grows.

Notably, `push_open_rate_30d` (derived from the `notification_received`/`notification_opened` pair) is genuinely useful here beyond just segmenting — it lets `query_segment` support "users who tend to ignore push" as a filter, and can inform the agent's **channel choice** (e.g., down-weight push, suggest email/in-app instead) — a nice example of behavioral data feeding the *content* decision, not just the *targeting* decision.

- **`query_segment` never lets the LLM write raw SQL** against `events` directly — it exposes a small constrained filter DSL (see §5.2) validated by pydantic, mapped to parameterized queries against `user_activity_summary` + `user_feature_adoption` joined to `users`. Avoids SQL-injection-via-LLM and keeps queries fast/predictable.
- Refresh strategy: rebuilt as part of `make ingest` for local dev; documented (not built) as a scheduled job for a real deployment — called out explicitly in "out of scope."

### 4.3 Campaigns store

```sql
CREATE TABLE campaigns (
  campaign_id      TEXT PRIMARY KEY,       -- generated (e.g. ULID)
  idempotency_key  TEXT UNIQUE NOT NULL,
  segment_def      JSON NOT NULL,          -- the DSL query used
  segment_size     INTEGER NOT NULL,
  channel          TEXT NOT NULL,
  copy             TEXT NOT NULL,
  image_prompt     TEXT,
  offer            JSON,
  guideline_citations JSON,                -- chunk ids used for grounding
  created_at       TIMESTAMP NOT NULL,
  status           TEXT NOT NULL           -- created | failed
);
```

**Addendum (added post-implementation, in response to review feedback):** `segment_def` + `segment_size` describe the segment as a *rule* + a *count* -- neither tells you who was actually targeted, and re-running the rule later can drift as `user_activity_summary` gets rebuilt from the ever-growing `events` log. A second table snapshots the real membership at creation time:

```sql
CREATE TABLE campaign_segment_members (
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  user_id     TEXT NOT NULL,
  PRIMARY KEY (campaign_id, user_id)
);
```

Populated by `create_campaign` itself (`src/tools/create_campaign.py`), in the same transaction as the `campaigns` row -- resolving `segment_def` via a new `resolve_segment_user_ids()` (the full match, not `query_segment`'s 20-row-capped LLM-facing sample) right before commit. An idempotent replay reuses the existing snapshot rather than re-resolving. See README.md's "Data modeling" section for the full rationale and tests.

---

## 5. Agent Design

### 5.1 Orchestration loop (plan → act → observe), via LangGraph

Built as a small `LangGraph` `StateGraph` rather than the legacy `AgentExecutor`, so the loop stays explicit and inspectable rather than hidden inside a framework's internal `.run()`:

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

class CopilotState(TypedDict):
    messages: list
    trace: list
    steps_taken: int

def call_model(state: CopilotState) -> CopilotState:
    response = llm_with_tools.invoke(state["messages"])
    state["messages"].append(response)
    state["trace"].append({"type": "llm_turn", "content": response})
    return state

def call_tools(state: CopilotState) -> CopilotState:
    for tool_call in state["messages"][-1].tool_calls:
        result = dispatch_tool_with_retry(tool_call)   # timeout + retry wrapper
        state["trace"].append({"tool": tool_call["name"],
                                "input": tool_call["args"],
                                "result_summary": summarize(result)})
        state["messages"].append(tool_result_message(tool_call["id"], result))
    state["steps_taken"] += 1
    return state

def should_continue(state: CopilotState) -> str:
    last = state["messages"][-1]
    if state["steps_taken"] >= MAX_STEPS:          # hard cost/latency cap
        return "fallback"
    return "tools" if getattr(last, "tool_calls", None) else "end"

graph = StateGraph(CopilotState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_node("fallback", fallback_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue,
                             {"tools": "tools", "end": END, "fallback": "fallback"})
graph.add_edge("tools", "agent")
copilot_graph = graph.compile()
```

Key properties:
- **The agent decides tool order** — the LLM node chooses which tool(s) to call each turn; the graph doesn't hardcode a step sequence, only a step *budget*.
- **Hard step cap** (`MAX_STEPS`, e.g. 6) is enforced in `should_continue`, routing to an explicit `fallback` node rather than looping unboundedly — the primary cost/latency control.
- Every LLM turn and tool call/result is appended to `state["trace"]`, returned in the API response for observability — this is plain Python state, not a LangChain-internal callback the reviewer has to dig for.
- Tools themselves are defined as framework-agnostic functions and wrapped with LangChain's `@tool` decorator only at the boundary, so `dispatch_tool_with_retry`, timeouts, and retry logic are testable without LangGraph in the loop at all (see `tests/test_orchestration.py`).

### 5.2 Tools

| Tool | Signature | Notes |
|---|---|---|
| `query_segment` | `query_segment(filters: {recency_days_max?, activity_window_days?, min_opens?, plan?, country?, platform?, feature_adopted?, feature_not_adopted?, push_open_rate_max?}) -> {size, sample_user_ids[:20], sql_used}` | Defined as a plain Python function with a pydantic `args_schema`, exposed to LangChain via `StructuredTool.from_function` (or the `@tool` decorator). Constrained filter object, not raw SQL — validated before hitting SQLite, mapped onto `user_activity_summary` + `user_feature_adoption`. Returns a **count + small sample**, never full PII dumps, to control token cost. |
| `search_guidelines` | `search_guidelines(query: str, k: int = 4) -> [{chunk_id, text, source_doc, score}]` | Wraps a `langchain_community.vectorstores.FAISS` retriever (or a thin custom wrapper over raw FAISS if we want more control than the LangChain retriever interface gives us) as a `@tool`. Returns chunk ids so the final output can cite exactly which guideline snippets informed the copy. |
| `create_campaign` | `create_campaign(idempotency_key: str, segment_def, channel, copy, image_prompt?, offer?, guideline_citations[]) -> {campaign_id, status}` | Plain function wrapped as a `@tool`. See §6 for idempotency mechanics. Fails loudly (structured error) on validation issues (e.g. missing offer for a channel that requires one) so the agent can self-correct in the next loop iteration. |

All three are implemented and unit-tested as ordinary Python functions first; the LangChain `@tool`/`StructuredTool` wrapping is a thin adapter layer added on top, so tool logic itself has zero LangChain dependency.

Tool failure handling: each tool call is wrapped with a timeout (e.g. 5s) and up to 2 retries with backoff; if it still fails, the tool returns a structured `{"error": "..."}` payload (not an exception) so the **agent itself sees the failure and can decide** to retry with different params, degrade gracefully (e.g., skip the image suggestion), or surface a clear failure to the caller — this is the "handles a tool returning nothing useful or failing" requirement.

### 5.3 Grounding

- Segment sizes in the final output are **always** the literal `size` returned by `query_segment` — the agent is instructed (and post-validated) never to state a segment size that doesn't match the last tool result in `trace`.
- Any guideline claim in the copy rationale must map to a `chunk_id` returned by `search_guidelines`; a lightweight post-check greps the final response for citation markers and rejects/retries the completion if citations don't trace back to a real retrieved chunk.

### 5.4 RAG Ingestion & Chunking Strategy

The corpus (per `guidelines/README.md`) is 17 topically distinct markdown docs — brand voice, re-engagement, push copy, channel/timing, segmentation, onboarding, winback, feature adoption, email, in-app, A/B testing, frequency capping, incentives, measuring success, **consent/compliance/opt-outs**, localization, and a lifecycle-stages glossary — **deliberately overlapping and cross-referencing**. That overlap is the main thing that makes naive chunking/retrieval risky here, so it shapes the approach:

**Chunking:**
- **Header-aware splitting**, not fixed-size windows: `MarkdownHeaderTextSplitter` (LangChain) splits each doc on its `##`/`###` structure first, preserving each guideline as a semantically complete unit (a chunk is "the winback timing recommendation," not an arbitrary 500-token slice that cuts a bullet list in half).
- A secondary `RecursiveCharacterTextSplitter` pass (token-length function, ~300–500 tokens, ~15% overlap) only kicks in for sections that are still too long after header splitting — most guideline sections are short enough not to need it.
- Every chunk is tagged with metadata: `{source_doc, doc_title, section_header, topic_slug}` — `topic_slug` derived from the filename (e.g. `07-winback-churned-users` → `winback`). This metadata is what lets citations in the final output say *which* doc and section informed the copy, not just a vector-store id.

**Retrieval:**
- **MMR (max marginal relevance) instead of pure top-k similarity** for `search_guidelines` — because the corpus intentionally overlaps (e.g. re-engagement, winback, and frequency-capping all touch "how often to message"), naive top-k tends to return 4 near-duplicate chunks all saying the same thing from slightly different docs. MMR trades a little relevance for diversity, which matters more here.
- `k=4–6`, with `topic_slug` available as an optional metadata filter the agent can pass if the LLM's own query already names an intent (e.g. it can bias toward `winback`/`re-engagement` docs for a "bring back lapsed users" goal) — but retrieval always stays hybrid (semantic + optional filter), never filter-only, since the agent's inferred topic could be wrong.
- **One rule sits outside pure semantic retrieval, on purpose**: for any campaign targeting an external channel (push/email — not in-app), the graph always additionally retrieves the top chunk from `15-consent-compliance-and-opt-outs.md`, regardless of its semantic score against the marketer's goal. Compliance guidance is exactly the kind of thing that's rarely the top semantic match for "win back lapsed users" but should never be silently skipped — a deliberate, small, explicit override rather than trusting embeddings alone for a correctness-sensitive doc.

**Grounding tie-in:** every citation surfaced in the final campaign output references `{source_doc, section_header}`, and the agent is instructed never to state a guideline claim without one — enforced by the same post-check described in §5.3.

**Problem:** a retried "create campaign" call (client timeout + retry, or the agent looping) must not double-create.

**Mechanism:**
1. Caller (or the agent, deterministically) derives an `idempotency_key` — a hash of `(goal_text, segment_def, channel)` normalized, OR a client-supplied key passed through the `/copilot/run` request.
2. `create_campaign` does an **`INSERT ... ON CONFLICT(idempotency_key) DO NOTHING`**-style upsert against the `campaigns` table (unique constraint on `idempotency_key`).
3. Before inserting, the tool first does a `SELECT` on `idempotency_key`; if found, it returns the **existing** `campaign_id` and `status` rather than creating a new row (idempotent read-through).
4. This is enforced at the **database constraint level** (unique index), not just application logic — so even concurrent/racing requests can't double-insert (the `INSERT` simply fails uniqueness and the app falls back to `SELECT`).

This is deliberately simple (single-node SQLite unique constraint) rather than a distributed idempotency service — appropriate for the scope; called out in README as a scaling tradeoff.

---

## 7. Resilience

- **Timeouts**: LLM calls wrapped with an explicit timeout (e.g. 20s); tool calls with a shorter one (5s for SQLite, 2s for the in-memory vector search since it's local and should never be slow).
- **Retries**: exponential backoff (e.g. 3 attempts, 1s/2s/4s) on transient LLM provider errors (429/5xx/timeout) and on tool timeouts.
- **Circuit breaker / fallback**: if the LLM provider fails after retries, fall back to a **deterministic template path**: use the raw filters the last successful tool call resolved (if any) and return a clearly-flagged `"degraded": true` response with a templated draft, rather than hanging or 500-ing the caller. This is a small, explicit fallback function — not a second LLM.
- **Step budget** (see §5.1) also protects against runaway agent loops racking up cost/latency.

**Addendum (added post-implementation, in response to review feedback about high-scale backend concerns):** the original design above assumed `POST /copilot/run` blocks synchronously for the whole agent loop. That doesn't scale -- a real run is a multi-turn LLM round-trip chain verified live to take anywhere from a few seconds to well over a minute, and holding an HTTP connection (plus a FastAPI thread-pool slot, since sync endpoints get thread-pooled) open for that whole duration caps throughput at "how many concurrent LLM round-trips can this process hold open."

Reworked to an async job pattern:
1. `POST /copilot/run` does two fast, synchronous things only -- a fail-fast check that the LLM/guidelines index are actually usable, and an INSERT into a new `runs` table (`run_id, goal, idempotency_key, status, result, error, created_at, updated_at`) with `status='pending'` -- then returns `202 Accepted` immediately (verified live: ~0.3s, regardless of the eventual run duration).
2. The actual agent loop runs via FastAPI's `BackgroundTasks`, executed *after* the response is sent. Self-contained (opens its own DB connection, since there's no HTTP request left by the time it runs to borrow resources from), and updates the same `runs` row through `pending -> running -> completed | failed`.
3. `GET /copilot/run/{run_id}` polls for status/result.

Deliberately `BackgroundTasks`, not Celery/RQ + Redis -- proportional to this project's in-memory/embedded scope, and Starlette already runs a sync callable added via `BackgroundTasks` in a worker thread automatically, so this needed zero changes to the graph/resilience layers above. Known gap, called out explicitly rather than glossed over: `BackgroundTasks` is in-process, in-memory scheduling -- a process crash/restart mid-run loses that run with nothing to retry it. A durable queue is the natural next step if this pattern needs to survive restarts. See README.md's "Async execution" section for the live verification (a real run's `pending -> running -> completed` transition observed over ~17 real seconds while the client's own connection was free the whole time).

---

## 8. Latency & Cost Awareness

- Hard cap on tool-calling steps (prevents unbounded loops).
- `query_segment` returns counts + a small sample, not full row dumps — keeps tool-result tokens small.
- `search_guidelines` caps `k` (top-4 by default) and truncates chunk text to a token budget.
- Embeddings are computed **once at ingest time** and cached to disk — no re-embedding of guideline docs per request.
- A single LLM call per loop iteration; system prompt + tool schemas are kept lean (no verbose few-shot bloat) to control input tokens.
- README documents an approximate cost-per-run budget (e.g. "~3-4 LLM calls, ~2-4k input tokens each, well under $0.01/run with Claude Haiku/Sonnet-class pricing" — actual number computed against current pricing).

---

## 9. Observability

- Every run gets a `run_id`; every tool call and LLM call logged with `run_id`, latency, token counts (from API response), and truncated input/output.
- Structured (JSON) logs to stdout — easy to pipe into any log aggregator.
- The full `trace` (ordered list of plan/tool-call/tool-result steps) is returned in the API response itself — this is the single most useful debugging artifact for "why did the agent do X," and doubles as the eval harness's primary input. This is built from our own `state["trace"]`, not solely from LangChain's internals, so it's readable even if someone isn't familiar with LangChain.
- A custom `BaseCallbackHandler` is registered on the LangChain/LangGraph run to capture `on_tool_start`/`on_tool_end`/`on_llm_end` events with timing — a cheap way to get token counts and latency per step without hand-rolling all of it.
- Optional: LangSmith tracing (if a `LANGCHAIN_API_KEY` is set) for a hosted trace viewer during development — explicitly optional, never required for the reviewer to run the project.
- Optional: wrap with OpenTelemetry spans per tool call if time allows (nice-to-have, not core).

---

## 10. Evaluation Approach

Given the non-determinism, the eval harness is a **small golden-set regression suite**, not a full eval framework:

1. **Fixture prompts** (8–12 realistic marketer goals) with **expected properties**, not exact string matches:
   - e.g. for "users active last month, no open in 14 days" → assert `segment.size > 0`, assert filters resolved include `recency_days_max: 14` and an activity window, assert `channel == "push"`, assert `offer is not None`, assert every guideline citation id exists in the retrieved index.
2. **Deterministic mode**: harness runs against the `MockLLMClient` (canned/deterministic tool-selection logic) for CI-safe, zero-cost regression testing of the orchestration/tool layer itself.
3. **Live mode**: same fixtures runnable against the real LLM (flagged, not run in CI) to spot-check real reasoning quality; asserts are relaxed to property-checks (segment non-empty, citations valid, idempotency key stable across 2 runs of the same goal).
4. **Idempotency test**: call `create_campaign` twice with the same key → assert single row in `campaigns`, same `campaign_id` returned both times.
5. **Failure-injection test**: force `search_guidelines` or the LLM call to raise/timeout → assert the agent's fallback path activates and the run still returns a well-formed (degraded) response rather than crashing.
6. **Retrieval spot-check**: a handful of hand-labeled queries (e.g. "winning back dormant users" → expect a top-k hit from `07-winback-churned-users.md` and/or `02-re-engagement-playbook.md`) asserting the expected `topic_slug`/`source_doc` shows up in results — cheap sanity check that chunking/MMR retrieval is behaving, given how much the guideline corpus intentionally overlaps.

This directly demonstrates "the instinct to measure quality in a non-deterministic system" the assignment asks for, without over-building a full eval framework.

---

## 11. Proposed Repo Structure

```
campaign-copilot/
├── README.md
├── Makefile                      # make setup / make ingest / make run / make test
├── docker-compose.yml            # optional, or just a venv-based Makefile
├── .env.example
├── data/                         # provided bundle (users/events/features, sqlite)
├── guidelines/                   # provided messaging docs
├── src/
│   ├── main.py                   # FastAPI app, POST /copilot/run
│   ├── agent/
│   │   ├── graph.py              # LangGraph StateGraph: agent/tools/fallback nodes
│   │   ├── prompts.py
│   │   └── llm_client.py         # real LangChain chat model + MockLLMClient, shared interface
│   ├── tools/
│   │   ├── query_segment.py      # plain function + pydantic args_schema
│   │   ├── search_guidelines.py  # plain function, wraps FAISS retriever
│   │   ├── create_campaign.py    # plain function
│   │   └── registry.py           # wraps the three as LangChain @tool / StructuredTool
│   ├── rag/
│   │   ├── ingest.py             # chunk → embed → index (LangChain text splitters), run via `make ingest`
│   │   └── index.py              # FAISS (langchain_community.vectorstores) load/query wrapper
│   ├── data_access/
│   │   ├── schema.sql
│   │   └── db.py
│   └── observability/
│       └── logging.py
└── tests/
    ├── fixtures/                 # golden-set prompts + expected properties
    ├── test_idempotency.py
    ├── test_orchestration.py
    └── test_eval_goldenset.py
```

---

## 12. Explicitly Out of Scope (call out in README)

- Multi-turn conversational refinement (single-shot goal → campaign only).
- Auth/multi-tenancy.
- A real scheduler for refreshing `user_activity_summary` (documented as a cron/Airflow job "with more time").
- Distributed idempotency (fine on single-node SQLite at this scale).
- A UI — CLI/API only, per the assignment's actual requirements.
- Hosted/managed vector DB — in-memory FAISS is sufficient and avoids provisioning.

## 13. What I'd Do With More Time

- Streaming the agent's trace to the client as it reasons (SSE), rather than a single blocking response.
- Real distributed idempotency (e.g., Redis SETNX + TTL) if this moved beyond single-node.
- A/B-style eval comparing chunking strategies (fixed-size vs. semantic/section-based) on retrieval precision for the guideline corpus.
- A scheduled refresh job for the activity summary table instead of build-at-startup.
- Confidence scoring on segment size (e.g., flag if segment is suspiciously tiny/huge relative to total user base) as an extra grounding safeguard.

---

## 14. Setup/Run (README skeleton)

```bash
git clone <repo>
cd campaign-copilot
cp .env.example .env          # set ANTHROPIC_API_KEY (or OPENAI_API_KEY, or leave blank for MockLLMClient)
make setup                    # installs deps, sets up venv
make ingest                   # chunks + embeds /guidelines into local FAISS index
make run                      # starts FastAPI on :8000

curl -X POST localhost:8000/copilot/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Win back users active last month who haven'"'"'t opened in 14 days, push notification with image and discount."}'

make test                     # runs unit + orchestration + idempotency + golden-set eval (mock LLM)
```