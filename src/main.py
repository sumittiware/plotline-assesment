"""
FastAPI app exposing the async job pattern for campaign creation:

- POST /copilot/run: enqueues a run and returns immediately (202) with a
  run_id + status="pending". The actual agent loop is a real multi-turn LLM
  round-trip chain that can take anywhere from a few seconds to well over a
  minute (observed live during development, not hypothetical) -- holding an
  HTTP connection open for that whole duration doesn't scale under
  concurrent load. The endpoint itself only does two fast, synchronous
  things: a fail-fast dependency check (is there a usable LLM/guidelines
  index at all), and inserting a `runs` row -- then hands the real work to a
  background task.
- GET /copilot/run/{run_id}: polls for the run's current status/result.

Background execution uses FastAPI's built-in BackgroundTasks rather than a
separate queue (Celery/RQ + Redis) -- proportional to this project's
"in-memory/embedded, no external services" scope, and Starlette already runs
a sync callable added via BackgroundTasks in a worker thread automatically,
so this needed zero changes to graph.py/resilience.py's existing
(synchronous) execution model.

Testing note: FastAPI's TestClient runs BackgroundTasks synchronously as
part of the same request/response cycle (verified directly), so tests don't
need to sleep/poll -- by the time client.post(...) returns, the background
task has already run to completion.

Dependencies (get_guidelines_store, get_agent_llm) are plain module-level
functions, not FastAPI Depends() -- the background task that actually calls
them runs outside any HTTP request context, so there's no request to bind a
Depends() override to. Tests instead monkeypatch these (and
connection_scope) directly by name on this module, per this project's
established convention (see CLAUDE.md: monkeypatch the consuming module's
already-imported name).

The `runs` table itself has no raw SQL here -- src/data_access/runs.py owns
that (plain functions + pydantic RunRecord, independently unit-tested in
tests/test_runs.py), the same pattern src/tools/create_campaign.py already
established for `campaigns`. Imported as `runs_repo` to avoid shadowing this
module's own GET /copilot/run/{run_id} handler, also named get_run.
"""
import uuid
from functools import lru_cache
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from src.agent.graph import CopilotState, last_tool_result, run as run_graph
from src.agent.llm_client import get_llm_client
from src.config import LLM_PROVIDER
from src.data_access import runs as runs_repo
from src.data_access.db import connection_scope
from src.observability.logging import configure_logging, log_run, log_trace
from src.rag.index import load_index
from src.tools.registry import build_tools

configure_logging()
app = FastAPI(title="Campaign Copilot")


@lru_cache(maxsize=1)
def get_guidelines_store() -> FAISS:
    try:
        return load_index()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="Guidelines index not found at VECTOR_INDEX_PATH. Run "
            "`python -m src.rag.ingest` first (requires GOOGLE_API_KEY for the "
            "default Gemini embedder, or set EMBEDDING_PROVIDER=local).",
        ) from e


@lru_cache(maxsize=1)
def _real_llm_client():
    return get_llm_client()


def get_agent_llm():
    if LLM_PROVIDER == "mock":
        raise HTTPException(
            status_code=503,
            detail="LLM_PROVIDER=mock cannot serve live requests (mock mode replays a "
            "caller-scripted response list, which has no meaning over HTTP). Set "
            "LLM_PROVIDER=gemini and GOOGLE_API_KEY to serve real traffic.",
        )
    return _real_llm_client()


class CopilotRunRequest(BaseModel):
    goal: str
    idempotency_key: Optional[str] = None


class CampaignResultPayload(BaseModel):
    """The agent's output once a run completes -- nested under RunStatusResponse.result."""

    campaign_id: Optional[str] = None
    status: Optional[str] = None
    idempotent_replay: bool = False
    segment_size: Optional[int] = None
    segment_member_count: Optional[int] = None
    channel: Optional[str] = None
    message_copy: Optional[str] = None
    image_prompt: Optional[str] = None
    offer: Optional[dict] = None
    guideline_citations: List[dict] = []
    degraded: bool
    final_message: str
    steps_taken: int
    trace: List[dict]


class RunStatusResponse(BaseModel):
    run_id: str
    status: str  # pending | running | completed | failed
    goal: str
    created_at: str
    updated_at: str
    result: Optional[CampaignResultPayload] = None
    error: Optional[str] = None  # populated only if status == "failed" (the worker itself crashed)


def _last_tool_call_input(state: CopilotState, tool_name: str) -> Optional[dict]:
    for entry in reversed(state["trace"]):
        if entry.get("tool") == tool_name:
            return entry.get("input")
    return None


def _build_result_payload(final_state: CopilotState) -> CampaignResultPayload:
    campaign_result = last_tool_result(final_state["messages"], "create_campaign") or {}
    segment_result = last_tool_result(final_state["messages"], "query_segment") or {}
    campaign_succeeded = "error" not in campaign_result and campaign_result.get("campaign_id") is not None
    # Only trust the last create_campaign call's INPUT args if that call
    # actually succeeded -- see CLAUDE.md for the real bug this guards
    # against (a rejected call's malformed args crashing response
    # construction with the same error the tool call already failed with).
    campaign_input = (_last_tool_call_input(final_state, "create_campaign") or {}) if campaign_succeeded else {}

    return CampaignResultPayload(
        campaign_id=campaign_result.get("campaign_id"),
        status=campaign_result.get("status"),
        idempotent_replay=campaign_result.get("idempotent_replay", False),
        segment_size=segment_result.get("size"),
        segment_member_count=campaign_result.get("segment_member_count"),
        channel=campaign_input.get("channel"),
        message_copy=campaign_input.get("message_copy"),
        image_prompt=campaign_input.get("image_prompt"),
        offer=campaign_input.get("offer"),
        guideline_citations=campaign_input.get("guideline_citations") or [],
        degraded=final_state["degraded"],
        final_message=final_state["messages"][-1].content,
        steps_taken=final_state["steps_taken"],
        trace=final_state["trace"],
    )


def _record_to_status_response(record: runs_repo.RunRecord) -> RunStatusResponse:
    return RunStatusResponse(
        run_id=record.run_id,
        status=record.status,
        goal=record.goal,
        created_at=record.created_at,
        updated_at=record.updated_at,
        result=CampaignResultPayload(**record.result) if record.result else None,
        error=record.error,
    )


def _execute_run(run_id: str, goal: str, idempotency_key: Optional[str]) -> None:
    """
    The actual agent loop, run as a background task (Starlette auto-offloads
    this sync callable to a worker thread). Fully self-contained: opens its
    own DB connection rather than reusing anything request-scoped, since
    this executes after the HTTP response has already been sent -- there's
    no request left to borrow resources from.
    """
    with log_run(run_id, goal) as run_log:
        try:
            with connection_scope() as conn:
                runs_repo.update_run_status(conn, run_id, status="running")
                store = get_guidelines_store()
                llm = get_agent_llm()
                tools = build_tools(conn, store)
                final_state = run_graph(goal, llm, tools, idempotency_key=idempotency_key, run_id=run_id)
                log_trace(run_id, final_state["trace"])

                result_payload = _build_result_payload(final_state)
                run_log.update(
                    degraded=final_state["degraded"],
                    steps_taken=final_state["steps_taken"],
                    campaign_id=result_payload.campaign_id,
                )
                runs_repo.update_run_status(conn, run_id, status="completed", result=result_payload.model_dump())
        except Exception as e:
            error_detail = e.detail if isinstance(e, HTTPException) else str(e)
            run_log.update(worker_error=f"{type(e).__name__}: {error_detail}")
            try:
                with connection_scope() as conn:
                    runs_repo.update_run_status(
                        conn, run_id, status="failed", error=f"{type(e).__name__}: {error_detail}"
                    )
            except Exception:
                pass  # best-effort -- if even this fails, there's nothing more we can do here


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/copilot/run", status_code=202, response_model=RunStatusResponse)
async def start_run(request: CopilotRunRequest, background_tasks: BackgroundTasks):
    # Fail fast, before enqueueing anything: a misconfigured server (no real
    # LLM key, no ingested guidelines index) should reject immediately with
    # a clear error, not accept a run it can never complete and silently
    # fail it later in the background where the caller can't see why.
    get_agent_llm()
    get_guidelines_store()

    run_id = f"run_{uuid.uuid4()}"
    with connection_scope() as conn:
        record = runs_repo.create_run(conn, run_id, request.goal, request.idempotency_key)

    background_tasks.add_task(_execute_run, run_id, request.goal, request.idempotency_key)

    return _record_to_status_response(record)


@app.get("/copilot/run/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str):
    with connection_scope() as conn:
        record = runs_repo.get_run(conn, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run found with run_id={run_id!r}")
    return _record_to_status_response(record)
