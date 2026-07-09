"""
Plain-function CRUD for the `runs` table (async job tracking for
POST /copilot/run) -- same pattern as src/tools/create_campaign.py: business
logic lives here, independently unit-testable (tests/test_runs.py) with zero
FastAPI/HTTP in the loop, rather than embedded as raw SQL directly in
src/main.py's route handlers. src/main.py just calls these functions.
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel


class RunRecord(BaseModel):
    run_id: str
    goal: str
    idempotency_key: Optional[str] = None
    status: str  # pending | running | completed | failed
    result: Optional[dict] = None
    error: Optional[str] = None  # populated only if status == "failed" (the worker itself crashed)
    created_at: str
    updated_at: str


def create_run(
    conn: sqlite3.Connection, run_id: str, goal: str, idempotency_key: Optional[str] = None
) -> RunRecord:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO runs (run_id, goal, idempotency_key, status, created_at, updated_at) "
        "VALUES (:run_id, :goal, :idempotency_key, 'pending', :now, :now)",
        {"run_id": run_id, "goal": goal, "idempotency_key": idempotency_key, "now": now},
    )
    conn.commit()
    return RunRecord(
        run_id=run_id, goal=goal, idempotency_key=idempotency_key, status="pending", created_at=now, updated_at=now
    )


def update_run_status(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    conn.execute(
        "UPDATE runs SET status = :status, result = :result, error = :error, updated_at = :updated_at "
        "WHERE run_id = :run_id",
        {
            "status": status,
            "result": json.dumps(result) if result is not None else None,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
        },
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[RunRecord]:
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return RunRecord(
        run_id=row["run_id"],
        goal=row["goal"],
        idempotency_key=row["idempotency_key"],
        status=row["status"],
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
