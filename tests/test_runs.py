"""
Plain-function CRUD for the `runs` table, tested in complete isolation from
FastAPI/HTTP -- same rationale as tests/test_idempotency.py testing
create_campaign directly rather than only through the API layer. If this
logic is wrong, it fails here in milliseconds, not three layers deep in an
HTTP integration test.
"""
import sqlite3

import pytest

from src.data_access.db import apply_schema
from src.data_access.runs import create_run, get_run, update_run_status


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    yield c
    c.close()


def test_create_run_starts_pending_with_no_result_or_error(conn):
    record = create_run(conn, "run_1", "some goal")
    assert record.status == "pending"
    assert record.goal == "some goal"
    assert record.result is None
    assert record.error is None
    assert record.idempotency_key is None


def test_create_run_stores_the_supplied_idempotency_key(conn):
    record = create_run(conn, "run_1", "some goal", idempotency_key="client-key-123")
    assert record.idempotency_key == "client-key-123"


def test_get_run_reads_back_a_created_run(conn):
    create_run(conn, "run_1", "some goal", idempotency_key="key-1")
    fetched = get_run(conn, "run_1")
    assert fetched is not None
    assert fetched.run_id == "run_1"
    assert fetched.goal == "some goal"
    assert fetched.idempotency_key == "key-1"
    assert fetched.status == "pending"


def test_get_run_returns_none_for_an_unknown_run_id(conn):
    assert get_run(conn, "does-not-exist") is None


def test_update_run_status_transitions_through_the_full_lifecycle(conn):
    create_run(conn, "run_1", "some goal")

    update_run_status(conn, "run_1", status="running")
    assert get_run(conn, "run_1").status == "running"

    update_run_status(conn, "run_1", status="completed", result={"campaign_id": "camp_123", "degraded": False})
    completed = get_run(conn, "run_1")
    assert completed.status == "completed"
    assert completed.result == {"campaign_id": "camp_123", "degraded": False}
    assert completed.error is None


def test_update_run_status_records_error_on_failure_and_leaves_result_empty(conn):
    create_run(conn, "run_1", "some goal")
    update_run_status(conn, "run_1", status="failed", error="ValueError: something broke")

    failed = get_run(conn, "run_1")
    assert failed.status == "failed"
    assert failed.error == "ValueError: something broke"
    assert failed.result is None


def test_update_run_status_bumps_updated_at_but_not_created_at(conn):
    original = create_run(conn, "run_1", "some goal")
    update_run_status(conn, "run_1", status="running")
    updated = get_run(conn, "run_1")

    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at
