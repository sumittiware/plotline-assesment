"""
Thin DB access layer. Two responsibilities only:
  1. Give the rest of the app a connection.
  2. Rebuild the derived tables (user_activity_summary, user_feature_adoption)
     from raw events, using the fixed DATASET_AS_OF_DATE so results are reproducible.

No ORM on purpose -- the schema is small and stable, and raw SQL keeps the
query_segment tool's generated WHERE clauses easy to reason about in tests.
"""
import os
import sqlite3
from contextlib import contextmanager

from src.config import SQLITE_PATH, DATASET_AS_OF_DATE

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def get_connection(db_path: str = SQLITE_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    # check_same_thread=False: tool calls now run inside resilience.py's
    # ThreadPoolExecutor (for the per-call timeout), so a connection created
    # on the request thread gets *used* from a worker thread. Safe here since
    # access is always sequential (the caller blocks on future.result()),
    # never truly concurrent from two threads at once.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")  # raw dataset may not be perfectly referentially clean
    return conn


@contextmanager
def connection_scope(db_path: str = SQLITE_PATH):
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def apply_schema(conn: sqlite3.Connection) -> None:
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()


def rebuild_derived_tables(conn: sqlite3.Connection, as_of: str = None) -> None:
    """
    Recomputes user_activity_summary and user_feature_adoption from raw events.
    Called by `make ingest` for local dev, and documented (not built) as a
    scheduled job in a real deployment -- see README "out of scope".
    """
    as_of = as_of or DATASET_AS_OF_DATE.isoformat()

    conn.execute("DELETE FROM user_activity_summary")
    conn.execute(
        """
        INSERT INTO user_activity_summary (
            user_id, signup_date, country, platform, plan,
            last_open_at, days_since_last_open, opens_last_30d, sessions_last_30d,
            last_purchase_at, lifetime_spend, push_open_rate_30d
        )
        SELECT
            u.user_id, u.signup_date, u.country, u.platform, u.plan,
            MAX(CASE WHEN e.event_name = 'app_open' THEN e.timestamp END),
            julianday(:as_of) - julianday(MAX(CASE WHEN e.event_name = 'app_open' THEN e.timestamp END)),
            COUNT(CASE WHEN e.event_name = 'app_open' AND e.timestamp >= datetime(:as_of, '-30 days') THEN 1 END),
            COUNT(CASE WHEN e.event_name = 'session_start' AND e.timestamp >= datetime(:as_of, '-30 days') THEN 1 END),
            MAX(CASE WHEN e.event_name = 'purchase' THEN e.timestamp END),
            SUM(CASE WHEN e.event_name = 'purchase' THEN json_extract(e.properties, '$.amount') END),
            CAST(COUNT(CASE WHEN e.event_name = 'notification_opened' AND e.timestamp >= datetime(:as_of, '-30 days') THEN 1 END) AS REAL)
                / NULLIF(COUNT(CASE WHEN e.event_name = 'notification_received' AND e.timestamp >= datetime(:as_of, '-30 days') THEN 1 END), 0)
        FROM users u
        LEFT JOIN events e ON e.user_id = u.user_id
        GROUP BY u.user_id
        """,
        {"as_of": as_of},
    )

    conn.execute("DELETE FROM user_feature_adoption")
    conn.execute(
        """
        INSERT INTO user_feature_adoption (user_id, feature_name, first_used_at)
        SELECT user_id, json_extract(properties, '$.feature_name') AS feature_name, MIN(timestamp)
        FROM events
        WHERE event_name = 'feature_used' AND json_extract(properties, '$.feature_name') IS NOT NULL
        GROUP BY user_id, feature_name
        """
    )
    conn.commit()


if __name__ == "__main__":
    # `python -m src.data_access.db` (wired to `make db-rebuild`): apply the
    # schema (creates campaigns/derived tables if missing; raw users/events/
    # features are IF NOT EXISTS, so this is safe to run against the already-
    # provided data.sqlite) and rebuild the derived tables from raw events.
    with connection_scope() as conn:
        apply_schema(conn)
        rebuild_derived_tables(conn)
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        print(f"Rebuilt derived tables for {total_users} users -> {SQLITE_PATH}")
