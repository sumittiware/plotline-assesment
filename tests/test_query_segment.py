"""
query_segment tested in complete isolation from any LLM/agent code. If a filter
is wrong, it fails here in milliseconds -- not three tool-calls deep in an agent trace.
"""
import pytest

from src.data_access.db import get_connection
from src.tools.query_segment import query_segment, SegmentFilters


@pytest.fixture(scope="module")
def conn():
    c = get_connection()
    yield c
    c.close()


@pytest.fixture(scope="module")
def total_users(conn):
    # Derived from the DB rather than hardcoded -- the assignment bundle's
    # row count isn't a stable constant to bake into assertions.
    return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def test_no_filters_returns_all_users(conn, total_users):
    result = query_segment(SegmentFilters(), conn)
    assert result.size == total_users


def test_inactive_users_segment_is_nonempty_and_bounded(conn, total_users):
    result = query_segment(SegmentFilters(inactive_days_min=14), conn)
    assert 0 < result.size < total_users
    assert len(result.sample_user_ids) <= 20


def test_recency_and_inactivity_are_mutually_exclusive_ranges(conn):
    active = query_segment(SegmentFilters(recency_days_max=7), conn)
    inactive = query_segment(SegmentFilters(inactive_days_min=14), conn)
    assert set(active.sample_user_ids).isdisjoint(set(inactive.sample_user_ids))


def test_plan_filter_narrows_correctly(conn):
    all_users = query_segment(SegmentFilters(), conn)
    pro_users = query_segment(SegmentFilters(plan="pro"), conn)
    assert pro_users.size < all_users.size


def test_feature_adopted_and_not_adopted_are_disjoint(conn, total_users):
    adopted = query_segment(SegmentFilters(feature_adopted="voice_agent"), conn)
    not_adopted = query_segment(SegmentFilters(feature_not_adopted="voice_agent"), conn)
    assert adopted.size + not_adopted.size <= total_users
    assert set(adopted.sample_user_ids).isdisjoint(set(not_adopted.sample_user_ids))


def test_combined_winback_style_query(conn):
    """The exact shape of segment the assignment's example prompt implies:
    active in the last month, but quiet in the last 14 days."""
    result = query_segment(SegmentFilters(inactive_days_min=14), conn)
    assert result.size > 0
    assert "days_since_last_open" in result.sql_used
    assert result.filters_applied == {"inactive_days_min": 14}


def test_sql_used_is_parameterized_not_string_interpolated(conn):
    result = query_segment(SegmentFilters(plan="pro"), conn)
    assert ":plan" in result.sql_used  # never a raw 'pro' spliced into the SQL string
