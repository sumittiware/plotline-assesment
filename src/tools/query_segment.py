"""
query_segment -- plain Python function, tested and correct on its own before any
LLM ever calls it. Wrapped as a LangChain tool in registry.py; this file has zero
knowledge that an agent exists.

Design choice: the LLM never writes raw SQL. It can only populate this constrained
SegmentFilters schema, which src/data_access/segments.py translates into a
parameterized query. That's what keeps a prompt-injected or just-wrong LLM output
from ever touching the database with arbitrary SQL. This module owns the
LLM-facing schemas (SegmentFilters/SegmentResult) and tool semantics only -- the
raw SQL itself lives in src/data_access/segments.py, same layering as
src/tools/create_campaign.py + src/data_access/campaigns.py.
"""
import sqlite3
from typing import List, Optional

from pydantic import BaseModel, Field

from src.data_access.segments import find_segment_user_ids


class SegmentFilters(BaseModel):
    recency_days_max: Optional[int] = Field(
        None, description="Max days since last app_open. E.g. 7 = opened within the last week."
    )
    inactive_days_min: Optional[int] = Field(
        None, description="Min days since last app_open. E.g. 14 = hasn't opened the app in 14+ days."
    )
    signed_up_within_days: Optional[int] = Field(
        None, description="User signed up within the last N days (for onboarding-style segments)."
    )
    plan: Optional[str] = Field(None, description="One of: free, pro, enterprise.")
    country: Optional[str] = Field(None, description="Two-letter country code, e.g. IN, US.")
    platform: Optional[str] = Field(None, description="One of: Android, iOS, Web.")
    min_opens_last_30d: Optional[int] = None
    min_lifetime_spend: Optional[float] = None
    feature_adopted: Optional[str] = Field(None, description="Feature name the user HAS used at least once.")
    feature_not_adopted: Optional[str] = Field(None, description="Feature name the user has NEVER used.")
    push_open_rate_max: Optional[float] = Field(
        None, description="Users whose push_open_rate_30d is at or below this (0-1). Useful for channel selection."
    )


class SegmentResult(BaseModel):
    size: int
    sample_user_ids: list[str]
    filters_applied: dict
    sql_used: str


def query_segment(filters: SegmentFilters, conn: sqlite3.Connection) -> SegmentResult:
    filters_dict = filters.model_dump(exclude_none=True)
    user_ids, sql_used = find_segment_user_ids(filters_dict, conn)

    return SegmentResult(
        size=len(user_ids),
        sample_user_ids=user_ids[:20],
        filters_applied=filters_dict,
        sql_used=sql_used,
    )


def resolve_segment_user_ids(filters: SegmentFilters, conn: sqlite3.Connection) -> List[str]:
    """
    The FULL matching user_id list (no 20-row cap) -- backend-only, never
    sent to the LLM (that's what query_segment's sample_user_ids is for,
    deliberately capped to control token cost). Used by create_campaign to
    snapshot exactly who a campaign targeted at creation time.
    """
    user_ids, _ = find_segment_user_ids(filters.model_dump(exclude_none=True), conn)
    return user_ids
