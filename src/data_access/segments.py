"""
Raw SQL for resolving user segments from user_activity_summary /
user_feature_adoption -- called by src/tools/query_segment.py, which owns the
LLM-facing SegmentFilters/SegmentResult schemas and tool semantics. This
module only knows plain dicts and sqlite3.Connection, no pydantic/LLM concepts.
"""
import sqlite3
from typing import List, Optional, Tuple


def build_segment_query(filters: dict) -> Tuple[str, dict]:
    """
    Shared by find_segment_user_ids() for both query_segment() (LLM-facing:
    count + small sample) and resolve_segment_user_ids() (backend-only: the
    FULL matching list, used to snapshot campaign_segment_members at
    create_campaign time) -- one place building the parameterized SQL so the
    two call sites can never drift into resolving a goal's segment differently.
    """
    where, params = ["1=1"], {}
    joins = ""

    if filters.get("recency_days_max") is not None:
        where.append("s.days_since_last_open <= :recency_days_max")
        params["recency_days_max"] = filters["recency_days_max"]
    if filters.get("inactive_days_min") is not None:
        where.append("s.days_since_last_open >= :inactive_days_min")
        params["inactive_days_min"] = filters["inactive_days_min"]
    if filters.get("signed_up_within_days") is not None:
        from src.config import DATASET_AS_OF_DATE
        where.append("(julianday(:as_of) - julianday(s.signup_date)) <= :signed_up_within_days")
        params["as_of"] = DATASET_AS_OF_DATE.isoformat()
        params["signed_up_within_days"] = filters["signed_up_within_days"]
    if filters.get("plan"):
        where.append("s.plan = :plan")
        params["plan"] = filters["plan"]
    if filters.get("country"):
        where.append("s.country = :country")
        params["country"] = filters["country"]
    if filters.get("platform"):
        where.append("s.platform = :platform")
        params["platform"] = filters["platform"]
    if filters.get("min_opens_last_30d") is not None:
        where.append("s.opens_last_30d >= :min_opens_last_30d")
        params["min_opens_last_30d"] = filters["min_opens_last_30d"]
    if filters.get("min_lifetime_spend") is not None:
        where.append("s.lifetime_spend >= :min_lifetime_spend")
        params["min_lifetime_spend"] = filters["min_lifetime_spend"]
    if filters.get("push_open_rate_max") is not None:
        where.append("(s.push_open_rate_30d IS NULL OR s.push_open_rate_30d <= :push_open_rate_max)")
        params["push_open_rate_max"] = filters["push_open_rate_max"]
    if filters.get("feature_adopted"):
        joins += (
            " INNER JOIN user_feature_adoption fa_yes"
            " ON fa_yes.user_id = s.user_id AND fa_yes.feature_name = :feature_adopted"
        )
        params["feature_adopted"] = filters["feature_adopted"]
    if filters.get("feature_not_adopted"):
        where.append(
            "s.user_id NOT IN (SELECT user_id FROM user_feature_adoption WHERE feature_name = :feature_not_adopted)"
        )
        params["feature_not_adopted"] = filters["feature_not_adopted"]

    sql = f"""
        SELECT s.user_id FROM user_activity_summary s
        {joins}
        WHERE {' AND '.join(where)}
    """
    return sql, params


def find_segment_user_ids(filters: dict, conn: sqlite3.Connection) -> Tuple[List[str], str]:
    """
    Executes the built query and returns (matching user_ids, the SQL used) --
    callers that need to display the query (query_segment's sql_used) don't
    have to rebuild it separately, so there's exactly one execution path.
    """
    sql, params = build_segment_query(filters)
    rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows], sql.strip()
