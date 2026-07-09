"""
create_campaign -- the idempotency-critical tool.

Design: INSERT first, catch the uniqueness violation, THEN read. Never
"SELECT-then-decide-to-INSERT" -- that ordering has a race window between the
check and the write. The unique constraint on campaigns.idempotency_key (see
schema.sql) is the actual source of truth; this function just reacts to it.
Raw SQL lives in src/data_access/campaigns.py -- this module owns the
LLM-facing schemas, idempotency-key derivation, and the transaction/retry
control flow around those data-access calls.

Segment snapshotting: segment_def (the filter DSL) and segment_size (a count)
alone don't tell you WHO a campaign actually targeted -- re-running the same
filters later can drift, since user_activity_summary is periodically rebuilt
from the ever-growing events log. On a real (non-replay) creation, this
function resolves segment_def into the actual FULL user_id list (via
resolve_segment_user_ids -- never query_segment's LLM-facing, 20-row-capped
sample) and snapshots it into campaign_segment_members, in the SAME
transaction as the campaign row -- either both land or neither does.
"""
import hashlib
import json
import sqlite3
import uuid
from typing import Optional

from pydantic import BaseModel

from src.data_access.campaigns import (
    count_segment_members,
    get_campaign_by_idempotency_key,
    insert_campaign,
    insert_segment_members,
)
from src.tools.query_segment import SegmentFilters, resolve_segment_user_ids


class CampaignInput(BaseModel):
    goal_text: str
    segment_def: dict
    segment_size: int
    channel: str
    message_copy: str
    image_prompt: Optional[str] = None
    offer: Optional[dict] = None
    guideline_citations: list[dict] = []
    idempotency_key: Optional[str] = None  # if not supplied, derived deterministically below


class CampaignResult(BaseModel):
    campaign_id: str
    status: str
    idempotent_replay: bool = False
    segment_member_count: int = 0


def derive_idempotency_key(goal_text: str, segment_def: dict, channel: str) -> str:
    """
    Deterministic fallback key so idempotency holds even if a caller doesn't supply
    one explicitly. Two requests describing the same goal + segment + channel are,
    by definition, the same campaign.
    """
    normalized = json.dumps(
        {"goal": goal_text.strip().lower(), "segment_def": segment_def, "channel": channel},
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def create_campaign(payload: CampaignInput, conn: sqlite3.Connection) -> CampaignResult:
    idempotency_key = payload.idempotency_key or derive_idempotency_key(
        payload.goal_text, payload.segment_def, payload.channel
    )
    campaign_id = f"camp_{uuid.uuid4()}"

    try:
        insert_campaign(
            conn,
            campaign_id=campaign_id,
            idempotency_key=idempotency_key,
            goal_text=payload.goal_text,
            segment_def=payload.segment_def,
            segment_size=payload.segment_size,
            channel=payload.channel,
            message_copy=payload.message_copy,
            image_prompt=payload.image_prompt,
            offer=payload.offer,
            guideline_citations=payload.guideline_citations,
        )

        # Snapshot exactly who this campaign targeted, atomically with the
        # campaign row itself (same transaction, committed together below).
        member_ids = resolve_segment_user_ids(SegmentFilters(**payload.segment_def), conn)
        insert_segment_members(conn, campaign_id, member_ids)

        conn.commit()
        return CampaignResult(campaign_id=campaign_id, status="created", segment_member_count=len(member_ids))

    except sqlite3.IntegrityError:
        # Someone (a retry, a race, or a genuine duplicate goal) already holds this key.
        # The failed INSERT *is* the idempotency check -- now just read back the winner,
        # including its ALREADY-snapshotted membership (a replay must not re-snapshot).
        conn.rollback()
        existing_campaign_id, existing_status = get_campaign_by_idempotency_key(conn, idempotency_key)
        member_count = count_segment_members(conn, existing_campaign_id)
        return CampaignResult(
            campaign_id=existing_campaign_id,
            status=existing_status,
            idempotent_replay=True,
            segment_member_count=member_count,
        )
