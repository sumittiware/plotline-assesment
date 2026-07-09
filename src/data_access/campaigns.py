"""
Raw SQL for the `campaigns` + `campaign_segment_members` tables -- called by
src/tools/create_campaign.py, which owns the LLM-facing CampaignInput/
CampaignResult schemas, idempotency-key derivation, and the transaction/retry
semantics. This module only knows plain values and sqlite3.Connection.
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Tuple


def insert_campaign(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    idempotency_key: str,
    goal_text: str,
    segment_def: dict,
    segment_size: int,
    channel: str,
    message_copy: str,
    image_prompt: Optional[str],
    offer: Optional[dict],
    guideline_citations: list,
) -> None:
    conn.execute(
        """
        INSERT INTO campaigns (
            campaign_id, idempotency_key, goal_text, segment_def, segment_size,
            channel, copy, image_prompt, offer, guideline_citations, status, created_at
        ) VALUES (
            :campaign_id, :idempotency_key, :goal_text, :segment_def, :segment_size,
            :channel, :message_copy, :image_prompt, :offer, :guideline_citations, 'created', :created_at
        )
        """,
        {
            "campaign_id": campaign_id,
            "idempotency_key": idempotency_key,
            "goal_text": goal_text,
            "segment_def": json.dumps(segment_def),
            "segment_size": segment_size,
            "channel": channel,
            "message_copy": message_copy,
            "image_prompt": image_prompt,
            "offer": json.dumps(offer) if offer else None,
            "guideline_citations": json.dumps(guideline_citations),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def insert_segment_members(conn: sqlite3.Connection, campaign_id: str, user_ids: List[str]) -> None:
    conn.executemany(
        "INSERT INTO campaign_segment_members (campaign_id, user_id) VALUES (?, ?)",
        [(campaign_id, user_id) for user_id in user_ids],
    )


def get_campaign_by_idempotency_key(conn: sqlite3.Connection, idempotency_key: str) -> Optional[Tuple[str, str]]:
    row = conn.execute(
        "SELECT campaign_id, status FROM campaigns WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    return (row[0], row[1]) if row is not None else None


def count_segment_members(conn: sqlite3.Connection, campaign_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM campaign_segment_members WHERE campaign_id = ?",
        (campaign_id,),
    ).fetchone()[0]
