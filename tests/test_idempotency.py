import concurrent.futures
import sqlite3

import pytest

from src.data_access.db import apply_schema
from src.tools.create_campaign import CampaignInput, create_campaign, derive_idempotency_key


@pytest.fixture
def fresh_conn():
    """In-memory DB per test -- isolated from the real data.sqlite and from other tests."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    yield conn
    conn.close()


def make_payload(**overrides) -> CampaignInput:
    base = dict(
        goal_text="Win back dormant users",
        segment_def={"inactive_days_min": 14},
        segment_size=125,
        channel="push",
        message_copy="We miss you! Come back for 20% off.",
        offer={"type": "discount", "value": "20%"},
    )
    base.update(overrides)
    return CampaignInput(**base)


def test_create_campaign_succeeds(fresh_conn):
    result = create_campaign(make_payload(), fresh_conn)
    assert result.status == "created"
    assert result.idempotent_replay is False


def test_same_key_called_twice_does_not_duplicate(fresh_conn):
    payload = make_payload(idempotency_key="fixed-key-123")
    r1 = create_campaign(payload, fresh_conn)
    r2 = create_campaign(payload, fresh_conn)

    assert r1.campaign_id == r2.campaign_id
    assert r2.idempotent_replay is True

    count = fresh_conn.execute(
        "SELECT COUNT(*) FROM campaigns WHERE idempotency_key = ?", ("fixed-key-123",)
    ).fetchone()[0]
    assert count == 1


def test_derived_key_is_stable_for_same_semantic_request(fresh_conn):
    """No explicit idempotency_key supplied -- same goal/segment/channel should still
    collapse to one campaign, because the key is derived deterministically."""
    p1 = make_payload()
    p2 = make_payload()  # separately constructed, same content
    assert derive_idempotency_key(p1.goal_text, p1.segment_def, p1.channel) == \
           derive_idempotency_key(p2.goal_text, p2.segment_def, p2.channel)

    r1 = create_campaign(p1, fresh_conn)
    r2 = create_campaign(p2, fresh_conn)
    assert r1.campaign_id == r2.campaign_id


def test_different_channel_is_a_different_campaign(fresh_conn):
    push = create_campaign(make_payload(channel="push"), fresh_conn)
    email = create_campaign(make_payload(channel="email"), fresh_conn)
    assert push.campaign_id != email.campaign_id


def _seed_users(conn, user_ids_and_inactivity: dict):
    """Insert minimal user_activity_summary rows directly -- enough for
    resolve_segment_user_ids() to have real data to match against."""
    for user_id, days_since_last_open in user_ids_and_inactivity.items():
        conn.execute(
            "INSERT INTO user_activity_summary (user_id, days_since_last_open) VALUES (?, ?)",
            (user_id, days_since_last_open),
        )
    conn.commit()


def test_create_campaign_snapshots_the_actual_segment_members(fresh_conn):
    """
    segment_def + segment_size alone don't tell you WHO was targeted --
    create_campaign must snapshot the real matching user_ids into
    campaign_segment_members, not just a count.
    """
    _seed_users(fresh_conn, {"u_dormant_1": 20, "u_dormant_2": 30, "u_active": 2})

    result = create_campaign(make_payload(segment_def={"inactive_days_min": 14}, segment_size=2), fresh_conn)

    members = {
        r[0]
        for r in fresh_conn.execute(
            "SELECT user_id FROM campaign_segment_members WHERE campaign_id = ?", (result.campaign_id,)
        ).fetchall()
    }
    assert members == {"u_dormant_1", "u_dormant_2"}  # matches the filter; u_active correctly excluded
    assert result.segment_member_count == 2


def test_idempotent_replay_does_not_duplicate_or_resnapshot_members(fresh_conn):
    """A retried create_campaign call must reuse the ORIGINAL snapshot, not
    write a second (possibly different, if data changed in between) one."""
    _seed_users(fresh_conn, {"u1": 20, "u2": 30})
    payload = make_payload(segment_def={"inactive_days_min": 14}, segment_size=2, idempotency_key="fixed-key-999")

    first = create_campaign(payload, fresh_conn)
    assert first.segment_member_count == 2

    # Data changes between the two calls -- a real replay must NOT re-resolve
    # against this new state; it should still report the ORIGINAL snapshot.
    fresh_conn.execute("INSERT INTO user_activity_summary (user_id, days_since_last_open) VALUES (?, ?)", ("u3", 40))
    fresh_conn.commit()

    second = create_campaign(payload, fresh_conn)
    assert second.idempotent_replay is True
    assert second.campaign_id == first.campaign_id
    assert second.segment_member_count == 2  # unchanged, not re-resolved to 3

    total_rows = fresh_conn.execute(
        "SELECT COUNT(*) FROM campaign_segment_members WHERE campaign_id = ?", (first.campaign_id,)
    ).fetchone()[0]
    assert total_rows == 2  # no duplicate insert on replay


def test_concurrent_creates_with_same_key_only_produce_one_row(tmp_path):
    """
    Proves race-safety, not just single-threaded correctness: fire the same
    idempotency key from multiple threads at once and assert exactly one row lands.
    Uses a file-backed DB (not :memory:) since each thread needs its own connection
    to the *same* database, which SQLite in-memory doesn't share across connections.
    """
    db_path = tmp_path / "race_test.sqlite"
    setup_conn = sqlite3.connect(str(db_path))
    apply_schema(setup_conn)
    setup_conn.close()

    key = "race-key-456"

    def attempt():
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            return create_campaign(make_payload(idempotency_key=key), conn)
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: attempt(), range(8)))

    campaign_ids = {r.campaign_id for r in results}
    assert len(campaign_ids) == 1, "concurrent requests with the same key produced different campaigns"

    verify_conn = sqlite3.connect(str(db_path))
    count = verify_conn.execute(
        "SELECT COUNT(*) FROM campaigns WHERE idempotency_key = ?", (key,)
    ).fetchone()[0]
    assert count == 1
    verify_conn.close()
