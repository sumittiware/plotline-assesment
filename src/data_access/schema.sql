-- Raw tables (matches DATA_README.md). If data.sqlite is already provided by the
-- assignment bundle, these CREATE TABLE statements are no-ops on top of it
-- (IF NOT EXISTS) -- this schema file is also what our synthetic data generator
-- uses when the real bundle isn't present yet.

CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    signup_date  TEXT NOT NULL,
    country      TEXT NOT NULL,
    platform     TEXT NOT NULL,
    app_version  TEXT NOT NULL,
    plan         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id),
    event_name  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    properties  TEXT  -- JSON blob
);
CREATE INDEX IF NOT EXISTS idx_events_user_id ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_name_ts ON events(event_name, timestamp);

CREATE TABLE IF NOT EXISTS features (
    feature_name TEXT PRIMARY KEY,
    description  TEXT
);

-- Derived tables -- rebuilt by `make ingest` / src/data_access/db.py::rebuild_derived_tables.
-- Kept as real tables (not views) so query_segment hits an indexed, pre-aggregated
-- table instead of re-scanning the full events log on every agent call.

DROP TABLE IF EXISTS user_activity_summary;
CREATE TABLE user_activity_summary (
    user_id              TEXT PRIMARY KEY,
    signup_date          TEXT,
    country              TEXT,
    platform             TEXT,
    plan                 TEXT,
    last_open_at         TEXT,
    days_since_last_open REAL,
    opens_last_30d       INTEGER,
    sessions_last_30d    INTEGER,
    last_purchase_at     TEXT,
    lifetime_spend       REAL,
    push_open_rate_30d   REAL
);

DROP TABLE IF EXISTS user_feature_adoption;
CREATE TABLE user_feature_adoption (
    user_id       TEXT NOT NULL,
    feature_name  TEXT NOT NULL,
    first_used_at TEXT NOT NULL,
    PRIMARY KEY (user_id, feature_name)
);

-- Campaigns -- the idempotent write target for the create_campaign tool.
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id         TEXT PRIMARY KEY,
    idempotency_key     TEXT UNIQUE NOT NULL,
    goal_text           TEXT,
    segment_def         TEXT NOT NULL,   -- JSON: the filters used
    segment_size        INTEGER NOT NULL,
    channel             TEXT NOT NULL,
    copy                TEXT NOT NULL,
    image_prompt         TEXT,
    offer                TEXT,           -- JSON
    guideline_citations  TEXT,           -- JSON list of {source_doc, section_header}
    status               TEXT NOT NULL,  -- created | failed
    created_at           TEXT NOT NULL
);

-- Segment membership -- the actual user_ids a campaign targeted, snapshotted
-- at creation time (not just segment_def + segment_size). segment_def alone
-- only tells you the FILTER; re-running it later can drift from what was
-- true at creation time, since user_activity_summary is periodically rebuilt
-- from the ever-growing events log. One row per (campaign, user) rather than
-- a JSON blob column so membership is directly queryable/indexable -- e.g.
-- "which campaigns has user X been targeted by" is a plain SQL query, not a
-- JSON scan.
CREATE TABLE IF NOT EXISTS campaign_segment_members (
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    user_id     TEXT NOT NULL,
    PRIMARY KEY (campaign_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_campaign_segment_members_user ON campaign_segment_members(user_id);

-- Async job tracking for POST /copilot/run -- an agent run can take anywhere
-- from a few seconds to well over a minute (real LLM round-trips, retries),
-- so the endpoint enqueues the run and returns immediately rather than
-- holding the HTTP connection open for the whole duration. This table is
-- what GET /copilot/run/{run_id} polls.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    goal            TEXT NOT NULL,
    idempotency_key TEXT,
    status          TEXT NOT NULL,  -- pending | running | completed | failed
    result          TEXT,           -- JSON: the full result payload once completed
    error           TEXT,           -- populated only if status = failed (the WORKER crashed --
                                     -- not to be confused with the agent's own "degraded" outcome,
                                     -- which is a normal, well-formed part of `result`)
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
