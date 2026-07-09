# Dataset

Synthetic data for the Campaign Copilot assignment. All timestamps are relative to a fixed **as-of date: 2026-06-24** — treat that as "today" when reasoning about recency (e.g., "last 14 days").

Provided in CSV and JSON, plus a prebuilt `data.sqlite` for convenience. Use whichever you like; you're free to load and model it however you see fit.

## `users` — profile attributes (one row per user)

| Field | Type | Notes |
|-------|------|-------|
| user_id | string | identity + join key |
| signup_date | date | `YYYY-MM-DD` |
| country | string | e.g. IN, US, ID, BR, UK, DE, NG |
| platform | string | Android / iOS / Web |
| app_version | string | e.g. 3.4.0 |
| plan | string | free / pro / enterprise |

The `users` table holds only profile attributes. **Anything behavioural — when a user was last active, how often, which features they've used, whether they've purchased — must be computed from `events`.**

## `events` — behavioural log (append-only)

| Field | Type | Notes |
|-------|------|-------|
| event_id | string | unique |
| user_id | string | references `users.user_id` |
| event_name | string | `app_open`, `session_start`, `feature_used`, `purchase`, `notification_received`, `notification_opened` |
| timestamp | ISO 8601 | e.g. `2026-06-02T09:14:00Z` |
| properties | JSON | event-specific, e.g. `{"feature_name":"voice_agent"}` or `{"amount":4900,"currency":"INR","item":"pro_monthly"}` |

## `features` — catalog

A flat list of the product features referenced in `feature_used` events, so "users who haven't tried feature X" is well-defined.

---

The data contains realistic lifecycle cohorts (new, active, dormant, churned users; payers; varying feature adoption), so natural-language targeting goals map to meaningful segments.
