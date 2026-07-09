# Lifecycle Stages — Glossary

A shared vocabulary for the user states our campaigns target. These stages are derived from behavior in the events log (relative to the dataset's as-of date), not stored on the user record — you compute them from activity.

## Stages

- **New:** signed up recently (typically within the last 7 days), still forming a first impression. Goal: activation. See onboarding guidance.
- **Active:** opening the app and taking actions regularly (within the last ~14 days). Goal: deepen usage and adoption.
- **Power user:** highly active, frequent sessions, broad feature usage. Goal: retain, learn from, and turn into advocates. Rarely need nudging.
- **Lapsing / recently lapsed:** were active but have gone quiet (roughly 7–14 days without an `app_open`). Goal: a light re-engagement nudge before the habit breaks.
- **Dormant:** inactive for a longer stretch (roughly 15–45 days), but recently enough to remember the product. Goal: re-engagement with stronger value.
- **Churned:** no activity in 30+ days; effectively gone. Goal: win-back with a fresh reason to return, off-app.

## Cross-cutting attributes

These overlay any lifecycle stage:

- **Payer vs. non-payer:** whether the user has any purchase history.
- **Plan tier:** free / pro / enterprise — affects what you can offer and pitch.
- **Feature adoption:** which features the user has and hasn't used — the basis for education campaigns.

## Why stages matter

The same message performs very differently across stages. "Try our new feature" is great for an active user, wasted on a churned one who won't see an in-app message, and premature for a brand-new user still learning the basics. Always identify the stage first, then choose the goal, channel, and message to match.
