# Segmentation & Personalization

A good campaign starts with a precise segment. Vague segments waste sends, annoy users, and make results impossible to interpret.

## Building segments

- **Combine profile and behavior.** "Free-plan users in India who haven't opened the app in 14 days" is far more useful than "inactive users." Profile attributes (plan, country, platform, signup date) come from the user record; behavior (recency, frequency, feature adoption, purchases) must be computed from events.
- **Define recency explicitly** against the as-of date — e.g., last `app_open` more than 14 days ago.
- **Use feature adoption to find opportunity.** Users who are active but haven't tried a high-value feature are strong targets for education campaigns.
- **Exclude users for whom the message doesn't apply.** Don't pitch a feature to users who already use it heavily; don't re-engage users who are currently active.

## Personalization that works

- Tie the message to the segment's defining trait. If the segment is "hasn't tried voice_agent," the copy should be about voice_agent's benefit — not a generic greeting.
- Use real data points (the feature they last used, their plan tier) to make the message feel specific.
- Keep it truthful. Never invent a user's history or imply knowledge you don't have.

## Segment size sanity checks

- An **empty** segment usually means the definition is wrong or the recency window is off.
- An **implausibly large** segment (e.g., "almost everyone") usually means the filter is too loose.
- Sanity-check the size against expectations before launching. A re-engagement campaign that targets 90% of your base is almost certainly mis-defined.

## Targeting precision vs. reach

There's a tradeoff between a tight, highly-relevant segment (better response, smaller reach) and a broad one (more reach, more noise and opt-outs). Default to precision: a smaller, well-matched segment almost always outperforms a large, loosely-defined blast on both response rate and long-term list health.
