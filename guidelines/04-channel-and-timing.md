# Channel & Timing

The right message on the wrong channel — or at the wrong time — is a wasted send. Choose both deliberately based on the user's current behavior.

## Choosing a channel

- **In-app message:** best for users who already open the app. Highest engagement, lowest intrusion. Prefer for active and recently-active users. Useless for users who aren't opening the app.
- **Push:** good for re-engaging lapsed users who still have notifications enabled. Interruptive — use sparingly and only when there's genuine value.
- **Email:** best for longer-form value, or for long-dormant and churned users who aren't opening the app at all. Most tolerant of length.

The guiding rule: **match the channel to where the user actually is.** If someone hasn't opened the app in three weeks, an in-app message will never be seen — reach them by push or email. If someone is active daily, an email is the wrong place for a nudge they'd see faster in-app.

## Timing

- **Respect quiet hours.** Avoid sending between roughly 9pm and 8am in the user's local time. Use their `country`/timezone to localize send windows.
- For time-insensitive nudges, prefer mid-morning or early evening local time, when engagement tends to be higher.
- Avoid bunching multiple messages together; spread touches across days, not hours.
- Tie timing to behavior when possible — e.g., shortly after a session, or aligned to when the user is usually active.

## Channel escalation

When a user doesn't respond, escalate **channel**, not frequency: in-app → push → email. Sending the same message three times on the same channel trains users to ignore (or disable) it.

## Suppression

Always respect opt-outs and channel-level consent. A user who disabled push must not receive push, and a campaign should route them elsewhere or skip them entirely.
