# Push Notification Copy

Push is interruptive and space-constrained. Respect both, and treat every push as borrowing the user's attention — spend it well or lose the permission.

## Hard limits

- **Title:** ≤ 50 characters. **Body:** ≤ 120 characters. Assume anything beyond is truncated, and that truncation happens at the worst possible word.
- One call to action. One idea per notification.
- The title should carry the message on its own; assume some users only read the title.

## Effective patterns

- **Specific over vague:** "Your draft campaign is ready to launch" beats "You have updates."
- **Benefit first:** lead the title with the value; use the body for supporting detail.
- **Curiosity with payoff** — never bait. The notification must deliver exactly what it implies, or you train users to ignore you.
- **Match the user's context:** reference their platform, their last action, or the feature relevant to them.

## What to avoid

- Emoji as a substitute for clarity (one at most, only if it adds meaning).
- Sending during the user's likely night-time hours (see channel & timing).
- Generic blasts that ignore the user's history or platform.
- Stacking multiple pushes close together — each additional push sharply raises opt-out and uninstall risk.

## Deep links

A push should take the user directly to the relevant screen, not the app's home. If the message is about a draft campaign, deep-link to that draft. Mismatched destinations are a top cause of immediate app-close and notification disablement.

## Fallback

Not every user has push enabled. A push-led campaign should degrade gracefully — fall back to in-app or email for users who've disabled notifications, rather than silently failing to reach them.
