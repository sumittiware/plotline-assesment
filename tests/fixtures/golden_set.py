"""
Golden-set fixtures for the eval harness (DESIGN.md SS10) -- a small,
hand-written set of realistic marketer goals with expected PROPERTIES, not
exact string matches, since real agent output isn't deterministic.

Each fixture pairs the SAME goal with two things:
- mock_responses: a scripted "well-behaved agent" trajectory, replayed by
  MockLLMClient for the deterministic/CI-safe mode (test_eval_goldenset.py) --
  this regression-tests the orchestration+tools+grounding+compliance
  pipeline, not the LLM's own reasoning (which is scripted, not produced).
- expected: property assertions checked against the resulting state. The
  SAME properties get checked, relaxed, against a REAL LLM run in live mode
  -- that's what actually evaluates reasoning quality, per DESIGN.md's own
  framing ("live mode... asserts relaxed to property-checks").

guideline_citations are deliberately left empty in every scripted
create_campaign call: SS5.4's compliance override auto-populates a real
citation for external channels (push/email), and an empty citations list
trivially satisfies SS5.3 grounding for non-external channels (there's
nothing to fabricate). This avoids hardcoding a chunk_id here that would
only be valid against one specific embedder/corpus snapshot.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from langchain_core.messages import AIMessage

ScriptedResponse = Union[AIMessage, Exception]


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@dataclass
class ExpectedProperties:
    filters_include: Sequence[str] = ()  # filter keys that must appear in query_segment's filters_applied
    channel: Optional[str] = None  # exact channel -- checked strictly in deterministic mode only
    offer_required: bool = False
    image_required: bool = False
    compliance_expected: bool = False  # channel is external (push/email) -> a compliance citation must exist


@dataclass
class GoldenFixture:
    name: str
    goal: str
    mock_responses: List[ScriptedResponse]
    expected: ExpectedProperties


GOLDEN_FIXTURES: List[GoldenFixture] = [
    GoldenFixture(
        # Matches the assignment's own headline example verbatim (goal, image, and
        # discount all called out explicitly) -- see Assignment.md's "The Problem".
        name="winback_push_image_discount",
        goal=(
            "Win back users who were active last month but haven't opened the app in the last "
            "14 days. Send them a push notification with an image and a discount offer to bring "
            "them back."
        ),
        mock_responses=[
            _tool_call("query_segment", {"recency_days_max": 30, "inactive_days_min": 14}, "1"),
            _tool_call("search_guidelines", {"query": "winback churned users discount push", "k": 4}, "2"),
            _tool_call(
                "create_campaign",
                {
                    "goal_text": "Win back users active last month, no open in 14 days, push with image and discount.",
                    "segment_def": {"recency_days_max": 30, "inactive_days_min": 14},
                    "segment_size": 0,
                    "channel": "push",
                    "message_copy": "We miss you! Come back for 20% off.",
                    "image_prompt": "A warm, inviting image of the app's home screen with a '20% off' badge overlay.",
                    "offer": {"type": "discount", "value": "20%"},
                    "guideline_citations": [],
                },
                "3",
            ),
            AIMessage(content="Created a push win-back campaign with an image and a 20% discount."),
        ],
        expected=ExpectedProperties(
            filters_include=["inactive_days_min"],
            channel="push",
            offer_required=True,
            image_required=True,
            compliance_expected=True,
        ),
    ),
    GoldenFixture(
        name="onboarding_nudge_in_app",
        goal="Nudge users who signed up in the last 3 days to finish onboarding, using an in-app message.",
        mock_responses=[
            _tool_call("query_segment", {"signed_up_within_days": 3}, "1"),
            _tool_call("search_guidelines", {"query": "onboarding new users in-app", "k": 4}, "2"),
            _tool_call(
                "create_campaign",
                {
                    "goal_text": "Nudge users who signed up in the last 3 days, in-app.",
                    "segment_def": {"signed_up_within_days": 3},
                    "segment_size": 0,
                    "channel": "in_app",
                    "message_copy": "Finish setting up your account to get the most out of the app!",
                    "guideline_citations": [],
                },
                "3",
            ),
            AIMessage(content="Created an in-app onboarding nudge for new signups."),
        ],
        expected=ExpectedProperties(
            filters_include=["signed_up_within_days"], channel="in_app", offer_required=False, compliance_expected=False
        ),
    ),
    GoldenFixture(
        name="feature_adoption_email",
        goal="Email free-plan users who have never tried the voice_agent feature, encouraging them to try it.",
        mock_responses=[
            _tool_call("query_segment", {"plan": "free", "feature_not_adopted": "voice_agent"}, "1"),
            _tool_call("search_guidelines", {"query": "feature adoption email campaign", "k": 4}, "2"),
            _tool_call(
                "create_campaign",
                {
                    "goal_text": "Email free-plan users who haven't tried voice_agent.",
                    "segment_def": {"plan": "free", "feature_not_adopted": "voice_agent"},
                    "segment_size": 0,
                    "channel": "email",
                    "message_copy": "Try voice_agent -- hands-free control, right in the app.",
                    "guideline_citations": [],
                },
                "3",
            ),
            AIMessage(content="Created an email feature-adoption campaign for voice_agent."),
        ],
        expected=ExpectedProperties(
            filters_include=["feature_not_adopted"], channel="email", offer_required=False, compliance_expected=True
        ),
    ),
    GoldenFixture(
        name="plan_upsell_push",
        goal="Offer pro-plan users on Android a limited-time discount on the annual plan, push notification.",
        mock_responses=[
            _tool_call("query_segment", {"plan": "pro", "platform": "Android"}, "1"),
            _tool_call("search_guidelines", {"query": "incentives discount upsell push", "k": 4}, "2"),
            _tool_call(
                "create_campaign",
                {
                    "goal_text": "Offer pro-plan Android users a discount on annual plan, push.",
                    "segment_def": {"plan": "pro", "platform": "Android"},
                    "segment_size": 0,
                    "channel": "push",
                    "message_copy": "Switch to annual and save -- limited time offer!",
                    "offer": {"type": "discount", "value": "15% off annual"},
                    "guideline_citations": [],
                },
                "3",
            ),
            AIMessage(content="Created a push upsell campaign for pro-plan Android users."),
        ],
        expected=ExpectedProperties(
            filters_include=["plan", "platform"], channel="push", offer_required=True, compliance_expected=True
        ),
    ),
    GoldenFixture(
        name="localization_in_app",
        goal="Send an in-app message to users in Germany on iOS about a new localized feature.",
        mock_responses=[
            _tool_call("query_segment", {"country": "DE", "platform": "iOS"}, "1"),
            _tool_call("search_guidelines", {"query": "localization timezone in-app messaging", "k": 4}, "2"),
            _tool_call(
                "create_campaign",
                {
                    "goal_text": "In-app message to German iOS users about a localized feature.",
                    "segment_def": {"country": "DE", "platform": "iOS"},
                    "segment_size": 0,
                    "channel": "in_app",
                    "message_copy": "Jetzt verfuegbar: die neue Funktion, extra fuer dich lokalisiert!",
                    "guideline_citations": [],
                },
                "3",
            ),
            AIMessage(content="Created an in-app localized feature announcement for German iOS users."),
        ],
        expected=ExpectedProperties(
            filters_include=["country", "platform"], channel="in_app", offer_required=False, compliance_expected=False
        ),
    ),
    GoldenFixture(
        name="push_fatigue_reengagement_email",
        goal=(
            "Re-engage users who tend to ignore push notifications (low push open rate) via email "
            "instead, with a friendly check-in."
        ),
        mock_responses=[
            _tool_call("query_segment", {"push_open_rate_max": 0.05}, "1"),
            _tool_call("search_guidelines", {"query": "frequency capping push fatigue channel choice", "k": 4}, "2"),
            _tool_call(
                "create_campaign",
                {
                    "goal_text": "Re-engage low push-open-rate users via email instead.",
                    "segment_def": {"push_open_rate_max": 0.05},
                    "segment_size": 0,
                    "channel": "email",
                    "message_copy": "Just checking in -- here's what's new since you've been away.",
                    "guideline_citations": [],
                },
                "3",
            ),
            AIMessage(content="Created an email re-engagement campaign for push-fatigued users."),
        ],
        expected=ExpectedProperties(
            filters_include=["push_open_rate_max"], channel="email", offer_required=False, compliance_expected=True
        ),
    ),
    GoldenFixture(
        name="broad_feature_announcement_in_app",
        goal="Announce a new feature to all users who opened the app in the last 7 days, in-app banner.",
        mock_responses=[
            _tool_call("query_segment", {"recency_days_max": 7}, "1"),
            _tool_call("search_guidelines", {"query": "feature announcement in-app messaging", "k": 4}, "2"),
            _tool_call(
                "create_campaign",
                {
                    "goal_text": "Announce new feature to recently active users, in-app.",
                    "segment_def": {"recency_days_max": 7},
                    "segment_size": 0,
                    "channel": "in_app",
                    "message_copy": "New: check out our latest feature, live now!",
                    "guideline_citations": [],
                },
                "3",
            ),
            AIMessage(content="Created an in-app feature announcement for recently active users."),
        ],
        expected=ExpectedProperties(
            filters_include=["recency_days_max"], channel="in_app", offer_required=False, compliance_expected=False
        ),
    ),
    GoldenFixture(
        name="long_dormant_winback_email",
        goal="Win back users who haven't opened the app in 90+ days with a strong discount, via email.",
        mock_responses=[
            _tool_call("query_segment", {"inactive_days_min": 90}, "1"),
            _tool_call("search_guidelines", {"query": "winback long dormant churned users incentive", "k": 4}, "2"),
            _tool_call(
                "create_campaign",
                {
                    "goal_text": "Win back 90+ day dormant users with a strong discount, email.",
                    "segment_def": {"inactive_days_min": 90},
                    "segment_size": 0,
                    "channel": "email",
                    "message_copy": "It's been a while -- come back for 30% off your next purchase.",
                    "offer": {"type": "discount", "value": "30%"},
                    "guideline_citations": [],
                },
                "3",
            ),
            AIMessage(content="Created an email win-back campaign for long-dormant users."),
        ],
        expected=ExpectedProperties(
            filters_include=["inactive_days_min"], channel="email", offer_required=True, compliance_expected=True
        ),
    ),
]


# Hand-labeled retrieval spot-checks (DESIGN.md SS10 item 6): a query and the
# topic_slug(s) we expect to see somewhere in the top-k, given how much the
# guideline corpus intentionally overlaps (re-engagement/winback/frequency-
# capping all touch "how often to message"). Queries are phrased to share
# literal vocabulary with their target doc (same rationale as
# tests/test_search_guidelines.py) since CI runs these against the
# deterministic hashing embedder, not a real semantic model -- keyword
# overlap is what that embedder actually captures.
RETRIEVAL_SPOT_CHECKS = [
    ("winning back dormant, churned users", {"winback-churned-users", "re-engagement-playbook"}),
    ("writing effective push notification copy", {"push-notification-copy"}),
    ("frequency capping and message fatigue, caps per week before users unsubscribe", {"frequency-capping-and-fatigue"}),
    ("consent, opt-outs and suppression lists", {"consent-compliance-and-opt-outs"}),
    ("activating brand new signups during onboarding", {"onboarding-new-users"}),
    ("choosing an incentive or discount for a campaign", {"incentives-and-offers"}),
]
