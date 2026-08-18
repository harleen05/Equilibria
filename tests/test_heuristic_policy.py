"""
tests/test_heuristic_policy.py — Verifies environment/heuristic_policy.py's
smart_policy_action(), a heavily defensive (try/except-guarded)
rule-based policy shared by demo.py, inference.py, and the server.

Before this file, coverage sat at 72% with the trust-recovery and
high-boredom branches, every exception-fallback path, and action_label()
entirely untested -- exactly the "test the exception-fallback branches
explicitly" gap flagged in the original review. Defensive code that's
never exercised by a test provides no actual assurance it works; it's
just code that LOOKS safe.
"""

from __future__ import annotations

from environment.heuristic_policy import smart_policy_action, action_label, _f, _float


# ── Core priority-order branches ──────────────────────────────────────────

def test_high_fatigue_triggers_pause() -> None:
    obs = {"visible_fatigue": 0.9, "visible_trust": 0.9, "visible_boredom": 0.1}
    result = smart_policy_action(obs)
    assert result["action_type"] == "pause_session"
    assert result["reasoning"] == "high fatigue"


def test_low_trust_triggers_explore_new_topic() -> None:
    obs = {"visible_fatigue": 0.1, "visible_trust": 0.2, "visible_boredom": 0.1}
    result = smart_policy_action(obs)
    assert result["action_type"] == "explore_new_topic"
    assert result["reasoning"] == "trust recovery"


def test_high_boredom_triggers_diversify() -> None:
    """Only reachable when fatigue and trust are both in the safe range --
    fatigue and low-trust checks come first in priority order."""
    obs = {"visible_fatigue": 0.1, "visible_trust": 0.9, "visible_boredom": 0.6}
    result = smart_policy_action(obs)
    assert result["action_type"] == "diversify_feed"
    assert result["reasoning"] == "high boredom"


def test_normal_conditions_recommend_best_ethical_match() -> None:
    obs = {
        "visible_fatigue": 0.1, "visible_trust": 0.9, "visible_boredom": 0.1,
        "interest_distribution": {"technology": 1.0},
        "recent_content_ids": [],
        "available_content": [
            {
                "content_id": "safe_item", "manipulation_score": 0.05, "addictiveness": 0.1,
                "topic_relevance": {"technology": 1.0}, "educational_value": 0.9,
            },
            {
                "content_id": "manipulative_item", "manipulation_score": 0.9, "addictiveness": 0.1,
                "topic_relevance": {"technology": 1.0}, "educational_value": 0.9,
            },
        ],
    }
    result = smart_policy_action(obs)
    assert result["action_type"] == "recommend"
    assert result["content_id"] == "safe_item"
    assert result["reasoning"] == "heuristic: best ethical match"


# ── Fallback and degenerate-input paths ───────────────────────────────────

def test_fallback_loop_can_recommend_recently_shown_content() -> None:
    """
    Real behavioral property, not just a coverage exercise: the fallback
    loop (triggered when nothing passes the primary manip/addict/recent
    filter) checks ONLY manipulation_score < 0.30 -- it does not check
    `recent`. So under degraded conditions (e.g. only one low-manipulation
    item exists and it was already shown), the heuristic will re-recommend
    it rather than return nothing. Worth knowing: the anti-repetition
    guarantee is NOT absolute, it only holds on the primary path.
    """
    obs = {
        "visible_fatigue": 0.1, "visible_trust": 0.9, "visible_boredom": 0.1,
        "interest_distribution": {"technology": 1.0},
        "recent_content_ids": ["a1"],
        "available_content": [
            {
                "content_id": "a1", "manipulation_score": 0.1, "addictiveness": 0.1,
                "topic_relevance": {"technology": 1.0}, "educational_value": 0.9,
            },
            {
                "content_id": "a2", "manipulation_score": 0.9, "addictiveness": 0.1,
                "topic_relevance": {"technology": 1.0}, "educational_value": 0.9,
            },
        ],
    }
    result = smart_policy_action(obs)
    assert result["action_type"] == "recommend"
    assert result["content_id"] == "a1"
    assert result["reasoning"] == "heuristic: low manipulation fallback"


def test_no_viable_content_returns_safe_default() -> None:
    obs = {
        "visible_fatigue": 0.1, "visible_trust": 0.9, "visible_boredom": 0.1,
        "interest_distribution": {"technology": 1.0},
        "recent_content_ids": [],
        "available_content": [
            {
                "content_id": "only_bad_item", "manipulation_score": 0.9, "addictiveness": 0.9,
                "topic_relevance": {"technology": 1.0}, "educational_value": 0.1,
            },
        ],
    }
    result = smart_policy_action(obs)
    assert result["action_type"] == "explore_new_topic"
    assert result["reasoning"] == "heuristic: safe default"


def test_malformed_items_in_available_content_do_not_crash() -> None:
    """None and a bare object() as list entries must be skipped gracefully
    (via the per-item except blocks), not raise."""
    obs = {
        "visible_fatigue": 0.1, "visible_trust": 0.9, "visible_boredom": 0.1,
        "interest_distribution": {"technology": 1.0},
        "recent_content_ids": [],
        "available_content": [None, object()],
    }
    result = smart_policy_action(obs)
    assert result["action_type"] in (
        "recommend", "explore_new_topic", "diversify_feed", "pause_session"
    )


def test_non_dict_topic_relevance_triggers_best_id_loop_except() -> None:
    """
    _f and _float swallow their own internal exceptions, so they can't be
    used to reach the OUTER except at line 69-70 in the best_id search
    loop. That except exists to guard code that runs AFTER _f/_float
    return -- specifically `rel.get(dominant)`, which raises AttributeError
    if topic_relevance isn't actually a dict. This exercises that exact
    path and confirms the item is skipped (not crashing the whole call),
    falling through to the low-manipulation fallback for the same item.
    """
    obs = {
        "visible_fatigue": 0.1, "visible_trust": 0.9, "visible_boredom": 0.1,
        "interest_distribution": {"technology": 1.0},
        "recent_content_ids": [],
        "available_content": [
            {
                "content_id": "malformed_relevance", "manipulation_score": 0.1,
                "addictiveness": 0.1, "topic_relevance": "not_a_dict",
                "educational_value": 0.9,
            },
        ],
    }
    result = smart_policy_action(obs)
    # Falls through to the fallback loop, which recommends the same item
    # since it doesn't touch topic_relevance at all -- confirms graceful
    # degradation rather than total failure.
    assert result["action_type"] == "recommend"
    assert result["content_id"] == "malformed_relevance"
    assert result["reasoning"] == "heuristic: low manipulation fallback"


def test_unboolable_content_id_triggers_fallback_loop_except() -> None:
    """
    Exercises the fallback loop's except at line 89-90: `if cid and ...`
    calls bool(cid), which can raise for a pathological content_id value
    even though _f/_float themselves never raised. Item must be excluded
    from the PRIMARY loop (via high addictiveness here) so execution
    actually reaches the fallback loop.
    """
    class BoolRaises:
        def __bool__(self):
            raise RuntimeError("bool boom")

    obs = {
        "visible_fatigue": 0.1, "visible_trust": 0.9, "visible_boredom": 0.1,
        "interest_distribution": {"technology": 1.0},
        "recent_content_ids": [],
        "available_content": [
            {
                "content_id": BoolRaises(), "manipulation_score": 0.1,
                "addictiveness": 0.9,  # excludes it from the primary loop
                "topic_relevance": {"technology": 1.0}, "educational_value": 0.9,
            },
        ],
    }
    result = smart_policy_action(obs)
    # No item survives either loop -> safe default
    assert result == {"action_type": "explore_new_topic", "reasoning": "heuristic: safe default"}


def test_completely_malformed_obs_returns_safe_default() -> None:
    """obs=None makes obs.get(...) raise AttributeError immediately -- the
    OUTER except (not a per-item one) must catch this and still return a
    well-formed action rather than propagating the exception."""
    result = smart_policy_action(None)
    assert result == {"action_type": "explore_new_topic", "reasoning": "heuristic: safe default"}


def test_empty_obs_dict_does_not_crash() -> None:
    result = smart_policy_action({})
    assert "action_type" in result


# ── Helper functions ───────────────────────────────────────────────────────

def test_f_reads_dict_and_object_attrs() -> None:
    assert _f({"content_id": "x1"}, "content_id") == "x1"

    class Item:
        content_id = "x2"
    assert _f(Item(), "content_id") == "x2"

    assert _f({}, "missing_key", "default") == "default"


def test_f_swallows_exceptions_from_getattr() -> None:
    class Bad:
        def __getattr__(self, name):
            raise RuntimeError("boom")
    assert _f(Bad(), "content_id", "fallback") == "fallback"


def test_float_converts_valid_values() -> None:
    assert _float("0.5") == 0.5
    assert _float(3) == 3.0
    assert _float(None, default=1.0) == 1.0


def test_float_swallows_conversion_errors() -> None:
    assert _float("not_a_number", default=9.9) == 9.9
    assert _float(object(), default=2.5) == 2.5


def test_action_label_recommend_includes_content_id() -> None:
    assert action_label({"action_type": "recommend", "content_id": "x1"}) == "recommend(x1)"


def test_action_label_recommend_missing_content_id() -> None:
    assert action_label({"action_type": "recommend"}) == "recommend(?)"


def test_action_label_other_action_types() -> None:
    assert action_label({"action_type": "pause_session"}) == "pause_session"
    assert action_label({"action_type": "diversify_feed"}) == "diversify_feed"


def test_action_label_missing_action_type() -> None:
    assert action_label({}) == "unknown"