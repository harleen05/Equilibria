"""
tests/test_interpretability.py — Verifies the pure analysis functions in
interpretability.py using synthetic episode data (fast, no real training
required). run_analysis_episodes() itself (which loads a real checkpoint
and drives real episodes) is exercised manually against real trained
checkpoints rather than unit-tested here, matching the pattern used for
train_rl.py/multi_seed.py's CLI entrypoints -- training a real model in
every CI run would be slow, and the interesting logic to verify is the
analysis functions, not the training/prediction plumbing already covered
by tests/test_masked_ppo.py and tests/test_a2c.py.
"""

from __future__ import annotations

from interpretability import (
    summarize_action_distribution,
    summarize_termination_reasons,
    find_failure_mode_candidates,
)


def _step(action_type="recommend", content_id=None, content_type=None, trust=0.5, addiction_risk=0.1):
    return {
        "action_type": action_type, "content_id": content_id, "content_type": content_type,
        "reward": 0.5, "trust": trust, "fatigue": 0.1, "satisfaction": 0.5,
        "addiction_risk": addiction_risk, "boredom": 0.1,
    }


def _episode(seed, termination_reason, steps, final_score=0.3):
    return {
        "seed": seed, "termination_reason": termination_reason,
        "episode_grade": {"final_score": final_score}, "steps": steps,
    }


def test_summarize_action_distribution_counts_actions() -> None:
    episodes = [
        _episode(0, "max_steps_reached", [
            _step("recommend", "a", "relevant"),
            _step("pause_session"),
        ]),
    ]
    dist = summarize_action_distribution(episodes)
    assert dist["action_counts"] == {"recommend": 1, "pause_session": 1}


def test_summarize_action_distribution_splits_by_trust_level() -> None:
    """The core interpretability question: does content_type choice differ
    when trust is fragile (<0.3) vs stable (>=0.3)?"""
    episodes = [
        _episode(0, "max_steps_reached", [
            _step("recommend", "a", "misleading", trust=0.1),
            _step("recommend", "b", "relevant", trust=0.9),
        ]),
    ]
    dist = summarize_action_distribution(episodes)
    assert dist["content_type_when_trust_fragile"] == {"misleading": 1}
    assert dist["content_type_when_trust_stable"] == {"relevant": 1}


def test_summarize_action_distribution_ignores_non_recommend_for_content_type() -> None:
    episodes = [_episode(0, "max_steps_reached", [_step("pause_session", trust=0.1)])]
    dist = summarize_action_distribution(episodes)
    assert dist["content_type_when_trust_fragile"] == {}
    assert dist["content_type_when_trust_stable"] == {}


def test_summarize_termination_reasons_counts_correctly() -> None:
    episodes = [
        _episode(0, "trust_collapse", [_step()]),
        _episode(1, "max_steps_reached", [_step()]),
        _episode(2, "max_steps_reached", [_step()]),
    ]
    counts = summarize_termination_reasons(episodes)
    assert counts == {"trust_collapse": 1, "max_steps_reached": 2}


def test_find_failure_mode_flags_repetitive_action_streak() -> None:
    episodes = [_episode(0, "max_steps_reached", [_step("pause_session")] * 5)]
    candidates = find_failure_mode_candidates(episodes, streak_threshold=4)
    assert len(candidates) == 1
    assert any("repetitive action streak" in f for f in candidates[0]["flags"])


def test_find_failure_mode_flags_rising_addiction_risk_streak() -> None:
    steps = [
        _step("recommend", "a", "addictive", addiction_risk=0.1 * i) for i in range(1, 6)
    ]
    episodes = [_episode(0, "max_steps_reached", steps)]
    candidates = find_failure_mode_candidates(episodes, streak_threshold=4)
    assert len(candidates) == 1
    assert any("rising addiction_risk streak" in f for f in candidates[0]["flags"])


def test_find_failure_mode_flags_trust_collapse_termination() -> None:
    episodes = [_episode(0, "trust_collapse", [_step()])]
    candidates = find_failure_mode_candidates(episodes, streak_threshold=4)
    assert len(candidates) == 1
    assert "ended in trust_collapse" in candidates[0]["flags"]


def test_find_failure_mode_does_not_flag_clean_episodes() -> None:
    """An episode with varied actions, stable addiction_risk, and a normal
    termination should NOT be flagged -- guards against over-flagging."""
    steps = [
        _step("recommend", "a", "relevant"),
        _step("diversify_feed"),
        _step("recommend", "b", "relevant"),
        _step("explore_new_topic"),
    ]
    episodes = [_episode(0, "max_steps_reached", steps)]
    candidates = find_failure_mode_candidates(episodes, streak_threshold=4)
    assert candidates == []


def test_find_failure_mode_empty_episodes_list() -> None:
    assert find_failure_mode_candidates([]) == []