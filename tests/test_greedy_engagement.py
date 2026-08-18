"""
tests/test_greedy_engagement.py — Verifies the greedy-engagement baseline
added to eval_rl.py. This baseline represents "what a platform optimizing
purely for engagement, with no ethical guardrails, would do" -- the
central contrast case the whole benchmark exists to argue against.

Empirically verified before this baseline was added (see chat history /
commit message): on the same task/seed, greedy-engagement selects content
with avg manipulation_score ~0.89 vs the ethics-aware heuristic's ~0.35,
and avg addictiveness ~0.575 vs ~0.27. These tests lock that contrast in.
"""

from __future__ import annotations

from environment.env_core import AttentionEconomyEnv
from environment.eval_rl import (
    _greedy_engagement,
    _heuristic,
    _run_greedy_engagement_episode,
    evaluate_greedy_engagement,
)


def test_greedy_engagement_returns_valid_action() -> None:
    env = AttentionEconomyEnv()
    obs = env.reset("medium", seed=1)
    action = _greedy_engagement(obs)

    assert action.action_type in (
        "recommend", "explore_new_topic", "diversify_feed", "pause_session"
    )
    if action.action_type == "recommend":
        assert action.content_id in {c.content_id for c in obs.available_content}


def test_greedy_engagement_selects_higher_manipulation_content_than_heuristic() -> None:
    """
    The core property this baseline exists to demonstrate: on the same
    task/seed, a pure engagement-maximizer picks systematically more
    manipulative/addictive content than the ethics-aware heuristic.
    """
    def collect_scores(policy_fn):
        env = AttentionEconomyEnv()
        obs = env.reset("hard", seed=1)
        manip_scores = []
        done = False
        while not done:
            action = policy_fn(obs)
            if action.action_type == "recommend":
                manip_scores.append(env.catalog[action.content_id].manipulation_score)
            obs, _, done, _ = env.step(action)
        return manip_scores

    greedy_scores = collect_scores(_greedy_engagement)
    heuristic_scores = collect_scores(_heuristic)

    assert greedy_scores, "greedy-engagement should recommend at least one item"
    assert heuristic_scores, "heuristic should recommend at least one item"

    avg_greedy = sum(greedy_scores) / len(greedy_scores)
    avg_heuristic = sum(heuristic_scores) / len(heuristic_scores)

    assert avg_greedy > avg_heuristic, (
        f"expected greedy-engagement (avg manip={avg_greedy:.3f}) to pick more "
        f"manipulative content than heuristic (avg manip={avg_heuristic:.3f})"
    )


def test_run_greedy_engagement_episode_returns_valid_grade() -> None:
    grade = _run_greedy_engagement_episode("easy", seed=1)

    assert "final_score" in grade
    assert 0.0 <= grade["final_score"] <= 1.0


def test_evaluate_greedy_engagement_multi_episode_aggregation() -> None:
    grade = evaluate_greedy_engagement("easy", n_eval=5, verbose=False)

    assert "final_score_mean" in grade
    assert "final_score_std" in grade
    assert 0.0 <= grade["final_score_mean"] <= 1.0


def test_greedy_engagement_avoids_recently_shown_content() -> None:
    """Mirrors the heuristic's anti-repetition behavior: it should not
    immediately re-recommend something already in recent_content_ids,
    as long as an alternative exists."""
    env = AttentionEconomyEnv()
    obs = env.reset("easy", seed=1)

    first_action = _greedy_engagement(obs)
    obs, _, _, _ = env.step(first_action)

    second_action = _greedy_engagement(obs)
    if second_action.action_type == "recommend" and first_action.action_type == "recommend":
        if len(obs.available_content) > 1:
            assert second_action.content_id != first_action.content_id