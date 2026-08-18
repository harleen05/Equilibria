import pytest

from environment.env_core import AttentionEconomyEnv
from environment.models import Action


def test_environment_reset_returns_observation() -> None:
    env = AttentionEconomyEnv()
    obs = env.reset("easy")

    assert obs.task_id == "easy"
    assert obs.step_count == 0
    assert obs.visible_fatigue == 0.0
    assert obs.visible_trust > 0.0
    assert len(obs.available_content) > 0
    assert obs.recent_diversity_score == 0.0


def test_environment_recommend_step_updates_state() -> None:
    env = AttentionEconomyEnv()
    env.reset("easy")
    action = Action(action_type="recommend", content_id="rel_tech_01")

    obs, reward, done, info = env.step(action)

    assert obs.step_count == 1
    assert reward >= 0.0001
    assert isinstance(done, bool)
    assert "diagnostics" in info
    assert info["diagnostics"]["interest_match"] >= 0.0


def test_environment_invalid_content_id_raises() -> None:
    env = AttentionEconomyEnv()
    env.reset("easy")

    with pytest.raises(ValueError, match="Invalid content_id"):
        env.step({"action_type": "recommend", "content_id": "does_not_exist"})


def test_environment_invalid_action_type_raises() -> None:
    env = AttentionEconomyEnv()
    env.reset("easy")

    with pytest.raises(Exception):
        env.step({"action_type": "invalid_action"})


def test_environment_completes_episode_and_returns_grade() -> None:
    env = AttentionEconomyEnv()
    env.reset("easy")
    info = {}

    while not env.done:
        _, _, done, info = env.step(Action(action_type="pause_session"))
        if done:
            break

    assert env.done is True
    assert "episode_grade" in info
    assert 0.0001 <= info["episode_grade"]["final_score"] < 1.0

def test_step_after_episode_done_raises_runtime_error() -> None:
    """Covers env_core.py line 100 -- calling step() after done=True must
    raise, not silently continue mutating a finished episode."""
    env = AttentionEconomyEnv()
    env.reset("easy")
    env.done = True

    with pytest.raises(RuntimeError, match="Episode finished"):
        env.step(Action(action_type="pause_session"))


def test_step_content_excluded_from_task_raises() -> None:
    """Covers env_core.py line 120. mis_outrage_01 exists in the full
    catalog but is deliberately excluded from the 'easy' task's
    allowed_content_ids (task difficulty is partly encoded by which
    content is reachable at all, not just reward weights)."""
    env = AttentionEconomyEnv()
    env.reset("easy")

    with pytest.raises(ValueError, match="Content not allowed in this task"):
        env.step({"action_type": "recommend", "content_id": "mis_outrage_01"})


def test_step_invalid_action_type_via_bypassed_validation() -> None:
    """
    Covers env_core.py line 112 (the internal `action_type not in [...]`
    check).

    This branch is NOT reachable through step()'s normal dict-construction
    path: Action.action_type is a pydantic Literal, so Action(**{"action_type":
    "bogus"}) is rejected by pydantic validation before this internal check
    ever runs (see test_environment_invalid_action_type_raises above -- it
    passes because it catches the broader pydantic ValidationError, not
    because it reaches line 112).

    The internal check exists as defense-in-depth against Action objects
    constructed via a path that skips pydantic validation, e.g.
    Action.model_construct(...) or an action_type mutated post-construction
    (pydantic v2 doesn't validate on assignment by default). This test
    exercises exactly that path so the branch isn't silently dead code.
    """
    action = Action.model_construct(action_type="bogus_action", content_id=None)

    env = AttentionEconomyEnv()
    env.reset("easy")

    with pytest.raises(ValueError, match="Invalid action type"):
        env.step(action)

def test_oracle_mode_is_default_and_exposes_true_manipulation_score() -> None:
    """Default behavior must be unchanged: oracle mode is implicit when
    observability isn't passed, and manipulation_score is the real value."""
    env = AttentionEconomyEnv()
    obs = env.reset("easy")  # no observability arg -> defaults to "oracle"

    true_scores = {
        item.content_id: item.manipulation_score
        for item in env.catalog.values()
        if item.content_id in env.allowed_content_ids
    }
    observed_scores = {item.content_id: item.manipulation_score for item in obs.available_content}

    assert observed_scores == true_scores


def test_partial_mode_masks_manipulation_score_to_neutral_constant() -> None:
    env = AttentionEconomyEnv()
    obs = env.reset("easy", observability="partial")

    scores_seen = {item.manipulation_score for item in obs.available_content}

    # Every item shows the SAME value -> no discriminating signal available
    assert len(scores_seen) == 1

    masked_value = scores_seen.pop()
    expected_mean = sum(i.manipulation_score for i in env.catalog.values()) / len(env.catalog)
    assert masked_value == pytest.approx(expected_mean)

    # And it must not just coincidentally match any single item's true score
    # in a way that could be mistaken for real signal -- spot check a known
    # highly-manipulative item is masked away from its true (high) score.
    true_high_score = env.catalog["mis_outrage_01"].manipulation_score
    assert masked_value != true_high_score


def test_invalid_observability_raises_value_error() -> None:
    env = AttentionEconomyEnv()
    with pytest.raises(ValueError, match="Invalid observability mode"):
        env.reset("easy", observability="omniscient")


def test_reward_identical_between_oracle_and_partial_modes() -> None:
    """
    The single most important invariant here: observability must ONLY
    change what the agent sees in Observation.available_content. It must
    NEVER change simulation dynamics or reward, since step() always reads
    the true content from self.catalog[action.content_id], not from the
    (possibly masked) observation. If this test ever fails, masking has
    leaked into the actual environment dynamics -- a serious correctness
    bug, since it would mean partial-observability runs are silently not
    comparable to oracle runs on ground truth.
    """
    actions = [
        {"action_type": "recommend", "content_id": "rel_tech_01"},
        {"action_type": "recommend", "content_id": "mis_outrage_01"},  # excluded from "easy" -- use "medium"
        {"action_type": "diversify_feed"},
    ]
    actions = [
        {"action_type": "recommend", "content_id": "rel_tech_01"},
        {"action_type": "recommend", "content_id": "add_gaming_01"},
        {"action_type": "diversify_feed"},
    ]

    env_oracle = AttentionEconomyEnv()
    env_oracle.reset("easy", seed=42, observability="oracle")

    env_partial = AttentionEconomyEnv()
    env_partial.reset("easy", seed=42, observability="partial")

    for action in actions:
        obs_o, reward_o, done_o, info_o = env_oracle.step(action)
        obs_p, reward_p, done_p, info_p = env_partial.step(action)

        assert reward_o == reward_p
        assert info_o["reward_breakdown"] == info_p["reward_breakdown"]
        assert info_o["user_state"] == info_p["user_state"]
        assert done_o == done_p

        # The only thing allowed to differ is what's exposed to the agent
        assert [c.manipulation_score for c in obs_o.available_content] != [
            c.manipulation_score for c in obs_p.available_content
        ]