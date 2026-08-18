"""
tests/test_determinism.py — Verifies the "fully deterministic state transition"
claim made in environment/simulation.py's module docstring.

Findings this file locks in:
  1. Given the same seed and the same action sequence, the full episode
     trajectory (rewards AND user-state observations) is byte-identical
     across independent AttentionEconomyEnv instances.
  2. User-state fields (fatigue, trust, satisfaction, boredom) do NOT
     depend on the RNG at all -- they're deterministic functions of the
     action sequence regardless of seed. Only `engagement` (and therefore
     `reward`, which is derived from it) carries stochastic noise.
  3. Different seeds actually DO produce different reward trajectories --
     this guards against a seed param that's silently ignored (see note
     below on why this matters).

Note: tests/test_rl_pipeline.py::test_attention_env_wrapper_reset_seed_reproducible
only compares the observation immediately after reset(), before any step()
is taken. Since the only randomness in the simulation lives inside
SimulationEngine.compute_engagement(), which fires during step(), that test
would pass even if seeding were completely broken. This file closes that gap.
"""

from __future__ import annotations

from environment.env_core import AttentionEconomyEnv

MIXED_ACTIONS = [
    {"action_type": "recommend", "content_id": "rel_tech_01"},
    {"action_type": "recommend", "content_id": "rel_sci_01"},
    {"action_type": "diversify_feed"},
    {"action_type": "recommend", "content_id": "add_gaming_01"},
    {"action_type": "explore_new_topic"},
    {"action_type": "recommend", "content_id": "mis_click_01"},
    {"action_type": "pause_session"},
]


def _run_episode(task: str, seed: int, actions: list[dict]) -> list[tuple]:
    """Run a fixed action sequence and capture the full per-step trajectory."""
    env = AttentionEconomyEnv()
    env.reset(task, seed=seed)
    trace = []
    for action in actions:
        obs, reward, done, info = env.step(action)
        trace.append(
            (
                round(reward, 10),
                round(obs.visible_fatigue, 10),
                round(obs.visible_trust, 10),
                round(obs.visible_satisfaction, 10),
                round(obs.visible_boredom, 10),
                done,
            )
        )
        if done:
            break
    return trace


def test_full_episode_identical_across_independent_envs_same_seed():
    """
    Two completely separate AttentionEconomyEnv instances, given the same
    seed and the same action sequence, must produce byte-identical
    trajectories -- including reward, which is the only stochastic signal.
    """
    trace_a = _run_episode("easy", seed=42, actions=MIXED_ACTIONS)
    trace_b = _run_episode("easy", seed=42, actions=MIXED_ACTIONS)

    assert trace_a == trace_b


def test_full_episode_identical_across_all_three_tasks():
    """Determinism should hold regardless of task difficulty."""
    for task in ("easy", "medium", "hard"):
        trace_a = _run_episode(task, seed=7, actions=MIXED_ACTIONS)
        trace_b = _run_episode(task, seed=7, actions=MIXED_ACTIONS)
        assert trace_a == trace_b, f"non-deterministic trajectory for task={task}"


def test_user_state_transitions_are_seed_independent():
    """
    Characterizes the environment precisely: fatigue/trust/satisfaction/
    boredom are deterministic functions of the action sequence ALONE --
    changing the seed must not move them at all, since none of the
    per-field update_* functions in SimulationEngine touch self.rng.
    Only `engagement` (and reward, derived from it) is seed-sensitive.
    """
    trace_seed_1 = _run_episode("medium", seed=1, actions=MIXED_ACTIONS)
    trace_seed_2 = _run_episode("medium", seed=2, actions=MIXED_ACTIONS)

    # Indices: 0=reward, 1=fatigue, 2=trust, 3=satisfaction, 4=boredom, 5=done
    for step_a, step_b in zip(trace_seed_1, trace_seed_2):
        assert step_a[1:] == step_b[1:], (
            "user-state fields diverged across seeds -- this would mean "
            "randomness leaked into state transitions, not just engagement"
        )


def test_different_seeds_actually_change_reward():
    """
    Guards against a seed parameter that's silently ignored: if the RNG
    were never wired into compute_engagement, this test (unlike the
    existing wrapper-level reset test) would catch it, since it inspects
    reward after real step()s instead of only reset().
    """
    trace_seed_1 = _run_episode("medium", seed=1, actions=MIXED_ACTIONS)
    trace_seed_2 = _run_episode("medium", seed=99, actions=MIXED_ACTIONS)

    rewards_1 = [step[0] for step in trace_seed_1]
    rewards_2 = [step[0] for step in trace_seed_2]

    assert rewards_1 != rewards_2


def test_unseeded_episodes_are_not_required_to_match():
    """
    Sanity check on the inverse case: reset(seed=None) draws from OS
    entropy (numpy default_rng(None) semantics), so two unseeded episodes
    are expected to diverge. This isn't a strict guarantee (extremely
    unlikely to collide), so it's documentation via test rather than a
    hard correctness requirement.
    """
    trace_a = _run_episode("easy", seed=None, actions=MIXED_ACTIONS[:3])
    trace_b = _run_episode("easy", seed=None, actions=MIXED_ACTIONS[:3])

    rewards_a = [step[0] for step in trace_a]
    rewards_b = [step[0] for step in trace_b]
    assert rewards_a != rewards_b