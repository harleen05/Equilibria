"""
tests/test_masked_ppo.py — Verifies the MaskablePPO integration added to
close the "action_masks() exists but is never used" gap flagged in the
original review: rl_wrapper.py's action_masks() was mechanically correct
but dead code, since train_rl.py only ever constructed plain PPO.

These tests use tiny timestep counts (fast, seconds not minutes) and check
mechanical correctness (masking is genuinely respected, checkpoints don't
collide, evaluate_ppo auto-detects masked vs unmasked) rather than
asserting anything about learning quality or sample efficiency -- that
empirical question is left to actual training runs via multi_seed.py at a
real timestep budget; see train_task()'s docstring for the honest
small-scale result this integration was based on (no statistically
decisive advantage at 4000 timesteps / 3 seeds).
"""

from __future__ import annotations

import os

from environment.train_rl import train_task
from environment.eval_rl import evaluate_ppo
from environment.rl_wrapper import build_env, ALL_CONTENT_IDS

TINY_TIMESTEPS = 256
TINY_N_ENVS = 2


def test_masked_and_unmasked_checkpoints_do_not_collide(tmp_path, monkeypatch) -> None:
    """Regression test for the save-path collision bug: training the same
    task/seed with masked=True and masked=False must produce two distinct
    checkpoint files, not overwrite each other."""
    checkpoint_masked = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=101, masked=True
    )
    checkpoint_unmasked = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=101, masked=False
    )

    assert checkpoint_masked != checkpoint_unmasked
    assert "_masked" in os.path.basename(checkpoint_masked)
    assert "_masked" not in os.path.basename(checkpoint_unmasked)
    assert os.path.exists(checkpoint_masked + ".zip")
    assert os.path.exists(checkpoint_unmasked + ".zip")


def test_masked_ppo_never_selects_disallowed_content_action() -> None:
    """
    The core correctness property of action masking: once trained (even
    briefly) with MaskablePPO, predict() must never return an action index
    corresponding to content excluded from the task's allowed set.
    """
    checkpoint = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=102, masked=True
    )

    env = build_env("easy", masked=True)
    obs, _ = env.reset(seed=1)

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(checkpoint, env=env)

    disallowed_indices = {
        i for i, cid in enumerate(ALL_CONTENT_IDS)
        if cid not in env.unwrapped._allowed_set
    }
    assert disallowed_indices, "test setup assumption failed: 'easy' should exclude some content"

    chosen_actions = []
    for _ in range(20):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        chosen_actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            obs, _ = env.reset(seed=1)

    violations = [a for a in chosen_actions if a in disallowed_indices]
    assert violations == [], f"masked policy selected disallowed actions: {violations}"


def test_evaluate_ppo_autodetects_masked_checkpoint() -> None:
    """evaluate_ppo() must correctly identify a _masked checkpoint by
    filename and load it via MaskablePPO with a properly wrapped env,
    without the caller having to specify anything extra."""
    checkpoint = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=103, masked=True
    )

    grade = evaluate_ppo("easy", model_path=checkpoint, n_eval=2, verbose=False)

    assert "final_score_mean" in grade
    assert 0.0 <= grade["final_score_mean"] <= 1.0


def test_evaluate_ppo_still_handles_unmasked_checkpoint() -> None:
    """Regression guard: the masked-detection logic must not break the
    existing unmasked evaluation path."""
    checkpoint = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=104, masked=False
    )

    grade = evaluate_ppo("easy", model_path=checkpoint, n_eval=2, verbose=False)

    assert "final_score_mean" in grade
    assert 0.0 <= grade["final_score_mean"] <= 1.0