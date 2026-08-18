"""
tests/test_a2c.py — Verifies the A2C alternative-algorithm integration
added to close the "only one RL algorithm tested" gap flagged in the
original review: the whole benchmark previously only ever demonstrated
PPO, so any result could plausibly be PPO-specific rather than a general
finding about the environment/reward design.

A2C_KWARGS deliberately shares every hyperparameter with PPO_KWARGS that
both algorithms' constructors accept (gamma, gae_lambda, ent_coef, vf_coef,
max_grad_norm, policy_kwargs/network architecture) and only diverges on
learning_rate and n_steps, which are core to each algorithm's update rule
-- see train_rl.py's A2C_KWARGS comment for the full rationale.
"""

from __future__ import annotations

import os

from environment.train_rl import train_task
from environment.eval_rl import evaluate_ppo


TINY_TIMESTEPS = 256
TINY_N_ENVS = 2


def test_a2c_and_ppo_checkpoints_do_not_collide() -> None:
    """algo is the filename PREFIX (not an add-on suffix), specifically to
    avoid the save-path-collision bug class this codebase hit twice
    already (seed, then warmstart)."""
    a2c_ckpt = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=401, algo="a2c"
    )
    ppo_ckpt = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=401, algo="ppo"
    )

    assert a2c_ckpt != ppo_ckpt
    assert os.path.basename(a2c_ckpt).startswith("a2c_")
    assert os.path.basename(ppo_ckpt).startswith("ppo_")
    assert os.path.exists(a2c_ckpt + ".zip")
    assert os.path.exists(ppo_ckpt + ".zip")


def test_invalid_algo_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="Unknown algo"):
        train_task("easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, algo="dqn")


def test_masked_with_a2c_raises() -> None:
    """There is no MaskableA2C in sb3-contrib -- masked=True must be
    rejected outright when algo='a2c', not silently ignored."""
    import pytest
    with pytest.raises(ValueError, match="masked=True requires algo='ppo'"):
        train_task(
            "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS,
            algo="a2c", masked=True,
        )


def test_evaluate_ppo_autodetects_a2c_checkpoint() -> None:
    """evaluate_ppo() must correctly identify an a2c_ checkpoint by
    filename prefix and load it via A2C, without the caller having to
    specify anything extra."""
    checkpoint = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=402, algo="a2c"
    )

    grade = evaluate_ppo("easy", model_path=checkpoint, n_eval=2, verbose=False)

    assert "final_score_mean" in grade
    assert 0.0 <= grade["final_score_mean"] <= 1.0


def test_evaluate_ppo_still_handles_ppo_checkpoint_alongside_a2c_detection() -> None:
    """Regression guard: adding A2C detection must not break the existing
    PPO/masked-PPO evaluation path."""
    checkpoint = train_task(
        "easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=403, algo="ppo"
    )

    grade = evaluate_ppo("easy", model_path=checkpoint, n_eval=2, verbose=False)

    assert "final_score_mean" in grade
    assert 0.0 <= grade["final_score_mean"] <= 1.0