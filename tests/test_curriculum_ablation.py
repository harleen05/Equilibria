"""
tests/test_curriculum_ablation.py — Verifies the curriculum-learning
ablation added to test the assumption baked into train_curriculum()'s
docstring ("the hard model benefits from the policy learned on medium").

The first test here (checkpoint path collision) is a REGRESSION test for a
real bug this ablation surfaced during development: training "hard" with
warmstart_path=<medium checkpoint> and training "hard" from scratch, at the
same seed, silently saved to the identical file path -- so the second
training run's save would overwrite the first's checkpoint before anyone
noticed the comparison was invalid. Fixed by adding a _warmstart suffix to
the save path whenever warmstart_path is provided.
"""

from __future__ import annotations

import os

from environment.train_rl import train_task
from environment.curriculum_ablation import run_curriculum_ablation

TINY_TIMESTEPS = 256
TINY_N_ENVS = 2


def test_warmstart_and_scratch_checkpoints_do_not_collide() -> None:
    """Regression test for the save-path collision bug this ablation
    surfaced: warm-started and from-scratch checkpoints for the same
    task/seed must be saved to distinct files."""
    medium_ckpt = train_task(
        "medium", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=201
    )
    warm_ckpt = train_task(
        "hard", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS,
        seed=201, warmstart_path=medium_ckpt,
    )
    scratch_ckpt = train_task(
        "hard", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=201
    )

    assert warm_ckpt != scratch_ckpt
    assert "_warmstart" in os.path.basename(warm_ckpt)
    assert "_warmstart" not in os.path.basename(scratch_ckpt)
    assert os.path.exists(warm_ckpt + ".zip")
    assert os.path.exists(scratch_ckpt + ".zip")


def test_run_curriculum_ablation_returns_expected_structure() -> None:
    result = run_curriculum_ablation(
        seeds=[301, 302], total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS,
        n_eval_episodes=2,
    )

    assert result["n_seeds"] == 2
    assert len(result["per_seed"]) == 2
    for row in result["per_seed"]:
        assert "warmstart_final_score" in row
        assert "scratch_final_score" in row
        assert 0.0 <= row["warmstart_final_score"] <= 1.0
        assert 0.0 <= row["scratch_final_score"] <= 1.0

    assert "warmstart_mean" in result
    assert "scratch_mean" in result
    assert "p_value" in result
    assert "significant_at_0.05" in result