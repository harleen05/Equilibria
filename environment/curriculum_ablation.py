"""
curriculum_ablation.py — Tests the assumption baked into train_curriculum()'s
docstring ("the hard model benefits from the policy learned on medium") by
training "hard" both WITH warm-start from a freshly-trained "medium"
checkpoint and WITHOUT (from scratch), at the same seed and same timestep
budget, and comparing final_score across seeds.

This ablation directly surfaced a real bug during development: the
checkpoint save-path logic didn't distinguish warm-started from
from-scratch models for the same task/seed, so they silently overwrote
each other -- see train_rl.py's save_path construction (warmstart_part).
Always verify with test_curriculum_ablation.py::
test_warmstart_and_scratch_checkpoints_do_not_collide before trusting any
numbers this script produces.

Usage:
    python -m environment.curriculum_ablation --seeds 1,2,3,4,5 --timesteps 3000
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
from scipy import stats

from environment.train_rl import train_task, TASK_CONFIGS
from environment.eval_rl import evaluate_ppo


def run_curriculum_ablation(
    seeds: List[int],
    total_timesteps: int,
    n_envs: int = 4,
    n_eval_episodes: int = 10,
) -> Dict:
    """
    For each seed: train medium, then train hard twice at that seed (once
    warm-started from that seed's medium checkpoint, once from scratch),
    and evaluate both. Returns per-seed and aggregated results plus a
    Welch's t-test comparing the two groups.
    """
    warmstart_scores, scratch_scores, per_seed = [], [], []

    for seed in seeds:
        medium_ckpt = train_task("medium", total_timesteps=total_timesteps, n_envs=n_envs, seed=seed)

        warm_ckpt = train_task(
            "hard", total_timesteps=total_timesteps, n_envs=n_envs,
            seed=seed, warmstart_path=medium_ckpt,
        )
        grade_warm = evaluate_ppo("hard", model_path=warm_ckpt, n_eval=n_eval_episodes, verbose=False)

        scratch_ckpt = train_task("hard", total_timesteps=total_timesteps, n_envs=n_envs, seed=seed)
        grade_scratch = evaluate_ppo("hard", model_path=scratch_ckpt, n_eval=n_eval_episodes, verbose=False)

        warmstart_scores.append(grade_warm["final_score_mean"])
        scratch_scores.append(grade_scratch["final_score_mean"])
        per_seed.append({
            "seed": seed,
            "warmstart_final_score": grade_warm["final_score_mean"],
            "scratch_final_score": grade_scratch["final_score_mean"],
        })

    t_stat, p_value = stats.ttest_ind(warmstart_scores, scratch_scores, equal_var=False)

    return {
        "per_seed": per_seed,
        "warmstart_mean": round(float(np.mean(warmstart_scores)), 4),
        "warmstart_std": round(float(np.std(warmstart_scores, ddof=1)), 4) if len(seeds) > 1 else 0.0,
        "scratch_mean": round(float(np.mean(scratch_scores)), 4),
        "scratch_std": round(float(np.std(scratch_scores, ddof=1)), 4) if len(seeds) > 1 else 0.0,
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
        "n_seeds": len(seeds),
    }


def _parse_seeds(raw: str) -> List[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum-learning (warm-start) ablation")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps (default: TASK_CONFIGS['hard'] value)",
    )
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    timesteps = args.timesteps or TASK_CONFIGS["hard"]["total_timesteps"]

    result = run_curriculum_ablation(
        seeds, total_timesteps=timesteps, n_envs=args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
    )

    print(f"\n{'#' * 60}\n  CURRICULUM ABLATION  (n_seeds={result['n_seeds']})\n{'#' * 60}")
    for row in result["per_seed"]:
        print(
            f"  seed={row['seed']}  warmstart={row['warmstart_final_score']:.4f}  "
            f"scratch={row['scratch_final_score']:.4f}"
        )
    print(f"\n  warmstart: mean={result['warmstart_mean']:.4f}  std={result['warmstart_std']:.4f}")
    print(f"  scratch:   mean={result['scratch_mean']:.4f}  std={result['scratch_std']:.4f}")
    print(
        f"  Welch t={result['t_statistic']:.4f}  p={result['p_value']:.6f}  "
        f"significant(p<0.05)={result['significant_at_0.05']}"
    )