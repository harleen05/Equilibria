"""
multi_seed.py — Multi-seed training + evaluation harness for the Attention
Economy Environment.

Why this exists: train_rl.py trains ONE model per task per invocation.
eval_rl.py's mean+-std is computed across EVALUATION EPISODES of a single
trained checkpoint -- it does not capture variance introduced by training
itself (different random weight initialization + different stochastic
rollout ordering per training seed). For any claim like "PPO outperforms
the heuristic baseline" to be defensible, that claim needs to hold across
independently-trained models, not just across eval episodes of one lucky
(or unlucky) training run.

This module trains N models per task (one per seed), evaluates each
independently, and reports BOTH the per-seed results and the aggregate
mean/std/95% CI ACROSS TRAINING SEEDS -- plus a Welch's t-test against the
heuristic baseline for the metric that matters most (final_score).

Usage:
    python -m environment.multi_seed --task easy --seeds 1,2,3,4,5
    python -m environment.multi_seed --task medium --seeds 1,2,3,4,5 --timesteps 10000
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

from environment.train_rl import train_task, TASK_CONFIGS
from environment.eval_rl import evaluate_ppo, _run_heuristic_episode


def train_and_evaluate_seeds(
    task_id: str,
    seeds: Sequence[int],
    total_timesteps: int,
    n_envs: int = 4,
    n_eval_episodes: int = 10,
    masked: bool = False,
) -> List[Dict]:
    """
    Train one PPO (or MaskablePPO, if masked=True) model per seed, evaluate
    each over n_eval_episodes, and return a list of per-seed result dicts:
        {"seed": int, "checkpoint": str, **aggregated_grade}
    where aggregated_grade is eval_rl.py's _aggregate() output for that
    seed's own model (within-seed episode variance -- distinct from the
    across-seed variance computed by summarize_across_seeds() below).
    """
    results = []
    for seed in seeds:
        print(
            f"\n{'=' * 60}\n  [multi-seed] task={task_id} seed={seed}"
            f"{'  [MASKED]' if masked else ''}\n{'=' * 60}"
        )
        checkpoint = train_task(
            task_id, total_timesteps=total_timesteps, n_envs=n_envs,
            seed=seed, masked=masked,
        )
        grade = evaluate_ppo(
            task_id, model_path=checkpoint, n_eval=n_eval_episodes, verbose=False
        )
        results.append({"seed": seed, "checkpoint": checkpoint, **grade})
    return results


def summarize_across_seeds(results: List[Dict], metric: str = "final_score_mean") -> Dict:
    """
    Aggregate ACROSS independently-trained seeds (not across eval episodes
    within one seed -- that's what eval_rl.py._aggregate already does).
    Each seed's own `metric` value is treated as one independent sample.
    """
    values = [r[metric] for r in results]
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    # 95% CI via normal approximation -- reasonable for n>=5, optimistic
    # for smaller n. n_seeds is always reported alongside so readers can
    # judge whether the CI should be trusted.
    ci95 = 1.96 * std / (n ** 0.5) if n > 1 else 0.0
    return {
        "metric": metric,
        "n_seeds": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95_halfwidth": round(ci95, 4),
        "per_seed_values": [round(v, 4) for v in values],
    }


def compare_to_heuristic(
    task_id: str,
    seed_results: List[Dict],
    metric: str = "final_score_mean",
    n_eval_episodes: int = 20,
) -> Dict:
    """
    Welch's t-test comparing PPO's across-seed distribution of `metric`
    against the heuristic baseline's across-episode distribution. Welch's
    (not Student's) t-test is used deliberately: the two groups have
    different, unequal variances by construction -- training-seed variance
    and episode-to-episode variance are different phenomena, and Welch's
    test does not assume equal variances the way Student's t-test does.
    """
    ppo_values = [r[metric] for r in seed_results]

    heuristic_metric = metric.replace("_mean", "")  # e.g. final_score_mean -> final_score
    heuristic_episode_values = [
        _run_heuristic_episode(task_id, seed=i).get(heuristic_metric, 0.0)
        for i in range(n_eval_episodes)
    ]

    t_stat, p_value = stats.ttest_ind(ppo_values, heuristic_episode_values, equal_var=False)

    # Cohen's d alongside the p-value: a significant p-value with a tiny
    # effect size is a materially different finding than a large effect,
    # and reporting only p-values invites exactly that ambiguity.
    pooled_var = (np.var(ppo_values, ddof=1) + np.var(heuristic_episode_values, ddof=1)) / 2
    pooled_std = pooled_var ** 0.5
    cohens_d = (
        (np.mean(ppo_values) - np.mean(heuristic_episode_values)) / pooled_std
        if pooled_std > 0 else 0.0
    )

    return {
        "metric": metric,
        "ppo_mean": round(float(np.mean(ppo_values)), 4),
        "heuristic_mean": round(float(np.mean(heuristic_episode_values)), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
        "cohens_d": round(float(cohens_d), 4),
    }


def _parse_seeds(raw: str) -> List[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed train+eval harness")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument(
        "--seeds", type=str, default="1,2,3,4,5",
        help="Comma-separated training seeds, e.g. 1,2,3,4,5",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps (default: task's TASK_CONFIGS value)",
    )
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument(
        "--n_eval_episodes", type=int, default=10,
        help="Eval episodes per trained seed",
    )
    parser.add_argument(
        "--masked", action="store_true",
        help="Train with MaskablePPO instead of plain PPO for every seed "
             "in this run. Run the same --seeds set once with and once "
             "without this flag to compare masked vs unmasked at your "
             "actual training budget.",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    timesteps = args.timesteps or TASK_CONFIGS[args.task]["total_timesteps"]

    seed_results = train_and_evaluate_seeds(
        args.task, seeds, total_timesteps=timesteps, n_envs=args.n_envs,
        n_eval_episodes=args.n_eval_episodes, masked=args.masked,
    )

    print(f"\n{'#' * 60}\n  MULTI-SEED SUMMARY  [{args.task.upper()}]  n_seeds={len(seeds)}\n{'#' * 60}")
    for metric in (
        "final_score_mean", "avg_engagement_mean",
        "final_trust_mean", "final_satisfaction_mean",
    ):
        summary = summarize_across_seeds(seed_results, metric=metric)
        print(
            f"  {metric:<24} mean={summary['mean']:.4f}  std={summary['std']:.4f}  "
            f"95% CI=+-{summary['ci95_halfwidth']:.4f}  (n={summary['n_seeds']})"
        )

    print(f"\n  Comparing PPO (across {len(seeds)} training seeds) vs heuristic baseline...")
    comparison = compare_to_heuristic(args.task, seed_results)
    print(f"  PPO mean={comparison['ppo_mean']:.4f}  Heuristic mean={comparison['heuristic_mean']:.4f}")
    print(
        f"  Welch t={comparison['t_statistic']:.4f}  p={comparison['p_value']:.6f}  "
        f"significant(p<0.05)={comparison['significant_at_0.05']}  "
        f"Cohen's d={comparison['cohens_d']:.4f}"
    )