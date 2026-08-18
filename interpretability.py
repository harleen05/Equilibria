"""
interpretability.py — Interpretability pass + failure-mode gallery for a
trained policy checkpoint.

Answers two concrete questions a paper reviewer will ask:
  1. What does the trained policy actually prioritize? (action-type
     distribution, and whether content-choice behavior shifts based on
     the user's current vulnerability -- specifically, does the policy
     behave differently when trust is already fragile?)
  2. What do its failure modes look like, concretely? Rather than only
     reporting aggregate scores, this collects and prints full qualitative
     trajectories where the policy's behavior is most exploitable /
     concerning, so the paper can show real examples rather than only
     summary statistics.

Usage:
    python -m interpretability --task hard --model models/ppo_hard_final --n_episodes 30
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict, List, Optional

from environment.eval_rl import _resolve_model_path


def _load_model(task_id: str, model_path: Optional[str]):
    """Reuses eval_rl.py's checkpoint-naming convention (a2c_ prefix,
    _masked suffix) to auto-detect algorithm/masking, same as evaluate_ppo."""
    from stable_baselines3 import PPO, A2C
    from sb3_contrib import MaskablePPO
    from environment.rl_wrapper import build_env

    model_path = _resolve_model_path(task_id, model_path)
    basename = os.path.basename(model_path)
    is_a2c = basename.startswith("a2c_")
    masked = (not is_a2c) and ("_masked" in basename)

    env = build_env(task_id, masked=masked)
    algo_cls = A2C if is_a2c else (MaskablePPO if masked else PPO)
    model = algo_cls.load(model_path, env=env)
    return model, masked


def run_analysis_episodes(task_id: str, model_path: Optional[str], n_episodes: int) -> List[Dict]:
    """
    Runs n_episodes with the given model and returns a rich per-episode
    trace: every step's action, content_type (if recommend), full user
    state, and reward -- plus the termination reason. Uses
    AttentionEnvWrapper's own internal AttentionEconomyEnv instance
    (wrapper._env) as the single source of truth, rather than running two
    separate env instances in parallel and hoping they stay in sync.
    """
    model, masked = _load_model(task_id, model_path)
    from environment.rl_wrapper import AttentionEnvWrapper
    wrapper = AttentionEnvWrapper(task_id=task_id)

    episodes = []
    for seed in range(n_episodes):
        gym_obs, _ = wrapper.reset(seed=seed)
        env = wrapper._env  # the real AttentionEconomyEnv driving this episode

        steps = []
        done = False
        while not done:
            if masked:
                action_int, _ = model.predict(
                    gym_obs, action_masks=wrapper.action_masks(), deterministic=True
                )
            else:
                action_int, _ = model.predict(gym_obs, deterministic=True)
            action = wrapper._decode_action(int(action_int))

            content_type = None
            if action.action_type == "recommend" and action.content_id in env.catalog:
                content_type = env.catalog[action.content_id].content_type

            gym_obs, reward, term, trunc, info = wrapper.step(int(action_int))
            done = term or trunc

            steps.append({
                "action_type": action.action_type,
                "content_id": action.content_id,
                "content_type": content_type,
                "reward": reward,
                "trust": env.user.trust,
                "fatigue": env.user.fatigue,
                "satisfaction": env.user.satisfaction,
                "addiction_risk": env.user.addiction_risk,
                "boredom": env.user.boredom,
            })

        max_steps_reached = env.step_count >= env.max_steps
        trust_collapse = env.user.trust <= 0.05
        fatigue_overload = env.user.fatigue >= 0.95
        if trust_collapse:
            termination_reason = "trust_collapse"
        elif fatigue_overload:
            termination_reason = "fatigue_overload"
        elif max_steps_reached:
            termination_reason = "max_steps_reached"
        else:
            termination_reason = "unknown"

        episodes.append({
            "seed": seed,
            "steps": steps,
            "termination_reason": termination_reason,
            "episode_grade": info.get("episode_grade", {}),
        })

    return episodes


def summarize_action_distribution(episodes: List[Dict]) -> Dict:
    """
    Action-type frequency, plus the interpretability question: among
    "recommend" actions, does the content_type distribution differ when
    trust is fragile (<0.3) vs stable (>=0.3) at the moment of that
    decision? A policy that shifts toward safer content specifically when
    trust is fragile is behaving adaptively; one that doesn't (or shifts
    the other way) is worth flagging.
    """
    action_counts: Counter = Counter()
    content_type_low_trust: Counter = Counter()
    content_type_high_trust: Counter = Counter()

    for ep in episodes:
        for step in ep["steps"]:
            action_counts[step["action_type"]] += 1
            if step["action_type"] == "recommend" and step["content_type"]:
                if step["trust"] < 0.3:
                    content_type_low_trust[step["content_type"]] += 1
                else:
                    content_type_high_trust[step["content_type"]] += 1

    return {
        "action_counts": dict(action_counts),
        "content_type_when_trust_fragile": dict(content_type_low_trust),
        "content_type_when_trust_stable": dict(content_type_high_trust),
    }


def summarize_termination_reasons(episodes: List[Dict]) -> Dict[str, int]:
    counts: Counter = Counter(ep["termination_reason"] for ep in episodes)
    return dict(counts)


def find_failure_mode_candidates(episodes: List[Dict], streak_threshold: int = 4) -> List[Dict]:
    """
    Flags episodes exhibiting patterns worth showing qualitatively in a
    paper's discussion section, rather than only reporting that they
    exist in aggregate:
      - longest same-action-type streak >= streak_threshold (possible
        degenerate/repetitive policy behavior)
      - addiction_risk rising for >= streak_threshold consecutive steps
        (possible engagement-farming pattern)
      - episode ended via trust_collapse (worth inspecting what led there)
    """
    candidates = []
    for ep in episodes:
        steps = ep["steps"]
        if not steps:
            continue

        # Longest same-action-type streak
        longest_streak, current_streak, current_action = 1, 1, steps[0]["action_type"]
        for step in steps[1:]:
            if step["action_type"] == current_action:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 1
                current_action = step["action_type"]

        # Longest monotonically-increasing addiction_risk streak
        longest_ar_streak, current_ar_streak = 1, 1
        for i in range(1, len(steps)):
            if steps[i]["addiction_risk"] > steps[i - 1]["addiction_risk"]:
                current_ar_streak += 1
                longest_ar_streak = max(longest_ar_streak, current_ar_streak)
            else:
                current_ar_streak = 1

        reasons = []
        if longest_streak >= streak_threshold:
            reasons.append(f"repetitive action streak ({longest_streak} steps)")
        if longest_ar_streak >= streak_threshold:
            reasons.append(f"rising addiction_risk streak ({longest_ar_streak} steps)")
        if ep["termination_reason"] == "trust_collapse":
            reasons.append("ended in trust_collapse")

        if reasons:
            candidates.append({**ep, "flags": reasons})

    return candidates


def print_trajectory(ep: Dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  seed={ep['seed']}  termination={ep['termination_reason']}"
          f"  flags={ep.get('flags', [])}")
    print(f"{'=' * 70}")
    print(f"  {'step':>4}  {'action':<24}  {'reward':>7}  {'trust':>6}  "
          f"{'fatigue':>7}  {'addict_risk':>11}")
    for i, step in enumerate(ep["steps"], 1):
        label = step["content_id"] or step["action_type"]
        if step["content_type"]:
            label = f"{label} [{step['content_type']}]"
        print(f"  {i:>4}  {label[:24]:<24}  {step['reward']:>7.4f}  "
              f"{step['trust']:>6.3f}  {step['fatigue']:>7.4f}  {step['addiction_risk']:>11.4f}")
    print(f"  final_score={ep['episode_grade'].get('final_score', 'n/a')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretability + failure-mode analysis")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--streak_threshold", type=int, default=4)
    parser.add_argument("--max_gallery", type=int, default=3,
                         help="Max number of flagged trajectories to print in full")
    args = parser.parse_args()

    episodes = run_analysis_episodes(args.task, args.model, args.n_episodes)

    print(f"\n{'#' * 70}\n  INTERPRETABILITY  [{args.task.upper()}]  n_episodes={args.n_episodes}\n{'#' * 70}")

    dist = summarize_action_distribution(episodes)
    print("\naction_type distribution:", dist["action_counts"])
    print("content_type chosen when trust FRAGILE (<0.3):", dist["content_type_when_trust_fragile"])
    print("content_type chosen when trust STABLE  (>=0.3):", dist["content_type_when_trust_stable"])

    term = summarize_termination_reasons(episodes)
    print("\ntermination reasons:", term)

    candidates = find_failure_mode_candidates(episodes, streak_threshold=args.streak_threshold)
    print(f"\n{'#' * 70}\n  FAILURE-MODE GALLERY  ({len(candidates)} flagged / {len(episodes)} episodes)\n{'#' * 70}")
    for ep in candidates[: args.max_gallery]:
        print_trajectory(ep)