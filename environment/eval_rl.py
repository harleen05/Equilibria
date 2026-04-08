"""
eval_rl.py — Evaluation script for trained PPO agents.

Loads a saved model and runs one full episode, printing step-by-step
actions and rewards in the same format as demo.py for easy comparison.

Usage:
    python eval_rl.py                          # evaluate medium (default)
    python eval_rl.py --task easy
    python eval_rl.py --task hard
    python eval_rl.py --task medium --model models/ppo_medium_final
    python eval_rl.py --compare                # run heuristic + PPO side-by-side
"""

from __future__ import annotations

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO

from rl_wrapper import AttentionEnvWrapper, ALL_CONTENT_IDS, META_ACTIONS, N_CONTENT
from env_core import AttentionEconomyEnv
from models import Action


# ─────────────────────────────────────────────
# Default model paths
# ─────────────────────────────────────────────

DEFAULT_MODEL_PATHS = {
    "easy":   "models/best/easy/best_model",
    "medium": "models/best/medium/best_model",
    "hard":   "models/best/hard/best_model",
}

FALLBACK_MODEL_PATHS = {
    "easy":   "models/ppo_easy_final",
    "medium": "models/ppo_medium_final",
    "hard":   "models/ppo_hard_final",
}


# ─────────────────────────────────────────────
# PPO Evaluation
# ─────────────────────────────────────────────

def evaluate_ppo(task_id: str, model_path: Optional[str] = None) -> dict:
    """
    Run one deterministic episode using the trained PPO model.
    Returns the episode grade dict.
    """
    # Resolve model path
    if model_path is None:
        model_path = DEFAULT_MODEL_PATHS.get(task_id, "")
        if not os.path.exists(model_path + ".zip"):
            model_path = FALLBACK_MODEL_PATHS.get(task_id, "")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}.zip'.\n"
            f"Run: python train_rl.py --task {task_id}"
        )

    env = AttentionEnvWrapper(task_id=task_id)
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()

    print(f"\n{'═' * 62}")
    print(f"  PPO AGENT  |  TASK: {task_id.upper()}  |  model: {os.path.basename(model_path)}")
    print(f"{'═' * 62}")
    print(f"  {'Step':>4}  {'Action':<22}  {'R':>7}  {'Trust':>6}  {'Fatigue':>7}  {'Sat':>5}")
    print(f"  {'─' * 57}")

    total_reward = 0.0
    final_info = {}
    step = 0

    while True:
        action_int, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action_int))

        step += 1
        total_reward += reward
        final_info = info

        # Human-readable label
        label = env.get_action_label(int(action_int))
        if len(label) > 20:
            label = label[:19] + "…"

        raw = env._last_obs
        print(
            f"  {step:>4}  {label:<22}  "
            f"{reward:.5f}  {raw.visible_trust:.4f}  "
            f"{raw.visible_fatigue:.5f}  {raw.visible_satisfaction:.3f}"
        )

        if terminated or truncated:
            break

    _print_grade(final_info, total_reward, step)
    env.close()
    return final_info.get("episode_grade", {})


# ─────────────────────────────────────────────
# Heuristic Baseline (same as demo.py)
# ─────────────────────────────────────────────

def evaluate_heuristic(task_id: str) -> dict:
    """Run the rule-based heuristic agent for comparison."""
    env = AttentionEconomyEnv()
    obs = env.reset(task_id)

    print(f"\n{'═' * 62}")
    print(f"  HEURISTIC  |  TASK: {task_id.upper()}")
    print(f"{'═' * 62}")
    print(f"  {'Step':>4}  {'Action':<22}  {'R':>7}  {'Trust':>6}  {'Fatigue':>7}  {'Sat':>5}")
    print(f"  {'─' * 57}")

    total_reward = 0.0
    final_info = {}
    step = 0

    while True:
        action = _heuristic(obs)
        obs, reward, done, info = env.step(action)

        step += 1
        total_reward += reward
        final_info = info

        label = action.content_id if action.content_id else action.action_type
        if len(label) > 20:
            label = label[:19] + "…"

        print(
            f"  {step:>4}  {label:<22}  "
            f"{reward:.5f}  {obs.visible_trust:.4f}  "
            f"{obs.visible_fatigue:.5f}  {obs.visible_satisfaction:.3f}"
        )
        if done:
            break

    _print_grade(final_info, total_reward, step)
    return final_info.get("episode_grade", {})


def _heuristic(obs) -> Action:
    if obs.visible_fatigue > 0.70:
        return Action(action_type="pause_session")
    if obs.visible_boredom > 0.50:
        return Action(action_type="diversify_feed")

    dominant = max(obs.interest_distribution, key=obs.interest_distribution.get)
    recent   = set(obs.recent_content_ids)
    best_item, best_score = None, -1.0

    for item in obs.available_content:
        if item.content_id in recent:
            continue
        match   = item.topic_relevance.get(dominant, 0.0)
        ethical = (1.0 - item.manipulation_score) * (1.0 - item.addictiveness)
        score   = match * ethical
        if score > best_score:
            best_score = score
            best_item  = item

    if best_item is None:
        return Action(action_type="explore_new_topic", topic=dominant)
    return Action(action_type="recommend", content_id=best_item.content_id)


def _print_grade(info: dict, total_reward: float, steps: int):
    print(f"\n  {'─' * 57}")
    print(f"  Total reward (sum)  : {total_reward:.4f}  over {steps} steps")
    if "episode_grade" in info:
        g = info["episode_grade"]
        print(f"  Final Score         : {g.get('final_score', 0):.4f}")
        print(f"  └─ avg_engagement   : {g.get('avg_engagement', 0):.4f}")
        print(f"  └─ final_trust      : {g.get('final_trust', 0):.4f}")
        print(f"  └─ final_satisf.    : {g.get('final_satisfaction', 0):.4f}")
    if "termination_reason" in info:
        print(f"  Termination         : {info['termination_reason']}")


# ─────────────────────────────────────────────
# Side-by-side comparison
# ─────────────────────────────────────────────

def compare(task_id: str, model_path: Optional[str] = None):
    print(f"\n{'#'*62}")
    print(f"  COMPARISON: Heuristic vs PPO  [{task_id.upper()}]")
    print(f"{'#'*62}")

    h_grade = evaluate_heuristic(task_id)
    p_grade = evaluate_ppo(task_id, model_path)

    print(f"\n{'─'*62}")
    print(f"  SUMMARY ({task_id.upper()})")
    print(f"{'─'*62}")
    metrics = ["final_score", "avg_engagement", "final_trust", "final_satisfaction"]
    print(f"  {'Metric':<22}  {'Heuristic':>10}  {'PPO':>10}  {'Δ':>8}")
    print(f"  {'─'*57}")
    for m in metrics:
        h = h_grade.get(m, 0.0)
        p = p_grade.get(m, 0.0)
        delta = p - h
        flag = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "≈")
        print(f"  {m:<22}  {h:>10.4f}  {p:>10.4f}  {flag} {abs(delta):>5.4f}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO agent on AttentionEconomyEnv")
    parser.add_argument(
        "--task", choices=["easy", "medium", "hard"], default="medium",
        help="Task to evaluate (default: medium)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model (no .zip). Defaults to best_model for the task."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run heuristic and PPO side-by-side and print a comparison table"
    )
    args = parser.parse_args()

    if args.compare:
        compare(args.task, args.model)
    else:
        evaluate_ppo(args.task, args.model)