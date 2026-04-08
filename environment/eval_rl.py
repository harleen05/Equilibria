"""
eval_rl.py — Evaluation script for trained PPO agents.

Usage:
    python eval_rl.py --task easy
    python eval_rl.py --task hard --compare       # heuristic + random + PPO
    python eval_rl.py --task medium --n_eval 20   # mean ± std over 20 episodes
"""

from __future__ import annotations

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from typing import Optional

from environment.rl_wrapper import AttentionEnvWrapper, ALL_CONTENT_IDS, N_CONTENT
from environment.env_core import AttentionEconomyEnv
from environment.models import Action


# ─────────────────────────────────────────────
# Model paths
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
# Single episode runners
# ─────────────────────────────────────────────

def _run_ppo_episode(env: AttentionEnvWrapper, model: PPO, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    final_info = {}
    while not done:
        action_int, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action_int))
        done = terminated or truncated
        final_info = info
    return final_info.get("episode_grade", {})


def _run_heuristic_episode(task_id: str, seed: int) -> dict:
    env = AttentionEconomyEnv()
    obs = env.reset(task_id, seed=seed)
    done = False
    final_info = {}
    while not done:
        action = _heuristic(obs)
        obs, _, done, info = env.step(action)
        final_info = info
    return final_info.get("episode_grade", {})


def _run_random_episode(task_id: str, seed: int) -> dict:
    env = AttentionEnvWrapper(task_id=task_id)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    done = False
    final_info = {}
    while not done:
        # Random action restricted to valid (allowed) content only
        valid = np.where(env.action_masks())[0]
        action = int(rng.choice(valid))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_info = info
    env.close()
    return final_info.get("episode_grade", {})


# ─────────────────────────────────────────────
# Multi-episode evaluation (mean ± std)
# ─────────────────────────────────────────────

def evaluate_ppo(task_id: str, model_path: Optional[str] = None,
                 n_eval: int = 1, verbose: bool = True) -> dict:
    model_path = _resolve_model_path(task_id, model_path)
    env = AttentionEnvWrapper(task_id=task_id)
    model = PPO.load(model_path, env=env)

    if verbose and n_eval == 1:
        # Single episode: print step-by-step like demo.py
        obs, _ = env.reset(seed=42)
        print(f"\n{'═'*62}")
        print(f"  PPO AGENT  |  TASK: {task_id.upper()}  |  {os.path.basename(model_path)}")
        print(f"{'═'*62}")
        print(f"  {'Step':>4}  {'Action':<22}  {'R':>7}  {'Trust':>6}  {'Fatigue':>7}  {'Sat':>5}")
        print(f"  {'─'*57}")
        done, step, total_r, final_info = False, 0, 0.0, {}
        while not done:
            action_int, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action_int))
            done = terminated or truncated
            step += 1; total_r += reward; final_info = info
            raw = env._last_obs
            label = env.get_action_label(int(action_int))[:20]
            print(f"  {step:>4}  {label:<22}  {reward:.5f}  "
                  f"{raw.visible_trust:.4f}  {raw.visible_fatigue:.5f}  "
                  f"{raw.visible_satisfaction:.3f}")
        _print_grade(final_info, total_r, step)
        env.close()
        return final_info.get("episode_grade", {})

    # Multi-episode: collect stats
    grades = [_run_ppo_episode(env, model, seed=i) for i in range(n_eval)]
    env.close()
    return _aggregate(grades)


def evaluate_heuristic(task_id: str, n_eval: int = 1, verbose: bool = True) -> dict:
    if verbose and n_eval == 1:
        env = AttentionEconomyEnv()
        obs = env.reset(task_id, seed=42)
        print(f"\n{'═'*62}")
        print(f"  HEURISTIC  |  TASK: {task_id.upper()}")
        print(f"{'═'*62}")
        print(f"  {'Step':>4}  {'Action':<22}  {'R':>7}  {'Trust':>6}  {'Fatigue':>7}  {'Sat':>5}")
        print(f"  {'─'*57}")
        done, step, total_r, final_info = False, 0, 0.0, {}
        while not done:
            action = _heuristic(obs)
            obs, reward, done, info = env.step(action)
            step += 1; total_r += reward; final_info = info
            label = (action.content_id or action.action_type)[:20]
            print(f"  {step:>4}  {label:<22}  {reward:.5f}  "
                  f"{obs.visible_trust:.4f}  {obs.visible_fatigue:.5f}  "
                  f"{obs.visible_satisfaction:.3f}")
        _print_grade(final_info, total_r, step)
        return final_info.get("episode_grade", {})

    grades = [_run_heuristic_episode(task_id, seed=i) for i in range(n_eval)]
    return _aggregate(grades)


def evaluate_random(task_id: str, n_eval: int = 20) -> dict:
    grades = [_run_random_episode(task_id, seed=i) for i in range(n_eval)]
    return _aggregate(grades)


# ─────────────────────────────────────────────
# Compare: Random vs Heuristic vs PPO
# ─────────────────────────────────────────────

def compare(task_id: str, model_path: Optional[str] = None, n_eval: int = 20):
    print(f"\n{'#'*65}")
    print(f"  COMPARISON [{task_id.upper()}]  —  {n_eval} episodes each  (mean ± std)")
    print(f"{'#'*65}")

    print(f"\n  Running random agent   ({n_eval} eps)...", end=" ", flush=True)
    r_grade = evaluate_random(task_id, n_eval)
    print("done")

    print(f"  Running heuristic      ({n_eval} eps)...", end=" ", flush=True)
    h_grade = evaluate_heuristic(task_id, n_eval=n_eval, verbose=False)
    print("done")

    print(f"  Running PPO            ({n_eval} eps)...", end=" ", flush=True)
    p_grade = evaluate_ppo(task_id, model_path, n_eval=n_eval, verbose=False)
    print("done")

    metrics = ["final_score", "avg_engagement", "final_trust", "final_satisfaction"]
    print(f"\n{'─'*65}")
    print(f"  {'Metric':<22}  {'Random':>14}  {'Heuristic':>14}  {'PPO':>14}")
    print(f"  {'─'*62}")

    for m in metrics:
        def fmt(g): return f"{g.get(m+'_mean', 0):.3f}±{g.get(m+'_std', 0):.3f}"
        r_val = r_grade.get(m + "_mean", 0)
        h_val = h_grade.get(m + "_mean", 0)
        p_val = p_grade.get(m + "_mean", 0)
        flag = "▲" if p_val > h_val + 0.005 else ("▼" if p_val < h_val - 0.005 else "≈")
        print(f"  {m:<22}  {fmt(r_grade):>14}  {fmt(h_grade):>14}  {fmt(p_grade):>14}  {flag}")

    print(f"\n  PPO vs Heuristic improvement: "
          f"{(p_grade.get('final_score_mean',0) - h_grade.get('final_score_mean',0)):+.3f} final_score")
    print(f"  PPO vs Random improvement:    "
          f"{(p_grade.get('final_score_mean',0) - r_grade.get('final_score_mean',0)):+.3f} final_score")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _aggregate(grades: list) -> dict:
    """Compute mean ± std across episodes for all grade metrics."""
    if not grades:
        return {}
    keys = grades[0].keys()
    result = {}
    for k in keys:
        vals = [g.get(k, 0.0) for g in grades]
        result[k] = round(float(np.mean(vals)), 4)
        result[k + "_mean"] = round(float(np.mean(vals)), 4)
        result[k + "_std"]  = round(float(np.std(vals)),  4)
    return result


def _resolve_model_path(task_id: str, model_path: Optional[str]) -> str:
    if model_path is None:
        model_path = DEFAULT_MODEL_PATHS.get(task_id, "")
    if not os.path.exists(model_path + ".zip"):
        model_path = FALLBACK_MODEL_PATHS.get(task_id, "")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"No model at '{model_path}.zip'. Run: python train_rl.py --task {task_id}")
    return model_path


def _heuristic(obs) -> Action:
    if obs.visible_fatigue > 0.70:
        return Action(action_type="pause_session")
    if obs.visible_boredom > 0.50:
        return Action(action_type="diversify_feed")
    dominant = max(obs.interest_distribution, key=obs.interest_distribution.get)
    recent = set(obs.recent_content_ids)
    best_item, best_score = None, -1.0
    for item in obs.available_content:
        if item.content_id in recent:
            continue
        match = item.topic_relevance.get(dominant, 0.0)
        ethical = (1.0 - item.manipulation_score) * (1.0 - item.addictiveness)
        score = match * ethical
        if score > best_score:
            best_score, best_item = score, item
    if best_item is None:
        return Action(action_type="explore_new_topic", topic=dominant)
    return Action(action_type="recommend", content_id=best_item.content_id)


def _print_grade(info: dict, total_reward: float, steps: int):
    print(f"\n  {'─'*57}")
    print(f"  Total reward : {total_reward:.4f}  over {steps} steps")
    if "episode_grade" in info:
        g = info["episode_grade"]
        print(f"  Final Score  : {g.get('final_score', 0):.4f}")
        print(f"  └─ engagement: {g.get('avg_engagement', 0):.4f}")
        print(f"  └─ trust     : {g.get('final_trust', 0):.4f}")
        print(f"  └─ satisf.   : {g.get('final_satisfaction', 0):.4f}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Random vs Heuristic vs PPO with mean±std")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Episodes per agent in --compare mode (default: 20)")
    args = parser.parse_args()

    if args.compare:
        compare(args.task, args.model, n_eval=args.n_eval)
    else:
        evaluate_ppo(args.task, args.model, n_eval=1, verbose=True)