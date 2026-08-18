"""
eval_rl.py — Evaluation script for trained PPO agents.

Usage:
    python eval_rl.py --task easy
    python eval_rl.py --task hard --compare       # heuristic + random + PPO
    python eval_rl.py --task medium --n_eval 20   # mean ± std over 20 episodes
"""

from __future__ import annotations

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import numpy as np
from stable_baselines3 import PPO, A2C
from sb3_contrib import MaskablePPO
from typing import Optional, cast

from environment.rl_wrapper import AttentionEnvWrapper, ALL_CONTENT_IDS, build_env
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

def _run_ppo_episode(env: AttentionEnvWrapper, model, seed: int, masked: bool = False) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    final_info = {}
    while not done:
        if masked:
            action_int, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        else:
            action_int, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action_int))
        done = terminated or truncated
        final_info = info
    return final_info.get("episode_grade", {})


def _greedy_engagement(obs) -> Action:
    """
    Naive baseline representing "what a platform optimizing purely for
    engagement, with no ethical guardrails, would do" -- this is the
    contrast case the whole benchmark exists to argue against, so it's
    worth having as an explicit, separately-reported baseline rather than
    just implying it.

    Estimates engagement using the SAME formula SimulationEngine.
    compute_engagement uses: interest_match (weighted dot product between
    interest_distribution and topic_relevance, matching
    compute_interest_match exactly) combined with addictiveness and
    novelty the same way compute_engagement's base/relevance_weight terms
    do. Deliberately EXCLUDES manipulation_score and repetition/fatigue
    awareness from the scoring -- a pure engagement-maximizer has no
    intrinsic reason to consider them, since they don't appear in the
    immediate engagement signal itself.

    Verified empirically (see tests/test_greedy_engagement.py) to select
    dramatically higher-manipulation, higher-addictiveness content than
    the ethics-aware _heuristic baseline on the same task/seed.
    """
    recent = set(obs.recent_content_ids)
    dist = obs.interest_distribution
    denom = sum(dist.values()) or 1.0

    def estimated_engagement(item) -> float:
        topics = set(item.topic_relevance) & set(dist)
        interest_match = (
            sum(dist[t] * item.topic_relevance[t] for t in topics) / denom
            if topics else 0.0
        )
        base = 0.4 + 0.6 * item.addictiveness
        relevance_weight = max(interest_match, 0.3 * item.addictiveness)
        novelty_factor = item.novelty  # repetition penalty ignored -- naive baseline
        return base * relevance_weight * (0.5 + 0.5 * novelty_factor)

    candidates = [item for item in obs.available_content if item.content_id not in recent]
    if not candidates:
        candidates = obs.available_content
    if not candidates:
        return Action(action_type="diversify_feed")

    best_item = max(candidates, key=estimated_engagement)
    return Action(action_type="recommend", content_id=best_item.content_id)


def _run_greedy_engagement_episode(task_id: str, seed: int) -> dict:
    env = AttentionEconomyEnv()
    obs = env.reset(task_id, seed=seed)
    done = False
    final_info = {}
    while not done:
        action = _greedy_engagement(obs)
        obs, _, done, info = env.step(action)
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
    basename = os.path.basename(model_path)
    # Masked checkpoints are named with a _masked suffix, and A2C checkpoints
    # with an a2c_ prefix (see train_rl.py's save-path convention) -- detect
    # both here so the right algorithm class loads the model and the right
    # env exposes action_masks() for predict() when applicable. masked=True
    # and algo="a2c" are mutually exclusive by construction (train_task
    # raises if both are requested), so is_a2c implies masked is False.
    is_a2c = basename.startswith("a2c_")
    masked = (not is_a2c) and ("_masked" in basename)
    env = cast(AttentionEnvWrapper, build_env(task_id, masked=masked))
    algo_cls = A2C if is_a2c else (MaskablePPO if masked else PPO)
    model = algo_cls.load(model_path, env=env)

    if verbose and n_eval == 1:
        # Single episode: print step-by-step like demo.py
        obs, _ = env.reset(seed=42)
        print(f"\n{'═'*62}")
        print(f"  PPO AGENT  |  TASK: {task_id.upper()}  |  {os.path.basename(model_path)}"
              f"{'  [MASKED]' if masked else ''}")
        print(f"{'═'*62}")
        print(f"  {'Step':>4}  {'Action':<22}  {'R':>7}  {'Trust':>6}  {'Fatigue':>7}  {'Sat':>5}")
        print(f"  {'─'*57}")
        done, step, total_r, final_info = False, 0, 0.0, {}
        while not done:
            if masked:
                action_int, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)  # type: ignore[call-arg]
            else:
                action_int, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action_int))
            done = terminated or truncated
            step += 1; total_r += reward; final_info = info
            raw = env._last_obs
            assert raw is not None  # populated by the step() call just above
            label = env.get_action_label(int(action_int))[:20]
            print(f"  {step:>4}  {label:<22}  {reward:.5f}  "
                  f"{raw.visible_trust:.4f}  {raw.visible_fatigue:.5f}  "
                  f"{raw.visible_satisfaction:.3f}")
        _print_grade(final_info, total_r, step)
        env.close()
        return final_info.get("episode_grade", {})

    # Multi-episode: collect stats
    grades = [_run_ppo_episode(env, model, seed=i, masked=masked) for i in range(n_eval)]
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


def evaluate_greedy_engagement(task_id: str, n_eval: int = 1, verbose: bool = True) -> dict:
    if verbose and n_eval == 1:
        env = AttentionEconomyEnv()
        obs = env.reset(task_id, seed=42)
        print(f"\n{'═'*62}")
        print(f"  GREEDY-ENGAGEMENT  |  TASK: {task_id.upper()}")
        print(f"{'═'*62}")
        print(f"  {'Step':>4}  {'Action':<22}  {'R':>7}  {'Trust':>6}  {'Fatigue':>7}  {'Sat':>5}")
        print(f"  {'─'*57}")
        done, step, total_r, final_info = False, 0, 0.0, {}
        while not done:
            action = _greedy_engagement(obs)
            obs, reward, done, info = env.step(action)
            step += 1; total_r += reward; final_info = info
            label = (action.content_id or action.action_type)[:20]
            print(f"  {step:>4}  {label:<22}  {reward:.5f}  "
                  f"{obs.visible_trust:.4f}  {obs.visible_fatigue:.5f}  "
                  f"{obs.visible_satisfaction:.3f}")
        _print_grade(final_info, total_r, step)
        return final_info.get("episode_grade", {})

    grades = [_run_greedy_engagement_episode(task_id, seed=i) for i in range(n_eval)]
    return _aggregate(grades)

# ─────────────────────────────────────────────
# Compare: Random vs Heuristic vs PPO
# ─────────────────────────────────────────────

def compare(task_id: str, model_path: Optional[str] = None, n_eval: int = 20):
    print(f"\n{'#'*65}")
    print(f"  COMPARISON [{task_id.upper()}]  —  {n_eval} episodes each  (mean ± std)")
    print(f"{'#'*65}")

    print(f"\n  Running random agent      ({n_eval} eps)...", end=" ", flush=True)
    r_grade = evaluate_random(task_id, n_eval)
    print("done")

    print(f"  Running greedy-engagement ({n_eval} eps)...", end=" ", flush=True)
    g_grade = evaluate_greedy_engagement(task_id, n_eval=n_eval, verbose=False)
    print("done")

    print(f"  Running heuristic         ({n_eval} eps)...", end=" ", flush=True)
    h_grade = evaluate_heuristic(task_id, n_eval=n_eval, verbose=False)
    print("done")

    print(f"  Running PPO               ({n_eval} eps)...", end=" ", flush=True)
    p_grade = evaluate_ppo(task_id, model_path, n_eval=n_eval, verbose=False)
    print("done")

    metrics = ["final_score", "avg_engagement", "final_trust", "final_satisfaction"]
    print(f"\n{'─'*80}")
    print(f"  {'Metric':<20}  {'Random':>14}  {'Greedy-Eng':>14}  {'Heuristic':>14}  {'PPO':>14}")
    print(f"  {'─'*77}")

    for m in metrics:
        def fmt(g): return f"{g.get(m+'_mean', 0):.3f}±{g.get(m+'_std', 0):.3f}"
        h_val = h_grade.get(m + "_mean", 0)
        p_val = p_grade.get(m + "_mean", 0)
        flag = "▲" if p_val > h_val + 0.005 else ("▼" if p_val < h_val - 0.005 else "≈")
        print(f"  {m:<20}  {fmt(r_grade):>14}  {fmt(g_grade):>14}  {fmt(h_grade):>14}  {fmt(p_grade):>14}  {flag}")

    print(f"\n  PPO vs Heuristic improvement:         "
          f"{(p_grade.get('final_score_mean',0) - h_grade.get('final_score_mean',0)):+.3f} final_score")
    print(f"  PPO vs Random improvement:             "
          f"{(p_grade.get('final_score_mean',0) - r_grade.get('final_score_mean',0)):+.3f} final_score")
    print(f"  PPO vs Greedy-Engagement improvement:  "
          f"{(p_grade.get('final_score_mean',0) - g_grade.get('final_score_mean',0)):+.3f} final_score  "
          f"(this is the comparison that matters most: does the agent do better "
          f"than a system with no ethical guardrails at all?)")


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