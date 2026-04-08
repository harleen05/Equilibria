"""
plot_results.py — Trust trajectory and score comparison plots.

Generates two figures saved to results/:
  1. trust_trajectory.png  — trust over steps: random vs heuristic vs PPO
  2. score_comparison.png  — bar chart of final_score across all tasks and agents

Usage:
    python plot_results.py --task easy
    python plot_results.py --task hard
    python plot_results.py --all         # generate plots for all three tasks
"""

from __future__ import annotations

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List

from rl_wrapper import AttentionEnvWrapper, N_CONTENT
from env_core import AttentionEconomyEnv
from models import Action
from stable_baselines3 import PPO

os.makedirs("results", exist_ok=True)

COLORS = {
    "random":    "#9e9e9e",
    "heuristic": "#f57c00",
    "ppo":       "#1976d2",
}

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
# Trajectory collectors
# ─────────────────────────────────────────────

def _collect_ppo_trajectory(task_id: str, seed: int) -> dict:
    path = DEFAULT_MODEL_PATHS[task_id]
    if not os.path.exists(path + ".zip"):
        path = FALLBACK_MODEL_PATHS[task_id]
    env = AttentionEnvWrapper(task_id=task_id)
    model = PPO.load(path, env=env)
    obs, _ = env.reset(seed=seed)
    trust, satisfaction, reward_hist = [], [], []
    done = False
    while not done:
        action_int, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action_int))
        done = terminated or truncated
        raw = env._last_obs
        trust.append(raw.visible_trust)
        satisfaction.append(raw.visible_satisfaction)
        reward_hist.append(reward)
    final = info.get("episode_grade", {})
    env.close()
    return {"trust": trust, "satisfaction": satisfaction,
            "reward": reward_hist, "grade": final}


def _collect_heuristic_trajectory(task_id: str, seed: int) -> dict:
    env = AttentionEconomyEnv()
    obs = env.reset(task_id, seed=seed)
    trust, satisfaction, reward_hist = [], [], []
    done = False
    final = {}
    while not done:
        action = _heuristic(obs)
        obs, reward, done, info = env.step(action)
        trust.append(obs.visible_trust)
        satisfaction.append(obs.visible_satisfaction)
        reward_hist.append(reward)
        final = info
    return {"trust": trust, "satisfaction": satisfaction,
            "reward": reward_hist, "grade": final.get("episode_grade", {})}


def _collect_random_trajectory(task_id: str, seed: int) -> dict:
    env = AttentionEnvWrapper(task_id=task_id)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    trust, satisfaction, reward_hist = [], [], []
    done = False
    final = {}
    while not done:
        valid = np.where(env.action_masks())[0]
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        raw = env._last_obs
        trust.append(raw.visible_trust)
        satisfaction.append(raw.visible_satisfaction)
        reward_hist.append(reward)
        final = info
    env.close()
    return {"trust": trust, "satisfaction": satisfaction,
            "reward": reward_hist, "grade": final.get("episode_grade", {})}


def _pad(lst: list, length: int, fill: float = None) -> list:
    """Pad a trajectory to fixed length (last value repeated)."""
    if fill is None:
        fill = lst[-1] if lst else 0.0
    return lst + [fill] * max(0, length - len(lst))


# ─────────────────────────────────────────────
# Plot: Trust Trajectory
# ─────────────────────────────────────────────

def plot_trust_trajectory(task_id: str, n_seeds: int = 10):
    max_steps = {"easy": 15, "medium": 20, "hard": 25}[task_id]
    steps = list(range(1, max_steps + 1))

    all_trust = {"random": [], "heuristic": [], "ppo": []}

    for seed in range(n_seeds):
        r = _collect_random_trajectory(task_id, seed)
        h = _collect_heuristic_trajectory(task_id, seed)
        p = _collect_ppo_trajectory(task_id, seed)
        all_trust["random"].append(_pad(r["trust"], max_steps, 0.0))
        all_trust["heuristic"].append(_pad(h["trust"], max_steps, 0.0))
        all_trust["ppo"].append(_pad(p["trust"], max_steps, p["trust"][-1]))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    for agent, color in COLORS.items():
        arr = np.array(all_trust[agent])   # (n_seeds, max_steps)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax.plot(steps, mean, color=color, linewidth=2.5, label=agent.upper())
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.15, color=color)

    ax.set_xlabel("Step", color="white", fontsize=12)
    ax.set_ylabel("User Trust", color="white", fontsize=12)
    ax.set_title(f"Trust Over Episode — {task_id.upper()} Task\n"
                 f"({n_seeds} seeds, mean ± 1σ)",
                 color="white", fontsize=13, pad=12)
    ax.set_xlim(1, max_steps)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, alpha=0.15, color="white")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=11,
              framealpha=0.9, loc="lower left")

    path = f"results/trust_trajectory_{task_id}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved → {path}")


# ─────────────────────────────────────────────
# Plot: Score Comparison Bar Chart (all tasks)
# ─────────────────────────────────────────────

def plot_score_comparison(n_seeds: int = 10):
    tasks = ["easy", "medium", "hard"]
    agents = ["random", "heuristic", "ppo"]
    collectors = {
        "random":    _collect_random_trajectory,
        "heuristic": _collect_heuristic_trajectory,
        "ppo":       _collect_ppo_trajectory,
    }

    scores = {a: [] for a in agents}
    errors = {a: [] for a in agents}

    for task in tasks:
        for agent in agents:
            vals = []
            for seed in range(n_seeds):
                try:
                    traj = collectors[agent](task, seed)
                    vals.append(traj["grade"].get("final_score", 0.0))
                except Exception:
                    vals.append(0.0)
            scores[agent].append(np.mean(vals))
            errors[agent].append(np.std(vals))

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    for i, agent in enumerate(agents):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, scores[agent], width,
                      yerr=errors[agent], capsize=4,
                      color=COLORS[agent], alpha=0.85,
                      label=agent.upper(),
                      error_kw=dict(ecolor="white", elinewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], color="white", fontsize=12)
    ax.set_ylabel("Final Score (mean ± std)", color="white", fontsize=12)
    ax.set_title(f"Agent Comparison Across Tasks  ({n_seeds} seeds)",
                 color="white", fontsize=13, pad=12)
    ax.set_ylim(0, 1.0)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, axis="y", alpha=0.15, color="white")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=11, framealpha=0.9)

    path = "results/score_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved → {path}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"],
                        default="all")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Episodes per agent per task (default: 10)")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    print(f"\nGenerating trust trajectory plots ({args.n_seeds} seeds)...")
    for task in tasks:
        print(f"  [{task.upper()}]")
        plot_trust_trajectory(task, n_seeds=args.n_seeds)

    if args.task == "all":
        print(f"\nGenerating score comparison chart...")
        plot_score_comparison(n_seeds=args.n_seeds)

    print(f"\nAll plots saved to results/")