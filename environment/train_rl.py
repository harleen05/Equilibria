"""
train_rl.py — PPO training script for the Attention Economy Environment.

Training strategy:
  - Three separate models are trained, one per task difficulty.
  - Each model is saved individually so they can be evaluated or fine-tuned
    independently.
  - Curriculum: easy → medium → hard. The hard model is warm-started from
    the medium checkpoint via policy cloning (load + continue training).
  - Callbacks: EvalCallback saves the best model checkpoint per task.

Usage:
    python train_rl.py                    # train all three tasks
    python train_rl.py --task easy        # train a single task
    python train_rl.py --task hard --timesteps 50000
"""

from __future__ import annotations

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.rl_wrapper import AttentionEnvWrapper


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "total_timesteps": 20_000,
        "n_envs": 4,               # parallel rollout workers
    },
    "medium": {
        "total_timesteps": 30_000,
        "n_envs": 4,
    },
    "hard": {
        "total_timesteps": 50_000,
        "n_envs": 4,
    },
}

# PPO hyperparameters — tuned for short-horizon episodic tasks
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=256,            # rollout buffer length per env (short eps → small buffer)
    batch_size=64,
    n_epochs=10,
    gamma=0.99,             # discount — trust the future
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,          # mild entropy bonus to encourage exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # SB3 >= v1.8.0 format
    ),
    verbose=1,
)

MODEL_DIR  = "models"
LOG_DIR    = "logs"
BEST_DIR   = "models/best"


# ─────────────────────────────────────────────
# Reward Logging Callback
# ─────────────────────────────────────────────

class EpisodeSummaryCallback(BaseCallback):
    """
    Logs episode final_score from info["episode_grade"] to console.
    Gives human-readable progress beyond raw SB3 output.
    """
    def __init__(self, task_id: str, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.task_id = task_id
        self.log_freq = log_freq
        self._episode_rewards: list = []
        self._episode_grades: list = []

    def _on_step(self) -> bool:
        # SB3 stores per-env infos in self.locals["infos"]
        for info in self.locals.get("infos", []):
            if "episode_grade" in info:
                grade = info["episode_grade"]
                self._episode_grades.append(grade["final_score"])

        if self.n_calls % self.log_freq == 0 and self._episode_grades:
            recent = self._episode_grades[-20:]
            mean_score = np.mean(recent)
            print(
                f"  [{self.task_id.upper()}] step={self.n_calls:>6}  "
                f"mean_episode_score={mean_score:.4f}  "
                f"(over last {len(recent)} eps)"
            )
        return True


# ─────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────

def make_env(task_id: str):
    """Factory function compatible with make_vec_env."""
    def _init():
        env = AttentionEnvWrapper(task_id=task_id)
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_task(
    task_id: str,
    total_timesteps: int,
    n_envs: int,
    warmstart_path: Optional[str] = None,
) -> str:
    """
    Train a PPO agent on the given task.

    Parameters
    ----------
    task_id          : "easy", "medium", or "hard"
    total_timesteps  : Total env steps to train for
    n_envs           : Number of parallel envs (DummyVecEnv)
    warmstart_path   : Path to a previous model to continue training from

    Returns
    -------
    Path to the saved final model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Training task: {task_id.upper()}  ({total_timesteps:,} steps, {n_envs} envs)")
    print(f"{'═'*60}")

    # ── Vectorised training envs ──────────────────────────────────────────
    vec_env = DummyVecEnv([make_env(task_id) for _ in range(n_envs)])

    # ── Eval env (single, unvectorised) ──────────────────────────────────
    eval_env = Monitor(AttentionEnvWrapper(task_id=task_id))

    # ── Callbacks ─────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(BEST_DIR, task_id),
        log_path=os.path.join(LOG_DIR, task_id),
        eval_freq=max(1000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )
    summary_cb = EpisodeSummaryCallback(task_id=task_id, log_freq=500)
    callbacks = CallbackList([eval_cb, summary_cb])

    # ── Model ─────────────────────────────────────────────────────────────
    if warmstart_path and os.path.exists(warmstart_path + ".zip"):
        print(f"  Warm-starting from: {warmstart_path}")
        model = PPO.load(warmstart_path, env=vec_env, **{
            k: v for k, v in PPO_KWARGS.items()
            if k not in ("verbose",)
        })
        model.verbose = 1
    else:
        model = PPO("MlpPolicy", vec_env, **PPO_KWARGS, seed=42)

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = os.path.join(MODEL_DIR, f"ppo_{task_id}_final")
    model.save(save_path)
    print(f"\n  ✓ Model saved → {save_path}.zip  ({elapsed:.1f}s)")

    vec_env.close()
    eval_env.close()
    return save_path


# ─────────────────────────────────────────────
# Curriculum entry point
# ─────────────────────────────────────────────

def train_curriculum():
    """
    Train easy → medium → hard with warm-starting.
    The hard model benefits from the policy learned on medium.
    """
    easy_path   = train_task("easy",   **{k: TASK_CONFIGS["easy"][k]   for k in ("total_timesteps","n_envs")})
    medium_path = train_task("medium", **{k: TASK_CONFIGS["medium"][k] for k in ("total_timesteps","n_envs")})
    _           = train_task("hard",   **{k: TASK_CONFIGS["hard"][k]   for k in ("total_timesteps","n_envs")},
                             warmstart_path=medium_path)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on AttentionEconomyEnv")
    parser.add_argument(
        "--task", choices=["easy", "medium", "hard", "all"], default="all",
        help="Which task to train (default: all via curriculum)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total timesteps"
    )
    parser.add_argument(
        "--warmstart", type=str, default=None,
        help="Path to model checkpoint to warm-start from (no .zip extension)"
    )
    args = parser.parse_args()

    if args.task == "all":
        train_curriculum()
    else:
        cfg = TASK_CONFIGS[args.task].copy()
        if args.timesteps:
            cfg["total_timesteps"] = args.timesteps
        train_task(args.task, warmstart_path=args.warmstart, **cfg)