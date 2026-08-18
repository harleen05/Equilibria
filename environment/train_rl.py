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
from typing import Optional, Type
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.rl_wrapper import AttentionEnvWrapper, build_env
from typing import Optional

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

# A2C has a genuinely different update rule (single-pass, short-rollout
# actor-critic) than PPO (clipped multi-epoch updates over a larger
# buffer), so it can't reuse PPO_KWARGS directly -- A2C's constructor
# doesn't even accept batch_size/n_epochs/clip_range. To keep the
# comparison as fair as possible, every hyperparameter A2C shares with PPO
# (gamma, gae_lambda, ent_coef, vf_coef, max_grad_norm, policy_kwargs -- and
# therefore network architecture) is held IDENTICAL to PPO_KWARGS.
# learning_rate and n_steps are set to stable-baselines3's own tuned A2C
# defaults instead of PPO's values, since those two are core to each
# algorithm's update rule (A2C's short 5-step rollout + higher LR is a
# fundamentally different design point, not an arbitrary choice) -- using
# PPO's n_steps=256/lr=3e-4 for A2C would be testing A2C configured to
# fail, not a fair algorithm comparison.
A2C_KWARGS = dict(
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
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
    seed: int = 42,
    masked: bool = False,
    algo: str = "ppo",
) -> str:
    """
    Train an RL agent on the given task.

    Parameters
    ----------
    task_id          : "easy", "medium", or "hard"
    total_timesteps  : Total env steps to train for
    n_envs           : Number of parallel envs (DummyVecEnv)
    warmstart_path   : Path to a previous model to continue training from.
                        Must have been trained with the SAME algo, or
                        loading will fail (PPO and A2C checkpoints are not
                        interchangeable).
    masked           : If True, train with sb3-contrib's MaskablePPO instead
                        of plain PPO. Only valid when algo="ppo" -- there is
                        no MaskableA2C in sb3-contrib.
    algo             : "ppo" or "a2c". Included as the checkpoint filename
                        PREFIX (not an add-on suffix flag) specifically to
                        avoid the save-path-collision bug class this
                        codebase has hit twice already (seed, then
                        warmstart) -- a prefix can't be forgotten the way an
                        optional suffix flag can.

    Returns
    -------
    Path to the saved final model.
    """
    if algo not in ("ppo", "a2c"):
        raise ValueError(f"Unknown algo '{algo}'. Must be 'ppo' or 'a2c'.")
    if masked and algo != "ppo":
        raise ValueError("masked=True requires algo='ppo' (no MaskableA2C exists in sb3-contrib).")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Training task: {task_id.upper()}  ({total_timesteps:,} steps, {n_envs} envs)"
          f"  [{algo.upper()}]{'  [MASKED]' if masked else ''}")
    print(f"{'═'*60}")

    # ── Vectorised training envs ──────────────────────────────────────────
    if masked:
        def _masked_thunk():
            return Monitor(build_env(task_id, masked=True))
        vec_env = DummyVecEnv([_masked_thunk for _ in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(task_id) for _ in range(n_envs)])

    # ── Eval env (single, unvectorised) ──────────────────────────────────
    eval_env: Monitor = Monitor(build_env(task_id, masked=masked))

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
    algo_cls: Type[BaseAlgorithm]
    if algo == "a2c":
        algo_cls, algo_kwargs = A2C, A2C_KWARGS
    else:
        algo_cls = MaskablePPO if masked else PPO
        algo_kwargs = PPO_KWARGS

    if warmstart_path and os.path.exists(warmstart_path + ".zip"):
        print(f"  Warm-starting from: {warmstart_path}")
        model = algo_cls.load(warmstart_path, env=vec_env, **{
            k: v for k, v in algo_kwargs.items()
            if k not in ("verbose",)
        })  # type: ignore[arg-type]
        model.verbose = 1
    else:
        model = algo_cls("MlpPolicy", vec_env, **algo_kwargs, seed=seed)  # type: ignore[arg-type]

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────────
    # Seed is part of the filename: without this, training multiple seeds
    # for the same task silently overwrites the previous seed's checkpoint,
    # making multi-seed comparison impossible. seed=42 keeps the original
    # unqualified filename for backward compatibility with any existing
    # deployment/download scripts that reference ppo_{task_id}_final.zip.
    # masked=True additionally gets a _masked suffix so masked and unmasked
    # checkpoints for the same task/seed never collide either. algo is the
    # filename PREFIX (see docstring) rather than another optional suffix.
    seed_part = "" if seed == 42 else f"_seed{seed}"
    masked_part = "_masked" if masked else ""
    warmstart_part = "_warmstart" if warmstart_path else ""
    save_path = os.path.join(
        MODEL_DIR, f"{algo}_{task_id}{seed_part}{warmstart_part}{masked_part}_final"
    )
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
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Training seed (torch/numpy init + env RNG). Default: 42."
    )
    parser.add_argument(
        "--masked", action="store_true",
        help="Train with sb3-contrib's MaskablePPO instead of plain PPO, "
             "excluding task-disallowed content from the action space "
             "entirely rather than letting the agent discover the no-op "
             "fallback via trial and error."
    )
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["ppo", "a2c"],
        help="RL algorithm to train with. Default: ppo. "
             "--masked is only valid with --algo ppo."
    )
    args = parser.parse_args()

    if args.task == "all":
        train_curriculum()
    else:
        cfg = TASK_CONFIGS[args.task].copy()
        if args.timesteps:
            cfg["total_timesteps"] = args.timesteps
        train_task(
            args.task, warmstart_path=args.warmstart,
            seed=args.seed, masked=args.masked, algo=args.algo, **cfg,
        )