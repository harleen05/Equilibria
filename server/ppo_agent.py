"""Lazy-loaded PPO inference for /step/ppo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from environment.models import Observation
from environment.rl_wrapper import AttentionEnvWrapper

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_CANDIDATES = (
    lambda task: _ROOT / "models" / "best" / task / "best_model.zip",
    lambda task: _ROOT / "models" / f"ppo_{task}_final.zip",
)

_cache: Dict[str, Any] = {}


def model_path_for_task(task_id: str) -> Optional[Path]:
    for resolver in _MODEL_CANDIDATES:
        path = resolver(task_id)
        if path.exists():
            return path
    return None


def ppo_available(task_id: str) -> bool:
    return model_path_for_task(task_id) is not None


def predict_action(task_id: str, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    from stable_baselines3 import PPO

    path = model_path_for_task(task_id)
    if path is None:
        raise FileNotFoundError(
            f"No PPO checkpoint for task '{task_id}'. "
            f"Train with: python environment/train_rl.py --task {task_id}"
        )

    key = str(path)
    if key not in _cache:
        wrapper = AttentionEnvWrapper(task_id=task_id)
        _cache[key] = (PPO.load(str(path), env=wrapper), wrapper)

    model, wrapper = _cache[key]
    obs = Observation(**obs_dict)
    vec = wrapper._encode_obs(obs)
    action_idx, _ = model.predict(vec, deterministic=True)
    action = wrapper._decode_action(int(action_idx))

    result: Dict[str, Any] = {"action_type": action.action_type, "reasoning": "ppo policy"}
    if action.content_id:
        result["content_id"] = action.content_id
    return result