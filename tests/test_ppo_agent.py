"""
tests/test_ppo_agent.py — Verifies server/ppo_agent.py's model resolution,
availability check, and the actual load+predict integration path.

Before this file, ppo_agent.py sat at 44% coverage with predict_action()
(lines 33-56) entirely untested -- the single most complex integration in
the server (loading a real torch/stable-baselines3 model + the
rl_wrapper's obs-encode/action-decode roundtrip) had zero test confidence.

These tests train a real (tiny) PPO checkpoint and place it at the exact
path server.ppo_agent's resolver expects, rather than mocking PPO.load --
mocking it would only prove the mock works, not that the real integration
does.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import server.ppo_agent as ppo_agent
from environment.train_rl import train_task
from environment.env_core import AttentionEconomyEnv

TINY_TIMESTEPS = 256
TINY_N_ENVS = 2


@pytest.fixture
def fake_model_root(tmp_path, monkeypatch):
    """
    Points server.ppo_agent's model resolution at an isolated tmp_path
    instead of the real models/ directory, and clears its module-level
    cache before and after so tests don't leak state into each other or
    into other test files that import the same module-level _cache dict.
    """
    monkeypatch.setattr(ppo_agent, "_ROOT", tmp_path)
    monkeypatch.setattr(
        ppo_agent, "_MODEL_CANDIDATES",
        (
            lambda task: tmp_path / "models" / "best" / task / "best_model.zip",
            lambda task: tmp_path / "models" / f"ppo_{task}_final.zip",
        ),
    )
    ppo_agent._cache.clear()
    yield tmp_path
    ppo_agent._cache.clear()


def test_model_path_for_task_returns_none_when_nothing_exists(fake_model_root) -> None:
    assert ppo_agent.model_path_for_task("easy") is None


def test_model_path_for_task_finds_fallback_checkpoint(fake_model_root) -> None:
    models_dir = fake_model_root / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "ppo_easy_final.zip").write_bytes(b"fake")

    resolved = ppo_agent.model_path_for_task("easy")
    assert resolved is not None
    assert resolved.name == "ppo_easy_final.zip"


def test_model_path_for_task_prefers_best_model_over_fallback(fake_model_root) -> None:
    """When both a best-checkpoint and a final-checkpoint exist, the
    best/{task}/best_model.zip candidate must win -- it's listed first in
    _MODEL_CANDIDATES specifically because EvalCallback's "best" checkpoint
    should be preferred over the unconditional final-timestep save."""
    best_dir = fake_model_root / "models" / "best" / "easy"
    best_dir.mkdir(parents=True)
    (best_dir / "best_model.zip").write_bytes(b"fake")
    (fake_model_root / "models" / "ppo_easy_final.zip").write_bytes(b"fake")

    resolved = ppo_agent.model_path_for_task("easy")
    assert resolved is not None
    assert "best" in str(resolved)


def test_ppo_available_true_and_false(fake_model_root) -> None:
    assert ppo_agent.ppo_available("easy") is False

    models_dir = fake_model_root / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "ppo_easy_final.zip").write_bytes(b"fake")

    assert ppo_agent.ppo_available("easy") is True
    assert ppo_agent.ppo_available("medium") is False


def test_predict_action_raises_with_helpful_message_when_no_checkpoint(fake_model_root) -> None:
    env = AttentionEconomyEnv()
    obs = env.reset("easy", seed=1)

    with pytest.raises(FileNotFoundError, match="No PPO checkpoint for task 'easy'"):
        ppo_agent.predict_action("easy", obs.model_dump())


def test_predict_action_full_integration_with_real_checkpoint(fake_model_root) -> None:
    """
    The core integration test: train a real (tiny) PPO checkpoint, place
    it where the resolver expects, and confirm predict_action() actually
    loads it and produces a valid, well-formed action -- not a mock, the
    real load -> _encode_obs -> predict -> _decode_action roundtrip.
    """
    checkpoint = train_task("easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=88)

    models_dir = fake_model_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint + ".zip", models_dir / "ppo_easy_final.zip")

    env = AttentionEconomyEnv()
    obs = env.reset("easy", seed=1)

    result = ppo_agent.predict_action("easy", obs.model_dump())

    assert "action_type" in result
    assert result["action_type"] in (
        "recommend", "explore_new_topic", "diversify_feed", "pause_session"
    )
    assert result["reasoning"] == "ppo policy"
    if result["action_type"] == "recommend":
        assert "content_id" in result
    else:
        assert "content_id" not in result


def test_predict_action_caches_loaded_model(fake_model_root) -> None:
    """The model should be loaded from disk once and reused across calls
    for the same task, not reloaded on every prediction."""
    checkpoint = train_task("easy", total_timesteps=TINY_TIMESTEPS, n_envs=TINY_N_ENVS, seed=89)

    models_dir = fake_model_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint + ".zip", models_dir / "ppo_easy_final.zip")

    env = AttentionEconomyEnv()
    obs = env.reset("easy", seed=1)
    obs_dict = obs.model_dump()

    assert len(ppo_agent._cache) == 0
    ppo_agent.predict_action("easy", obs_dict)
    assert len(ppo_agent._cache) == 1

    ppo_agent.predict_action("easy", obs_dict)
    assert len(ppo_agent._cache) == 1  # unchanged -- reused, not reloaded