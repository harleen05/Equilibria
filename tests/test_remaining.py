import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import environment.demo as demo
import environment.eval_rl as eval_rl
import inference
import environment.plot_results as plot_results
import environment.train_rl as train_rl
from environment.env_core import AttentionEconomyEnv
from environment.rl_wrapper import AttentionEnvWrapper
from environment.models import Action


def test_train_make_env_returns_callable():
    env_init = train_rl.make_env("easy")
    env = env_init()
    assert isinstance(env.action_space.n, (int, np.integer))
    env.close()


def test_train_task_uses_patched_learn_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr(train_rl, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(train_rl, "BEST_DIR", str(tmp_path / "models" / "best"))
    monkeypatch.setattr(train_rl, "LOG_DIR", str(tmp_path / "logs"))

    monkeypatch.setattr(train_rl.PPO, "learn", lambda self, total_timesteps, callback=None: None)
    monkeypatch.setattr(train_rl.PPO, "save", lambda self, path: None)

    path = train_rl.train_task("easy", total_timesteps=1, n_envs=1)

    assert path.endswith("ppo_easy_final")
    assert (tmp_path / "models").exists()


def test_eval_ppo_loads_saved_model(tmp_path):
    model = PPO("MlpPolicy", DummyVecEnv([lambda: AttentionEnvWrapper(task_id="easy")]), verbose=0)
    model_path = str(tmp_path / "ppo_easy_test")
    model.save(model_path)

    result = eval_rl.evaluate_ppo("easy", model_path=model_path, n_eval=1, verbose=False)

    assert isinstance(result, dict)
    assert "final_score_mean" in result
    assert result["final_score_mean"] == result.get("final_score")


def test_evaluate_heuristic_and_random_return_dicts():
    h = eval_rl.evaluate_heuristic("easy", n_eval=1, verbose=False)
    r = eval_rl.evaluate_random("easy", n_eval=1)

    assert isinstance(h, dict)
    assert isinstance(r, dict)
    assert "final_score_mean" in h
    assert "final_score_mean" in r


def test_plot_trust_trajectory_creates_png(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(plot_results, "_collect_random_trajectory", lambda task_id, seed: {"trust": [0.1, 0.2], "satisfaction": [0.1, 0.2], "reward": [0.2, 0.3], "grade": {}})
    monkeypatch.setattr(plot_results, "_collect_heuristic_trajectory", lambda task_id, seed: {"trust": [0.2, 0.3], "satisfaction": [0.2, 0.3], "reward": [0.3, 0.4], "grade": {}})
    monkeypatch.setattr(plot_results, "_collect_ppo_trajectory", lambda task_id, seed: {"trust": [0.3, 0.4], "satisfaction": [0.3, 0.4], "reward": [0.4, 0.5], "grade": {}})

    (tmp_path / "results").mkdir()
    plot_results.plot_trust_trajectory("easy", n_seeds=2)
    assert (tmp_path / "results" / "trust_trajectory_easy.png").exists()


def test_plot_score_comparison_creates_png(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_collector(task_id, seed):
        return {"trust": [0.1], "satisfaction": [0.1], "reward": [0.1], "grade": {"final_score": 0.2}}

    monkeypatch.setattr(plot_results, "_collect_random_trajectory", fake_collector)
    monkeypatch.setattr(plot_results, "_collect_heuristic_trajectory", fake_collector)
    monkeypatch.setattr(plot_results, "_collect_ppo_trajectory", fake_collector)

    (tmp_path / "results").mkdir()
    plot_results.plot_score_comparison(n_seeds=2)
    assert (tmp_path / "results" / "score_comparison.png").exists()


def test_demo_heuristic_agent_and_run_episode(capsys):
    env = AttentionEconomyEnv()
    obs = env.reset("easy", seed=42)
    action = demo.heuristic_agent(obs)
    assert isinstance(action, Action)

    demo.run_episode("easy")
    captured = capsys.readouterr()
    assert "TASK: EASY" in captured.out
    assert "Final Score" in captured.out


def test_inference_helpers_parsing_and_fallback():
    obs = {
        "visible_fatigue": 0.2,
        "visible_trust": 0.8,
        "visible_satisfaction": 0.5,
        "visible_boredom": 0.1,
        "interest_distribution": {"technology": 1.0},
        "available_content": [{"content_id": "rel_tech_01", "topic_relevance": {"technology": 1.0}, "manipulation_score": 0.05, "addictiveness": 0.1, "educational_value": 0.9}],
        "recent_content_ids": [],
        "task_id": "easy",
    }

    raw = '{"action_type":"recommend","content_id":"rel_tech_01","reasoning":"use safe tech content"}'
    parsed = inference._parse_llm_response(raw, obs)
    assert parsed["action_type"] == "recommend"
    assert parsed["content_id"] == "rel_tech_01"

    raw_bad = "I recommend rel_tech_01 because it is safe"
    parsed2 = inference._parse_llm_response(raw_bad, obs)
    assert parsed2["action_type"] == "recommend"
    assert parsed2["content_id"] == "rel_tech_01"

    assert inference._action_str({"action_type": "pause_session"}) == "pause_session"
    assert inference._action_str({"action_type": "recommend", "content_id": "rel_tech_01"}) == "recommend(content_id=rel_tech_01)"

    message = inference._build_user_message(obs, 1, "easy", 0.0)
    assert "Step 1" in message
    assert "USER STATE" in message

    fallback = inference._smart_policy(obs)
    assert fallback["action_type"] in {"recommend", "diversify_feed", "explore_new_topic", "pause_session"}

    fake = inference._fake_reset("easy")
    assert fake["observation"]["task_id"] == "easy"


def test_inference_call_reset_and_step_request(monkeypatch):
    class DummyResponse:
        def __init__(self, status_code=200, json_data=None, text="OK"):
            self.status_code = status_code
            self._json_data = json_data or {}
            self.text = text
            self.ok = status_code == 200

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if not self.ok:
                raise Exception("http error")

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/reset"):
            return DummyResponse(json_data={"observation": {"visible_fatigue": 0.1}})
        return DummyResponse(json_data={"observation": {"visible_fatigue": 0.2}, "reward": 0.5, "done": False, "info": {}})

    monkeypatch.setattr(inference.requests, "post", fake_post)
    reset_data = inference.call_reset("easy")
    step_data = inference.call_step({"action_type": "pause_session"})

    assert reset_data["observation"]["visible_fatigue"] == 0.1
    assert step_data["reward"] == 0.5


def test_train_episode_summary_callback_records_grade():
    cb = train_rl.EpisodeSummaryCallback(task_id="easy", log_freq=1)
    cb.locals = {"infos": [{"episode_grade": {"final_score": 0.42}}]}
    cb.n_calls = 1

    result = cb._on_step()

    assert result is True
    assert cb._episode_grades == [0.42]


def test_train_task_warmstart_path(tmp_path, monkeypatch):
    warm_path = tmp_path / "warm_model"
    warm_model = PPO(
        "MlpPolicy",
        DummyVecEnv([lambda: AttentionEnvWrapper(task_id="easy")]),
        verbose=0,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    )
    warm_model.save(str(warm_path))

    monkeypatch.setattr(train_rl, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(train_rl, "BEST_DIR", str(tmp_path / "models" / "best"))
    monkeypatch.setattr(train_rl, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(train_rl.PPO, "learn", lambda self, total_timesteps, callback=None: None)
    monkeypatch.setattr(train_rl.PPO, "save", lambda self, path: None)

    path = train_rl.train_task("easy", total_timesteps=1, n_envs=1, warmstart_path=str(warm_path))

    # Previously this asserted path.endswith("ppo_easy_final") -- i.e. warm-started
    # and from-scratch training at the same task/seed produced the SAME save path.
    # That was a real bug: it meant training "hard" both with and without warm-start
    # (e.g. to compare them, as in curriculum_ablation.py) silently overwrote one
    # checkpoint with the other. Fixed by adding a _warmstart suffix whenever
    # warmstart_path is provided; this test now locks in the corrected behavior.
    assert path.endswith("ppo_easy_warmstart_final")


def test_eval_ppo_verbose_prints(tmp_path, capsys):
    model = PPO("MlpPolicy", DummyVecEnv([lambda: AttentionEnvWrapper(task_id="easy")]), verbose=0)
    model_path = str(tmp_path / "ppo_verbose_test")
    model.save(model_path)

    result = eval_rl.evaluate_ppo("easy", model_path=model_path, n_eval=1, verbose=True)
    captured = capsys.readouterr()

    assert "PPO AGENT" in captured.out
    assert isinstance(result, dict)
    assert "final_score" in result or "final_score_mean" in result


def test_eval_compare_prints(monkeypatch, capsys):
    monkeypatch.setattr(eval_rl, "evaluate_random", lambda task_id, n_eval: {"final_score_mean": 0.1, "final_score_std": 0.0, "avg_engagement_mean": 0.1, "final_trust_mean": 0.1, "final_satisfaction_mean": 0.1})
    monkeypatch.setattr(eval_rl, "evaluate_heuristic", lambda task_id, n_eval, verbose=True: {"final_score_mean": 0.2, "final_score_std": 0.0, "avg_engagement_mean": 0.2, "final_trust_mean": 0.2, "final_satisfaction_mean": 0.2})
    monkeypatch.setattr(eval_rl, "evaluate_ppo", lambda task_id, model_path=None, n_eval=1, verbose=False: {"final_score_mean": 0.3, "final_score_std": 0.0, "avg_engagement_mean": 0.3, "final_trust_mean": 0.3, "final_satisfaction_mean": 0.3})

    eval_rl.compare("easy", model_path=None, n_eval=1)
    captured = capsys.readouterr()

    assert "COMPARISON" in captured.out
    assert "PPO" in captured.out


def test_plot_trajectory_collectors_and_ppo(tmp_path, monkeypatch):
    model = PPO("MlpPolicy", DummyVecEnv([lambda: AttentionEnvWrapper(task_id="easy")]), verbose=0)
    model_path = tmp_path / "ppo_plot_test"
    model.save(str(model_path))

    monkeypatch.setattr(plot_results, "DEFAULT_MODEL_PATHS", {"easy": str(model_path)})
    monkeypatch.setattr(plot_results, "FALLBACK_MODEL_PATHS", {"easy": str(model_path)})

    random_traj = plot_results._collect_random_trajectory("easy", seed=0)
    heuristic_traj = plot_results._collect_heuristic_trajectory("easy", seed=0)
    ppo_traj = plot_results._collect_ppo_trajectory("easy", seed=0)

    assert random_traj["trust"]
    assert heuristic_traj["trust"]
    assert ppo_traj["trust"]


def test_inference_run_episode_dry_run():
    result = inference.run_episode("easy", max_steps_override=3, dry_run=True)

    assert isinstance(result, dict)
    assert result["steps"] == 3
    assert "score" in result


def test_dockerfile_contains_healthcheck():
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
    text = dockerfile.read_text()
    assert "HEALTHCHECK" in text
    assert "uvicorn" in text
    assert "server.main:app" in text
