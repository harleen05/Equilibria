"""
server/main.py — FastAPI wrapper for AttentionEconomyEnv

Exposes the OpenEnv HTTP API:
  POST /reset           — start new episode (optional X-Session-Id header)
  POST /step            — advance one step (manual action)
  POST /step/heuristic  — step using ethical rule-based policy
  POST /step/ppo        — step using trained PPO (if checkpoint exists)
  GET  /observation     — current observation without stepping
  GET  /state        — full internal state (debug)
  GET  /health       — liveness check
  DELETE /session    — drop a session
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from environment.env_core import AttentionEconomyEnv
from environment.heuristic_policy import smart_policy_action
from environment.models import Action
from server.errors import api_error
from server.sessions import DEFAULT_SESSION_ID, sessions

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Attention Economy OpenEnv",
    description="Multi-objective RL environment for ethical content recommendation",
    version="0.2.0",
)

_cors_raw = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
)
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
_allow_all = "*" in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _cors_origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_MAP = {
    "easy_recommendation": "easy",
    "diverse_feed": "medium",
    "trust_preservation": "hard",
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}

# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────


class ResetRequest(BaseModel):
    task: str = Field(default="medium", description="Task id: easy, medium, or hard")
    task_id: Optional[str] = Field(default=None, description="Alias for task")
    seed: Optional[int] = Field(default=None, description="Optional RNG seed")
    observability: str = Field(
        default="oracle",
        description=(
            "'oracle' (default): agent observes true manipulation_score per "
            "item. 'partial': manipulation_score is masked to a neutral "
            "constant, forcing the agent to infer manipulativeness from "
            "engagement/trust/fatigue behavior instead of reading it directly."
        ),
    )
    new_session: bool = Field(
        default=False,
        description="If true, allocate a fresh session id",
    )


class StepRequest(BaseModel):
    action: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    session_id: str


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    session_id: str
    policy: Optional[str] = None
    policy_action: Optional[Dict[str, Any]] = None


# ──────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────


def get_session_id(
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
) -> str:
    return sessions.resolve_id(x_session_id)


def get_env(session_id: str = Depends(get_session_id)) -> AttentionEconomyEnv:
    return sessions.get(session_id)


def normalize_task(raw: str) -> str:
    resolved = TASK_MAP.get(raw)
    if resolved is None:
        raise api_error(
            400,
            "UNKNOWN_TASK",
            f"Unknown task '{raw}'. Choose one of: {sorted(set(TASK_MAP.values()))} "
            f"(aliases accepted: {sorted(TASK_MAP.keys())})",
        )
    return resolved


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "attention-economy-env",
        "version": "0.2.0",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": [
            "/reset",
            "/step",
            "/step/heuristic",
            "/step/ppo",
            "/observation",
            "/state",
            "/health",
            "/session",
        ],
        "policies": ["manual", "heuristic", "ppo"],
        "headers": {"X-Session-Id": "optional client session id"},
        "message": "Attention Economy Env is live",
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(
    req: ResetRequest,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
):
    if req.new_session or not x_session_id:
        session_id = sessions.new_id() if req.new_session else sessions.resolve_id(x_session_id)
    else:
        session_id = sessions.resolve_id(x_session_id)

    env = sessions.get(session_id)
    raw = req.task_id or req.task
    task = normalize_task(raw)

    try:
        obs = env.reset(task, seed=req.seed, observability=req.observability)
        return ResetResponse(observation=obs.model_dump(), session_id=session_id)
    except ValueError as e:
        raise api_error(400, "INVALID_OBSERVABILITY", str(e)) from e
    except Exception as e:
        raise api_error(500, "RESET_FAILED", f"reset() failed: {e}") from e


def _run_step(
    env: AttentionEconomyEnv,
    action: Action,
    session_id: str,
    policy: Optional[str] = None,
    policy_action: Optional[Dict[str, Any]] = None,
) -> StepResponse:
    if env.user is None:
        raise api_error(400, "NOT_RESET", "Call /reset before /step.")
    if env.done:
        raise api_error(400, "EPISODE_DONE", "Episode finished. Call /reset to start a new one.")
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=reward,
            done=done,
            info=info,
            session_id=session_id,
            policy=policy,
            policy_action=policy_action,
        )
    except ValueError as e:
        raise api_error(400, "STEP_REJECTED", str(e)) from e
    except Exception as e:
        raise api_error(
            500,
            "STEP_FAILED",
            f"step() failed: {e}\n{traceback.format_exc()}",
        ) from e


@app.post("/step", response_model=StepResponse)
async def step(
    req: StepRequest,
    session_id: str = Depends(get_session_id),
    env: AttentionEconomyEnv = Depends(get_env),
):
    try:
        action = Action(**req.action)
    except Exception as e:
        raise api_error(422, "INVALID_ACTION", f"Invalid action: {e}") from e
    return _run_step(env, action, session_id, policy="manual")


@app.post("/step/heuristic", response_model=StepResponse)
async def step_heuristic(
    session_id: str = Depends(get_session_id),
    env: AttentionEconomyEnv = Depends(get_env),
):
    if env.user is None:
        raise api_error(400, "NOT_RESET", "Call /reset before /step/heuristic.")
    obs_dict = env._get_observation().model_dump()
    policy_action = smart_policy_action(obs_dict)
    action = Action(
        action_type=policy_action["action_type"],
        content_id=policy_action.get("content_id"),
    )
    return _run_step(env, action, session_id, policy="heuristic", policy_action=policy_action)


@app.post("/step/ppo", response_model=StepResponse)
async def step_ppo(
    session_id: str = Depends(get_session_id),
    env: AttentionEconomyEnv = Depends(get_env),
):
    if env.user is None:
        raise api_error(400, "NOT_RESET", "Call /reset before /step/ppo.")
    from server.ppo_agent import ppo_available, predict_action as ppo_predict_action

    task = env.task_id or "medium"
    if not ppo_available(task):
        raise api_error(
            404,
            "PPO_NOT_FOUND",
            f"No PPO model for task '{task}'. Run: python environment/train_rl.py --task {task}",
        )
    obs_dict = env._get_observation().model_dump()
    try:
        policy_action = ppo_predict_action(task, obs_dict)
    except FileNotFoundError as e:
        raise api_error(404, "PPO_NOT_FOUND", str(e)) from e
    except Exception as e:
        raise api_error(500, "PPO_PREDICT_FAILED", str(e)) from e

    action = Action(
        action_type=policy_action["action_type"],
        content_id=policy_action.get("content_id"),
    )
    return _run_step(env, action, session_id, policy="ppo", policy_action=policy_action)


@app.get("/policies")
def list_policies(task: str = "easy"):
    """Report which automated policies have checkpoints for a task."""
    from server.ppo_agent import model_path_for_task

    task = normalize_task(task)
    path = model_path_for_task(task)
    return {
        "task": task,
        "heuristic": True,
        "ppo": path is not None,
        "ppo_model": str(path) if path else None,
    }


@app.get("/observation")
def observation(env: AttentionEconomyEnv = Depends(get_env)):
    if env.user is None:
        raise api_error(400, "NOT_RESET", "Call /reset first.")
    return {
        "observation": env._get_observation().model_dump(),
        "done": env.done,
    }


@app.get("/state")
def state(env: AttentionEconomyEnv = Depends(get_env)):
    if env.user is None:
        raise api_error(400, "NOT_RESET", "Call /reset first.")
    return env.state()


@app.delete("/session")
def delete_session(session_id: str = Depends(get_session_id)):
    removed = sessions.delete(session_id)
    if not removed and session_id == DEFAULT_SESSION_ID:
        # Recreate a clean default env
        sessions.get(DEFAULT_SESSION_ID).reset("medium")
        return {"deleted": False, "reset": True, "session_id": session_id}
    return {"deleted": removed, "session_id": session_id}


# ──────────────────────────────────────────────
# Test / legacy compatibility
# ──────────────────────────────────────────────

# Re-export for conftest: `from server.main import env`
env = sessions.get(DEFAULT_SESSION_ID)

# ──────────────────────────────────────────────
# Static UI (production / HF Space)
# ──────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file():
    _assets = STATIC_DIR / "assets"
    if _assets.is_dir():
        app.mount("/ui/assets", StaticFiles(directory=_assets), name="ui-assets")

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def ui_index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/ui/{path:path}", include_in_schema=False)
    def ui_spa(path: str):
        candidate = STATIC_DIR / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
