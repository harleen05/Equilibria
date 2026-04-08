"""
server/main.py — FastAPI wrapper for AttentionEconomyEnv
This is what runs on HuggingFace Space. Exposes the OpenEnv HTTP API:
  POST /reset   — start new episode
  POST /step    — advance one step
  GET  /state   — current state (debug)
  GET  /health  — liveness check (required by HF Space HEALTHCHECK)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import traceback

from environment.env_core import AttentionEconomyEnv
from environment.models import Action

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Attention Economy OpenEnv",
    description="Multi-objective RL environment for ethical content recommendation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global env instance (stateful per session)
env = AttentionEconomyEnv()

# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "medium"

class StepRequest(BaseModel):
    action: Dict[str, Any]

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — HF Space HEALTHCHECK pings this."""
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "name": "attention-economy-env",
        "version": "0.1.0",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
        "message": "Attention Economy Env is live 🚀",
    }

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
        task = data.get("task", "medium")   # ✅ flexible

        if task not in ("easy", "medium", "hard"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task: {task}. Choose easy/medium/hard."
            )

        obs = env.reset(task)

        return {
            "observation": obs.model_dump()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"reset() failed: {e}"
        )

@app.post("/step")
def step(req: StepRequest):
    """
    Advance the environment one step.
    Action format:
      {"action_type": "recommend", "content_id": "rel_tech_01"}
      {"action_type": "pause_session"}
      {"action_type": "diversify_feed"}
      {"action_type": "explore_new_topic"}
    """
    if env.user is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    if env.done:
        raise HTTPException(status_code=400, detail="Episode finished. Call /reset to start a new one.")

    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"step() failed: {e}\n{traceback.format_exc()}")

@app.get("/state")
def state():
    """Return full internal state for debugging."""
    if env.user is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return env.state()