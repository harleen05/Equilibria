from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from environment.models import (
    Action, ContentItem, Observation, UserState
)
from environment.content import get_full_catalog
from environment.simulation import SimulationEngine
from environment.reward import RewardFunction
from environment.tasks import get_task


# ─────────────────────────────────────────────
# Episode Grader
# ─────────────────────────────────────────────

class EpisodeGrader:
    WEIGHTS = {
        "avg_engagement": 0.40,
        "final_trust": 0.35,
        "final_satisfaction": 0.25,
    }

    @classmethod
    def grade(cls, engagement_history: List[float], final_user: UserState):
        avg_eng = sum(engagement_history) / len(engagement_history) if engagement_history else 0.0

        score = (
            cls.WEIGHTS["avg_engagement"] * avg_eng +
            cls.WEIGHTS["final_trust"] * final_user.trust +
            cls.WEIGHTS["final_satisfaction"] * final_user.satisfaction
        )

        return {
            "final_score": round(min(max(score, 0.0001), 0.9999), 4),
            "avg_engagement": round(avg_eng, 4),
            "final_trust": round(final_user.trust, 4),
            "final_satisfaction": round(final_user.satisfaction, 4),
        }


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class AttentionEconomyEnv:

    def __init__(self):
        self.catalog = get_full_catalog()
        self.engine = SimulationEngine()

        self.user: Optional[UserState] = None
        self.history: List[str] = []
        self.step_count = 0
        self.max_steps = 20

        self.reward_fn: Optional[RewardFunction] = None
        self.allowed_content_ids: List[str] = []

        self.engagement_history: List[float] = []
        self.done = False
        self.task_id = None
        self.consecutive_pauses: int = 0   # tracks pause spamming

    # ─────────────────────────────
    # RESET
    # ─────────────────────────────

    def reset(self, task_id: str = "medium", seed: Optional[int] = None) -> Observation:
        task_cfg, user = get_task(task_id)

        self.user = user
        self.history = []
        self.step_count = 0
        self.done = False
        self.task_id = task_id

        self.max_steps = task_cfg.max_steps
        self.allowed_content_ids = task_cfg.allowed_content_ids
        self.reward_fn = RewardFunction(task_cfg.reward_weights)

        self.engagement_history = []
        self.consecutive_pauses = 0

        # Re-seed the simulation engine for reproducibility
        self.engine = SimulationEngine(seed=seed)

        return self._get_observation()

    # ─────────────────────────────
    # STEP
    # ─────────────────────────────

    def step(self, action) -> Tuple[Observation, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        if isinstance(action, dict):
            action = Action(**action)

    # ── Validate action ──
        if action.action_type not in [
        "recommend", "explore_new_topic", "diversify_feed", "pause_session"
    ]:
            raise ValueError("Invalid action type")

    # ── Resolve content ──
        content = None
        if action.action_type == "recommend":
            if action.content_id not in self.catalog:
                raise ValueError("Invalid content_id")
            if action.content_id not in self.allowed_content_ids:
                raise ValueError("Content not allowed in this task")
    

            content = self.catalog[action.content_id]

    # ── Simulation ──
        updated_user, diagnostics = self.engine.apply_transition(
        user=self.user,
        action=action,
        content=content,
        history=self.history,
        catalog=self.catalog,
    )
    

        engagement = diagnostics["engagement"]

    # ── Update state ──
        self.user = updated_user
        self.step_count += 1

        # Track consecutive pauses for exploit prevention
        if action.action_type == "pause_session":
            self.consecutive_pauses += 1
        else:
            self.consecutive_pauses = 0

        if content:
          self.history.append(content.content_id)

    # ── Reward ──
        reward, breakdown = self.reward_fn.compute(
        engagement=engagement,
        satisfaction=self.user.satisfaction,
        trust=self.user.trust,
        fatigue=self.user.fatigue,
        manipulation_score=content.manipulation_score if content else 0.0,
        addiction_risk=self.user.addiction_risk,
        diversity_score=diagnostics["diversity_score"],
    )

        # Hard penalty for spamming pause — more than 2 consecutive = diminishing returns
        if self.consecutive_pauses > 2:
            reward = max(0.0, reward - 0.15 * (self.consecutive_pauses - 2))

        self.engagement_history.append(engagement)

    # ── Done conditions ──
        max_steps_reached = self.step_count >= self.max_steps
        trust_collapse = self.user.trust <= 0.05
        fatigue_overload = self.user.fatigue >= 0.95

        self.done = max_steps_reached or trust_collapse or fatigue_overload

    # ── Info ──
        info = {
        "step": self.step_count,
        "task": self.task_id,
        "diagnostics": diagnostics,
        "reward_breakdown": breakdown,
        "user_state": {
            "trust": round(self.user.trust, 4),
            "fatigue": round(self.user.fatigue, 4),
            "addiction_risk": round(self.user.addiction_risk, 4),
        },
    }

        if self.done:
          info["episode_grade"] = EpisodeGrader.grade(
            self.engagement_history,
            self.user
        )

        return self._get_observation(), reward, self.done, info
    
    # ─────────────────────────────
    # OBSERVATION
    # ─────────────────────────────

    def _get_observation(self) -> Observation:
        return Observation(
        # Visible user signals
        visible_fatigue=self.user.fatigue,
        visible_trust=self.user.trust,
        visible_satisfaction=self.user.satisfaction,
        visible_boredom=self.user.boredom,

        # Session
        session_length=self.user.session_length,
        step_count=self.step_count,   # ✅ FIX 1

        # Preferences
        interest_distribution=self.user.interest_distribution,

        # Expose only allowed content for this task, as ContentItem objects
        available_content=[
            self.catalog[cid]
            for cid in self.allowed_content_ids
            if cid in self.catalog
        ],

        # History
        recent_content_ids=self.history[-5:],

        # Diversity proxy
        recent_diversity_score=min(len(set(self.history[-5:])), 5) / 5.0,

        # Task
        task_id=self.task_id,
    )

    # ─────────────────────────────
    # STATE (DEBUG)
    # ─────────────────────────────

    def state(self) -> Dict:
        return {
            "user": self.user.model_dump(),
            "step": self.step_count,
            "history": self.history,
            "done": self.done,
        }