"""
tasks/easy.py — Easy task configuration for the Attention Economy Environment.

Profile:
  - Single dominant interest (technology: 70%)
  - Low fatigue sensitivity (0.6×) — user doesn't tire quickly
  - Low trust decay rate (0.7×) — trust is stable under mild manipulation
  - Short episode (15 steps)
  - Reward weights favour engagement slightly

Expected optimal agent strategy:
  Serve technology and science content with high educational value.
  Use diversify_feed every 4–5 steps to prevent repetition fatigue.
  Avoid misleading content entirely — unnecessary given the catalog depth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from environment.models import UserState
from environment.reward import EASY_WEIGHTS, RewardWeights


@dataclass(frozen=True)
class EasyTaskConfig:
    task_id: str = "easy"
    description: str = (
        "Single dominant interest (technology). "
        "Low fatigue sensitivity. "
        "Straightforward interest matching with light ethical constraints."
    )
    max_steps: int = 15
    reward_weights: RewardWeights = field(default_factory=lambda: EASY_WEIGHTS)

    # Content IDs available to the agent in this task
    allowed_content_ids: List[str] = field(default_factory=lambda: [
        "rel_sci_01", "rel_tech_01", "rel_fin_01", "rel_hist_01",
        "rel_health_01", "rel_health_02", "rel_news_01", "rel_env_01",
        "rnd_film_01", "rnd_music_01", "rnd_food_01", "rnd_sport_01",
        "add_gaming_01", "add_social_01",
        "mis_click_01",
    ])

    expected_strategy: str = (
        "Serve rel_tech_01 and rel_sci_01 in rotation. "
        "Diversify every 4-5 steps using diversify_feed. "
        "Avoid mis_click_01 — manipulation penalty not worth short-term engagement. "
        "Limit add_gaming_01 to one use maximum."
    )


def get_task_config() -> EasyTaskConfig:
    return EasyTaskConfig()


def get_initial_user() -> UserState:
    """Return the deterministic starting user state for the easy task."""
    return UserState(
        user_id="user_easy_01",
        interest_distribution={
            "technology":   0.70,
            "science":      0.20,
            "entertainment":0.10,
        },
        fatigue=0.00,
        trust=0.85,
        addiction_risk=0.05,
        satisfaction=0.50,
        boredom=0.00,
        session_length=0,
        fatigue_sensitivity=0.60,   # Low — user doesn't tire easily
        trust_decay_rate=0.70,      # Low — trust is resilient
    )