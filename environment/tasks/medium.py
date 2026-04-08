"""
tasks/medium.py — Medium task configuration for the Attention Economy Environment.

Profile:
  - Five active interests (science, health, entertainment, politics, technology)
  - Normal fatigue sensitivity (1.0×)
  - Normal trust decay rate (1.0×)
  - Medium episode length (20 steps)
  - Reward weights balanced across engagement, retention, and trust

Expected optimal agent strategy:
  Rotate content across all five interest domains to prevent boredom.
  Use diversify_feed when boredom exceeds ~0.4.
  Strongly avoid misleading content — trust penalty competes with engagement gain
  and tips negative after 2+ exposures.
  Introduce educational content to build trust recovery buffer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from environment.models import UserState
from environment.reward import MEDIUM_WEIGHTS, RewardWeights


@dataclass(frozen=True)
class MediumTaskConfig:
    task_id: str = "medium"
    description: str = (
        "Five active interests. "
        "Normal fatigue and trust sensitivity. "
        "Diversity required to prevent boredom-driven reward decay. "
        "Outrage content is a local maximum trap."
    )
    max_steps: int = 20
    reward_weights: RewardWeights = field(default_factory=lambda: MEDIUM_WEIGHTS)

    allowed_content_ids: List[str] = field(default_factory=lambda: [
        "rel_sci_01", "rel_tech_01", "rel_fin_01", "rel_hist_01",
        "rel_health_01", "rel_health_02", "rel_news_01", "rel_env_01",
        "rnd_film_01", "rnd_music_01", "rnd_food_01", "rnd_sport_01",
        "add_scroll_01", "add_satisfy_01", "add_gaming_01",
        "add_social_01", "add_social_02",
        "mis_outrage_01", "mis_click_01", "mis_click_02",
    ])

    expected_strategy: str = (
        "Rotate: science → health → entertainment → politics → technology each 4 steps. "
        "Use diversify_feed when boredom > 0.4. "
        "Avoid mis_outrage_01 — short spike, severe trust collapse over 3 steps. "
        "Use pause_session once around step 12 to reset fatigue accumulation."
    )


def get_task_config() -> MediumTaskConfig:
    return MediumTaskConfig()


def get_initial_user() -> UserState:
    """Return the deterministic starting user state for the medium task."""
    return UserState(
        user_id="user_medium_01",
        interest_distribution={
            "science":      0.30,
            "health":       0.25,
            "entertainment":0.20,
            "politics":     0.15,
            "technology":   0.10,
        },
        fatigue=0.10,
        trust=0.75,
        addiction_risk=0.15,
        satisfaction=0.50,
        boredom=0.05,
        session_length=0,
        fatigue_sensitivity=1.00,   # Normal
        trust_decay_rate=1.00,      # Normal
    )