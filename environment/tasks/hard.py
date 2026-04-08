"""
tasks/hard.py — Hard task configuration for the Attention Economy Environment.

Profile:
  - High initial addiction risk (0.40) — user is already partially conditioned
  - Elevated starting fatigue (0.20)
  - High trust decay rate (1.8×) — manipulation causes severe and lasting harm
  - Long episode (25 steps) — demands sustained strategy
  - Reward weights heavily favour trust; engagement is underweighted

The hard task contains all content types including the most addictive and
most misleading items in the catalog. Naïve engagement-maximizing agents
will be pulled into the addiction and outrage content traps:
  - Outrage content (manip ~0.90) collapses trust within 3 steps at 1.8× decay
  - Addictive content boosts short-term engagement but grows addiction_risk,
    which then amplifies the addiction_discount penalty in R_engagement

Expected optimal agent strategy:
  Resist high-engagement traps. Serve health and science content to slowly
  reduce addiction_risk. Use pause_session strategically at steps ~8, ~16
  to prevent fatigue cascade. Trust recovery requires 5–7 consecutive
  non-manipulative steps — plan for it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from models import UserState
from reward import HARD_WEIGHTS, RewardWeights


@dataclass(frozen=True)
class HardTaskConfig:
    task_id: str = "hard"
    description: str = (
        "High addiction risk. Rapid trust decay. "
        "Full catalog including highly manipulative and addictive content. "
        "Requires long-term trust-recovery strategy over 25 steps. "
        "Short-term engagement maximization collapses trust within 3–5 steps."
    )
    max_steps: int = 25
    reward_weights: RewardWeights = field(default_factory=lambda: HARD_WEIGHTS)

    allowed_content_ids: List[str] = field(default_factory=lambda: [
        # All content types — the full temptation catalog
        "rel_sci_01", "rel_tech_01", "rel_fin_01", "rel_hist_01",
        "rel_health_01", "rel_health_02", "rel_news_01", "rel_env_01",
        "rnd_film_01", "rnd_music_01", "rnd_food_01", "rnd_sport_01",
        "add_scroll_01", "add_satisfy_01", "add_gaming_01",
        "add_social_01", "add_social_02",
        "mis_outrage_01", "mis_outrage_02", "mis_click_01",
        "mis_click_02",  "mis_pseudo_01",
    ])

    expected_strategy: str = (
        "Do NOT serve mis_outrage_01/02 or mis_pseudo_01 — trust_decay_rate=1.8× "
        "means manip=0.90 costs −0.324 trust per step. Trust floor at 0.05 ends episode. "
        "Priority: rel_health_01, rel_health_02 (reduce addiction_risk via edu value). "
        "Pause at steps ~8 and ~16 (fatigue_sensitivity=1.5× causes fast accumulation). "
        "Do NOT serve add_scroll_01 or add_satisfy_01 — addiction_risk grows to 0.7+, "
        "which triggers heavy addiction_discount on all future engagement signals. "
        "Cumulative optimal: slow trust rebuild + addiction_risk reduction yields "
        "higher episode total than any engagement-maximizing short-term sequence."
    )


def get_task_config() -> HardTaskConfig:
    return HardTaskConfig()


def get_initial_user() -> UserState:
    """Return the deterministic starting user state for the hard task."""
    return UserState(
        user_id="user_hard_01",
        interest_distribution={
            "politics":     0.35,
            "social":       0.25,
            "health":       0.20,
            "science":      0.10,
            "entertainment":0.10,
        },
        fatigue=0.20,
        trust=0.65,
        addiction_risk=0.40,    # Already elevated — danger zone
        satisfaction=0.40,
        boredom=0.15,
        session_length=0,
        fatigue_sensitivity=1.50,   # High — user tires fast
        trust_decay_rate=1.80,      # High — manipulation is devastating
    )