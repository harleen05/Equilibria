"""
reward.py — Multi-objective reward function for the Attention Economy Environment.

Reward is a weighted combination of positive signals (engagement, retention, trust)
and negative penalties (fatigue, manipulation).

Output is normalized to [0, 1].

Design rationale:
  - Trust carries the highest weight in harder tasks: it's the hardest to rebuild
    and most consequential for long-term user well-being.
  - Manipulation penalty is SUBTRACTIVE and cannot be offset by high engagement —
    this closes the "manipulate but engage" exploitation loophole.
  - Fatigue penalty is convex (fatigue^1.5) to strongly discourage pushing users
    past moderate exhaustion levels.
  - The addiction discount in R_engagement prevents agents from farming engagement
    via addictive content — inflated engagement from addiction is partially clawed back.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Reward Weight Profiles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RewardWeights:
    """
    Defines the multi-objective trade-off surface for a given task difficulty.
    All five weights must sum exactly to 1.0.
    """
    engagement:          float  # Weight on R_engagement component
    retention:           float  # Weight on R_retention component
    trust:               float  # Weight on R_trust component
    fatigue_penalty:     float  # Weight on P_fatigue deduction
    manipulation_penalty: float  # Weight on P_manipulation deduction

    def __post_init__(self) -> None:
        total = (
            self.engagement + self.retention + self.trust
            + self.fatigue_penalty + self.manipulation_penalty
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"RewardWeights must sum to 1.0, got {total:.6f}")


# Three pre-defined profiles — task files import these directly
EASY_WEIGHTS = RewardWeights(
    engagement=0.35,
    retention=0.25,
    trust=0.25,
    fatigue_penalty=0.08,
    manipulation_penalty=0.07,
)

MEDIUM_WEIGHTS = RewardWeights(
    engagement=0.25,
    retention=0.25,
    trust=0.30,
    fatigue_penalty=0.10,
    manipulation_penalty=0.10,
)

HARD_WEIGHTS = RewardWeights(
    engagement=0.15,
    retention=0.25,
    trust=0.35,
    fatigue_penalty=0.12,
    manipulation_penalty=0.13,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Function
# ─────────────────────────────────────────────────────────────────────────────

class RewardFunction:
    """
    Computes the per-step reward as a normalized multi-objective score.

    Formula:
        R = w_eng × R_eng
          + w_ret × R_ret
          + w_trust × R_trust
          − w_fat × P_fatigue
          − w_manip × P_manipulation

    Positive components:
        R_eng   = engagement × (1 − 0.5 × addiction_risk)   [addiction-discounted]
        R_ret   = satisfaction × (0.7 + 0.3 × trust)        [trust-amplified retention]
        R_trust = trust × (0.9 + 0.1 × diversity_score)     [diversity bonus]

    Penalty components:
        P_fatigue     = fatigue^1.5                           [convex — steep above 0.7]
        P_manipulation= manip_score × (0.5 + 0.5 × trust)   [trust-weighted severity]

    Final reward is clipped to [0, 1].
    """

    def __init__(self, weights: RewardWeights = MEDIUM_WEIGHTS) -> None:
        self.weights = weights

    def compute(
        self,
        engagement: float,
        satisfaction: float,
        trust: float,
        fatigue: float,
        manipulation_score: float,
        addiction_risk: float,
        diversity_score: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward and return a full component breakdown.

        Parameters
        ----------
        engagement        : Instantaneous engagement from SimulationEngine
        satisfaction      : Current user satisfaction (retention proxy)
        trust             : Current trust level
        fatigue           : Current fatigue level
        manipulation_score: Content manipulation score (0.0 for non-content actions)
        addiction_risk    : Current addiction risk (hidden from agent)
        diversity_score   : Recent feed diversity score

        Returns
        -------
        (reward, component_breakdown_dict)
        """
        w = self.weights

        # ── Positive components ───────────────────────────────────────────────

        # R_engagement: discount engagement inflated by addiction exploitation
        addiction_discount = 1.0 - 0.5 * addiction_risk
        R_eng = engagement * addiction_discount

        # R_retention: satisfaction quality is amplified by trust
        # A satisfied-but-distrustful user will churn — trust is the multiplier
        R_ret = satisfaction * (0.7 + 0.3 * trust)

        # R_trust: base trust with a small bonus for diverse feeds
        # Diverse feeds signal system integrity to the user
        R_trust = trust * (0.9 + 0.1 * diversity_score)

        # ── Penalty components ────────────────────────────────────────────────

        # P_fatigue: convex penalty (accelerates sharply above ~0.65)
        # Prevents engagement-maximizing loops that push users to exhaustion
        P_fatigue = fatigue ** 1.5

        # P_manipulation: penalty scales with current trust
        # The more the user trusts the system, the more damaging a manipulation is
        P_manipulation = manipulation_score * (0.5 + 0.5 * trust)

        # ── Weighted sum ──────────────────────────────────────────────────────
        raw_reward = (
            w.engagement          * R_eng
            + w.retention         * R_ret
            + w.trust             * R_trust
            - w.fatigue_penalty   * P_fatigue
            - w.manipulation_penalty * P_manipulation
        )

        # Clip to [0, 1] — penalties can push below 0 for severely harmful actions
        reward = max(0.0001, min(raw_reward, 0.9999))

        breakdown: Dict[str, float] = {
            "R_engagement":      round(R_eng,          4),
            "R_retention":       round(R_ret,          4),
            "R_trust":           round(R_trust,        4),
            "P_fatigue":         round(P_fatigue,      4),
            "P_manipulation":    round(P_manipulation, 4),
            "addiction_discount":round(addiction_discount, 4),
            "raw_reward":        round(raw_reward,     4),
            "reward":            round(reward,         4),
        }

        return round(reward, 4), breakdown