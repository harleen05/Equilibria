"""
reward.py — Multi-objective reward function for the Attention Economy Environment.

Reward is a weighted combination of positive signals (engagement, retention, trust)
and negative penalties (fatigue, manipulation).

Output is normalized to [0, 1] via a per-task-weight-profile affine rescale
(see RewardWeights.raw_reward_lower_bound / raw_reward_upper_bound).

Design rationale:
  - Trust carries the highest weight in harder tasks: it's the hardest to rebuild
    and most consequential for long-term user well-being.
  - Manipulation penalty is SUBTRACTIVE and cannot be offset by high engagement —
    this closes the "manipulate but engage" exploitation loophole.
  - Fatigue penalty uses exponent 1.5, which is CONVEX in the calculus sense
    (its marginal/derivative penalty rate increases with fatigue -- crossing
    above what a plain linear penalty's constant rate would charge once
    fatigue exceeds ~0.44). This is a narrower claim than "harsher than
    linear": fatigue^1.5 is actually LESS severe in absolute terms than a
    linear penalty at every fatigue level below 1.0 (e.g. at fatigue=0.7,
    linear=0.700 vs fatigue^1.5=0.586). What it actually encodes is
    front-loaded leniency + back-loaded severity: mild fatigue costs
    little, but the marginal cost of pushing fatigue higher accelerates the
    closer the user gets to exhaustion. See tests/test_reward.py::
    test_fatigue_penalty_is_convex_but_milder_than_linear for the exact
    numeric verification this claim is based on.
  - The addiction discount in R_engagement prevents agents from farming engagement
    via addictive content — inflated engagement from addiction is partially clawed back.

IMPORTANT CAVEAT: the specific numeric constants throughout this module
(the 1.5 exponent, the 0.7/0.3 split in R_retention, the 0.9/0.1 split in
R_trust, the 0.5/0.5 split in P_manipulation, the 0.5 coefficient in the
addiction discount) are engineering priors chosen to encode a DIRECTION
(e.g. "trust should amplify retention reward", "manipulation should hurt
more when trust is high") rather than values fit to any behavioral data.
They have not been empirically calibrated or sensitivity-tested against
alternative choices. Treat any reported result as conditional on this
specific reward specification, not as a general claim about attention-economy
dynamics, until a sensitivity analysis (sweeping each constant and
re-measuring downstream policy behavior) has been run.
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

    @property
    def raw_reward_upper_bound(self) -> float:
        """Approx. theoretical max of raw_reward: positive components ~1, penalties ~0."""
        return self.engagement + self.retention + self.trust

    @property
    def raw_reward_lower_bound(self) -> float:
        """Approx. theoretical min of raw_reward: positive components ~0, penalties maxed."""
        return -(self.fatigue_penalty + self.manipulation_penalty)

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

        # R_engagement: discount engagement inflated by addiction exploitation.
        # Bounded multiplier in [0.5, 1.0] -- addiction can claw back AT MOST
        # 50% of raw engagement reward, never fully zero it out.
        addiction_discount = 1.0 - 0.5 * addiction_risk
        R_eng = engagement * addiction_discount

        # R_retention: satisfaction quality is amplified by trust.
        # Bounded multiplier in [0.7, 1.0] -- even a fully distrustful-but-
        # satisfied user (trust=0) still yields 70% of raw satisfaction
        # reward. Trust can dampen retention reward by at most 30%, not
        # eliminate it -- satisfaction alone is treated as partially valuable
        # regardless of trust level.
        R_ret = satisfaction * (0.7 + 0.3 * trust)

        # R_trust: base trust with a small bonus for diverse feeds.
        # Bounded multiplier in [0.9, 1.0] -- diversity contributes at most a
        # 10% bonus on top of trust; it is a minor secondary signal, not a
        # primary reward driver.
        R_trust = trust * (0.9 + 0.1 * diversity_score)

        # ── Penalty components ────────────────────────────────────────────────

        # P_fatigue: exponent 1.5 is convex in the derivative sense (marginal
        # penalty rate accelerates as fatigue rises, overtaking a plain
        # linear penalty's constant rate once fatigue exceeds ~0.44) but is
        # NOT harsher than linear in absolute terms anywhere below fatigue=1.0
        # (e.g. at fatigue=0.7: linear=0.700, fatigue^1.5=0.586). Net effect:
        # front-loaded leniency, back-loaded severity -- mild fatigue is
        # cheap, pushing fatigue near exhaustion gets disproportionately
        # expensive. See test_fatigue_penalty_is_convex_but_milder_than_linear.
        P_fatigue = fatigue ** 1.5

        # P_manipulation: penalty scales with current trust.
        # Bounded multiplier in [0.5, 1.0] -- manipulating a fully trusting
        # user is penalized at FULL severity, but manipulating an already-
        # distrustful user (trust=0) is penalized at only HALF severity.
        # CAVEAT: this is a deliberate choice ("betraying trust is worse than
        # exploiting an already-skeptical user") but it also creates a
        # potential perverse incentive -- once trust has already collapsed,
        # further manipulation becomes relatively CHEAPER for the agent, not
        # more expensive. In practice this window is narrow because the
        # environment terminates the episode once trust <= 0.05 (see
        # env_core.py's trust_collapse condition), but this interaction
        # between the done-condition and this penalty's trust-weighting has
        # not been separately stress-tested and is worth flagging as a
        # candidate reward-hacking path for the ablation study.
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
        eps = 0.02
        lo = w.raw_reward_lower_bound - eps
        hi = w.raw_reward_upper_bound + eps
        scaled = (raw_reward - lo) / (hi - lo)
        reward = max(0.0001, min(scaled, 0.9999))

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