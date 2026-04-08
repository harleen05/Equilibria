"""
Attention Economy Environment — Simulation Engine
Fully deterministic state transition and reward computation.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
from models import UserState, ContentItem, Action


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

HISTORY_WINDOW = 5
FATIGUE_CAP = 1.0
TRUST_FLOOR = 0.0
BOREDOM_DECAY = 0.05
FATIGUE_DECAY = 0.03


class SimulationEngine:

    # ── Interest Match ──────────────────────────────────────────────────
    @staticmethod
    def compute_interest_match(content: ContentItem, user: UserState) -> float:
        """
        Weighted dot product between content.topic_relevance and
        user.interest_distribution. Normalized to [0, 1].
        """
        topics = set(content.topic_relevance) & set(user.interest_distribution)
        if not topics:
            return 0.0
        numerator = sum(
            user.interest_distribution[t] * content.topic_relevance[t]
            for t in topics
        )
        denominator = sum(user.interest_distribution.values()) or 1.0
        return min(numerator / denominator, 1.0)

    # ── Repetition Penalty ──────────────────────────────────────────────
    @staticmethod
    def compute_repetition_penalty(
        content_id: str,
        history: List[str],
        window: int = HISTORY_WINDOW
    ) -> float:
        recent = history[-window:] if len(history) >= window else history
        count = recent.count(content_id)
        return min(count / max(window, 1), 1.0)

    # ── Diversity Score ─────────────────────────────────────────────────
    @staticmethod
    def compute_diversity_score(
        history: List[str],
        catalog: Dict[str, ContentItem],
        window: int = HISTORY_WINDOW
    ) -> float:
        """
        Entropy-based diversity over recent content topics (dominant topic per item).
        Returns 0 (no diversity) → 1 (maximum diversity).
        """
        recent_ids = history[-window:]
        if not recent_ids:
            return 1.0

        cats: Dict[str, int] = {}
        for cid in recent_ids:
            if cid in catalog:
                item = catalog[cid]
                # Use dominant topic as category proxy
                dominant = max(item.topic_relevance, key=item.topic_relevance.get)
                cats[dominant] = cats.get(dominant, 0) + 1

        n = len(recent_ids)
        entropy = 0.0
        for count in cats.values():
            p = count / n
            entropy -= p * math.log2(p) if p > 0 else 0.0

        max_entropy = math.log2(max(len(cats), 2))
        return min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0

    # ── Fatigue Transition ──────────────────────────────────────────────
    @staticmethod
    def update_fatigue(
        user: UserState,
        content: Optional[ContentItem],
        action_type: str
    ) -> float:
        f = user.fatigue
        if action_type == "pause_session":
            f -= 0.20
        elif action_type == "diversify_feed":
            f -= 0.10
        elif content is not None:
            delta = 0.08 + 0.15 * content.addictiveness
            f += delta * user.fatigue_sensitivity
        else:
            f -= FATIGUE_DECAY
        return max(0.0, min(f, FATIGUE_CAP))

    # ── Trust Transition ────────────────────────────────────────────────
    @staticmethod
    def update_trust(
        user: UserState,
        content: Optional[ContentItem],
        interest_match: float,
        action_type: str
    ) -> float:
        t = user.trust
        if action_type == "pause_session":
            t += 0.05
        elif action_type == "diversify_feed":
            t += 0.02
        elif action_type == "explore_new_topic":
            t += 0.01
        elif content is not None:
            t -= 0.20 * content.manipulation_score * user.trust_decay_rate
            t -= 0.05 * (1.0 - interest_match)
            t += 0.03 * interest_match * content.educational_value
        return max(TRUST_FLOOR, min(t, 1.0))

    # ── Satisfaction Transition ─────────────────────────────────────────
    @staticmethod
    def update_satisfaction(
        user: UserState,
        content: Optional[ContentItem],
        interest_match: float,
        repetition_penalty: float,
        action_type: str
    ) -> float:
        s = user.satisfaction
        if action_type in ("pause_session", "diversify_feed"):
            s -= 0.01
        elif content is not None:
            engagement_quality = (
                interest_match
                * (1.0 - repetition_penalty)
                * (1.0 - content.manipulation_score)
            )
            alpha = 0.12
            beta = 0.08
            s += alpha * engagement_quality - beta * user.fatigue
            s = max(0.0, min(s, 1.0))
        return s

    # ── Addiction Risk Transition ────────────────────────────────────────
    @staticmethod
    def update_addiction_risk(
        user: UserState,
        content: Optional[ContentItem],
        action_type: str
    ) -> float:
        ar = user.addiction_risk
        if action_type == "pause_session":
            ar -= 0.05
        elif action_type == "diversify_feed":
            ar -= 0.02
        elif content is not None:
            ar += 0.08 * content.addictiveness
            ar -= 0.03 * content.educational_value
        return max(0.0, min(ar, 1.0))

    # ── Boredom Transition ───────────────────────────────────────────────
    @staticmethod
    def update_boredom(
        user: UserState,
        content: Optional[ContentItem],
        repetition_penalty: float,
        diversity_score: float
    ) -> float:
        b = user.boredom
        b -= BOREDOM_DECAY
        if content is not None:
            b += repetition_penalty * 0.2
            b += (1.0 - diversity_score) * 0.05
            b -= content.novelty * 0.05
        return max(0.0, min(b, 1.0))

    # ── Engagement Signal ────────────────────────────────────────────────
    @staticmethod
    def compute_engagement(
        content: ContentItem,
        user: UserState,
        interest_match: float,
        repetition_penalty: float
    ) -> float:
        """
        Engagement derived from interest match, addictiveness, novelty, and fatigue.
        Uses addictiveness as a stand-in for engagement_potential since the model
        does not have a dedicated engagement_potential field.
        """
        # Base engagement potential: addictive content can hook even with low match
        base = 0.4 + 0.6 * content.addictiveness

        addiction_amp = 1.0 + user.addiction_risk * content.addictiveness * 0.5
        novelty_factor = content.novelty * (1.0 - repetition_penalty)
        fatigue_drag = max(0.0, 1.0 - user.fatigue * 0.6)

        # Blend: relevant content uses interest_match; addictive content partially bypasses it
        relevance_weight = max(interest_match, 0.3 * content.addictiveness)

        eng = base * relevance_weight * addiction_amp * (0.5 + 0.5 * novelty_factor) * fatigue_drag
        return min(eng, 1.0)

    # ── Full Step Transition ─────────────────────────────────────────────
    def apply_transition(
        self,
        user: UserState,
        action: Action,
        content: Optional[ContentItem],
        history: List[str],
        catalog: Dict[str, ContentItem]
    ) -> Tuple[UserState, Dict[str, float]]:

        interest_match = (
            self.compute_interest_match(content, user) if content else 0.0
        )
        repetition_penalty = (
            self.compute_repetition_penalty(content.content_id, history)
            if content else 0.0
        )
        diversity_score = self.compute_diversity_score(history, catalog)
        engagement = (
            self.compute_engagement(content, user, interest_match, repetition_penalty)
            if content else 0.0
        )

        new_fatigue = self.update_fatigue(user, content, action.action_type)
        new_trust = self.update_trust(user, content, interest_match, action.action_type)
        new_satisfaction = self.update_satisfaction(
            user, content, interest_match, repetition_penalty, action.action_type
        )
        new_addiction_risk = self.update_addiction_risk(user, content, action.action_type)
        new_boredom = self.update_boredom(user, content, repetition_penalty, diversity_score)

        updated_user = user.model_copy(update={
            "fatigue": new_fatigue,
            "trust": new_trust,
            "satisfaction": new_satisfaction,
            "addiction_risk": new_addiction_risk,
            "boredom": new_boredom,
            "session_length": user.session_length + (1 if content else 0),
        })

        diagnostics = {
            "interest_match": round(interest_match, 4),
            "repetition_penalty": round(repetition_penalty, 4),
            "diversity_score": round(diversity_score, 4),
            "engagement": round(engagement, 4),
            "delta_fatigue": round(new_fatigue - user.fatigue, 4),
            "delta_trust": round(new_trust - user.trust, 4),
            "delta_satisfaction": round(new_satisfaction - user.satisfaction, 4),
            "delta_addiction_risk": round(new_addiction_risk - user.addiction_risk, 4),
        }

        return updated_user, diagnostics