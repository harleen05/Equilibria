"""
state_manager.py — Deterministic state update logic for the Attention Economy Environment.

StateManager owns the mechanics of mutating UserState given simulation outputs.
It is the single source of truth for all state transitions — no other module
should directly mutate UserState fields.

Responsibilities:
  - Apply fatigue, trust, satisfaction, addiction_risk, boredom deltas
  - Clamp all values to [0, 1]
  - Append content_id to history
  - Increment step counter
  - Compute and return post-transition diagnostics
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from environment.models import UserState, ContentItem
from environment.utils import clip


# ─────────────────────────────────────────────────────────────────────────────
# StateManager
# ─────────────────────────────────────────────────────────────────────────────

class StateManager:
    """
    Manages all mutable episode state:
      - UserState fields (fatigue, trust, satisfaction, addiction_risk, boredom)
      - Recommendation history (list of content_ids)
      - Step counter

    All updates are deterministic: same (state, deltas) → same output.
    """

    def __init__(self) -> None:
        self._user: Optional[UserState] = None
        self._history: List[str] = []
        self._step_count: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self, user: UserState) -> None:
        """Set initial user state and clear all episode history."""
        # Deep-copy via model serialization to ensure isolation
        self._user = user.model_copy(deep=True)
        self._history = []
        self._step_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # State Update
    # ─────────────────────────────────────────────────────────────────────────

    def apply_step(
        self,
        content: Optional[ContentItem],
        fatigue_delta: float,
        trust_delta: float,
        satisfaction_delta: float,
        addiction_risk_delta: float,
        boredom_delta: float,
    ) -> Dict[str, float]:
        """
        Apply all deltas to the current UserState and return a snapshot
        of the before/after changes for the info payload.

        Parameters
        ----------
        content               : Content item served (None for management actions)
        fatigue_delta         : Change to apply to user.fatigue
        trust_delta           : Change to apply to user.trust
        satisfaction_delta    : Change to apply to user.satisfaction
        addiction_risk_delta  : Change to apply to user.addiction_risk
        boredom_delta         : Change to apply to user.boredom

        Returns
        -------
        Dict of pre→post values for each updated field.
        """
        if self._user is None:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")

        u = self._user

        # ── Record pre-transition values ──────────────────────────────────────
        prev = {
            "fatigue":       u.fatigue,
            "trust":         u.trust,
            "satisfaction":  u.satisfaction,
            "addiction_risk":u.addiction_risk,
            "boredom":       u.boredom,
        }

        # ── Apply deltas with clamping ────────────────────────────────────────
        new_fatigue        = clip(u.fatigue        + fatigue_delta)
        new_trust          = clip(u.trust          + trust_delta)
        new_satisfaction   = clip(u.satisfaction   + satisfaction_delta)
        new_addiction_risk = clip(u.addiction_risk + addiction_risk_delta)
        new_boredom        = clip(u.boredom        + boredom_delta)
        new_session_length = u.session_length + (1 if content is not None else 0)

        # ── Write back (via model_copy for Pydantic immutability pattern) ──────
        self._user = u.model_copy(update={
            "fatigue":        new_fatigue,
            "trust":          new_trust,
            "satisfaction":   new_satisfaction,
            "addiction_risk": new_addiction_risk,
            "boredom":        new_boredom,
            "session_length": new_session_length,
        })

        # ── Update history ────────────────────────────────────────────────────
        if content is not None:
            self._history.append(content.content_id)

        # ── Increment step ────────────────────────────────────────────────────
        self._step_count += 1

        # ── Build change summary ──────────────────────────────────────────────
        changes = {
            f"delta_{k}": round(getattr(self._user, k) - prev[k], 4)
            for k in prev
        }
        changes["step_count"] = self._step_count
        return changes

    # ─────────────────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def user(self) -> UserState:
        if self._user is None:
            raise RuntimeError("StateManager not initialized.")
        return self._user

    @property
    def history(self) -> List[str]:
        """Read-only view of the content_id history."""
        return list(self._history)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def recent_history(self) -> List[str]:
        """Last 5 content_ids served (for repetition/diversity checks)."""
        return self._history[-5:]

    # ─────────────────────────────────────────────────────────────────────────
    # History-based signals
    # ─────────────────────────────────────────────────────────────────────────

    def consecutive_same_type_count(
        self,
        current_content_type: str,
        catalog: Dict,
    ) -> int:
        """
        Count how many consecutive steps at the END of history served
        content of the same type as current_content_type.

        Used by the environment to detect and warn about homogeneous feed runs.
        """
        count = 0
        for cid in reversed(self._history):
            if cid in catalog and catalog[cid].content_type == current_content_type:
                count += 1
            else:
                break
        return count

    def has_seen_recently(self, content_id: str, window: int = 5) -> bool:
        """Return True if content_id appears in the last `window` steps."""
        return content_id in self._history[-window:]