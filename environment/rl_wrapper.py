"""
rl_wrapper.py — Gymnasium-compatible wrapper for AttentionEconomyEnv.

Design decisions:
  - Action space: Discrete(N+3) where N = len(allowed_content_ids for task).
    The last 3 indices map to: diversify_feed, explore_new_topic, pause_session.
    This keeps actions bounded and enumerable for PPO.

  - Observation space: Box(float32) of fixed size 4 + 9 + N*4 + 5 + 1 + 1:
      [fatigue, trust, satisfaction, boredom]          4  user scalars
      [interest_dist over ALL_TOPICS]                  9  canonical topic order
      [add, manip, edu, novelty per allowed item]      N*4 content features
      [recent_seen flags per allowed item]             N  binary recency
      [session_length_norm, step_norm, diversity]      3  session signals
    Total = 4 + 9 + N*5 + 3 fixed scalars

  Why this encoding?
    PPO needs a *fixed-size* vector — the content catalog size per task is
    constant after reset(), so we size the space at init using the UNION of
    all tasks (hard task has the most content: 22 items). Tasks with fewer
    items zero-pad the unused content slots.

  - task_id: configurable at wrapper construction; defaults to "medium".
    For curriculum learning, instantiate multiple wrappers with different task_ids.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from environment.env_core import AttentionEconomyEnv
from environment.models import Action, Observation


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Canonical topic order — fixed across all tasks so the interest vector is stable
ALL_TOPICS: List[str] = [
    "technology", "science", "health", "politics", "entertainment",
    "social", "finance", "sports", "general",
]

# Full content catalog IDs (hard task = superset). Used to size obs space.
# Order is fixed so the index → content mapping is deterministic across episodes.
ALL_CONTENT_IDS: List[str] = [
    "rel_sci_01", "rel_tech_01", "rel_fin_01", "rel_hist_01",
    "rel_health_01", "rel_health_02", "rel_news_01", "rel_env_01",
    "rnd_film_01", "rnd_music_01", "rnd_food_01", "rnd_sport_01",
    "add_scroll_01", "add_satisfy_01", "add_gaming_01",
    "add_social_01", "add_social_02",
    "mis_outrage_01", "mis_outrage_02", "mis_click_01",
    "mis_click_02", "mis_pseudo_01",
]

# Non-content actions appended at the end of the discrete action space
META_ACTIONS: List[str] = ["diversify_feed", "explore_new_topic", "pause_session"]

# Observation vector layout (sizes)
N_USER_SCALARS   = 4          # fatigue, trust, satisfaction, boredom
N_TOPICS         = len(ALL_TOPICS)   # 9
N_CONTENT        = len(ALL_CONTENT_IDS)  # 22
N_CONTENT_FEATS  = 4          # addictiveness, manipulation, educational, novelty per item
N_RECENCY        = N_CONTENT  # binary: seen in last 5 steps?
N_SESSION        = 3          # session_length_norm, step_norm, diversity_score
OBS_SIZE = N_USER_SCALARS + N_TOPICS + N_CONTENT * N_CONTENT_FEATS + N_RECENCY + N_SESSION
# = 4 + 9 + 22*4 + 22 + 3 = 126


# ─────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────

class AttentionEnvWrapper(gym.Env):
    """
    Gymnasium wrapper around AttentionEconomyEnv.

    Parameters
    ----------
    task_id : str
        One of "easy", "medium", "hard". Determines the user profile,
        episode length, allowed content, and reward weights.
    max_steps : int, optional
        Override the task's default max_steps. Useful for curriculum.
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "medium", max_steps: Optional[int] = None):
        super().__init__()

        self.task_id = task_id
        self._max_steps_override = max_steps

        # Inner environment (untouched)
        self._env = AttentionEconomyEnv()

        # ── Action space ──────────────────────────────────────────────────
        # Discrete: indices 0..N_CONTENT-1 → recommend ALL_CONTENT_IDS[i]
        #           indices N_CONTENT..N_CONTENT+2 → META_ACTIONS
        self.action_space = spaces.Discrete(N_CONTENT + len(META_ACTIONS))

        # ── Observation space ─────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        # Pre-build content feature matrix from the full catalog
        # Shape: (N_CONTENT, 4) — static across all episodes
        self._env.reset(task_id)   # initialise catalog
        self._content_feat_matrix = self._build_content_matrix()

        # Cache which content IDs are allowed this task (set at reset)
        self._allowed_set: set = set()

        # Track last raw observation for rendering
        self._last_obs: Optional[Observation] = None

    # ─────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        task_id = (options or {}).get("task_id", self.task_id)
        raw_obs = self._env.reset(task_id)
        self._last_obs = raw_obs
        self._allowed_set = set(self._env.allowed_content_ids)

        return self._encode_obs(raw_obs), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Map integer action → domain Action, call inner env.step(), encode obs.

        Returns (obs, reward, terminated, truncated, info) — Gymnasium v26 API.
        """
        domain_action = self._decode_action(action)
        raw_obs, reward, done, info = self._env.step(domain_action)

        self._last_obs = raw_obs
        obs_vec = self._encode_obs(raw_obs)

        # Gymnasium v26: terminated = natural end (trust collapse / max steps)
        #                truncated  = time limit imposed externally (we fold both into terminated)
        terminated = done
        truncated  = False

        return obs_vec, float(reward), terminated, truncated, info

    def render(self):
        """Print current user state — useful for debugging."""
        if self._last_obs is None:
            print("No observation yet.")
            return
        o = self._last_obs
        print(
            f"  step={self._env.step_count:>3}  "
            f"trust={o.visible_trust:.3f}  "
            f"fatigue={o.visible_fatigue:.3f}  "
            f"satisfaction={o.visible_satisfaction:.3f}  "
            f"boredom={o.visible_boredom:.3f}"
        )

    # ─────────────────────────────────────────────
    # Action mapping
    # ─────────────────────────────────────────────

    def _decode_action(self, action: int) -> Action:
        """
        Map integer action index to a domain Action object.

        Indices 0..N_CONTENT-1: recommend ALL_CONTENT_IDS[i]
          - If the content is NOT in the allowed set for this task,
            we fall back to `diversify_feed` (safe no-op) to avoid
            crashing during training. The negative reward signal from
            the resulting step teaches PPO to avoid such actions.
        Indices N_CONTENT+: meta-actions (diversify, explore, pause)
        """
        if action < N_CONTENT:
            content_id = ALL_CONTENT_IDS[action]
            if content_id in self._allowed_set:
                return Action(action_type="recommend", content_id=content_id)
            else:
                # Masked action — safe fallback with implicit penalty
                return Action(action_type="diversify_feed")
        else:
            meta_idx = action - N_CONTENT
            meta_type = META_ACTIONS[meta_idx]
            return Action(action_type=meta_type)

    def action_masks(self) -> np.ndarray:
        """
        Boolean mask over the action space: True = valid action for this task.
        Compatible with MaskablePPO (sb3-contrib) for improved sample efficiency.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        for i, cid in enumerate(ALL_CONTENT_IDS):
            if cid in self._allowed_set:
                mask[i] = True
        # Meta-actions always valid
        mask[N_CONTENT:] = True
        return mask

    # ─────────────────────────────────────────────
    # Observation encoding
    # ─────────────────────────────────────────────

    def _encode_obs(self, obs: Observation) -> np.ndarray:
        """
        Convert Pydantic Observation → fixed-size float32 numpy vector.

        Layout:
          [0:4]         user scalars: fatigue, trust, satisfaction, boredom
          [4:13]        interest distribution over ALL_TOPICS (zero if topic absent)
          [13:13+N*4]   content features (add, manip, edu, novelty) × N_CONTENT items
          [13+N*4:
           13+N*5]      recency flags: 1.0 if content_id in last-5 history
          [-3:]         session_length / max_steps, step / max_steps, diversity_score
        """
        vec = np.zeros(OBS_SIZE, dtype=np.float32)
        ptr = 0

        # ── User scalars ──────────────────────────────────────────────────
        vec[ptr] = obs.visible_fatigue;    ptr += 1
        vec[ptr] = obs.visible_trust;      ptr += 1
        vec[ptr] = obs.visible_satisfaction; ptr += 1
        vec[ptr] = obs.visible_boredom;    ptr += 1

        # ── Interest distribution ─────────────────────────────────────────
        for topic in ALL_TOPICS:
            vec[ptr] = obs.interest_distribution.get(topic, 0.0)
            ptr += 1

        # ── Content features ─────────────────────────────────────────────
        # Use the pre-built matrix; rows ordered by ALL_CONTENT_IDS
        feat_start = ptr
        for i in range(N_CONTENT):
            vec[ptr:ptr + N_CONTENT_FEATS] = self._content_feat_matrix[i]
            ptr += N_CONTENT_FEATS

        # ── Recency flags ─────────────────────────────────────────────────
        recent_set = set(obs.recent_content_ids)
        for cid in ALL_CONTENT_IDS:
            vec[ptr] = 1.0 if cid in recent_set else 0.0
            ptr += 1

        # ── Session signals ───────────────────────────────────────────────
        max_steps = self._env.max_steps or 1
        vec[ptr] = min(obs.session_length / max_steps, 1.0); ptr += 1
        vec[ptr] = min(obs.step_count    / max_steps, 1.0); ptr += 1
        vec[ptr] = obs.recent_diversity_score;               ptr += 1

        assert ptr == OBS_SIZE, f"Observation size mismatch: {ptr} != {OBS_SIZE}"
        return vec

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _build_content_matrix(self) -> np.ndarray:
        """
        Build a (N_CONTENT, 4) matrix of static content features.
        Rows correspond to ALL_CONTENT_IDS in order.
        Columns: [addictiveness, manipulation_score, educational_value, novelty]
        """
        catalog = self._env.catalog
        matrix = np.zeros((N_CONTENT, N_CONTENT_FEATS), dtype=np.float32)
        for i, cid in enumerate(ALL_CONTENT_IDS):
            if cid in catalog:
                item = catalog[cid]
                matrix[i] = [
                    item.addictiveness,
                    item.manipulation_score,
                    item.educational_value,
                    item.novelty,
                ]
        return matrix

    def get_action_label(self, action: int) -> str:
        """Human-readable label for a given integer action."""
        if action < N_CONTENT:
            return ALL_CONTENT_IDS[action]
        return META_ACTIONS[action - N_CONTENT]