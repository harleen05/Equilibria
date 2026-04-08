"""
env/__init__.py — Public exports for the Attention Economy Environment package.

Quick start:
    from env import AttentionEconomyEnv, Action

    env = AttentionEconomyEnv()
    obs = env.reset("medium")
    obs, reward, done, info = env.step(Action(action_type="recommend", content_id="rel_tech_01"))
"""

from environment.env_core import AttentionEconomyEnv
from environment.models import (
    UserState, ContentItem, Action, Observation, EnvironmentState
)
from environment.reward import RewardFunction, RewardWeights, EASY_WEIGHTS, MEDIUM_WEIGHTS, HARD_WEIGHTS
from environment.simulation import SimulationEngine
from environment.state_manager import StateManager
from environment.content import get_full_catalog, get_catalog_by_type, get_catalog_subset
from environment.utils import clip, normalize, diversity_score, weighted_average, format_metrics
from environment.tasks import get_task

__all__ = [
    # Core environment
    "AttentionEconomyEnv",
    # Data models
    "UserState", "ContentItem", "Action", "Observation", "EnvironmentState",
    # Reward system
    "RewardFunction", "RewardWeights",
    "EASY_WEIGHTS", "MEDIUM_WEIGHTS", "HARD_WEIGHTS",
    # Sub-systems
    "SimulationEngine", "StateManager",
    # Content catalog
    "get_full_catalog", "get_catalog_by_type", "get_catalog_subset",
    # Utilities
    "clip", "normalize", "diversity_score", "weighted_average", "format_metrics",
    # Task registry
    "get_task",
]