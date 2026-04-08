from __future__ import annotations
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy
# ─────────────────────────────────────────────────────────────────────────────

ContentType = Literal["relevant", "random", "addictive", "misleading"]

ActionType = Literal[
    "recommend",
    "diversify_feed",
    "explore_new_topic",
    "pause_session",
]


# ─────────────────────────────────────────────────────────────────────────────
# Content Item
# ─────────────────────────────────────────────────────────────────────────────

class ContentItem(BaseModel):
    """A single piece of content that can be recommended to the user."""

    content_id: str
    title: str

    # ❌ REMOVED: content_type
    # ❌ REMOVED: base_engagement

    topic_relevance: Dict[str, float] = Field(
        description="Relevance weight per interest topic, each in [0, 1]"
    )

    addictiveness: float = Field(ge=0.0, le=1.0)
    manipulation_score: float = Field(ge=0.0, le=1.0)
    educational_value: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)

    @field_validator("topic_relevance")
    @classmethod
    def validate_relevance_bounds(cls, v: Dict[str, float]) -> Dict[str, float]:
        for topic, weight in v.items():
            if not (0.0 <= weight <= 1.0):
                raise ValueError(f"Relevance for '{topic}' must be in [0, 1], got {weight}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# User State
# ─────────────────────────────────────────────────────────────────────────────

class UserState(BaseModel):

    user_id: str

    interest_distribution: Dict[str, float] = Field(
        description="User interest weights per topic, each in [0, 1]"
    )

    fatigue: float = Field(ge=0.0, le=1.0, default=0.0)
    trust: float = Field(ge=0.0, le=1.0, default=0.8)
    addiction_risk: float = Field(ge=0.0, le=1.0, default=0.1)
    satisfaction: float = Field(ge=0.0, le=1.0, default=0.5)
    boredom: float = Field(ge=0.0, le=1.0, default=0.0)

    session_length: int = Field(default=0, ge=0)

    fatigue_sensitivity: float = Field(ge=0.0, le=2.0, default=1.0)
    trust_decay_rate: float = Field(ge=0.0, le=2.0, default=1.0)

    @field_validator("interest_distribution")
    @classmethod
    def validate_interest_bounds(cls, v: Dict[str, float]) -> Dict[str, float]:
        for topic, weight in v.items():
            if not (0.0 <= weight <= 1.0):
                raise ValueError(f"Interest weight for '{topic}' must be in [0, 1], got {weight}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────────────────────────────────────

class Action(BaseModel):

    action_type: ActionType
    content_id: Optional[str] = None
    topic: Optional[str] = None

    @field_validator("content_id")
    @classmethod
    def content_id_required_for_recommend(cls, v: Optional[str], info) -> Optional[str]:
        if info.data.get("action_type") == "recommend" and v is None:
            raise ValueError("content_id is required when action_type is 'recommend'")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):

    visible_fatigue: float
    visible_trust: float
    visible_satisfaction: float
    visible_boredom: float
    session_length: int

    interest_distribution: Dict[str, float]

    available_content: List[ContentItem]

    recent_content_ids: List[str]

    recent_diversity_score: float

    step_count: int
    task_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Environment State
# ─────────────────────────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):

    user: UserState
    step_count: int
    max_steps: int
    history: List[str]
    content_pool: List[ContentItem]
    done: bool
    task_id: str
    engagement_history: List[float]
    reward_history: List[float]
    action_log: List[Dict]

    class Config:
        arbitrary_types_allowed = True