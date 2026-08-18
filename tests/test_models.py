import pytest
from pydantic import ValidationError

from environment.models import Action, ContentItem, UserState


def test_content_item_topic_relevance_bounds() -> None:
    with pytest.raises(ValidationError):
        ContentItem(
            content_id="bad_rel",
            title="Bad Relevance",
            content_type="relevant",
            topic_relevance={"technology": 1.2},
            addictiveness=0.1,
            manipulation_score=0.1,
            educational_value=0.1,
            novelty=0.1,
        )


def test_action_requires_content_id_for_recommend() -> None:
    with pytest.raises(ValidationError):
        Action(action_type="recommend")

    action = Action(action_type="pause_session")
    assert action.action_type == "pause_session"
    assert action.content_id is None


def test_user_state_interest_distribution_bounds() -> None:
    with pytest.raises(ValidationError):
        UserState(
            user_id="user_test",
            interest_distribution={"technology": 1.1},
            fatigue=0.0,
            trust=0.5,
            addiction_risk=0.1,
            satisfaction=0.5,
            boredom=0.0,
            session_length=0,
            fatigue_sensitivity=1.0,
            trust_decay_rate=1.0,
        )
