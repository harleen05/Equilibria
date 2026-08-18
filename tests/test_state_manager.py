from environment.models import UserState, ContentItem
from environment.state_manager import StateManager


def test_state_manager_initialize_and_apply_step() -> None:
    user = UserState(
        user_id="test_user",
        interest_distribution={"technology": 1.0},
        fatigue=0.0,
        trust=0.5,
        addiction_risk=0.1,
        satisfaction=0.5,
        boredom=0.0,
        session_length=0,
        fatigue_sensitivity=1.0,
        trust_decay_rate=1.0,
    )
    content = ContentItem(
        content_id="rel_tech_01",
        title="Tech",
        content_type="relevant",
        topic_relevance={"technology": 1.0},
        addictiveness=0.1,
        manipulation_score=0.05,
        educational_value=0.9,
        novelty=0.8,
    )

    manager = StateManager()
    manager.initialize(user)

    changes = manager.apply_step(
        content=content,
        fatigue_delta=0.1,
        trust_delta=0.05,
        satisfaction_delta=0.1,
        addiction_risk_delta=0.05,
        boredom_delta=0.02,
    )

    assert manager.step_count == 1
    assert manager.history == ["rel_tech_01"]
    assert changes["step_count"] == 1
    assert manager.user.fatigue == 0.1
    assert manager.user.session_length == 1


def test_state_manager_history_recently() -> None:
    user = UserState(
        user_id="test_user",
        interest_distribution={"technology": 1.0},
        fatigue=0.0,
        trust=0.5,
        addiction_risk=0.1,
        satisfaction=0.5,
        boredom=0.0,
        session_length=0,
        fatigue_sensitivity=1.0,
        trust_decay_rate=1.0,
    )
    manager = StateManager()
    manager.initialize(user)
    manager.apply_step(
        content=ContentItem(
            content_id="rel_tech_01",
            title="Tech",
            content_type="relevant",
            topic_relevance={"technology": 1.0},
            addictiveness=0.1,
            manipulation_score=0.05,
            educational_value=0.9,
            novelty=0.8,
        ),
        fatigue_delta=0.0,
        trust_delta=0.0,
        satisfaction_delta=0.0,
        addiction_risk_delta=0.0,
        boredom_delta=0.0,
    )

    assert manager.has_seen_recently("rel_tech_01") is True
    assert manager.has_seen_recently("does_not_exist") is False

def _mk_item(content_id: str, content_type: str) -> ContentItem:
    return ContentItem(
        content_id=content_id,
        title=content_id,
        content_type=content_type,
        topic_relevance={"technology": 1.0},
        addictiveness=0.1,
        manipulation_score=0.05,
        educational_value=0.9,
        novelty=0.8,
    )


def test_consecutive_same_type_count() -> None:
    """
    Regression test: this method previously referenced ContentItem.content_type,
    a field that had been removed from the model, so any call raised
    AttributeError. It was never exercised by the existing suite because
    env_core.py doesn't use StateManager at all.
    """
    user = UserState(
        user_id="test_user",
        interest_distribution={"technology": 1.0},
        fatigue=0.0, trust=0.5, addiction_risk=0.1,
        satisfaction=0.5, boredom=0.0, session_length=0,
        fatigue_sensitivity=1.0, trust_decay_rate=1.0,
    )
    catalog = {
        "a1": _mk_item("a1", "addictive"),
        "a2": _mk_item("a2", "addictive"),
        "a3": _mk_item("a3", "addictive"),
        "r1": _mk_item("r1", "relevant"),
    }

    manager = StateManager()
    manager.initialize(user)

    for cid in ("r1", "a1", "a2", "a3"):
        manager.apply_step(
            content=catalog[cid],
            fatigue_delta=0.0, trust_delta=0.0,
            satisfaction_delta=0.0, addiction_risk_delta=0.0,
            boredom_delta=0.0,
        )

    assert manager.consecutive_same_type_count("addictive", catalog) == 3
    assert manager.consecutive_same_type_count("relevant", catalog) == 0

    manager._history.append("unknown_id")
    assert manager.consecutive_same_type_count("addictive", catalog) == 0