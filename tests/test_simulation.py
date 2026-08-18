from environment.content import get_full_catalog
from environment.models import UserState, ContentItem, Action
from environment.simulation import SimulationEngine


def make_test_user() -> UserState:
    return UserState(
        user_id="user_test",
        interest_distribution={"technology": 0.5, "science": 0.5},
        fatigue=0.2,
        trust=0.5,
        addiction_risk=0.1,
        satisfaction=0.5,
        boredom=0.2,
        session_length=0,
        fatigue_sensitivity=1.0,
        trust_decay_rate=1.0,
    )


def make_test_content() -> ContentItem:
    return ContentItem(
        content_id="rel_tech_01",
        title="Tech Innovation Weekly",
        content_type="relevant",
        topic_relevance={"technology": 1.0, "science": 0.4},
        addictiveness=0.15,
        manipulation_score=0.05,
        educational_value=0.85,
        novelty=0.75,
    )


def test_interest_match_and_repetition_penalty() -> None:
    user = make_test_user()
    content = make_test_content()
    engine = SimulationEngine(seed=123)

    match = engine.compute_interest_match(content, user)
    assert 0.0 <= match <= 1.0

    penalty = engine.compute_repetition_penalty(content.content_id, [content.content_id, "other"])
    assert penalty == 0.2


def test_diversity_score() -> None:
    catalog = get_full_catalog()
    score = SimulationEngine.compute_diversity_score(["rel_tech_01", "rel_sci_01"], catalog)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_update_transitions_and_engagement() -> None:
    user = make_test_user()
    content = make_test_content()
    engine = SimulationEngine(seed=42)

    new_fatigue = engine.update_fatigue(user, content, "recommend")
    assert new_fatigue >= user.fatigue

    new_trust = engine.update_trust(user, content, engine.compute_interest_match(content, user), "recommend")
    assert 0.0 <= new_trust <= 1.0

    new_satisfaction = engine.update_satisfaction(user, content, 0.8, 0.0, "recommend")
    assert 0.0 <= new_satisfaction <= 1.0

    new_addiction = engine.update_addiction_risk(user, content, "recommend")
    assert 0.0 <= new_addiction <= 1.0

    new_boredom = engine.update_boredom(user, content, 0.0, 1.0)
    assert 0.0 <= new_boredom <= 1.0

    engagement = engine.compute_engagement(content, user, 0.8, 0.0)
    assert 0.0 <= engagement <= 1.0


def test_apply_transition_returns_diagnostics() -> None:
    user = make_test_user()
    content = make_test_content()
    engine = SimulationEngine(seed=7)
    action = Action(action_type="recommend", content_id=content.content_id)

    updated, diagnostics = engine.apply_transition(user, action, content, [], get_full_catalog())

    assert updated.user_id == user.user_id
    assert diagnostics["engagement"] >= 0.0
    assert "diversity_score" in diagnostics
    assert updated.session_length == 1
