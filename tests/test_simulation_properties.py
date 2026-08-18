"""
tests/test_simulation_properties.py — Property-based (Hypothesis) tests for
environment/simulation.py's state-transition functions.

Why property-based tests, on top of the example-based tests in
tests/test_simulation.py: this substantiates the "fully deterministic,
bounded" claim in simulation.py's module docstring with an actual proof
sweep across hundreds of randomized inputs per function, rather than a
handful of hand-picked examples. Writing these caught a REAL bug (see
below) that no hand-written example test had found.

BUG FOUND AND FIXED WHILE WRITING THIS FILE:
SimulationEngine.update_satisfaction() previously only applied its
[0.0001, 0.9999] clamp INSIDE specific branches (the content-recommend
branch, and -- in a first-pass fix -- the pause/diversify branch), unlike
every other update_* function (update_fatigue, update_trust,
update_addiction_risk, update_boredom), which all clamp UNCONDITIONALLY as
their last line regardless of which branch executed. This meant any
action_type/content combination that fell through both branches (e.g.
"explore_new_topic" with content=None, which happens on every real
explore_new_topic step) returned an UNCLAMPED value. Concretely
reproducible via completely ordinary gameplay: spamming pause_session from
the default starting satisfaction drove it to -0.06, which then silently
persisted in the live UserState object (Pydantic's model_copy(update=...)
does not re-run field validators, so the ge=0.0 constraint on
UserState.satisfaction was silently bypassed). Fixed by moving the clamp
to be unconditional, matching the other four update_* functions' pattern.
"""

from __future__ import annotations

from hypothesis import given, strategies as st, settings, HealthCheck

from environment.simulation import SimulationEngine, TRUST_FLOOR, FATIGUE_CAP
from environment.models import UserState, ContentItem, Action


# ── Strategies ──────────────────────────────────────────────────────────────
# Bounds mirror the actual Pydantic Field(ge=..., le=...) constraints in
# models.py, so generated inputs match what the real validated domain
# objects can actually contain -- not arbitrary/unrealistic floats.

unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
sensitivity_float = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
TOPICS = [
    "technology", "science", "health", "politics",
    "entertainment", "social", "finance", "sports", "general",
]
topic_dist_strategy = st.dictionaries(
    keys=st.sampled_from(TOPICS), values=unit_float, min_size=1, max_size=len(TOPICS)
)
action_types = st.sampled_from(
    ["recommend", "explore_new_topic", "diversify_feed", "pause_session"]
)


@st.composite
def user_states(draw):
    return UserState(
        user_id="prop_test_user",
        interest_distribution=draw(topic_dist_strategy),
        fatigue=draw(unit_float),
        trust=draw(unit_float),
        addiction_risk=draw(unit_float),
        satisfaction=draw(unit_float),
        boredom=draw(unit_float),
        session_length=draw(st.integers(min_value=0, max_value=200)),
        fatigue_sensitivity=draw(sensitivity_float),
        trust_decay_rate=draw(sensitivity_float),
    )


@st.composite
def content_items(draw, content_id=None):
    return ContentItem(
        content_id=content_id or draw(st.sampled_from(["c1", "c2", "c3"])),
        title="prop test item",
        content_type=draw(st.sampled_from(["relevant", "random", "addictive", "misleading"])),
        topic_relevance=draw(topic_dist_strategy),
        addictiveness=draw(unit_float),
        manipulation_score=draw(unit_float),
        educational_value=draw(unit_float),
        novelty=draw(unit_float),
    )


optional_content = st.one_of(content_items(), st.none())
_settings = dict(max_examples=200, suppress_health_check=[HealthCheck.too_slow])


# ── Per-field bound properties ───────────────────────────────────────────────

@given(user=user_states(), content=optional_content, action_type=action_types)
@settings(**_settings)
def test_fatigue_always_bounded(user, content, action_type) -> None:
    result = SimulationEngine.update_fatigue(user, content, action_type)
    assert 0.0 <= result <= FATIGUE_CAP


@given(user=user_states(), content=optional_content, im=unit_float, action_type=action_types)
@settings(**_settings)
def test_trust_always_bounded(user, content, im, action_type) -> None:
    result = SimulationEngine.update_trust(user, content, im, action_type)
    assert TRUST_FLOOR <= result <= 0.9999


@given(
    user=user_states(), content=optional_content,
    im=unit_float, rp=unit_float, action_type=action_types,
)
@settings(**_settings)
def test_satisfaction_always_bounded(user, content, im, rp, action_type) -> None:
    """Regression test for the bug described in this file's module
    docstring -- this exact property (checked across all four action
    types, including the ones that previously fell through unclamped) is
    what caught it."""
    result = SimulationEngine.update_satisfaction(user, content, im, rp, action_type)
    assert 0.0001 <= result <= 0.9999


@given(user=user_states(), content=optional_content, action_type=action_types)
@settings(**_settings)
def test_addiction_risk_always_bounded(user, content, action_type) -> None:
    result = SimulationEngine.update_addiction_risk(user, content, action_type)
    assert 0.0001 <= result <= 0.9999


@given(user=user_states(), content=optional_content, rp=unit_float, ds=unit_float)
@settings(**_settings)
def test_boredom_always_bounded(user, content, rp, ds) -> None:
    result = SimulationEngine.update_boredom(user, content, rp, ds)
    assert 0.0001 <= result <= 0.9999


@given(content=content_items(), user=user_states())
@settings(**_settings)
def test_interest_match_always_bounded(content, user) -> None:
    result = SimulationEngine.compute_interest_match(content, user)
    assert 0.0 <= result <= 1.0


@given(
    cid=st.text(min_size=1, max_size=10),
    history=st.lists(st.text(min_size=1, max_size=10), max_size=20),
)
@settings(**_settings)
def test_repetition_penalty_always_bounded(cid, history) -> None:
    result = SimulationEngine.compute_repetition_penalty(cid, history)
    assert 0.0 <= result <= 1.0


@given(history=st.lists(st.sampled_from(["c1", "c2", "c3", "c4"]), max_size=20))
@settings(**_settings)
def test_diversity_score_always_bounded(history) -> None:
    catalog = {
        cid: ContentItem(
            content_id=cid, title="t", content_type="relevant",
            topic_relevance={"technology": 1.0}, addictiveness=0.1,
            manipulation_score=0.1, educational_value=0.5, novelty=0.5,
        )
        for cid in ["c1", "c2", "c3", "c4"]
    }
    result = SimulationEngine.compute_diversity_score(history, catalog)
    assert 0.0 <= result <= 1.0


@given(content=content_items(), user=user_states(), im=unit_float, rp=unit_float)
@settings(**_settings)
def test_engagement_always_bounded(content, user, im, rp) -> None:
    # Seeded so the stochastic noise term is reproducible across runs --
    # the bound must hold regardless of the specific noise draw, but a
    # fixed seed keeps CI failures reproducible if this ever breaks.
    engine = SimulationEngine(seed=42)
    result = engine.compute_engagement(content, user, im, rp)
    assert 0.0001 <= result <= 0.9999


# ── End-to-end composed pipeline ─────────────────────────────────────────────

@given(
    user=user_states(),
    content=st.one_of(content_items(), st.none()),
    action_type=st.sampled_from(["explore_new_topic", "diversify_feed", "pause_session"]),
    history=st.lists(st.sampled_from(["c1", "c2", "c3"]), max_size=10),
)
@settings(**_settings)
def test_apply_transition_all_fields_bounded(user, content, action_type, history) -> None:
    """
    The most valuable property here: even though every individual update_*
    function is bounded (checked above), that doesn't guarantee the FULL
    composed apply_transition() pipeline is bug-free -- e.g. a wrong
    argument order, a field written from the wrong intermediate variable,
    etc. This exercises the real end-to-end path env_core.py actually
    calls, using non-"recommend" action types so content is used the same
    way env_core.py itself uses it (None for meta-actions).
    """
    catalog = {
        cid: ContentItem(
            content_id=cid, title="t", content_type="relevant",
            topic_relevance={"technology": 1.0}, addictiveness=0.3,
            manipulation_score=0.2, educational_value=0.5, novelty=0.5,
        )
        for cid in ["c1", "c2", "c3"]
    }
    engine = SimulationEngine(seed=1)
    action = Action(action_type=action_type)  # meta-actions never carry content_id
    updated_user, diagnostics = engine.apply_transition(
        user, action, content=None, history=history, catalog=catalog,
    )

    assert 0.0 <= updated_user.fatigue <= 1.0
    assert 0.0 <= updated_user.trust <= 1.0
    assert 0.0 <= updated_user.satisfaction <= 1.0
    assert 0.0 <= updated_user.addiction_risk <= 1.0
    assert 0.0 <= updated_user.boredom <= 1.0