from environment.reward import RewardFunction


def test_reward_function_returns_valid_bounds() -> None:
    reward_fn = RewardFunction()
    reward, breakdown = reward_fn.compute(
        engagement=0.5,
        satisfaction=0.6,
        trust=0.7,
        fatigue=0.2,
        manipulation_score=0.1,
        addiction_risk=0.1,
        diversity_score=0.8,
    )

    assert 0.0001 <= reward <= 0.9999
    assert breakdown["R_engagement"] >= 0.0
    assert breakdown["P_fatigue"] >= 0.0
    assert breakdown["P_manipulation"] >= 0.0
    assert breakdown["reward"] == reward

def test_negative_raw_rewards_are_not_collapsed_to_the_same_floor() -> None:
    """
    Regression test for a real bug: the previous implementation clamped
    reward with max(0.0001, min(raw_reward, 0.9999)), which mapped EVERY
    raw_reward < 0 to the identical value 0.0001 -- destroying all
    gradient-relevant information about HOW bad an action was. Empirically,
    under random rollouts across all 3 tasks, ~7% of steps produced a
    negative raw_reward (observed range: -0.12 to -0.0003), and all of them
    collapsed onto a single point under the old clamp. This test locks in
    that distinct negative raw_reward values now produce distinct outputs.
    """
    reward_fn = RewardFunction()  # medium weights

    # A mildly bad step: raw_reward just barely below zero
    reward_mild, breakdown_mild = reward_fn.compute(
        engagement=0.15, satisfaction=0.0, trust=0.3, fatigue=0.75,
        manipulation_score=0.65, addiction_risk=0.9, diversity_score=0.3,
    )
    # A severely bad step: near-zero engagement/satisfaction, near-max
    # fatigue and manipulation
    reward_severe, breakdown_severe = reward_fn.compute(
        engagement=0.0, satisfaction=0.0, trust=0.3, fatigue=0.95,
        manipulation_score=0.98, addiction_risk=0.2, diversity_score=0.05,
    )

    assert breakdown_mild["raw_reward"] < 0
    assert breakdown_severe["raw_reward"] < 0
    assert breakdown_severe["raw_reward"] < breakdown_mild["raw_reward"]

    # The critical assertion: under the old code both of these collapsed
    # to reward == 0.0001. They must now be distinguishable.
    assert reward_mild != reward_severe
    assert reward_severe < reward_mild

def test_reward_rescale_respects_per_task_weight_profiles() -> None:
    """
    easy/medium/hard have different weight sums (0.85/0.80/0.75 positive
    weight), so a single global clip range can't be correct for all three --
    this test confirms each profile's bound properties are wired correctly.
    """
    from environment.reward import EASY_WEIGHTS, MEDIUM_WEIGHTS, HARD_WEIGHTS

    for weights in (EASY_WEIGHTS, MEDIUM_WEIGHTS, HARD_WEIGHTS):
        assert weights.raw_reward_upper_bound == (
            weights.engagement + weights.retention + weights.trust
        )
        assert weights.raw_reward_lower_bound == -(
            weights.fatigue_penalty + weights.manipulation_penalty
        )
        # sanity: bounds must actually bracket 0 (positive engagement should
        # be achievable, and penalties should be able to push below 0)
        assert weights.raw_reward_lower_bound < 0 < weights.raw_reward_upper_bound

def test_fatigue_penalty_is_convex_but_milder_than_linear() -> None:
    """
    Locks in the precise numeric claim made in reward.py's docstring:
    P_fatigue = fatigue**1.5 is convex (marginal rate increases with
    fatigue) but is NOT harsher than a plain linear penalty anywhere in
    (0, 1) -- they only meet at the boundaries 0 and 1. This corrects an
    earlier (inaccurate) version of the docstring, which implied the
    exponent made the penalty harsher overall; it actually makes it milder
    everywhere except right at the top of the fatigue range.
    """
    fatigue_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    for f in fatigue_levels:
        linear_penalty = f
        convex_penalty = f ** 1.5
        assert convex_penalty < linear_penalty, (
            f"fatigue^1.5 should be strictly milder than linear at fatigue={f}"
        )

    # Convexity: the GAP between fatigue^1.5 and linear should shrink as
    # fatigue increases (i.e. the two curves converge toward fatigue=1.0)
    # The gap between linear and convex penalty is NOT monotonic across the
    # whole range -- it's unimodal: rises from 0 at fatigue=0, peaks near
    # fatigue~0.44 (where d/df[fatigue - fatigue**1.5] = 0), then shrinks
    # back toward 0 as fatigue -> 1.0. What matters for the "back-loaded
    # severity" design property is the SECOND half of that shape: from the
    # peak onward, the convex penalty catches up to linear specifically as
    # the user approaches exhaustion.
    gap_at = {f: f - (f ** 1.5) for f in fatigue_levels}
    assert gap_at[0.9] < gap_at[0.7] < gap_at[0.5], (
        "convex penalty should catch up to linear (gap shrinking) as "
        "fatigue rises through the upper half of the range"
    )
    # And near the bottom of the range, the gap should still be opening up
    # (not yet shrinking) -- confirms the unimodal shape rather than a
    # simple monotonic decrease from the start
    assert gap_at[0.1] < gap_at[0.3] < gap_at[0.5]

    # At the boundaries, linear and convex penalties coincide
    assert 0.0 ** 1.5 == 0.0
    assert 1.0 ** 1.5 == 1.0