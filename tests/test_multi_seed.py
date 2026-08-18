"""
tests/test_multi_seed.py — Unit tests for the statistics logic in
environment/multi_seed.py. These deliberately do NOT train real models
(that would make the test suite slow and flaky) -- they feed
summarize_across_seeds() and the Welch's-t-test math synthetic data with
known properties, so the arithmetic itself is regression-tested independent
of anything RL-related.
"""

from __future__ import annotations

import numpy as np
import pytest

from environment.multi_seed import summarize_across_seeds, _parse_seeds


def test_parse_seeds_basic() -> None:
    assert _parse_seeds("1,2,3,4,5") == [1, 2, 3, 4, 5]


def test_parse_seeds_handles_whitespace() -> None:
    assert _parse_seeds(" 1, 2 ,3 ") == [1, 2, 3]


def test_parse_seeds_single_value() -> None:
    assert _parse_seeds("7") == [7]


def test_summarize_across_seeds_matches_known_values() -> None:
    """Feed synthetic per-seed results with a known mean/std and confirm
    the aggregation arithmetic is correct, independent of any RL code."""
    fake_results = [
        {"seed": 1, "final_score_mean": 0.2},
        {"seed": 2, "final_score_mean": 0.4},
        {"seed": 3, "final_score_mean": 0.6},
    ]
    summary = summarize_across_seeds(fake_results, metric="final_score_mean")

    assert summary["n_seeds"] == 3
    assert summary["mean"] == pytest.approx(0.4, abs=1e-6)
    # sample std (ddof=1) of [0.2, 0.4, 0.6] is 0.2
    assert summary["std"] == pytest.approx(0.2, abs=1e-6)
    assert summary["per_seed_values"] == [0.2, 0.4, 0.6]


def test_summarize_across_seeds_single_seed_has_zero_std() -> None:
    """With n=1, there's no variance to compute -- std and CI should be 0,
    not NaN or an error (numpy's ddof=1 std of a single value is NaN,
    which the function must guard against explicitly)."""
    fake_results = [{"seed": 1, "final_score_mean": 0.5}]
    summary = summarize_across_seeds(fake_results, metric="final_score_mean")

    assert summary["n_seeds"] == 1
    assert summary["mean"] == 0.5
    assert summary["std"] == 0.0
    assert summary["ci95_halfwidth"] == 0.0


def test_summarize_across_seeds_ci_widens_with_more_variance() -> None:
    """Sanity check on direction: a set of seed results with more spread
    should produce a wider 95% CI half-width than a tighter set, for the
    same n_seeds."""
    tight = [{"seed": i, "final_score_mean": v} for i, v in enumerate([0.40, 0.41, 0.39, 0.40])]
    spread = [{"seed": i, "final_score_mean": v} for i, v in enumerate([0.10, 0.70, 0.20, 0.60])]

    tight_summary = summarize_across_seeds(tight, metric="final_score_mean")
    spread_summary = summarize_across_seeds(spread, metric="final_score_mean")

    assert tight_summary["ci95_halfwidth"] < spread_summary["ci95_halfwidth"]