"""
utils.py — Shared utility functions for the Attention Economy Environment.

Provides:
  - clip()             : Clamp a float to [lo, hi]
  - normalize()        : Min-max normalize a value into [0, 1]
  - diversity_score()  : Shannon entropy-based category diversity
  - safe_divide()      : Division with zero-denominator guard
  - weighted_average() : Weighted mean of a dict of (value, weight) pairs
  - format_metrics()   : Pretty-print a metrics dict for logging
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional


def clip(value: float, lo: float = 0.0001, hi: float = 0.9999) -> float:
    """
    Clamp `value` to the closed interval [lo, hi].

    Examples
    --------
    >>> clip(1.5)       # → 1.0
    >>> clip(-0.1)      # → 0.0
    >>> clip(0.7)       # → 0.7
    >>> clip(3.0, 0, 2) # → 2.0
    """
    return max(lo, min(value, hi))


def normalize(value: float, lo: float, hi: float) -> float:
    """
    Min-max normalize `value` from range [lo, hi] into [0, 1].
    Returns 0.0 if lo == hi (degenerate range guard).

    Examples
    --------
    >>> normalize(5.0, 0.0, 10.0)  # → 0.5
    >>> normalize(0.0, 0.0, 10.0)  # → 0.0
    >>> normalize(10.0, 5.0, 5.0)  # → 0.0  (degenerate)
    """
    if abs(hi - lo) < 1e-9:
        return 0.0
    return clip((value - lo) / (hi - lo))


def diversity_score(
    history: List[str],
    category_map: Dict[str, str],
    window: int = 5,
) -> float:
    """
    Compute Shannon entropy-based diversity of content categories
    in the most recent `window` steps.

    Returns a score in [0, 1]:
      0.0 → all items in the window are the same category (no diversity)
      1.0 → all items are different categories (maximum diversity)

    Parameters
    ----------
    history      : List of content_ids in chronological order
    category_map : Dict mapping content_id → category string
    window       : Number of recent steps to consider

    Algorithm
    ---------
    H = −Σ p(c) × log2(p(c))  for each unique category c in the window
    diversity = H / log2(num_unique_categories)
    """
    recent = history[-window:]
    if not recent:
        return 1.0  # Empty history → assume maximum diversity

    counts: Dict[str, int] = {}
    for cid in recent:
        cat = category_map.get(cid, "__unknown__")
        counts[cat] = counts.get(cat, 0) + 1

    n = len(recent)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(max(len(counts), 2))
    return clip(entropy / max_entropy) if max_entropy > 0 else 0.0


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """
    Divide numerator by denominator, returning `fallback` if denominator is zero.

    Examples
    --------
    >>> safe_divide(1.0, 2.0)     # → 0.5
    >>> safe_divide(1.0, 0.0)     # → 0.0
    >>> safe_divide(5.0, 0.0, 1.0)# → 1.0
    """
    if abs(denominator) < 1e-9:
        return fallback
    return numerator / denominator


def weighted_average(values_and_weights: Dict[str, tuple]) -> float:
    """
    Compute the weighted mean of a set of (value, weight) pairs.

    Parameters
    ----------
    values_and_weights : {label: (value, weight), ...}

    Returns
    -------
    Σ(value × weight) / Σ(weight), clipped to [0, 1].

    Examples
    --------
    >>> weighted_average({"a": (0.8, 0.6), "b": (0.4, 0.4)})  # → 0.64
    """
    total_weight = sum(w for _, w in values_and_weights.values())
    if abs(total_weight) < 1e-9:
        return 0.0
    weighted_sum = sum(v * w for v, w in values_and_weights.values())
    return clip(weighted_sum / total_weight)


def format_metrics(metrics: Dict[str, float], indent: int = 2) -> str:
    """
    Pretty-format a dict of float metrics for logging/debug output.

    Parameters
    ----------
    metrics : Dict mapping label → float value
    indent  : Number of leading spaces per line

    Returns
    -------
    Multi-line string, one "label: value" per line.

    Example output:
        trust:        0.7430
        fatigue:      0.2100
        engagement:   0.5812
    """
    pad = " " * indent
    max_key_len = max((len(k) for k in metrics), default=0)
    lines = [
        f"{pad}{k.ljust(max_key_len + 2)}: {v:.4f}"
        for k, v in metrics.items()
    ]
    return "\n".join(lines)