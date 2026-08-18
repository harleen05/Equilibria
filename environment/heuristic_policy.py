"""
heuristic_policy.py — Ethical rule-based policy (shared by demo, inference, server, UI).

Mirrors the smart fallback in inference.py: fatigue → pause, low trust → explore,
boredom → diversify, otherwise recommend safe high-relevance content.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _f(item: Any, key: str, default: Any = None) -> Any:
    try:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)
    except Exception:
        return default


def _float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def smart_policy_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Choose an action dict suitable for Action(**dict) / POST /step.

    Returns keys: action_type, optional content_id, optional reasoning.
    """
    try:
        fatigue = _float(obs.get("visible_fatigue"))
        trust = _float(obs.get("visible_trust"), 1.0)
        boredom = _float(obs.get("visible_boredom"))

        if fatigue > 0.65:
            return {"action_type": "pause_session", "reasoning": "high fatigue"}
        if trust < 0.35:
            return {"action_type": "explore_new_topic", "reasoning": "trust recovery"}
        if boredom > 0.45:
            return {"action_type": "diversify_feed", "reasoning": "high boredom"}

        interest_dist = obs.get("interest_distribution") or {}
        dominant = max(interest_dist, key=lambda k: interest_dist[k]) if interest_dist else "technology"
        recent = set(obs.get("recent_content_ids") or [])

        best_id: Optional[str] = None
        best_score = -1.0
        for item in obs.get("available_content") or []:
            try:
                cid = _f(item, "content_id", "")
                manip = _float(_f(item, "manipulation_score"))
                addict = _float(_f(item, "addictiveness"))
                rel = _f(item, "topic_relevance") or {}
                edu = _float(_f(item, "educational_value"))

                if cid in recent or manip > 0.40 or addict > 0.60:
                    continue

                score = 0.50 * _float(rel.get(dominant)) + 0.20 * edu - 0.20 * addict - 0.30 * manip
                if score > best_score:
                    best_score, best_id = score, cid
            except Exception:
                continue

        if best_id:
            return {
                "action_type": "recommend",
                "content_id": best_id,
                "reasoning": "heuristic: best ethical match",
            }

        for item in obs.get("available_content") or []:
            try:
                cid = _f(item, "content_id", "")
                manip = _float(_f(item, "manipulation_score"))
                if cid and manip < 0.30:
                    return {
                        "action_type": "recommend",
                        "content_id": cid,
                        "reasoning": "heuristic: low manipulation fallback",
                    }
            except Exception:
                continue

    except Exception:
        pass

    return {"action_type": "explore_new_topic", "reasoning": "heuristic: safe default"}


def action_label(action: Dict[str, Any]) -> str:
    atype = action.get("action_type", "unknown")
    if atype == "recommend":
        return f"recommend({action.get('content_id', '?')})"
    return atype