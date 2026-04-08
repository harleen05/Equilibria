"""
demo.py — Runs all three tasks with a heuristic agent to verify correctness.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env_core import AttentionEconomyEnv
from environment.models import Action


def heuristic_agent(obs) -> Action:
    """Simple ethical heuristic — avoids manipulation, respects fatigue."""

    # Safe field access (works with both naming styles)
    fatigue = getattr(obs, "visible_fatigue", getattr(obs, "fatigue", 0))
    boredom = getattr(obs, "visible_boredom", getattr(obs, "boredom", 0))

    # High fatigue → pause
    if fatigue > 0.70:
        return Action(action_type="pause_session")

    # High boredom → diversify
    if boredom > 0.50:
        return Action(action_type="diversify_feed")

    dominant = max(obs.interest_distribution, key=obs.interest_distribution.get)
    recent   = set(obs.recent_content_ids)

    best_item  = None
    best_score = -1.0

    for item in obs.available_content:
        if item.content_id in recent:
            continue

        match = item.topic_relevance.get(dominant, 0.0)
        ethical_score = (1.0 - item.manipulation_score) * (1.0 - item.addictiveness)
        score = match * ethical_score

        if score > best_score:
            best_score = score
            best_item = item

    if best_item is None:
        return Action(action_type="explore_new_topic", topic=dominant)

    return Action(action_type="recommend", content_id=best_item.content_id)


def run_episode(task_id: str) -> None:
    env = AttentionEconomyEnv()
    obs = env.reset(task_id)

    print(f"\n{'═' * 60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'═' * 60}")
    print(f"  {'Step':>4}  {'Action':<22}  {'R':>6}  {'Trust':>6}  {'Fatigue':>7}  {'Sat':>5}")
    print(f"  {'─'*55}")

    final_info = {}
    step = 0

    while True:
        action = heuristic_agent(obs)
        obs, reward, done, info = env.step(action)

        step += 1
        final_info = info

        trust = getattr(obs, "visible_trust", getattr(obs, "trust", 0))
        fatigue = getattr(obs, "visible_fatigue", getattr(obs, "fatigue", 0))
        sat = getattr(obs, "visible_satisfaction", getattr(obs, "satisfaction", 0))

        label = action.content_id if action.content_id else action.action_type
        if label and len(label) > 20:
            label = label[:19] + "…"

        print(
            f"  {step:>4}  {label:<22}  "
            f"{reward:.4f}  {trust:.4f}  "
            f"{fatigue:.5f}  {sat:.3f}"
        )

        if done:
            break

    print(f"\n  {'─'*55}")

    if "episode_grade" in final_info:
        g = final_info["episode_grade"]

        print(f"  Final Score        : {g.get('final_score', 0):.4f}")
        print(f"  └─ avg_engagement  : {g.get('avg_engagement', 0):.4f}")
        print(f"  └─ final_trust     : {g.get('final_trust', 0):.4f}")
        print(f"  └─ final_satisf.   : {g.get('final_satisfaction', 0):.4f}")

        if "final_addiction_risk" in g:
            print(f"  └─ final_addiction : {g['final_addiction_risk']:.4f}")

        if "termination_reason" in final_info:
            print(f"  Termination        : {final_info['termination_reason']}")


if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_episode(task)