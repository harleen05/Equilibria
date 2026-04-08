"""
inference.py — Attention Economy OpenEnv
Person 2 deliverable: agent loop, logging, scoring.

WIRED TO: Person 1's AttentionEconomyEnv (Equilibria repo)
  - Actions: {"action_type": "recommend", "content_id": "rel_tech_01"}
  - Observation fields: visible_fatigue, visible_trust, visible_satisfaction,
    visible_boredom, available_content, interest_distribution, recent_content_ids
  - Episode grade: info["episode_grade"] at done=True
    formula: 0.40*avg_engagement + 0.35*final_trust + 0.25*final_satisfaction

Usage:
  python inference.py                        # runs all 3 tasks
  python inference.py --task easy
  python inference.py --task all --steps 15
  python inference.py --dry-run              # test log format, no API/env calls
"""

import os
import sys
import json
import argparse
import requests
import re
from typing import Optional

# ──────────────────────────────────────────────
# ENV CONFIG — never hardcode these
# ──────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME   = os.environ.get("MODEL_NAME", "claude-sonnet-4-20250514")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")

# ──────────────────────────────────────────────
# TASK CONSTANTS — from Person 1's task configs
# easy: 15 steps | medium: 20 steps | hard: 25 steps
# ──────────────────────────────────────────────
TASKS = ["easy", "medium", "hard"]

TASK_CONFIG = {
    "easy":   {"max_steps": 15, "max_reward_per_step": 1.0, "success_threshold": 0.60},
    "medium": {"max_steps": 20, "max_reward_per_step": 1.0, "success_threshold": 0.55},
    "hard":   {"max_steps": 25, "max_reward_per_step": 1.0, "success_threshold": 0.50},
}

# Known content IDs from Person 1's content.py
SAFE_CONTENT = [
    "rel_sci_01", "rel_tech_01", "rel_fin_01", "rel_hist_01",
    "rel_health_01", "rel_health_02", "rel_news_01", "rel_env_01",
    "rnd_film_01", "rnd_music_01", "rnd_food_01", "rnd_sport_01",
]
ADDICTIVE_CONTENT = [
    "add_scroll_01", "add_satisfy_01", "add_gaming_01",
    "add_social_01", "add_social_02",
]
MANIPULATIVE_CONTENT = [
    "mis_outrage_01", "mis_outrage_02", "mis_click_01",
    "mis_click_02", "mis_pseudo_01",
]

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are an ethical content recommendation agent in a simulated attention economy.

EPISODE SCORE (what judges measure):
  final_score = 0.40 * avg_engagement + 0.35 * final_trust + 0.25 * final_satisfaction

YOUR 4 POSSIBLE ACTIONS:
1. recommend a specific content item
2. diversify_feed — reduces boredom, slightly reduces fatigue
3. explore_new_topic — small trust boost
4. pause_session — reduces fatigue by 0.20, boosts trust by 0.05. Use when fatigue > 0.65.

ETHICAL RULES:
- NEVER recommend mis_outrage_01, mis_outrage_02, mis_pseudo_01 — manipulation ~0.90 collapses trust
- AVOID repeating the same content_id — repetition kills engagement
- LIMIT addictive content (add_scroll_01, add_satisfy_01) — addiction_risk grows over time

DECISION PROCESS:
1. If fatigue > 0.65: use pause_session
2. If boredom > 0.45: use diversify_feed
3. Find dominant interest from interest_distribution
4. Pick content where: topic_relevance[dominant_interest] is high, manipulation_score < 0.20, NOT in recent_content_ids
5. Prefer rel_* content — educational value builds trust over time

RESPONSE FORMAT — ONLY valid JSON, no markdown:
For recommend: {"action_type": "recommend", "content_id": "rel_tech_01", "reasoning": "one sentence"}
For other actions: {"action_type": "pause_session", "reasoning": "fatigue is 0.72"}"""

# ──────────────────────────────────────────────
# MEMORY TRACKER (NEW)
# ──────────────────────────────────────────────
def init_history():
    return {
        "trust_trend": [],
        "fatigue_trend": [],
        "recent_actions": []
    }

def is_trust_dropping(trend):
    if len(trend) < 3:
        return False
    return trend[-1] < trend[-2] < trend[-3]

def is_fatigue_spiking(trend):
    if len(trend) < 3:
        return False
    return trend[-1] > trend[-2] > trend[-3]

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print("[START]", flush=True)
    print(f"task={task}", flush=True)
    print(f"env={env}", flush=True)
    print(f"model={model}", flush=True)
    print(flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    print("[STEP]", flush=True)
    print(f"step={step}", flush=True)
    print(f"action={action}", flush=True)
    print(f"reward={reward:+.2f}", flush=True)
    print(f"done={str(done).lower()}", flush=True)
    if error:
        print(f"error={error}", flush=True)
    print(flush=True)

def log_end(success: bool, steps: int, score: float):
    print("[END]", flush=True)
    print(f"success={str(success).lower()}", flush=True)
    print(f"steps={steps}", flush=True)
    print(f"score={score:.4f}", flush=True)
    print(flush=True)

# ──────────────────────────────────────────────
# ENV API CALLS
# ──────────────────────────────────────────────
def call_reset(task: str) -> dict:
    resp = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def call_step(action: dict) -> dict:
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ──────────────────────────────────────────────
# HEURISTIC FALLBACK — mirrors demo.py exactly
# ──────────────────────────────────────────────
def _heuristic_action(observation: dict) -> dict:
    fatigue = observation.get("visible_fatigue", 0)
    boredom = observation.get("visible_boredom", 0)

    if fatigue > 0.65:
        return {"action_type": "pause_session", "reasoning": "heuristic: high fatigue"}
    if boredom > 0.45:
        return {"action_type": "diversify_feed", "reasoning": "heuristic: high boredom"}

    interest_dist = observation.get("interest_distribution", {})
    dominant = max(interest_dist, key=interest_dist.get) if interest_dist else "technology"
    recent = set(observation.get("recent_content_ids", []))
    available = observation.get("available_content", [])

    best_id, best_score = None, -1.0
    for item in available:
        cid   = item.get("content_id", "") if isinstance(item, dict) else getattr(item, "content_id", "")
        manip = item.get("manipulation_score", 0) if isinstance(item, dict) else getattr(item, "manipulation_score", 0)
        rel   = item.get("topic_relevance", {}) if isinstance(item, dict) else getattr(item, "topic_relevance", {})
        edu   = item.get("educational_value", 0) if isinstance(item, dict) else getattr(item, "educational_value", 0)
        if cid in recent or manip > 0.30:
            continue
        score = rel.get(dominant, 0.0) * (1.0 - manip) * (0.7 + 0.3 * edu)
        if score > best_score:
            best_score, best_id = score, cid

    if best_id:
        return {"action_type": "recommend", "content_id": best_id, "reasoning": "heuristic safe pick"}

    for item in available:
        cid   = item.get("content_id", "") if isinstance(item, dict) else getattr(item, "content_id", "")
        manip = item.get("manipulation_score", 0) if isinstance(item, dict) else getattr(item, "manipulation_score", 0)
        if manip < 0.30:
            return {"action_type": "recommend", "content_id": cid, "reasoning": "last resort"}

    return {"action_type": "explore_new_topic", "reasoning": "absolute fallback"}

# ──────────────────────────────────────────────
# SMART POLICY (NEW — HARD TASK OPTIMIZED)
# ──────────────────────────────────────────────
def smart_policy(observation: dict, history: dict) -> dict:

    fatigue = observation.get("visible_fatigue", 0)
    trust   = observation.get("visible_trust", 0)
    boredom = observation.get("visible_boredom", 0)

    trust_trend   = history["trust_trend"]
    fatigue_trend = history["fatigue_trend"]

    recent_actions = history.get("recent_actions", [])
    force_recommend = recent_actions.count("diversify_feed") >= 3

    # 🚨 EMERGENCY CONTROL
    if fatigue > 0.7 or is_fatigue_spiking(fatigue_trend):
        return {"action_type": "pause_session", "reasoning": "fatigue control"}

    if trust < 0.35 or is_trust_dropping(trust_trend):
        return {"action_type": "explore_new_topic", "reasoning": "trust recovery"}

    if boredom > 0.5 and not force_recommend and len(history["recent_actions"]) % 2 == 0:
        return {"action_type": "diversify_feed", "reasoning": "controlled diversity"}

    # 🎯 CONTENT SELECTION
    interest_dist = observation.get("interest_distribution", {})
    dominant = max(interest_dist, key=interest_dist.get) if interest_dist else "technology"

    recent_list = observation.get("recent_content_ids", [])
    recent = set(recent_list)

    if len(set(recent_list)) < 3 and len(recent_list) >= 3:
        return {"action_type": "diversify_feed", "reasoning": "low diversity fix"}

    best_id = None
    best_score = -1

    for item in observation.get("available_content", []):
        cid   = item.get("content_id")
        manip = item.get("manipulation_score", 0)
        addict= item.get("addictiveness", 0)
        rel   = item.get("topic_relevance", {})
        edu   = item.get("educational_value", 0)

        # ❌ HARD FILTERS (VERY IMPORTANT)
        if manip > 0.4:
            continue
        if trust < 0.6 and manip > 0.2:
            continue
        if addict > 0.6:
            continue
        if cid in recent:
            continue

        score = (
            0.5 * rel.get(dominant, 0)
            + 0.2 * edu
            - 0.2 * addict
            - 0.3 * manip
        )

        if score > best_score:
            best_score = score
            best_id = cid

    if best_id:
        return {"action_type": "recommend", "content_id": best_id, "reasoning": "safe optimized pick"}

    return {"action_type": "diversify_feed", "reasoning": "fallback"}

# ──────────────────────────────────────────────
# LLM AGENT
# ──────────────────────────────────────────────
def call_llm(observation: dict, step_num: int, task: str) -> dict:
    obs_summary = {
        "step": step_num,
        "task": task,
        "user": {
            "fatigue":      observation.get("visible_fatigue", 0),
            "trust":        observation.get("visible_trust", 0),
            "satisfaction": observation.get("visible_satisfaction", 0),
            "boredom":      observation.get("visible_boredom", 0),
        },
        "interest_distribution": observation.get("interest_distribution", {}),
        "recent_content_ids":    observation.get("recent_content_ids", []),
        "available_content": [
            {
                "content_id":        c.get("content_id", "") if isinstance(c, dict) else getattr(c, "content_id", ""),
                "topic_relevance":   c.get("topic_relevance", {}) if isinstance(c, dict) else getattr(c, "topic_relevance", {}),
                "manipulation_score":c.get("manipulation_score", 0) if isinstance(c, dict) else getattr(c, "manipulation_score", 0),
                "addictiveness":     c.get("addictiveness", 0) if isinstance(c, dict) else getattr(c, "addictiveness", 0),
                "educational_value": c.get("educational_value", 0) if isinstance(c, dict) else getattr(c, "educational_value", 0),
            }
            for c in observation.get("available_content", [])
        ],
    }

    headers = {"Content-Type": "application/json", "x-api-key": HF_TOKEN}
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Current state:\n{json.dumps(obs_summary, indent=2)}\n\nChoose your action. Return ONLY JSON."}],
    }

    resp = requests.post(f"{API_BASE_URL}/v1/messages", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    raw_text = "".join(
        b["text"] for b in resp.json().get("content", []) if b.get("type") == "text"
    )
    return parse_llm_response(raw_text, observation)

def parse_llm_response(raw: str, observation: dict) -> dict:
    available_ids = [
        c.get("content_id", "") if isinstance(c, dict) else getattr(c, "content_id", "")
        for c in observation.get("available_content", [])
    ]

    # Strategy 1: clean JSON
    try:
        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
        parsed = json.loads(clean.strip())
        atype = parsed.get("action_type", "recommend")
        if atype in ("diversify_feed", "explore_new_topic", "pause_session"):
            return {"action_type": atype, "reasoning": parsed.get("reasoning", "")}
        if atype == "recommend" and parsed.get("content_id"):
            return {"action_type": "recommend", "content_id": str(parsed["content_id"]), "reasoning": parsed.get("reasoning", "")}
    except Exception:
        pass

    # Strategy 2: regex for content_id string
    m = re.search(r'"content_id"\s*:\s*"([^"]+)"', raw)
    if m and m.group(1) in available_ids:
        return {"action_type": "recommend", "content_id": m.group(1), "reasoning": "regex fallback"}

    # Strategy 3: known safe content_id in raw text
    for cid in SAFE_CONTENT:
        if cid in raw and cid in available_ids:
            return {"action_type": "recommend", "content_id": cid, "reasoning": "substring fallback"}

    # Strategy 4: heuristic
    return _heuristic_action(observation)

# ──────────────────────────────────────────────
# DRY RUN FAKE ENV
# ──────────────────────────────────────────────
def fake_reset(task: str) -> dict:
    content_pool = [
        {"content_id": "rel_tech_01",    "topic_relevance": {"technology": 1.0, "science": 0.4},   "addictiveness": 0.15, "manipulation_score": 0.05, "educational_value": 0.85, "novelty": 0.75},
        {"content_id": "rel_sci_01",     "topic_relevance": {"science": 1.0, "technology": 0.3},    "addictiveness": 0.10, "manipulation_score": 0.05, "educational_value": 0.90, "novelty": 0.70},
        {"content_id": "rel_health_01",  "topic_relevance": {"health": 1.0, "science": 0.3},        "addictiveness": 0.08, "manipulation_score": 0.04, "educational_value": 0.92, "novelty": 0.60},
        {"content_id": "rnd_film_01",    "topic_relevance": {"entertainment": 1.0, "general": 0.3}, "addictiveness": 0.30, "manipulation_score": 0.10, "educational_value": 0.30, "novelty": 0.80},
        {"content_id": "add_gaming_01",  "topic_relevance": {"entertainment": 0.9},                 "addictiveness": 0.75, "manipulation_score": 0.20, "educational_value": 0.10, "novelty": 0.65},
        {"content_id": "mis_click_01",   "topic_relevance": {"entertainment": 0.6},                 "addictiveness": 0.50, "manipulation_score": 0.70, "educational_value": 0.03, "novelty": 0.75},
    ]
    profiles = {
        "easy":   {"visible_fatigue": 0.00, "visible_trust": 0.85, "visible_satisfaction": 0.50, "visible_boredom": 0.00,
                   "interest_distribution": {"technology": 0.70, "science": 0.20, "entertainment": 0.10}},
        "medium": {"visible_fatigue": 0.10, "visible_trust": 0.75, "visible_satisfaction": 0.50, "visible_boredom": 0.05,
                   "interest_distribution": {"science": 0.30, "health": 0.25, "entertainment": 0.20, "politics": 0.15, "technology": 0.10}},
        "hard":   {"visible_fatigue": 0.20, "visible_trust": 0.65, "visible_satisfaction": 0.40, "visible_boredom": 0.15,
                   "interest_distribution": {"politics": 0.35, "social": 0.25, "health": 0.20, "science": 0.10, "entertainment": 0.10}},
    }
    p = profiles.get(task, profiles["medium"])
    return {"observation": {**p, "available_content": content_pool, "recent_content_ids": [],
                            "recent_diversity_score": 1.0, "session_length": 0, "step_count": 0, "task_id": task}}

def fake_step(action: dict, step_num: int, max_steps: int, observation: dict) -> dict:
    import random
    random.seed(step_num + abs(hash(action.get("content_id", action.get("action_type", "")))) % 999)
    reward = round(random.uniform(0.35, 0.75), 4)
    done   = step_num >= max_steps

    prev_f = observation.get("visible_fatigue", 0.1)
    prev_t = observation.get("visible_trust", 0.8)
    prev_s = observation.get("visible_satisfaction", 0.5)
    atype  = action.get("action_type", "recommend")
    cid    = action.get("content_id", "")

    if atype == "pause_session":
        nf, nt, ns = max(0.0, prev_f-0.18), min(1.0, prev_t+0.04), max(0.0, prev_s-0.01)
    elif atype == "diversify_feed":
        nf, nt, ns = max(0.0, prev_f-0.08), min(1.0, prev_t+0.01), prev_s
    else:
        is_m = cid in MANIPULATIVE_CONTENT
        is_a = cid in ADDICTIVE_CONTENT
        nf = min(1.0, prev_f + (0.12 if is_a else 0.07))
        nt = max(0.0, prev_t - (0.18 if is_m else 0.01))
        ns = min(1.0, prev_s + (-0.02 if is_m else 0.05))

    new_obs = dict(observation)
    new_obs.update({
        "visible_fatigue": round(nf, 4), "visible_trust": round(nt, 4),
        "visible_satisfaction": round(ns, 4), "step_count": step_num,
        "recent_content_ids": (observation.get("recent_content_ids", []) + ([cid] if cid else []))[-5:],
    })

    avg_eng = round(random.uniform(0.4, 0.75), 4)
    info = {
        "step": step_num, "task": observation.get("task_id", "unknown"),
        "diagnostics": {"engagement": avg_eng, "diversity_score": round(random.uniform(0.4, 1.0), 4)},
        "reward_breakdown": {"reward": reward},
        "user_state": {"trust": round(nt, 4), "fatigue": round(nf, 4), "addiction_risk": 0.10},
    }
    if done:
        info["episode_grade"] = {
            "final_score":        round(0.40*avg_eng + 0.35*nt + 0.25*ns, 4),
            "avg_engagement":     avg_eng,
            "final_trust":        round(nt, 4),
            "final_satisfaction": round(ns, 4),
        }

    return {"observation": new_obs, "reward": reward, "done": done, "info": info}

# ──────────────────────────────────────────────
# ACTION → LOG STRING
# ──────────────────────────────────────────────
def action_to_str(action: dict) -> str:
    atype = action.get("action_type", "unknown")
    if atype == "recommend":
        return f"recommend(content_id={action.get('content_id', '?')})"
    return atype

# ──────────────────────────────────────────────
# EPISODE RUNNER
# ──────────────────────────────────────────────
def run_episode(task: str, max_steps: int, dry_run: bool = False) -> dict:
    cfg              = TASK_CONFIG[task]
    max_steps        = max_steps or cfg["max_steps"]
    success_threshold= cfg["success_threshold"]
    max_total_reward = max_steps * cfg["max_reward_per_step"]

    log_start(task=task, env="attention-economy-env", model=MODEL_NAME)

    try:
        reset_data = fake_reset(task) if dry_run else call_reset(task)
    except Exception as e:
        print(f"[ERROR] reset() failed: {e}", file=sys.stderr)
        log_end(success=False, steps=0, score=0.0)
        return {"score": 0.0, "success": False, "steps": 0, "rewards": [], "episode_grade": {}}

    observation   = reset_data.get("observation", reset_data)
    history = init_history()
    rewards       = []
    step_num      = 0
    done          = False
    episode_grade = {}

    while not done and step_num < max_steps:
        step_num += 1
        history["trust_trend"].append(observation.get("visible_trust", 0))
        history["fatigue_trend"].append(observation.get("visible_fatigue", 0))

        try:
            if dry_run:
                action = smart_policy(observation, history)
                history["recent_actions"].append(action["action_type"])
            else:
                action = smart_policy(observation, history)
                history["recent_actions"].append(action["action_type"])
        except Exception as e:
            action = _heuristic_action(observation)
            print(f"[WARN] LLM error step {step_num}: {e}", file=sys.stderr)

        action_str = action_to_str(action)
        env_action = {"action_type": action["action_type"]}
        if action.get("content_id"):
            env_action["content_id"] = action["content_id"]
        if action.get("topic"):
            env_action["topic"] = action["topic"]

        try:
            step_result = (fake_step(env_action, step_num, max_steps, observation)
                           if dry_run else call_step(env_action))
        except Exception as e:
            log_step(step_num, action_str, 0.0, True, error=str(e)[:80])
            break

        reward      = float(step_result.get("reward", 0.0))
        done        = bool(step_result.get("done", False))
        observation = step_result.get("observation", observation)
        info        = step_result.get("info", {})

        if done and "episode_grade" in info:
            episode_grade = info["episode_grade"]

        rewards.append(reward)
        log_step(step_num, action_str, reward, done)

    # Use Person 1's authoritative grade if available, else normalize cumulative reward
    if episode_grade and "final_score" in episode_grade:
        score = round(episode_grade["final_score"], 4)
    else:
        cumulative = sum(rewards)
        score = round(min(cumulative / max_total_reward, 1.0), 4) if max_total_reward > 0 else 0.0

    success = score >= success_threshold
    log_end(success=success, steps=step_num, score=score)
    return {"score": score, "success": success, "steps": step_num, "rewards": rewards, "episode_grade": episode_grade}

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    choices=["easy","medium","hard","all"], default="all")
    parser.add_argument("--steps",   type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        missing = [v for v in ["API_BASE_URL","MODEL_NAME","HF_TOKEN"] if not os.environ.get(v)]
        if missing:
            print(f"[ERROR] Missing env vars: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    results = {}
    for task in tasks_to_run:
        results[task] = run_episode(task=task, max_steps=args.steps, dry_run=args.dry_run)

    print("\n" + "="*60, file=sys.stderr)
    print("FINAL RESULTS", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"{'Task':<10} {'Score':<10} {'OK':<6} {'Steps':<8} {'eng / trust / sat'}", file=sys.stderr)
    print("-"*60, file=sys.stderr)
    for task, r in results.items():
        g = r.get("episode_grade", {})
        detail = (f"{g.get('avg_engagement',0):.2f} / {g.get('final_trust',0):.2f} / {g.get('final_satisfaction',0):.2f}"
                  if g else "—")
        print(f"{task:<10} {r['score']:<10.4f} {'✓' if r['success'] else '✗':<6} {r['steps']:<8} {detail}", file=sys.stderr)
    print("="*60, file=sys.stderr)
    overall = sum(r["score"] for r in results.values()) / len(results)
    print(f"Overall avg: {overall:.4f}", file=sys.stderr)

if __name__ == "__main__":
    main()