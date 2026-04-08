"""
inference.py — Attention Economy OpenEnv
Person 2 deliverable: agent loop, structured logging, scoring.

WIRED TO: Person 1's AttentionEconomyEnv (Equilibria repo)
  Actions  : {"action_type": "recommend", "content_id": "rel_tech_01"}
             {"action_type": "diversify_feed"}
             {"action_type": "explore_new_topic"}
             {"action_type": "pause_session"}
  Obs keys : visible_fatigue, visible_trust, visible_satisfaction,
             visible_boredom, available_content, interest_distribution,
             recent_content_ids, recent_diversity_score, step_count, task_id
  Episode grade: info["episode_grade"]["final_score"] at done=True
                 formula: 0.40*avg_engagement + 0.35*final_trust + 0.25*final_satisfaction

Submission compliance:
  ✓ OpenAI client for ALL LLM calls (mandatory per rules)
  ✓ API_BASE_URL / MODEL_NAME / HF_TOKEN from environment variables
  ✓ [START] / [STEP] / [END] on stdout, exact field names and order
  ✓ LLM called first every step; smart_policy is fallback only
  ✓ inference.py in repo root
  ✓ Runs all 3 tasks; completes < 20 min on 2vCPU / 8 GB

Usage:
  python inference.py                         # all 3 tasks
  python inference.py --task easy
  python inference.py --task all --steps 15   # override max_steps
  python inference.py --dry-run               # no API / env calls, verifies log format
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI   # mandatory per submission rules

# ─────────────────────────────────────────────────────────
# CONFIG — read from environment, never hardcoded
# ─────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4o")
HF_TOKEN:     str = os.environ.get("HF_TOKEN",     "")
ENV_URL:      str = os.environ.get("ENV_URL",       "http://localhost:8000")

BENCHMARK = "attention-economy-env"

# Task constants aligned with Person 1's tasks.py
TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy":   {"max_steps": 10, "success_threshold": 0.60},
    "medium": {"max_steps": 15, "success_threshold": 0.55},
    "hard":   {"max_steps": 20, "success_threshold": 0.50},
}
TASKS = ["easy", "medium", "hard"]

# Content classification for heuristic fallback
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

# ─────────────────────────────────────────────────────────
# OPENAI CLIENT  — mandatory per submission rules
# Initialised lazily so --dry-run works without credentials
# ─────────────────────────────────────────────────────────

_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return _client

# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an ethical content recommendation agent in a simulated attention economy.

EPISODE SCORE (what judges measure):
  final_score = 0.40 * avg_engagement + 0.35 * final_trust + 0.25 * final_satisfaction

YOUR 4 POSSIBLE ACTIONS:
1. recommend a specific content_id from available_content
2. diversify_feed   — reduces boredom, lowers fatigue slightly
3. explore_new_topic — small trust boost
4. pause_session    — reduces fatigue by ~0.20, boosts trust by ~0.05

STRATEGY:
- If visible_fatigue > 0.65 → pause_session
- If visible_boredom > 0.45 → diversify_feed
- If visible_trust < 0.35   → explore_new_topic
- Otherwise → recommend the highest-scoring safe content:
    score = 0.5 * topic_relevance[dominant_interest]
          + 0.2 * educational_value
          - 0.2 * addictiveness
          - 0.3 * manipulation_score
  Exclude: content_id in recent_content_ids, manipulation_score > 0.40, addictiveness > 0.60

RESPONSE FORMAT — valid JSON only, no markdown fences:
recommend    : {"action_type": "recommend", "content_id": "rel_tech_01", "reasoning": "one sentence"}
other action : {"action_type": "pause_session", "reasoning": "fatigue is 0.72"}"""

# ─────────────────────────────────────────────────────────
# STRICT LOG HELPERS  — field names and order are fixed
# ALL output goes to stdout so the eval harness captures it
# ─────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print("[START]",      flush=True)
    print(f"task={task}", flush=True)
    print(f"env={env}",   flush=True)
    print(f"model={model}", flush=True)

def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    print("[STEP]",                    flush=True)
    print(f"step={step}",              flush=True)
    print(f"action={action}",          flush=True)
    print(f"reward={reward:+.2f}",     flush=True)
    print(f"done={str(done).lower()}", flush=True)
    if error:
        print(f"error={error}",        flush=True)

def log_end(success: bool, steps: int, score: float) -> None:
    print("[END]",                          flush=True)
    print(f"success={str(success).lower()}", flush=True)
    print(f"steps={steps}",                  flush=True)
    print(f"score={score:.4f}",              flush=True)

# ─────────────────────────────────────────────────────────
# ENV HTTP CLIENT
# ─────────────────────────────────────────────────────────

def call_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def call_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ─────────────────────────────────────────────────────────
# LLM AGENT  — uses OpenAI client as required
# ─────────────────────────────────────────────────────────

def _build_user_message(
    obs: Dict[str, Any],
    step: int,
    task: str,
    last_reward: float,
) -> str:
    content_pool = obs.get("available_content", [])

    def _field(item, key, default=None):
        return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

    pool_lines = [
        f"  id={_field(c,'content_id')}  "
        f"relevance={json.dumps(_field(c,'topic_relevance',{}))}  "
        f"manip={_field(c,'manipulation_score',0):.2f}  "
        f"addict={_field(c,'addictiveness',0):.2f}  "
        f"edu={_field(c,'educational_value',0):.2f}"
        for c in content_pool
    ]

    return (
        f"Step {step} (task={task}, last_reward={last_reward:+.2f})\n\n"
        f"USER STATE:\n"
        f"  fatigue={obs.get('visible_fatigue',0):.2f}  "
        f"trust={obs.get('visible_trust',0):.2f}  "
        f"satisfaction={obs.get('visible_satisfaction',0):.2f}  "
        f"boredom={obs.get('visible_boredom',0):.2f}\n\n"
        f"INTERESTS: {json.dumps(obs.get('interest_distribution',{}))}\n"
        f"RECENT IDs: {obs.get('recent_content_ids',[])}\n\n"
        f"AVAILABLE CONTENT:\n" + "\n".join(pool_lines) + "\n\n"
        f"Respond with JSON only."
    )


def call_llm(
    obs: Dict[str, Any],
    step: int,
    task: str,
    last_reward: float,
) -> Dict[str, Any]:
    """
    Call the LLM via OpenAI client (mandatory per submission rules).
    Returns a parsed action dict.
    Falls back to smart_policy on any exception.
    """
    user_msg = _build_user_message(obs, step, task, last_reward)

    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=200,
    )

    raw = response.choices[0].message.content or ""
    return _parse_llm_response(raw, obs)


def _parse_llm_response(raw: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Parse LLM JSON output with three fallback strategies."""
    def _field(item, key, default=None):
        return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

    available_ids = [_field(c, "content_id") for c in obs.get("available_content", [])]

    # Strategy 1: clean and parse JSON
    try:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        parsed = json.loads(clean.strip())
        atype = parsed.get("action_type", "recommend")
        if atype in ("diversify_feed", "explore_new_topic", "pause_session"):
            return {"action_type": atype, "reasoning": parsed.get("reasoning", "")}
        if atype == "recommend":
            cid = str(parsed.get("content_id", ""))
            if cid in available_ids:
                return {"action_type": "recommend", "content_id": cid,
                        "reasoning": parsed.get("reasoning", "")}
    except Exception:
        pass

    # Strategy 2: regex extract content_id from raw text
    m = re.search(r'"content_id"\s*:\s*"([^"]+)"', raw)
    if m and m.group(1) in available_ids:
        return {"action_type": "recommend", "content_id": m.group(1),
                "reasoning": "regex-extracted"}

    # Strategy 3: known safe id anywhere in text
    for cid in SAFE_CONTENT:
        if cid in raw and cid in available_ids:
            return {"action_type": "recommend", "content_id": cid,
                    "reasoning": "substring-extracted"}

    # Strategy 4: heuristic (last resort)
    return _smart_policy(obs)

# ─────────────────────────────────────────────────────────
# SMART POLICY  — heuristic fallback, never the primary path
# ─────────────────────────────────────────────────────────

def _smart_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based fallback. Mirrors demo.py heuristic agent.
    Used only when LLM call fails or parse fails entirely.
    """
    def _field(item, key, default=None):
        return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

    fatigue = obs.get("visible_fatigue", 0)
    trust   = obs.get("visible_trust", 1)
    boredom = obs.get("visible_boredom", 0)

    if fatigue > 0.65:
        return {"action_type": "pause_session",    "reasoning": "fallback: high fatigue"}
    if trust < 0.35:
        return {"action_type": "explore_new_topic","reasoning": "fallback: trust recovery"}
    if boredom > 0.45:
        return {"action_type": "diversify_feed",   "reasoning": "fallback: high boredom"}

    interest_dist = obs.get("interest_distribution", {})
    dominant = max(interest_dist, key=interest_dist.get) if interest_dist else "technology"
    recent   = set(obs.get("recent_content_ids", []))

    best_id, best_score = None, -1.0
    for item in obs.get("available_content", []):
        cid   = _field(item, "content_id", "")
        manip = _field(item, "manipulation_score", 0)
        addict= _field(item, "addictiveness", 0)
        rel   = _field(item, "topic_relevance", {})
        edu   = _field(item, "educational_value", 0)

        if cid in recent or manip > 0.40 or addict > 0.60:
            continue

        score = (
            0.50 * rel.get(dominant, 0)
            + 0.20 * edu
            - 0.20 * addict
            - 0.30 * manip
        )
        if score > best_score:
            best_score, best_id = score, cid

    if best_id:
        return {"action_type": "recommend", "content_id": best_id,
                "reasoning": "fallback: heuristic best pick"}

    # absolute last resort: first non-manipulative available item
    for item in obs.get("available_content", []):
        cid   = _field(item, "content_id", "")
        manip = _field(item, "manipulation_score", 0)
        if cid and manip < 0.30:
            return {"action_type": "recommend", "content_id": cid,
                    "reasoning": "fallback: last resort"}

    return {"action_type": "explore_new_topic", "reasoning": "fallback: no content available"}

# ─────────────────────────────────────────────────────────
# ACTION → CANONICAL LOG STRING
# ─────────────────────────────────────────────────────────

def _action_str(action: Dict[str, Any]) -> str:
    atype = action.get("action_type", "unknown")
    if atype == "recommend":
        return f"recommend(content_id={action.get('content_id','?')})"
    return atype

# ─────────────────────────────────────────────────────────
# DRY-RUN FAKE ENV  — validates log format without any calls
# ─────────────────────────────────────────────────────────

def _fake_reset(task_id: str) -> Dict[str, Any]:
    content_pool = [
        {"content_id":"rel_tech_01",   "topic_relevance":{"technology":1.0,"science":0.4},  "addictiveness":0.15,"manipulation_score":0.05,"educational_value":0.85,"novelty":0.75},
        {"content_id":"rel_sci_01",    "topic_relevance":{"science":1.0,"technology":0.3},   "addictiveness":0.10,"manipulation_score":0.05,"educational_value":0.90,"novelty":0.70},
        {"content_id":"rel_health_01", "topic_relevance":{"health":1.0,"science":0.3},       "addictiveness":0.08,"manipulation_score":0.04,"educational_value":0.92,"novelty":0.60},
        {"content_id":"rnd_film_01",   "topic_relevance":{"entertainment":1.0,"general":0.3},"addictiveness":0.30,"manipulation_score":0.10,"educational_value":0.30,"novelty":0.80},
        {"content_id":"add_gaming_01", "topic_relevance":{"entertainment":0.9},              "addictiveness":0.75,"manipulation_score":0.20,"educational_value":0.10,"novelty":0.65},
        {"content_id":"mis_click_01",  "topic_relevance":{"entertainment":0.6},              "addictiveness":0.50,"manipulation_score":0.70,"educational_value":0.03,"novelty":0.75},
    ]
    profiles = {
        "easy":   {"visible_fatigue":0.10,"visible_trust":0.90,"visible_satisfaction":0.50,"visible_boredom":0.10,
                   "interest_distribution":{"technology":0.85,"science":0.60,"health":0.30}},
        "medium": {"visible_fatigue":0.15,"visible_trust":0.80,"visible_satisfaction":0.50,"visible_boredom":0.20,
                   "interest_distribution":{"science":0.60,"technology":0.55,"health":0.50,"entertainment":0.45}},
        "hard":   {"visible_fatigue":0.20,"visible_trust":0.70,"visible_satisfaction":0.45,"visible_boredom":0.25,
                   "interest_distribution":{"entertainment":0.70,"social":0.65,"politics":0.40,"technology":0.30}},
    }
    p = profiles.get(task_id, profiles["medium"])
    return {"observation": {**p, "available_content": content_pool,
                            "recent_content_ids": [], "recent_diversity_score": 1.0,
                            "session_length": 0, "step_count": 0, "task_id": task_id}}

def _fake_step(
    action: Dict[str, Any],
    step_num: int,
    max_steps: int,
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    import random
    rng = random.Random(step_num * 17 + abs(hash(action.get("content_id", action.get("action_type", "")))) % 997)

    atype = action.get("action_type", "recommend")
    cid   = action.get("content_id", "")

    f = obs.get("visible_fatigue", 0.1)
    t = obs.get("visible_trust", 0.8)
    s = obs.get("visible_satisfaction", 0.5)

    if atype == "pause_session":
        f, t, s = max(0.0,f-0.20), min(1.0,t+0.05), s
    elif atype == "diversify_feed":
        f, t, s = max(0.0,f-0.08), min(1.0,t+0.02), s
    elif atype == "explore_new_topic":
        f, t, s = f, min(1.0,t+0.01), s
    else:  # recommend
        is_manip = cid in MANIPULATIVE_CONTENT
        is_addict = cid in ADDICTIVE_CONTENT
        f = min(1.0, f + (0.12 if is_addict else 0.07))
        t = max(0.0, t - (0.20 if is_manip else 0.01))
        s = min(1.0, s + (-0.03 if is_manip else 0.05))

    reward = round(rng.uniform(0.35, 0.75), 4)
    done   = step_num >= max_steps

    new_obs = dict(obs)
    new_obs.update({
        "visible_fatigue":      round(f, 4),
        "visible_trust":        round(t, 4),
        "visible_satisfaction": round(s, 4),
        "step_count":           step_num,
        "recent_content_ids":   (obs.get("recent_content_ids", []) + ([cid] if cid else []))[-5:],
    })

    eng  = round(rng.uniform(0.40, 0.75), 4)
    info: Dict[str, Any] = {
        "step": step_num, "task": obs.get("task_id", "unknown"),
        "diagnostics": {"engagement": eng, "diversity_score": round(rng.uniform(0.4,1.0),4)},
        "reward_breakdown": {"reward": reward},
        "user_state": {"trust": round(t,4), "fatigue": round(f,4), "addiction_risk": 0.10},
    }
    if done:
        info["episode_grade"] = {
            "final_score":        round(0.40*eng + 0.35*t + 0.25*s, 4),
            "avg_engagement":     eng,
            "final_trust":        round(t, 4),
            "final_satisfaction": round(s, 4),
        }

    return {"observation": new_obs, "reward": reward, "done": done, "info": info}

# ─────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────

def run_episode(
    task_id: str,
    max_steps_override: int = 0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cfg       = TASK_CONFIG[task_id]
    max_steps = max_steps_override or cfg["max_steps"]
    threshold = cfg["success_threshold"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # ── reset ──────────────────────────────────────────────────────────
    try:
        reset_data = _fake_reset(task_id) if dry_run else call_reset(task_id)
    except Exception as e:
        log_end(success=False, steps=0, score=0.0)
        return {"score": 0.0, "success": False, "steps": 0,
                "rewards": [], "episode_grade": {}}

    obs           = reset_data.get("observation", reset_data)
    rewards:      List[float] = []
    step_num:     int         = 0
    done:         bool        = False
    last_reward:  float       = 0.0
    episode_grade: Dict       = {}

    # ── step loop ──────────────────────────────────────────────────────
    while not done and step_num < max_steps:
        step_num += 1
        error_str: Optional[str] = None

        # ── decide action: LLM first, smart_policy fallback ──────────
        if dry_run:
            # dry-run skips API calls, uses heuristic to test log format
            action = _smart_policy(obs)
        else:
            try:
                action = call_llm(obs, step_num, task_id, last_reward)
            except Exception as e:
                # LLM unavailable / parse fail → heuristic, still continues
                action = _smart_policy(obs)
                error_str = f"llm_fallback:{str(e)[:60]}"

        action_str = _action_str(action)

        # ── build env action payload ──────────────────────────────────
        env_action: Dict[str, Any] = {"action_type": action["action_type"]}
        if action.get("content_id"):
            env_action["content_id"] = action["content_id"]
        if action.get("topic"):
            env_action["topic"] = action["topic"]

        # ── step env ──────────────────────────────────────────────────
        try:
            result = (
                _fake_step(env_action, step_num, max_steps, obs)
                if dry_run
                else call_step(env_action)
            )
        except Exception as e:
            log_step(step_num, action_str, 0.0, True,
                     error=f"step_error:{str(e)[:60]}")
            done = True
            break

        reward      = float(result.get("reward", 0.0))
        done        = bool(result.get("done", False))
        obs         = result.get("observation", obs)
        info        = result.get("info", {})
        last_reward = reward

        if done and "episode_grade" in info:
            episode_grade = info["episode_grade"]

        rewards.append(reward)
        log_step(step_num, action_str, reward, done, error=error_str)

    # ── score ──────────────────────────────────────────────────────────
    # Use Person 1's authoritative grade when available (preferred)
    if episode_grade and "final_score" in episode_grade:
        score = round(float(episode_grade["final_score"]), 4)
    else:
        # Fallback: normalised cumulative reward
        max_total = max_steps * 1.0
        score = round(min(sum(rewards) / max_total, 1.0), 4) if max_total > 0 else 0.0

    score   = max(0.0, min(score, 1.0))
    success = score >= threshold

    log_end(success=success, steps=step_num, score=score)

    return {
        "score":         score,
        "success":       success,
        "steps":         step_num,
        "rewards":       rewards,
        "episode_grade": episode_grade,
    }

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AttentionEconomyEnv baseline inference agent"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Override max steps per episode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use fake env + heuristic; validates log format without any network calls",
    )
    args = parser.parse_args()

    if not args.dry_run:
        missing = [
            v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
            if not os.environ.get(v)
        ]
        if missing:
            print(
                f"[ERROR] Missing required env vars: {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    results: Dict[str, Dict] = {}

    for task_id in tasks_to_run:
        results[task_id] = run_episode(
            task_id=task_id,
            max_steps_override=args.steps,
            dry_run=args.dry_run,
        )

    # ── human-readable summary to stderr (does not affect eval parsing) ──
    print("\n" + "=" * 62, file=sys.stderr)
    print("  BASELINE SUMMARY", file=sys.stderr)
    print("=" * 62, file=sys.stderr)
    print(f"  {'task':<10} {'score':<10} {'ok':<5} {'steps':<7} {'eng / trust / sat'}", file=sys.stderr)
    print(f"  {'-'*57}", file=sys.stderr)

    for task_id, r in results.items():
        g      = r.get("episode_grade", {})
        status = "PASS" if r["success"] else "FAIL"
        detail = (
            f"{g.get('avg_engagement',0):.2f} / "
            f"{g.get('final_trust',0):.2f} / "
            f"{g.get('final_satisfaction',0):.2f}"
            if g else "—"
        )
        print(
            f"  {task_id:<10} {r['score']:<10.4f} {status:<5} "
            f"{r['steps']:<7} {detail}",
            file=sys.stderr,
        )

    overall = sum(r["score"] for r in results.values()) / max(len(results), 1)
    print(f"\n  Overall avg score: {overall:.4f}", file=sys.stderr)
    print("=" * 62, file=sys.stderr)

    all_pass = all(r["success"] for r in results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()