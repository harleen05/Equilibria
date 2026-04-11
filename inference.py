"""
inference.py — Attention Economy OpenEnv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "").rstrip("/")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")
API_KEY:      str = os.environ.get("API_KEY", HF_TOKEN)
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_URL:      str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK = "attention-economy-env"

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy":   {"max_steps": 10, "success_threshold": 0.60},
    "medium": {"max_steps": 15, "success_threshold": 0.55},
    "hard":   {"max_steps": 20, "success_threshold": 0.50},
}
TASKS = ["easy", "medium", "hard"]

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
# OPENAI CLIENT
# ─────────────────────────────────────────────────────────

_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
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
- Pick content where manipulation_score < 0.20, not in recent_content_ids
- Prefer rel_* content (high educational_value builds trust)
- NEVER recommend mis_outrage_01/02 or mis_pseudo_01

For recommend: {"action_type": "recommend", "content_id": "rel_tech_01", "reasoning": "one sentence"}
For other action : {"action_type": "pause_session", "reasoning": "fatigue is 0.72"}"""

# ─────────────────────────────────────────────────────────
# LOG HELPERS — SINGLE LINE FORMAT (spec-compliant)
# Format: [START] task=X env=Y model=Z
#         [STEP] step=N action=A reward=0.00 done=false error=null
#         [END] success=true steps=N score=0.00 rewards=r1,r2,...
# ─────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    # reward formatted to 2 decimal places, no sign prefix
    reward_str = f"{reward:.2f}"
    done_str   = "true" if done else "false"
    error_str  = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward_str} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Score must be strictly between 0 and 1 — not 0.0, not 1.0
    score        = max(0.0001, min(score, 0.9999))
    success_str  = "true" if success else "false"
    score_str    = f"{score:.2f}"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={success_str} steps={steps} score={score_str} rewards={rewards_str}", flush=True)

# ─────────────────────────────────────────────────────────
# SAFE FIELD ACCESSOR
# ─────────────────────────────────────────────────────────

def _f(item: Any, key: str, default: Any = None) -> Any:
    """Safely get a field from dict or object, never raises."""
    try:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)
    except Exception:
        return default

def _float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float, never raises."""
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default

# ─────────────────────────────────────────────────────────
# ENV HTTP CLIENT
# ─────────────────────────────────────────────────────────

def call_reset(task_id: str) -> Dict[str, Any]:
    payload = {"task": task_id, "task_id": task_id}
    print(f"[DEBUG] POST {ENV_URL}/reset payload={payload}", file=sys.stderr)
    resp = requests.post(f"{ENV_URL}/reset", json=payload, timeout=30)
    print(f"[DEBUG] reset status={resp.status_code} body={resp.text[:300]}", file=sys.stderr)
    resp.raise_for_status()
    return resp.json()

def call_step(action: Dict[str, Any]) -> Dict[str, Any]:
    """Try wrapped format first (server expects {"action": {...}}), then flat."""
    print(f"[DEBUG] POST {ENV_URL}/step action={action}", file=sys.stderr)
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    print(f"[DEBUG] step status={resp.status_code}", file=sys.stderr)
    if resp.ok:
        return resp.json()
    resp2 = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    print(f"[DEBUG] step fallback status={resp2.status_code} body={resp2.text[:200]}", file=sys.stderr)
    resp2.raise_for_status()
    return resp2.json()

# ─────────────────────────────────────────────────────────
# LLM AGENT
# ─────────────────────────────────────────────────────────

def _build_user_message(obs: Dict[str, Any], step: int, task: str, last_reward: float) -> str:
    try:
        content_pool = obs.get("available_content", [])
        pool_lines = []
        for c in content_pool:
            try:
                line = (
                    f"  id={_f(c,'content_id','?')}  "
                    f"relevance={json.dumps(_f(c,'topic_relevance') or {})}  "
                    f"manip={_float(_f(c,'manipulation_score')):.2f}  "
                    f"addict={_float(_f(c,'addictiveness')):.2f}  "
                    f"edu={_float(_f(c,'educational_value')):.2f}"
                )
                pool_lines.append(line)
            except Exception:
                pool_lines.append(f"  id={_f(c,'content_id','?')} (parse error)")

        return (
            f"Step {step} (task={task}, last_reward={last_reward:.2f})\n\n"
            f"USER STATE:\n"
            f"  fatigue={_float(obs.get('visible_fatigue')):.2f}  "
            f"trust={_float(obs.get('visible_trust')):.2f}  "
            f"satisfaction={_float(obs.get('visible_satisfaction')):.2f}  "
            f"boredom={_float(obs.get('visible_boredom')):.2f}\n\n"
            f"INTERESTS: {json.dumps(obs.get('interest_distribution') or {})}\n"
            f"RECENT IDs: {obs.get('recent_content_ids') or []}\n\n"
            f"AVAILABLE CONTENT:\n" + "\n".join(pool_lines) + "\n\n"
            f"Respond with JSON only."
        )
    except Exception as e:
        print(f"[ERROR] _build_user_message failed: {e}", file=sys.stderr)
        return f"Step {step}. Choose one action: recommend, diversify_feed, explore_new_topic, pause_session. Respond with JSON only."


def call_llm(obs: Dict[str, Any], step: int, task: str, last_reward: float) -> Dict[str, Any]:
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
    try:
        available_ids = [_f(c, "content_id") for c in (obs.get("available_content") or [])]

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

        m = re.search(r'"content_id"\s*:\s*"([^"]+)"', raw)
        if m and m.group(1) in available_ids:
            return {"action_type": "recommend", "content_id": m.group(1), "reasoning": "regex-extracted"}

        for cid in SAFE_CONTENT:
            if cid in raw and cid in available_ids:
                return {"action_type": "recommend", "content_id": cid, "reasoning": "substring-extracted"}
    except Exception as e:
        print(f"[ERROR] _parse_llm_response failed: {e}", file=sys.stderr)

    return _smart_policy(obs)

# ─────────────────────────────────────────────────────────
# SMART POLICY — heuristic fallback
# ─────────────────────────────────────────────────────────

def _smart_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        fatigue = _float(obs.get("visible_fatigue"))
        trust   = _float(obs.get("visible_trust"), 1.0)
        boredom = _float(obs.get("visible_boredom"))

        if fatigue > 0.65:
            return {"action_type": "pause_session",     "reasoning": "fallback: high fatigue"}
        if trust < 0.35:
            return {"action_type": "explore_new_topic", "reasoning": "fallback: trust recovery"}
        if boredom > 0.45:
            return {"action_type": "diversify_feed",    "reasoning": "fallback: high boredom"}

        interest_dist = obs.get("interest_distribution") or {}
        dominant = max(interest_dist, key=interest_dist.get) if interest_dist else "technology"
        recent   = set(obs.get("recent_content_ids") or [])

        best_id, best_score = None, -1.0
        for item in (obs.get("available_content") or []):
            try:
                cid    = _f(item, "content_id", "")
                manip  = _float(_f(item, "manipulation_score"))
                addict = _float(_f(item, "addictiveness"))
                rel    = _f(item, "topic_relevance") or {}
                edu    = _float(_f(item, "educational_value"))

                if cid in recent or manip > 0.40 or addict > 0.60:
                    continue

                score = 0.50 * _float(rel.get(dominant)) + 0.20 * edu - 0.20 * addict - 0.30 * manip
                if score > best_score:
                    best_score, best_id = score, cid
            except Exception:
                continue

        if best_id:
            return {"action_type": "recommend", "content_id": best_id, "reasoning": "fallback: heuristic"}

        for item in (obs.get("available_content") or []):
            try:
                cid   = _f(item, "content_id", "")
                manip = _float(_f(item, "manipulation_score"))
                if cid and manip < 0.30:
                    return {"action_type": "recommend", "content_id": cid, "reasoning": "fallback: last resort"}
            except Exception:
                continue

    except Exception as e:
        print(f"[ERROR] _smart_policy failed: {e}", file=sys.stderr)

    return {"action_type": "explore_new_topic", "reasoning": "fallback: safe default"}

# ─────────────────────────────────────────────────────────
# ACTION STRING
# ─────────────────────────────────────────────────────────

def _action_str(action: Dict[str, Any]) -> str:
    try:
        atype = action.get("action_type", "unknown")
        if atype == "recommend":
            return f"recommend(content_id={action.get('content_id','?')})"
        return atype
    except Exception:
        return "unknown"

# ─────────────────────────────────────────────────────────
# DRY-RUN FAKE ENV
# ─────────────────────────────────────────────────────────

def _fake_reset(task_id: str) -> Dict[str, Any]:
    content_pool = [
        {"content_id":"rel_tech_01",   "topic_relevance":{"technology":1.0,"science":0.4},   "addictiveness":0.15,"manipulation_score":0.05,"educational_value":0.85,"novelty":0.75},
        {"content_id":"rel_sci_01",    "topic_relevance":{"science":1.0,"technology":0.3},    "addictiveness":0.10,"manipulation_score":0.05,"educational_value":0.90,"novelty":0.70},
        {"content_id":"rel_health_01", "topic_relevance":{"health":1.0,"science":0.3},        "addictiveness":0.08,"manipulation_score":0.04,"educational_value":0.92,"novelty":0.60},
        {"content_id":"rnd_film_01",   "topic_relevance":{"entertainment":1.0,"general":0.3}, "addictiveness":0.30,"manipulation_score":0.10,"educational_value":0.30,"novelty":0.80},
        {"content_id":"add_gaming_01", "topic_relevance":{"entertainment":0.9},               "addictiveness":0.75,"manipulation_score":0.20,"educational_value":0.10,"novelty":0.65},
        {"content_id":"mis_click_01",  "topic_relevance":{"entertainment":0.6},               "addictiveness":0.50,"manipulation_score":0.70,"educational_value":0.03,"novelty":0.75},
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

def _fake_step(action: Dict[str, Any], step_num: int, max_steps: int, obs: Dict[str, Any]) -> Dict[str, Any]:
    import random
    rng = random.Random(step_num * 17 + abs(hash(action.get("content_id", action.get("action_type", "")))) % 997)
    atype = action.get("action_type", "recommend")
    cid   = action.get("content_id", "")
    f = _float(obs.get("visible_fatigue"), 0.1)
    t = _float(obs.get("visible_trust"), 0.8)
    s = _float(obs.get("visible_satisfaction"), 0.5)

    if atype == "pause_session":
        f, t, s = max(0.0001, f-0.20), min(0.9999, t+0.05), s
    elif atype == "diversify_feed":
        f, t, s = max(0.0001, f-0.08), min(0.9999, t+0.02), s
    elif atype == "explore_new_topic":
        f, t, s = f, min(0.9999, t+0.01), s
    else:
        is_manip  = cid in MANIPULATIVE_CONTENT
        is_addict = cid in ADDICTIVE_CONTENT
        f = min(0.9999, f + (0.12 if is_addict else 0.07))
        t = max(0.0001, t - (0.20 if is_manip  else 0.01))
        s = min(0.9999, s + (-0.03 if is_manip else 0.05))

    reward = round(rng.uniform(0.35, 0.75), 4)
    done   = step_num >= max_steps
    new_obs = dict(obs)
    existing_ids = obs.get("recent_content_ids") or []
    new_obs.update({
        "visible_fatigue":      round(f, 4),
        "visible_trust":        round(t, 4),
        "visible_satisfaction": round(s, 4),
        "step_count":           step_num,
        "recent_content_ids":   (existing_ids + ([cid] if cid else []))[-5:],
    })
    eng = round(rng.uniform(0.40, 0.75), 4)
    info: Dict[str, Any] = {
        "step": step_num, "task": obs.get("task_id", "unknown"),
        "diagnostics": {"engagement": eng, "diversity_score": round(rng.uniform(0.4, 1.0), 4)},
        "reward_breakdown": {"reward": reward},
        "user_state": {"trust": round(t, 4), "fatigue": round(f, 4), "addiction_risk": 0.10},
    }
    if done:
        raw_score = 0.40*eng + 0.35*t + 0.25*s
        info["episode_grade"] = {
            "final_score":        round(max(0.0001, min(raw_score, 0.9999)), 4),
            "avg_engagement":     eng,
            "final_trust":        round(t, 4),
            "final_satisfaction": round(s, 4),
        }
    return {"observation": new_obs, "reward": reward, "done": done, "info": info}

# ─────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────

def run_episode(task_id: str, max_steps_override: int = 0, dry_run: bool = False) -> Dict[str, Any]:
    cfg       = TASK_CONFIG[task_id]
    max_steps = max_steps_override or cfg["max_steps"]
    threshold = cfg["success_threshold"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = _fake_reset(task_id) if dry_run else call_reset(task_id)
    except Exception as e:
        print(f"[ERROR] reset failed: {e}\n{traceback.format_exc()}", file=sys.stderr)
        log_end(success=False, steps=0, score=0.0001, rewards=[])
        return {"score": 0.0001, "success": False, "steps": 0, "rewards": [], "episode_grade": {}}

    try:
        obs = reset_data.get("observation", reset_data)
        if not isinstance(obs, dict):
            obs = {}
    except Exception:
        obs = {}

    rewards:       List[float] = []
    step_num:      int         = 0
    done:          bool        = False
    last_reward:   float       = 0.0
    episode_grade: Dict        = {}

    while not done and step_num < max_steps:
        step_num += 1
        error_str: Optional[str] = None

        try:
            if dry_run:
                action = _smart_policy(obs)
            else:
                try:
                    action = call_llm(obs, step_num, task_id, last_reward)
                except Exception as e:
                    print(f"[ERROR] LLM call failed at step {step_num}: {e}\n{traceback.format_exc()}", file=sys.stderr)
                    action = _smart_policy(obs)
                    error_str = f"llm_fallback:{str(e)[:60]}"
        except Exception as e:
            print(f"[ERROR] action decision failed: {e}", file=sys.stderr)
            action = {"action_type": "explore_new_topic", "reasoning": "emergency fallback"}
            error_str = f"action_error:{str(e)[:60]}"

        action_str = _action_str(action)

        try:
            env_action: Dict[str, Any] = {"action_type": action.get("action_type", "explore_new_topic")}
            if action.get("content_id"):
                env_action["content_id"] = action["content_id"]
        except Exception:
            env_action = {"action_type": "explore_new_topic"}

        try:
            result = (
                _fake_step(env_action, step_num, max_steps, obs)
                if dry_run
                else call_step(env_action)
            )
        except Exception as e:
            print(f"[ERROR] step failed at step={step_num}: {e}\n{traceback.format_exc()}", file=sys.stderr)
            log_step(step_num, action_str, 0.0, True, error=f"step_error:{str(e)[:60]}")
            done = True
            break

        try:
            reward      = _float(result.get("reward"), 0.0)
            done        = bool(result.get("done", False))
            new_obs     = result.get("observation", obs)
            obs         = new_obs if isinstance(new_obs, dict) else obs
            info        = result.get("info") or {}
            last_reward = reward

            if done and "episode_grade" in info:
                episode_grade = info["episode_grade"]

            rewards.append(reward)
            log_step(step_num, action_str, reward, done, error=error_str)
        except Exception as e:
            print(f"[ERROR] result parsing failed: {e}", file=sys.stderr)
            log_step(step_num, action_str, 0.0, done, error=f"parse_error:{str(e)[:60]}")

    try:
        if episode_grade and "final_score" in episode_grade:
            score = round(_float(episode_grade["final_score"]), 4)
        else:
            max_total = float(max_steps)
            score = round(min(sum(rewards) / max_total, 1.0), 4) if max_total > 0 else 0.0001
        score   = max(0.0001, min(score, 0.9999))
        success = score >= threshold
    except Exception:
        score, success = 0.0001, False

    log_end(success=success, steps=step_num, score=score, rewards=rewards)
    return {"score": score, "success": success, "steps": step_num,
            "rewards": rewards, "episode_grade": episode_grade}

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AttentionEconomyEnv inference agent")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[DEBUG] ENV_URL={ENV_URL}",             file=sys.stderr)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}",   file=sys.stderr)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}",       file=sys.stderr)
    print(f"[DEBUG] HF_TOKEN={'SET' if HF_TOKEN else 'NOT SET'}", file=sys.stderr)
    print(f"[DEBUG] API_KEY={'SET' if API_KEY else 'NOT SET'}",   file=sys.stderr)

    if not args.dry_run:
        missing = []
        if not API_BASE_URL:
            missing.append("API_BASE_URL")
        if not API_KEY:
            missing.append("HF_TOKEN / API_KEY")
        if missing:
            print(f"[ERROR] Required environment variables not set: {', '.join(missing)}", file=sys.stderr)
            print("[ERROR] Set API_BASE_URL, MODEL_NAME, and HF_TOKEN before running.", file=sys.stderr)
            sys.exit(1)

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    results: Dict[str, Dict] = {}

    for task_id in tasks_to_run:
        try:
            results[task_id] = run_episode(
                task_id=task_id,
                max_steps_override=args.steps,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"[ERROR] run_episode crashed for {task_id}: {e}\n{traceback.format_exc()}", file=sys.stderr)
            results[task_id] = {"score": 0.0001, "success": False, "steps": 0, "rewards": [], "episode_grade": {}}
            log_end(success=False, steps=0, score=0.0001, rewards=[])

    print("\n" + "=" * 62, file=sys.stderr)
    print("  BASELINE SUMMARY", file=sys.stderr)
    print("=" * 62, file=sys.stderr)
    print(f"  {'task':<10} {'score':<10} {'ok':<5} {'steps':<7} {'eng / trust / sat'}", file=sys.stderr)
    print(f"  {'-'*57}", file=sys.stderr)

    for task_id, r in results.items():
        g      = r.get("episode_grade") or {}
        status = "PASS" if r.get("success") else "FAIL"
        detail = (
            f"{_float(g.get('avg_engagement')):.2f} / "
            f"{_float(g.get('final_trust')):.2f} / "
            f"{_float(g.get('final_satisfaction')):.2f}"
            if g else "—"
        )
        print(f"  {task_id:<10} {r.get('score',0):<10.4f} {status:<5} {r.get('steps',0):<7} {detail}", file=sys.stderr)

    overall = sum(r.get("score", 0) for r in results.values()) / max(len(results), 1)
    print(f"\n  Overall avg score: {overall:.4f}", file=sys.stderr)
    print("=" * 62, file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()