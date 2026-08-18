"""
tests/test_inference.py — Verifies inference.py, the harness that drives a
real LLM as a policy against the live server (via HTTP /reset and /step)
and computes a comparable episode_grade -- the mechanism behind the
"compare an LLM agent against PPO/heuristic" checklist item.

Before this file, inference.py sat at 49% coverage AND was silently
excluded from every coverage report this whole project's pytest config
produces, since pyproject.toml's addopts only tracked --cov=environment.
inference.py lives at the repo root, not under environment/, so nobody
would notice a regression here without explicitly targeting it -- see the
pyproject.toml change alongside this file.

The most safety-critical function tested here is _parse_llm_response():
if it silently breaks, every LLM-driven run degrades to the heuristic
fallback (smart_policy_action) with NO visible signal that the LLM's
response was never actually used -- the episode still "succeeds", just
not with the intended policy. These tests lock in every parsing path.
"""

from __future__ import annotations

from inference import (
    _parse_llm_response, _action_str, _f, _float, run_episode,
)


AVAILABLE = {
    "available_content": [
        {"content_id": "rel_tech_01"},
        {"content_id": "rel_sci_01"},
    ]
}


# ── _parse_llm_response: the safety-critical parsing chain ────────────────

def test_parse_clean_json_recommend() -> None:
    raw = '{"action_type": "recommend", "content_id": "rel_tech_01", "reasoning": "good match"}'
    result = _parse_llm_response(raw, AVAILABLE)
    assert result == {
        "action_type": "recommend", "content_id": "rel_tech_01", "reasoning": "good match",
    }


def test_parse_clean_json_meta_action() -> None:
    raw = '{"action_type": "pause_session", "reasoning": "tired"}'
    result = _parse_llm_response(raw, AVAILABLE)
    assert result == {"action_type": "pause_session", "reasoning": "tired"}


def test_parse_code_fenced_json() -> None:
    """LLMs very commonly wrap JSON in ```json ... ``` fences despite being
    told not to -- this must still parse correctly."""
    raw = '```json\n{"action_type": "pause_session", "reasoning": "tired"}\n```'
    result = _parse_llm_response(raw, AVAILABLE)
    assert result["action_type"] == "pause_session"


def test_parse_recommend_with_unavailable_content_id_falls_back() -> None:
    """A content_id not in available_content must NOT be trusted --
    falls through to the heuristic fallback rather than sending an
    invalid action to the server."""
    raw = '{"action_type": "recommend", "content_id": "nonexistent_id"}'
    result = _parse_llm_response(raw, AVAILABLE)
    assert result["action_type"] == "recommend"
    assert result["content_id"] in {"rel_tech_01", "rel_sci_01"}
    assert result["reasoning"].startswith("fallback:")


def test_parse_invalid_action_type_falls_back() -> None:
    raw = '{"action_type": "explode", "reasoning": "chaos"}'
    result = _parse_llm_response(raw, AVAILABLE)
    assert result["reasoning"].startswith("fallback:")


def test_parse_regex_extraction_fallback() -> None:
    """Not valid JSON, but content_id is regex-extractable from prose."""
    raw = 'I choose "content_id": "rel_sci_01" because it is safe'
    result = _parse_llm_response(raw, AVAILABLE)
    assert result == {
        "action_type": "recommend", "content_id": "rel_sci_01", "reasoning": "regex-extracted",
    }


def test_parse_substring_extraction_fallback() -> None:
    """Not valid JSON, no quoted content_id key, but a known safe content
    ID appears as bare text in the response."""
    raw = "I will pick rel_tech_01 for this user"
    result = _parse_llm_response(raw, AVAILABLE)
    assert result == {
        "action_type": "recommend", "content_id": "rel_tech_01", "reasoning": "substring-extracted",
    }


def test_parse_complete_garbage_falls_back_to_heuristic() -> None:
    raw = "asdkjaslkdj not json at all"
    result = _parse_llm_response(raw, AVAILABLE)
    assert result["reasoning"].startswith("fallback:")
    assert result["action_type"] in (
        "recommend", "explore_new_topic", "diversify_feed", "pause_session"
    )


def test_parse_empty_string_falls_back() -> None:
    result = _parse_llm_response("", AVAILABLE)
    assert result["reasoning"].startswith("fallback:")


# ── Helper functions ───────────────────────────────────────────────────────

def test_action_str_recommend() -> None:
    assert _action_str({"action_type": "recommend", "content_id": "x1"}) == "recommend(content_id=x1)"


def test_action_str_other_action() -> None:
    assert _action_str({"action_type": "pause_session"}) == "pause_session"


def test_action_str_malformed_input_does_not_crash() -> None:
    assert _action_str(None) == "unknown"
    assert _action_str({}) == "unknown"


def test_f_reads_dict_values() -> None:
    assert _f({"content_id": "x1"}, "content_id") == "x1"
    assert _f({}, "missing", "default") == "default"


def test_float_conversion() -> None:
    assert _float("0.5") == 0.5
    assert _float(None, default=1.0) == 1.0
    assert _float("not_a_number", default=9.9) == 9.9


# ── Dry-run structural tests (existing + expanded across all tasks) ────────

def test_inference_dry_run_easy() -> None:
    result = run_episode("easy", dry_run=True)

    assert isinstance(result, dict)
    assert result["steps"] > 0
    assert 0.0001 <= result["score"] <= 0.9999
    assert isinstance(result["episode_grade"], dict)
    assert "final_score" in result["episode_grade"]
    assert isinstance(result["success"], bool)


def test_inference_dry_run_all_tasks() -> None:
    """
    NOTE: dry-run mode's rewards come from _fake_step(), which uses
    rng.uniform(0.35, 0.75) -- pure random noise, NOT the real reward
    function in environment/reward.py. This test only verifies the
    harness's PLUMBING (loop, logging, episode_grade structure) works
    across all three tasks -- it is not a substantive evaluation. Getting
    a real LLM-vs-PPO comparison requires running against a live server
    with a real API key (dry_run=False); see inference.py's module
    docstring / README for that invocation.
    """
    for task in ("easy", "medium", "hard"):
        result = run_episode(task, dry_run=True)
        assert result["steps"] > 0
        assert 0.0001 <= result["score"] <= 0.9999


def test_cli_entrypoint_dry_run_runs_end_to_end() -> None:
    """
    Subprocess-level smoke test for main(): catches import errors, syntax
    issues, or CLI argument-parsing regressions that a direct function
    call (run_episode) wouldn't exercise, since it goes through the real
    `python inference.py ...` invocation path a person would actually run.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "inference.py", "--task", "easy", "--dry-run"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "[START]" in result.stdout
    assert "[END]" in result.stdout
    assert "BASELINE SUMMARY" in result.stderr