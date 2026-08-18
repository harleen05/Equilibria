from fastapi.testclient import TestClient

from server.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_reset_then_step_then_state_cycle() -> None:
    client = TestClient(app)

    reset_response = client.post("/reset", json={"task": "easy"})
    assert reset_response.status_code == 200
    reset_body = reset_response.json()
    assert reset_body["observation"]["task_id"] == "easy"
    assert reset_body["observation"]["step_count"] == 0
    assert "session_id" in reset_body

    step_response = client.post(
        "/step",
        json={"action": {"action_type": "recommend", "content_id": "rel_tech_01"}},
    )
    assert step_response.status_code == 200
    step_body = step_response.json()
    assert step_body["reward"] >= 0.0001
    assert step_body["observation"]["step_count"] == 1
    assert "session_id" in step_body

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert state_response.json()["step"] == 1


def test_step_before_reset_returns_400() -> None:
    client = TestClient(app)
    response = client.post("/step", json={"action": {"action_type": "pause_session"}})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "NOT_RESET"


def test_observation_endpoint() -> None:
    client = TestClient(app)
    client.post("/reset", json={"task": "easy"})

    obs_response = client.get("/observation")
    assert obs_response.status_code == 200
    body = obs_response.json()
    assert body["observation"]["task_id"] == "easy"
    assert body["done"] is False


def test_session_isolation() -> None:
    client = TestClient(app)

    r1 = client.post(
        "/reset",
        json={"task": "easy"},
        headers={"X-Session-Id": "session-a"},
    )
    r2 = client.post(
        "/reset",
        json={"task": "hard"},
        headers={"X-Session-Id": "session-b"},
    )

    assert r1.json()["observation"]["task_id"] == "easy"
    assert r2.json()["observation"]["task_id"] == "hard"

    obs_a = client.get("/observation", headers={"X-Session-Id": "session-a"})
    obs_b = client.get("/observation", headers={"X-Session-Id": "session-b"})

    assert obs_a.json()["observation"]["task_id"] == "easy"
    assert obs_b.json()["observation"]["task_id"] == "hard"


def test_step_heuristic() -> None:
    client = TestClient(app)
    client.post("/reset", json={"task": "easy"})

    response = client.post("/step/heuristic")
    assert response.status_code == 200
    body = response.json()
    assert body["policy"] == "heuristic"
    assert body["policy_action"]["action_type"] in (
        "recommend",
        "pause_session",
        "diversify_feed",
        "explore_new_topic",
    )
    assert body["observation"]["step_count"] == 1


def test_policies_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/policies", params={"task": "easy"})
    assert response.status_code == 200
    body = response.json()
    assert body["heuristic"] is True
    assert "ppo" in body


def test_structured_episode_done_error() -> None:
    client = TestClient(app)
    client.post("/reset", json={"task": "easy"})

    for _ in range(50):
        resp = client.post(
            "/step",
            json={"action": {"action_type": "pause_session"}},
        )
        if resp.status_code != 200:
            break

    final = client.post("/step", json={"action": {"action_type": "pause_session"}})
    assert final.status_code == 400
    assert final.json()["detail"]["code"] == "EPISODE_DONE"

def test_reset_unknown_task_returns_400() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"task": "esay"})  # typo, not "easy"

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UNKNOWN_TASK"


def test_policies_unknown_task_returns_400() -> None:
    client = TestClient(app)
    response = client.get("/policies", params={"task": "not_a_real_task"})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UNKNOWN_TASK"


def test_reset_accepts_documented_task_aliases() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"task": "trust_preservation"})

    assert response.status_code == 200
    assert response.json()["observation"]["task_id"] == "hard"

def test_reset_default_observability_is_oracle() -> None:
    """No observability param -> defaults to oracle, matching pre-existing
    behavior exactly (backward compatible)."""
    client = TestClient(app)
    response = client.post("/reset", json={"task": "easy"})

    assert response.status_code == 200
    content = response.json()["observation"]["available_content"]
    scores = {item["manipulation_score"] for item in content}
    # oracle mode -> real, varied scores across items (not all masked to one value)
    assert len(scores) > 1


def test_reset_partial_observability_masks_manipulation_score() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"task": "easy", "observability": "partial"})

    assert response.status_code == 200
    content = response.json()["observation"]["available_content"]
    scores = {item["manipulation_score"] for item in content}
    # partial mode -> every item shows the identical masked constant
    assert len(scores) == 1


def test_reset_invalid_observability_returns_400() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"task": "easy", "observability": "omniscient"})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_OBSERVABILITY"