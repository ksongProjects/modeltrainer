from __future__ import annotations

import time

from fastapi.testclient import TestClient

from quant_platform.main import app


client = TestClient(app)


def _wait_for_state(path: str, terminal_states: set[str], timeout_seconds: float = 25.0) -> dict:
    deadline = time.time() + timeout_seconds
    latest = {}
    while time.time() < deadline:
        response = client.get(path)
        response.raise_for_status()
        latest = response.json()
        if latest["state"] in terminal_states:
            return latest
        time.sleep(0.5)
    raise AssertionError(f"Run did not reach terminal state before timeout: {latest}")


def test_end_to_end_training_and_testing_flow() -> None:
    dataset_response = client.post("/api/datasets", json={})
    assert dataset_response.status_code == 200
    dataset = dataset_response.json()

    feature_response = client.post("/api/features", json={"dataset_version_id": dataset["id"]})
    assert feature_response.status_code == 200
    feature_set = feature_response.json()

    training_response = client.post(
        "/api/training-runs",
        json={
            "dataset_version_id": dataset["id"],
            "feature_set_version_id": feature_set["id"],
            "model_kind": "lightgbm",
            "epochs": 1,
            "name": "API Smoke",
        },
    )
    assert training_response.status_code == 200
    training_run = training_response.json()
    completed_training = _wait_for_state(f"/api/training-runs/{training_run['id']}", {"completed", "failed", "stopped"})
    assert completed_training["state"] == "completed"

    model_versions = client.get("/api/model-versions").json()
    assert model_versions
    model_version = model_versions[0]

    testing_response = client.post(
        "/api/testing-runs",
        json={
            "model_version_id": model_version["id"],
            "feature_set_version_id": feature_set["id"],
            "stress_iterations": 60,
        },
    )
    assert testing_response.status_code == 200
    testing_run = testing_response.json()
    completed_testing = _wait_for_state(f"/api/testing-runs/{testing_run['id']}", {"completed", "failed", "stopped"})
    assert completed_testing["state"] == "completed"

    events = client.get(f"/api/runs/testing/{testing_run['id']}/events").json()
    metrics = client.get(f"/api/runs/testing/{testing_run['id']}/metrics").json()
    traces = client.get(f"/api/runs/testing/{testing_run['id']}/traces").json()
    assert events
    assert any(metric["name"] == "sharpe" for metric in metrics)
    assert traces
