from __future__ import annotations

import time

from fastapi.testclient import TestClient

from quant_platform.main import app
from quant_platform.research_layers import (
    LAYER_FUSION_DECISION,
    LAYER_PRICE_SIGNAL,
    LAYER_SENTIMENT_SIGNAL,
)

client = TestClient(app)


def _wait_for_state(path: str, terminal_states: set[str], timeout_seconds: float = 30.0) -> dict:
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


def test_layered_decision_training_and_testing_flow() -> None:
    dataset = client.post("/api/datasets", json={"name": "Layered Stack Dataset"}).json()
    assert dataset["summary"]["news_events"]["rows"] > 0
    feature_set = client.post(
        "/api/features",
        json={"dataset_version_id": dataset["id"], "name": "Layered Stack Feature Set"},
    ).json()
    assert "text_embedding_columns" in feature_set["summary"]["text_embedding_summary"]

    training_response = client.post(
        "/api/training-runs",
        json={
            "dataset_version_id": dataset["id"],
            "feature_set_version_id": feature_set["id"],
            "model_kind": "layered_decision",
            "epochs": 1,
            "name": "Layered Stack Training",
            "layer_configs": {
                "layer_price_signal": {"model_kind": "gru", "hidden_dim": 16},
                "layer_fundamental_signal": {"model_kind": "lightgbm"},
                "layer_sentiment_signal": {"model_kind": "pytorch_mlp", "hidden_dim": 16},
                "layer_macro_regime": {"model_kind": "logistic_fusion"},
                "layer_fusion_decision": {"model_kind": "lightgbm"},
            },
        },
    )
    assert training_response.status_code == 200
    training_run = training_response.json()
    completed_training = _wait_for_state(
        f"/api/training-runs/{training_run['id']}",
        {"completed", "failed", "stopped"},
    )
    assert completed_training["state"] == "completed"

    model_version = client.get("/api/model-versions").json()[0]
    assert model_version["summary"]["model_kind"] == "layered_decision"
    assert LAYER_FUSION_DECISION in model_version["summary"]["layers"]
    assert "candidate_metrics" in model_version["summary"]["layers"][LAYER_PRICE_SIGNAL]
    assert any("text_embedding_" in column for column in model_version["summary"]["layers"][LAYER_SENTIMENT_SIGNAL]["feature_columns"])

    testing_response = client.post(
        "/api/testing-runs",
        json={
            "model_version_id": model_version["id"],
            "feature_set_version_id": feature_set["id"],
            "stress_iterations": 60,
            "decision_top_k": 5,
            "name": "Layered Stack Testing",
        },
    )
    assert testing_response.status_code == 200
    testing_run = testing_response.json()
    completed_testing = _wait_for_state(
        f"/api/testing-runs/{testing_run['id']}",
        {"completed", "failed", "stopped"},
    )
    assert completed_testing["state"] == "completed"

    artifacts = client.get(f"/api/runs/testing/{testing_run['id']}/artifacts").json()
    assert any(artifact["artifact_type"] == "final_decision_report" for artifact in artifacts)

    training_artifacts = client.get(f"/api/runs/training/{training_run['id']}/artifacts").json()
    assert any(
        artifact["artifact_type"] == "comparison_report" and artifact["metadata"]["layer_id"] == LAYER_PRICE_SIGNAL
        for artifact in training_artifacts
    )

    price_observability = client.get(f"/api/research-layers/{LAYER_PRICE_SIGNAL}/observability").json()
    assert "rank_ic" in price_observability["latest_metrics"]

    sentiment_observability = client.get(f"/api/research-layers/{LAYER_SENTIMENT_SIGNAL}/observability").json()
    assert "directional_accuracy" in sentiment_observability["latest_metrics"]

    fusion_observability = client.get(f"/api/research-layers/{LAYER_FUSION_DECISION}/observability").json()
    assert "rank_ic" in fusion_observability["latest_metrics"]
    assert "final_decision_report" in fusion_observability["latest_artifacts"]
