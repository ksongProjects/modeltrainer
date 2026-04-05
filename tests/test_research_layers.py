from __future__ import annotations

import time

from fastapi.testclient import TestClient

from quant_platform.main import app
from quant_platform.research_layers import (
    LAYER_EXECUTION_POLICY,
    LAYER_FEATURE_STORE,
    LAYER_FUSION_DECISION,
    LAYER_PORTFOLIO_CONSTRUCTION,
    LAYER_PRICE_SIGNAL,
    LAYER_SENTIMENT_SIGNAL,
    LAYER_SNAPSHOT_SIGNAL,
)

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


def test_research_architecture_exposes_layer_contracts() -> None:
    response = client.get("/api/research-architecture")
    assert response.status_code == 200

    payload = response.json()
    layer_ids = {layer["id"] for layer in payload["layers"]}
    assert LAYER_SNAPSHOT_SIGNAL in layer_ids
    assert LAYER_FUSION_DECISION in layer_ids

    snapshot_layer = next(layer for layer in payload["layers"] if layer["id"] == LAYER_SNAPSHOT_SIGNAL)
    assert snapshot_layer["data_contract"]["input_columns"]
    assert snapshot_layer["control_surface"]["tunable_parameters"]
    assert snapshot_layer["latest_observability"]["latest_metrics"] is not None


def test_research_layer_controls_are_persisted() -> None:
    response = client.post(
        f"/api/research-layers/{LAYER_PRICE_SIGNAL}/controls",
        json={
            "preferred_model_kind": "lightgbm",
            "candidate_model_kinds": ["lightgbm", "pytorch_mlp"],
            "process_step_state": {
                "include_cross_signal_composite": False,
                "include_volume_context": True,
            },
            "runtime_settings": {
                "compute_target": "cpu",
                "precision_mode": "fp32",
                "batch_size": 48,
                "sequence_length": 18,
                "gradient_clip_norm": 0.8,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["control_state"]["preferred_model_kind"] == "lightgbm"
    assert payload["control_state"]["candidate_model_kinds"] == ["lightgbm", "pytorch_mlp"]
    assert payload["control_state"]["process_step_state"]["include_cross_signal_composite"] is False
    assert payload["control_state"]["runtime_settings"]["compute_target"] == "cpu"
    assert payload["control_state"]["runtime_settings"]["sequence_length"] == 18

    feature_layer = client.get(f"/api/research-layers/{LAYER_FEATURE_STORE}").json()
    assert any(step["id"] == "winsorize_value_factor" for step in feature_layer["process_steps"])
    sentiment_layer = client.get(f"/api/research-layers/{LAYER_SENTIMENT_SIGNAL}").json()
    assert any(step["id"] == "include_text_embeddings" for step in sentiment_layer["process_steps"])


def test_runtime_capabilities_endpoint_exposes_local_backends() -> None:
    response = client.get("/api/runtime-capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommended_compute_target"] in {"cpu", "cuda", "directml"}
    assert any(device["kind"] == "cpu" for device in payload["devices"])
    assert "supported_compute_targets" in payload


def test_research_layer_observability_tracks_latest_runs() -> None:
    dataset_response = client.post("/api/datasets", json={"name": "Layer Obs Dataset"})
    assert dataset_response.status_code == 200
    dataset = dataset_response.json()

    feature_response = client.post(
        "/api/features",
        json={"dataset_version_id": dataset["id"], "name": "Layer Obs Feature Set"},
    )
    assert feature_response.status_code == 200
    feature_set = feature_response.json()

    training_response = client.post(
        "/api/training-runs",
        json={
            "dataset_version_id": dataset["id"],
            "feature_set_version_id": feature_set["id"],
            "model_kind": "lightgbm",
            "epochs": 1,
            "name": "Layer Obs Training",
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

    testing_response = client.post(
        "/api/testing-runs",
        json={
            "model_version_id": model_version["id"],
            "feature_set_version_id": feature_set["id"],
            "stress_iterations": 60,
            "name": "Layer Obs Testing",
        },
    )
    assert testing_response.status_code == 200
    testing_run = testing_response.json()
    completed_testing = _wait_for_state(
        f"/api/testing-runs/{testing_run['id']}",
        {"completed", "failed", "stopped"},
    )
    assert completed_testing["state"] == "completed"

    snapshot_observability = client.get(f"/api/research-layers/{LAYER_SNAPSHOT_SIGNAL}/observability").json()
    assert snapshot_observability["latest_run"]["id"] == training_run["id"]
    assert "rank_ic" in snapshot_observability["latest_metrics"]

    portfolio_observability = client.get(f"/api/research-layers/{LAYER_PORTFOLIO_CONSTRUCTION}/observability").json()
    assert portfolio_observability["latest_run"]["id"] == testing_run["id"]
    assert "sharpe" in portfolio_observability["latest_metrics"]

    execution_observability = client.get(f"/api/research-layers/{LAYER_EXECUTION_POLICY}/observability").json()
    assert execution_observability["latest_run"]["id"] == testing_run["id"]
    assert "avg_slippage_bps" in execution_observability["latest_metrics"]
