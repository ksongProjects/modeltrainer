from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from fastapi.testclient import TestClient

from quant_platform.main import app
from tests.support import write_news_events_parquet, write_realistic_parquet


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

def _write_parquet_with_gaps_and_duplicates(path: Path) -> None:
    write_realistic_parquet(path)
    frame = pd.read_parquet(path)
    missing_row_index = frame.index[(frame["ticker"] == "BBB") & (frame["effective_at"] == "2025-01-16T00:00:00")][0]
    duplicate_row = frame.loc[(frame["ticker"] == "AAA") & (frame["effective_at"] == "2025-01-21T00:00:00")].iloc[[0]]
    adjusted = pd.concat([frame.drop(index=missing_row_index), duplicate_row], ignore_index=True)
    adjusted.to_parquet(path, index=False)


def test_import_parquet_dataset_and_materialize_features(tmp_path: Path) -> None:
    parquet_path = tmp_path / "findf_snapshot.parquet"
    write_realistic_parquet(parquet_path)
    write_news_events_parquet(tmp_path / "news_events.parquet")

    saved_tag_response = client.post("/api/dataset-tags", json={"name": "production"})
    assert saved_tag_response.status_code == 200
    saved_tag = saved_tag_response.json()

    duplicate_tag_response = client.post("/api/dataset-tags", json={"name": " Production "})
    assert duplicate_tag_response.status_code == 200
    assert duplicate_tag_response.json()["id"] == saved_tag["id"]

    import_response = client.post(
        "/api/datasets/import-parquet",
        json={"path": str(parquet_path), "name": "findf import", "tags": ["production", "large cap", "production", " "]},
    )
    assert import_response.status_code == 200
    dataset = import_response.json()
    assert dataset["source_id"] == "source_findf_parquet"
    assert dataset["manual_tags"] == ["production", "large cap"]
    assert {"ohlcv", "fundamental", "momentum", "sentiment", "macro", "event", "multi-ticker", "micro-universe", "multi-sector", "parquet-import"}.issubset(set(dataset["auto_tags"]))
    assert {"ticker:AAA", "ticker:BBB", "ticker:CCC", "ticker:DDD"}.issubset(set(dataset["auto_tags"]))
    assert {"production", "large cap"}.issubset(set(dataset["tags"]))
    assert dataset["summary"]["artifacts"]["source_path"] == str(parquet_path.resolve())
    assert dataset["summary"]["sample_tickers"] == ["AAA", "BBB", "CCC", "DDD"]
    assert dataset["summary"]["assessment"]["data_level"] == "high"
    assert dataset["summary"]["assessment"]["status"] == "healthy"
    assert dataset["summary"]["assessment"]["gaps"]["missing_sessions"] == 0
    assert dataset["summary"]["assessment"]["gaps"]["duplicate_key_rows"] == 0
    assert dataset["summary"]["news_events"]["rows"] > 0

    saved_tags = client.get("/api/dataset-tags")
    assert saved_tags.status_code == 200
    assert any(str(tag["name"]).lower() == "production" for tag in saved_tags.json())

    datasets = client.get("/api/datasets")
    assert datasets.status_code == 200
    imported_dataset = next(item for item in datasets.json() if item["id"] == dataset["id"])
    assert imported_dataset["manual_tags"] == ["production", "large cap"]
    assert {"production", "ohlcv", "ticker:AAA"}.issubset(set(imported_dataset["tags"]))

    feature_response = client.post("/api/features", json={"dataset_version_id": dataset["id"]})
    assert feature_response.status_code == 200
    feature_set = feature_response.json()
    assert feature_set["summary"]["rows"] > 0
    assert "composite_score" in feature_set["summary"]["feature_columns"]

    delete_tag_response = client.delete(f"/api/dataset-tags/{saved_tag['id']}")
    assert delete_tag_response.status_code == 200
    remaining_tags = client.get("/api/dataset-tags")
    assert remaining_tags.status_code == 200
    assert all(tag["id"] != saved_tag["id"] for tag in remaining_tags.json())
    persisted_dataset = next(item for item in client.get("/api/datasets").json() if item["id"] == dataset["id"])
    assert persisted_dataset["manual_tags"] == ["production", "large cap"]
    assert {"large cap", "macro", "ticker:DDD"}.issubset(set(persisted_dataset["tags"]))


def test_import_parquet_dataset_surfaces_gaps_and_duplicates(tmp_path: Path) -> None:
    parquet_path = tmp_path / "findf_snapshot_with_gaps.parquet"
    _write_parquet_with_gaps_and_duplicates(parquet_path)

    import_response = client.post(
        "/api/datasets/import-parquet",
        json={"path": str(parquet_path), "name": "findf import with gaps"},
    )
    assert import_response.status_code == 200
    dataset = import_response.json()
    assessment = dataset["summary"]["assessment"]

    assert assessment["data_level"] == "medium"
    assert assessment["status"] == "warning"
    assert assessment["gaps"]["missing_sessions"] == 1
    assert assessment["gaps"]["duplicate_key_rows"] == 1
    assert assessment["gaps"]["instruments_with_gaps"] == 1
    assert any(issue["title"] == "Missing trading sessions" for issue in assessment["issues"])
    assert any(issue["title"] == "Duplicate entity-date keys" for issue in assessment["issues"])


def test_synthetic_dataset_creation_is_disabled() -> None:
    dataset_response = client.post("/api/datasets", json={})
    assert dataset_response.status_code == 400
    assert "disabled" in dataset_response.json()["detail"].lower()


def test_end_to_end_training_and_testing_flow(tmp_path: Path) -> None:
    parquet_path = tmp_path / "e2e_snapshot.parquet"
    write_realistic_parquet(parquet_path)
    write_news_events_parquet(tmp_path / "news_events.parquet")

    dataset_response = client.post(
        "/api/datasets/import-parquet",
        json={"path": str(parquet_path), "name": "E2E Import"},
    )
    assert dataset_response.status_code == 200
    dataset = dataset_response.json()
    assert {"parquet-import", "ohlcv", "fundamental", "momentum", "sentiment", "macro", "event", "multi-ticker"}.issubset(set(dataset["tags"]))
    assert dataset["summary"]["assessment"]["data_level"] == "high"

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

    visualization = client.get(
        f"/api/datasets/{dataset['id']}/visualization",
        params={
            "ticker": "AAA",
            "feature_set_version_id": feature_set["id"],
            "model_version_id": model_version["id"],
        },
    )
    assert visualization.status_code == 200
    payload = visualization.json()
    assert payload["ticker"] == "AAA"
    assert payload["price_series"]
    assert payload["prediction_series"]
    assert any(item["event_scope"] == "ticker" for item in payload["news_events"])
    assert any(item["category"] == "earnings" for item in payload["event_markers"])


def test_runtime_self_check_endpoint_runs_smoke_probe() -> None:
    response = client.post(
        "/api/runtime-self-check",
        json={
            "compute_target": "cpu",
            "precision_mode": "fp32",
            "batch_size": 16,
            "sequence_length": 12,
            "gradient_clip_norm": 1.0,
            "model_kind": "pytorch_mlp",
            "input_dim": 8,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved_runtime"]["resolved_compute_target"] == "cpu"
    assert "tensor_allocation" in payload["checks"]
    assert "elapsed_ms" in payload["metrics"]
