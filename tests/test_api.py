from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

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


def _write_realistic_parquet(path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=140)
    tickers = [
        ("AAA", "eq_0001", "Technology", 80.0),
        ("BBB", "eq_0002", "Financials", 42.0),
        ("CCC", "eq_0003", "Health Care", 61.0),
        ("DDD", "eq_0004", "Industrials", 55.0),
    ]
    rows: list[dict[str, object]] = []
    for ticker_index, (ticker, entity_id, sector, base_price) in enumerate(tickers):
        for day_index, effective_at in enumerate(dates):
            close = base_price + day_index * (0.14 + ticker_index * 0.02)
            rows.append(
                {
                    "entity_id": entity_id,
                    "ticker": ticker,
                    "sector": sector,
                    "effective_at": effective_at.isoformat(),
                    "known_at": (effective_at + pd.Timedelta(hours=16, minutes=5)).isoformat(),
                    "ingested_at": (effective_at + pd.Timedelta(hours=18)).isoformat(),
                    "source_version": "findf_v1",
                    "open": close - 0.35,
                    "high": close + 0.7,
                    "low": close - 0.9,
                    "close": close,
                    "volume": 120_000 + (day_index * 900) + (ticker_index * 1_500),
                    "ev_ebitda": 8.0 + ticker_index + (day_index % 5) * 0.15,
                    "roic": 0.08 + ticker_index * 0.01 + (day_index % 7) * 0.001,
                    "momentum_20d": (day_index % 20) * 0.004,
                    "momentum_60d": (day_index % 60) * 0.002,
                    "sentiment_1d": ((day_index % 5) - 2) * 0.05,
                    "sentiment_5d": ((day_index % 7) - 3) * 0.04,
                    "macro_surprise": ((day_index % 9) - 4) * 0.02,
                    "earnings_signal": 0.06 if day_index % 30 == 0 else 0.0,
                    "macro_rate": 4.5,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_import_parquet_dataset_and_materialize_features(tmp_path: Path) -> None:
    parquet_path = tmp_path / "findf_snapshot.parquet"
    _write_realistic_parquet(parquet_path)

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


def test_end_to_end_training_and_testing_flow() -> None:
    dataset_response = client.post("/api/datasets", json={})
    assert dataset_response.status_code == 200
    dataset = dataset_response.json()
    assert {"synthetic", "ohlcv", "fundamental", "momentum", "sentiment", "macro", "event", "multi-ticker"}.issubset(set(dataset["tags"]))

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
