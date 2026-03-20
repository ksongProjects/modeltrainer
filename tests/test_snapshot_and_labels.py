from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from markets_pipeline.contracts.findf import ArtifactPaths, FindfRunManifest
from markets_pipeline.datasets.snapshots import build_snapshot_daily
from markets_pipeline.features.sentiment import build_sentiment_features
from markets_pipeline.settings import Settings


def make_settings(project_root: Path, findf_root: Path) -> Settings:
    artifacts_dir = project_root / "artifacts"
    settings = Settings(
        project_root=project_root,
        findf_root=findf_root,
        artifacts_dir=artifacts_dir,
        manifests_dir=artifacts_dir / "manifests",
        datasets_dir=artifacts_dir / "datasets",
        reports_dir=artifacts_dir / "reports",
        models_dir=artifacts_dir / "models",
        registry_dir=artifacts_dir / "registry",
        configs_dir=project_root / "configs",
    )
    settings.ensure_directories()
    return settings


def write_configs(project_root: Path) -> None:
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "feature_view.json").write_text(
        json.dumps(
            {
                "version": "v1_daily_specialists",
                "base_frequency": "1d",
                "price_windows": [5, 10, 20, 60],
                "sentiment_windows": [1, 3, 7, 30],
                "buy_threshold": 0.6,
                "sell_threshold": 0.4,
            }
        ),
        encoding="utf-8",
    )
    (configs_dir / "horizons.json").write_text(
        json.dumps({"horizons": [{"name": "1d", "days": 1}, {"name": "5d", "days": 5}, {"name": "10d", "days": 10}]}),
        encoding="utf-8",
    )
    (configs_dir / "folds.json").write_text(json.dumps({"folds": []}), encoding="utf-8")


def test_sentiment_features_are_trailing_only() -> None:
    price_index = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    scored_news = pd.DataFrame(
        {
            "id": ["n1"],
            "published_at": ["2024-01-02T10:00:00Z"],
            "tickers": ["AAPL"],
            "sentiment_score": [0.8],
            "prob_negative": [0.1],
            "prob_neutral": [0.1],
            "prob_positive": [0.8],
            "embedding_norm": [1.0],
            "embedding_mean": [0.0],
            "embedding_std": [0.1],
        }
    )

    features = build_sentiment_features(price_index, scored_news, [1, 3])
    day1 = features.loc[features["trade_date"] == pd.Timestamp("2024-01-01")].iloc[0]
    day2 = features.loc[features["trade_date"] == pd.Timestamp("2024-01-02")].iloc[0]
    day3 = features.loc[features["trade_date"] == pd.Timestamp("2024-01-03")].iloc[0]

    assert day1["news_count_1d"] == 0.0
    assert day2["news_count_1d"] == 1.0
    assert day3["news_count_3d"] == 1.0


def test_snapshot_builder_forward_fills_macro_and_generates_labels(tmp_path: Path) -> None:
    project_root = tmp_path / "trainer"
    findf_root = tmp_path / "findf"
    silver_dir = findf_root / "data" / "silver"
    silver_dir.mkdir(parents=True)
    write_configs(project_root)

    job_id = "job456"
    prices = pd.DataFrame(
        [
            {"ticker": "AAPL", "timestamp": "2024-01-02", "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.0, "volume": 100.0, "adj_close": 10.0, "source": "polygon"},
            {"ticker": "AAPL", "timestamp": "2024-01-03", "open": 10.5, "high": 11.5, "low": 10.0, "close": 11.0, "volume": 120.0, "adj_close": 11.0, "source": "polygon"},
            {"ticker": "AAPL", "timestamp": "2024-01-04", "open": 11.0, "high": 12.0, "low": 10.5, "close": 12.0, "volume": 130.0, "adj_close": 12.0, "source": "polygon"},
        ]
    )
    prices.to_parquet(silver_dir / f"{job_id}_prices.parquet", index=False)
    pd.DataFrame(
        columns=["id", "title", "summary", "tickers", "published_at", "source", "url"]
    ).to_parquet(silver_dir / f"{job_id}_news.parquet", index=False)
    macro = pd.DataFrame(
        [
            {"series_id": "FEDFUNDS", "date": "2024-01-03", "value": 5.0, "source": "fred"},
        ]
    )
    macro.to_parquet(silver_dir / f"{job_id}_macro.parquet", index=False)
    pd.DataFrame(columns=["ticker", "date", "metric", "value", "source"]).to_parquet(
        silver_dir / f"{job_id}_fundamentals.parquet",
        index=False,
    )

    settings = make_settings(project_root, findf_root)
    snapshot_version = f"{job_id}_v1_daily_specialists"
    snapshot_dir = settings.datasets_dir / snapshot_version
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "id",
            "published_at",
            "tickers",
            "sentiment_score",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
            "embedding_norm",
            "embedding_mean",
            "embedding_std",
        ]
    ).to_parquet(snapshot_dir / "news_scores.parquet", index=False)

    manifest = FindfRunManifest(
        job_id=job_id,
        tickers=["AAPL"],
        start_date="2024-01-02",
        end_date="2024-01-04",
        providers={},
        artifact_paths=ArtifactPaths(
            prices_silver=silver_dir / f"{job_id}_prices.parquet",
            news_silver=silver_dir / f"{job_id}_news.parquet",
            macro_silver=silver_dir / f"{job_id}_macro.parquet",
            fundamentals_silver=silver_dir / f"{job_id}_fundamentals.parquet",
        ),
    )

    result = build_snapshot_daily(settings, manifest, settings.load_feature_view())
    snapshot = pd.read_parquet(result.snapshot_path)
    snapshot["trade_date"] = pd.to_datetime(snapshot["trade_date"])

    jan2 = snapshot.loc[snapshot["trade_date"] == pd.Timestamp("2024-01-02")].iloc[0]
    jan4 = snapshot.loc[snapshot["trade_date"] == pd.Timestamp("2024-01-04")].iloc[0]

    assert pd.isna(jan2["fedfunds"])
    assert jan4["fedfunds"] == 5.0
    assert jan2["forward_return_1d"] == 0.1
    assert jan2["target_up_1d"] == 1.0
