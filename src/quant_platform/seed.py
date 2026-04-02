from __future__ import annotations

import json
from uuid import uuid4

from .database import connect, utcnow


def _seed_table(connection, table: str, rows: list[dict[str, str]]) -> None:
    existing = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
    if existing:
        return
    for row in rows:
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        connection.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(row.values()),
        )


def seed_defaults() -> None:
    now = utcnow()
    with connect() as connection:
        _seed_table(
            connection,
            "data_sources",
            [
                {
                    "id": "source_synthetic",
                    "name": "Synthetic Daily + Events",
                    "kind": "synthetic",
                    "description": "Deterministic local dataset for OHLCV, events, macro, and factor demos.",
                    "config_json": json.dumps({"granularity": "daily", "events": True}),
                    "created_at": now,
                },
                {
                    "id": "source_findf_parquet",
                    "name": "findf Parquet PIT Import",
                    "kind": "parquet_import",
                    "description": "Local parquet import path for point-in-time datasets emitted by findf.",
                    "config_json": json.dumps({"format": "parquet", "ingest_mode": "local_path"}),
                    "created_at": now,
                }
            ],
        )
        _seed_table(
            connection,
            "universe_definitions",
            [
                {
                    "id": "universe_sp500",
                    "name": "S&P 500",
                    "description": "Built-in large-cap universe.",
                    "config_json": json.dumps({"benchmark": "SPY", "size": 500}),
                    "created_at": now,
                },
                {
                    "id": "universe_r1000",
                    "name": "Russell 1000",
                    "description": "Built-in broad US universe.",
                    "config_json": json.dumps({"benchmark": "IWB", "size": 1000}),
                    "created_at": now,
                },
                {
                    "id": "universe_custom",
                    "name": "Custom Watchlist",
                    "description": "User-managed single-tenant watchlist universe.",
                    "config_json": json.dumps({"editable": True}),
                    "created_at": now,
                },
            ],
        )
        _seed_table(
            connection,
            "factor_definitions",
            [
                {
                    "id": "factor_value",
                    "name": "Value",
                    "category": "value",
                    "formula": "-zscore(ev_ebitda)",
                    "config_json": json.dumps({"weights": {"composite": 0.3}, "eligibility": "all"}),
                    "created_at": now,
                },
                {
                    "id": "factor_quality",
                    "name": "Quality",
                    "category": "quality",
                    "formula": "sector_zscore(roic)",
                    "config_json": json.dumps({"weights": {"composite": 0.25}, "eligibility": "all"}),
                    "created_at": now,
                },
                {
                    "id": "factor_momentum",
                    "name": "Momentum",
                    "category": "momentum",
                    "formula": "zscore(momentum_20d)",
                    "config_json": json.dumps({"weights": {"composite": 0.2}, "eligibility": "all"}),
                    "created_at": now,
                },
                {
                    "id": "factor_sentiment",
                    "name": "NLP Sentiment",
                    "category": "alt_data",
                    "formula": "zscore(sentiment_5d)",
                    "config_json": json.dumps({"weights": {"composite": 0.15}, "eligibility": "news_available"}),
                    "created_at": now,
                },
                {
                    "id": "factor_macro",
                    "name": "Macro Shock",
                    "category": "macro",
                    "formula": "zscore(macro_shock_3d)",
                    "config_json": json.dumps({"weights": {"composite": 0.1}, "eligibility": "all"}),
                    "created_at": now,
                },
            ],
        )
        _seed_table(
            connection,
            "model_specs",
            [
                {
                    "id": "spec_lightgbm",
                    "name": "Gradient Boosted Snapshot",
                    "kind": "lightgbm",
                    "description": "Primary tabular baseline with LightGBM-style interface and sklearn fallback.",
                    "config_json": json.dumps({"task": "regression", "family": "tree"}),
                    "created_at": now,
                },
                {
                    "id": "spec_logistic_fusion",
                    "name": "Logistic Fusion",
                    "kind": "logistic_fusion",
                    "description": "Meta-model for combining tabular and event-driven signals.",
                    "config_json": json.dumps({"task": "classification", "family": "fusion"}),
                    "created_at": now,
                },
                {
                    "id": "spec_pytorch_mlp",
                    "name": "PyTorch MLP Snapshot",
                    "kind": "pytorch_mlp",
                    "description": "Checkpointable MLP baseline for feature snapshots.",
                    "config_json": json.dumps({"task": "regression", "family": "torch"}),
                    "created_at": now,
                },
                {
                    "id": "spec_gru",
                    "name": "PyTorch GRU",
                    "kind": "gru",
                    "description": "Sequence baseline for time-window experiments.",
                    "config_json": json.dumps({"task": "regression", "family": "torch"}),
                    "created_at": now,
                },
                {
                    "id": "spec_temporal_cnn",
                    "name": "Temporal CNN",
                    "kind": "temporal_cnn",
                    "description": "Local temporal-pattern detector for event response experiments.",
                    "config_json": json.dumps({"task": "regression", "family": "torch"}),
                    "created_at": now,
                },
            ],
        )
        _seed_table(
            connection,
            "acceptance_policies",
            [
                {
                    "id": "policy_rank_quality",
                    "name": "Rank Quality Gate",
                    "description": "Require positive rank IC and directional accuracy over baseline.",
                    "config_json": json.dumps({"rank_ic_min": 0.02, "directional_accuracy_min": 0.52}),
                    "created_at": now,
                },
                {
                    "id": "policy_oos_performance",
                    "name": "OOS Portfolio Gate",
                    "description": "Require positive Sharpe and max drawdown under budget.",
                    "config_json": json.dumps({"sharpe_min": 0.5, "max_drawdown_max": 0.2}),
                    "created_at": now,
                },
                {
                    "id": "policy_pit_integrity",
                    "name": "PIT Integrity Gate",
                    "description": "Require zero look-ahead violations and reproducible feature lineage.",
                    "config_json": json.dumps({"pit_violations_max": 0, "lineage_required": True}),
                    "created_at": now,
                },
                {
                    "id": "policy_drift_tolerance",
                    "name": "Drift Tolerance Gate",
                    "description": "Require feature and alpha drift within thresholds.",
                    "config_json": json.dumps({"feature_drift_max": 0.15, "alpha_half_life_min": 3}),
                    "created_at": now,
                },
            ],
        )
        connection.commit()
