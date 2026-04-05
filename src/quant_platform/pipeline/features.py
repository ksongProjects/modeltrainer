from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..config import FEATURE_DIR, WAREHOUSE_PATH, ensure_directories
from .data import load_dataset_frame
from .text_embeddings import (
    MACRO_TEXT_EMBEDDING_COLUMNS,
    TEXT_EMBEDDING_COLUMNS,
    TextEmbeddingBuildResult,
    materialize_text_embedding_features,
)

FEATURE_COLUMNS = [
    "value_z",
    "quality_sector_z",
    "momentum_z",
    "sentiment_z",
    "macro_z",
    "earnings_z",
    "composite_score",
    "volume_z",
]

TEXT_FEATURE_COLUMNS = [*TEXT_EMBEDDING_COLUMNS, "text_event_count", "text_event_weight"]
MACRO_TEXT_FEATURE_COLUMNS = [*MACRO_TEXT_EMBEDDING_COLUMNS, "macro_text_event_count", "macro_text_event_weight"]


@dataclass
class FeatureBuildResult:
    feature_path: Path
    summary: dict[str, object]
    traces: list[dict[str, object]]


def winsorize_series(series: pd.Series, limit: float = 3.0) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    z_values = (series - mean) / std
    return z_values.clip(-limit, limit)


def zscore_series(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def sector_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(["effective_at", "sector"], sort=False)[column].transform(zscore_series).fillna(0.0)


def materialize_features(
    feature_set_id: str,
    dataset_version_id: str,
    name: str,
    winsor_limit: float,
    forecast_horizon_days: int,
    process_step_state: dict[str, bool] | None = None,
) -> FeatureBuildResult:
    ensure_directories()
    frame = load_dataset_frame(dataset_version_id).copy()
    frame["effective_at"] = pd.to_datetime(frame["effective_at"], utc=True).dt.normalize()
    frame["known_at"] = pd.to_datetime(frame["known_at"], utc=True)
    frame["ingested_at"] = pd.to_datetime(frame["ingested_at"], utc=True)
    frame["close"] = frame["close"].astype(float)
    frame["volume"] = frame["volume"].astype(float)
    process_state = _resolve_feature_process_state(process_step_state)

    frame["value_z"] = frame.groupby("effective_at", group_keys=False)["ev_ebitda"].transform(
        lambda values: -(
            winsorize_series(values, winsor_limit)
            if process_state["winsorize_value_factor"]
            else zscore_series(values)
        )
    )
    frame["quality_raw"] = frame["roic"].fillna(frame["roic"].median())
    if process_state["sector_neutralize_quality"]:
        frame["quality_sector_z"] = sector_zscore(frame, "quality_raw")
    else:
        frame["quality_sector_z"] = frame.groupby("effective_at", group_keys=False)["quality_raw"].transform(zscore_series)
    frame["momentum_raw"] = 0.65 * frame["momentum_20d"] + 0.35 * frame["momentum_60d"]
    frame["momentum_z"] = frame.groupby("effective_at", group_keys=False)["momentum_raw"].transform(zscore_series)
    frame["sentiment_raw"] = 0.35 * frame["sentiment_1d"] + 0.65 * frame["sentiment_5d"]
    frame["sentiment_z"] = frame.groupby("effective_at", group_keys=False)["sentiment_raw"].transform(zscore_series)
    if process_state["demean_macro_surprise"]:
        frame["macro_raw"] = frame["macro_surprise"] - frame.groupby("effective_at")["macro_surprise"].transform("mean")
    else:
        frame["macro_raw"] = frame["macro_surprise"]
    frame["macro_z"] = frame.groupby("effective_at", group_keys=False)["macro_raw"].transform(zscore_series)
    frame["earnings_z"] = frame.groupby("effective_at", group_keys=False)["earnings_signal"].transform(zscore_series)
    frame["volume_z"] = frame.groupby("effective_at", group_keys=False)["volume"].transform(
        lambda values: winsorize_series(
            np.log1p(values) if process_state["log_transform_volume"] else values,
            winsor_limit,
        )
    )
    frame["missing_flag"] = frame[["roic", "ev_ebitda"]].isna().any(axis=1).astype(int)
    frame["outlier_flag"] = (frame["value_z"].abs() >= winsor_limit).astype(int)
    frame["composite_score"] = (
        0.30 * frame["value_z"]
        + 0.25 * frame["quality_sector_z"]
        + 0.20 * frame["momentum_z"]
        + 0.15 * frame["sentiment_z"]
        + 0.05 * frame["macro_z"]
        + 0.05 * frame["earnings_z"]
    )
    frame["forward_return"] = frame.groupby("ticker")["close"].shift(-forecast_horizon_days) / frame["close"] - 1.0
    frame["direction_label"] = (frame["forward_return"] > 0).astype(int)
    frame["split"] = "train"
    unique_dates = np.array(sorted(frame["effective_at"].dt.strftime("%Y-%m-%d").unique()))
    train_cut = int(len(unique_dates) * 0.7)
    val_cut = int(len(unique_dates) * 0.85)
    frame.loc[frame["effective_at"].dt.strftime("%Y-%m-%d").isin(unique_dates[train_cut:val_cut]), "split"] = "validation"
    frame.loc[frame["effective_at"].dt.strftime("%Y-%m-%d").isin(unique_dates[val_cut:]), "split"] = "test"

    if process_state["aggregate_text_embeddings"]:
        text_embedding_result = materialize_text_embedding_features(dataset_version_id, frame)
    else:
        zero_features = frame[["ticker", "effective_at"]].copy()
        for column in [*TEXT_FEATURE_COLUMNS, *MACRO_TEXT_FEATURE_COLUMNS]:
            zero_features[column] = 0.0
        text_embedding_result = TextEmbeddingBuildResult(
            features=zero_features,
            summary={
                "rows": int(len(zero_features)),
                "news_event_rows": 0,
                "ticker_text_coverage": 0.0,
                "macro_text_coverage": 0.0,
                "text_embedding_columns": TEXT_EMBEDDING_COLUMNS,
                "macro_text_embedding_columns": MACRO_TEXT_EMBEDDING_COLUMNS,
                "lookback_days": 5,
                "half_life_days": 2.0,
            },
            traces=[],
        )
    frame = frame.merge(text_embedding_result.features, on=["ticker", "effective_at"], how="left")
    for column in [*TEXT_FEATURE_COLUMNS, *MACRO_TEXT_FEATURE_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    feature_frame = frame.dropna(subset=["forward_return"]).copy()
    feature_folder = FEATURE_DIR / feature_set_id
    feature_folder.mkdir(parents=True, exist_ok=True)
    feature_path = feature_folder / "features.parquet"
    feature_frame.to_parquet(feature_path, index=False)
    _register_feature_set(feature_set_id, feature_path)

    trace_rows = feature_frame.head(10)
    traces = []
    for _, row in trace_rows.iterrows():
        traces.append(
            {
                "formula_id": "phase2.standardize.composite",
                "label": f"{row['ticker']} {row['effective_at'].date()}",
                "inputs": {
                    "ev_ebitda": row["ev_ebitda"],
                    "roic": row["roic"],
                    "momentum_20d": row["momentum_20d"],
                    "sentiment_5d": row["sentiment_5d"],
                    "macro_surprise": row["macro_surprise"],
                },
                "transformed_inputs": {
                    "value_z": row["value_z"],
                    "quality_sector_z": row["quality_sector_z"],
                    "momentum_z": row["momentum_z"],
                    "sentiment_z": row["sentiment_z"],
                    "macro_z": row["macro_z"],
                    "earnings_z": row["earnings_z"],
                    "text_embedding_00": row.get(TEXT_EMBEDDING_COLUMNS[0], 0.0),
                    "macro_text_embedding_00": row.get(MACRO_TEXT_EMBEDDING_COLUMNS[0], 0.0),
                    "process_step_state": process_state,
                },
                "output": {"composite_score": row["composite_score"]},
                "units": "zscore",
                "provenance": {
                    "dataset_version_id": dataset_version_id,
                    "feature_set_id": feature_set_id,
                    "transform": name,
                    "process_step_state": process_state,
                },
            }
        )
    traces.extend(text_embedding_result.traces[:4])

    summary = {
        "rows": int(len(feature_frame)),
        "feature_columns": FEATURE_COLUMNS,
        "text_feature_columns": TEXT_FEATURE_COLUMNS,
        "macro_text_feature_columns": MACRO_TEXT_FEATURE_COLUMNS,
        "forecast_horizon_days": forecast_horizon_days,
        "splits": feature_frame["split"].value_counts().to_dict(),
        "missing_flags": int(feature_frame["missing_flag"].sum()),
        "outlier_flags": int(feature_frame["outlier_flag"].sum()),
        "text_embedding_summary": text_embedding_result.summary,
        "process_step_state": process_state,
        "artifacts": {"feature_path": str(feature_path)},
    }
    (feature_folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return FeatureBuildResult(feature_path=feature_path, summary=summary, traces=traces)


def _resolve_feature_process_state(process_step_state: dict[str, bool] | None) -> dict[str, bool]:
    state = {
        "winsorize_value_factor": True,
        "sector_neutralize_quality": True,
        "aggregate_text_embeddings": True,
        "demean_macro_surprise": True,
        "log_transform_volume": True,
    }
    if process_step_state:
        for key, value in process_step_state.items():
            if key in state:
                state[key] = bool(value)
    return state


def _register_feature_set(feature_set_id: str, feature_path: Path) -> None:
    ensure_directories()
    connection = duckdb.connect(str(WAREHOUSE_PATH))
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE features_{feature_set_id} AS
            SELECT * FROM read_parquet('{feature_path.as_posix()}')
            """
        )
    finally:
        connection.close()


def load_feature_frame(feature_set_id: str) -> pd.DataFrame:
    feature_path = FEATURE_DIR / feature_set_id / "features.parquet"
    return pd.read_parquet(feature_path)
