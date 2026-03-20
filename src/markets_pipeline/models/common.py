from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..settings import Settings

TARGET_PREFIX = "target_up_"
RETURN_PREFIX = "forward_return_"


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class PassthroughCalibrator:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0.0, 1.0)


def timestamp_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def load_snapshot(settings: Settings, snapshot_version: str) -> pd.DataFrame:
    path = settings.datasets_dir / snapshot_version / "snapshot_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot dataset not found: {path}")
    frame = pd.read_parquet(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.tz_localize(None)
    return frame


def load_fold_specs(settings: Settings) -> list[FoldSpec]:
    payload = settings.load_folds()
    specs: list[FoldSpec] = []
    for fold in payload["folds"]:
        specs.append(
            FoldSpec(
                fold_id=fold["id"],
                train_start=pd.Timestamp(fold["train_start"]),
                train_end=pd.Timestamp(fold["train_end"]),
                validation_start=pd.Timestamp(fold["validation_start"]),
                validation_end=pd.Timestamp(fold["validation_end"]),
                test_start=pd.Timestamp(fold["test_start"]),
                test_end=pd.Timestamp(fold["test_end"]),
            )
        )
    return specs


def resolve_fold_specs(frame: pd.DataFrame, settings: Settings) -> list[FoldSpec]:
    configured = load_fold_specs(settings)
    usable = []
    for fold in configured:
        train_df, val_df, test_df = split_by_fold(frame, fold)
        if not train_df.empty and not val_df.empty and not test_df.empty:
            usable.append(fold)
    if usable:
        return usable

    unique_dates = pd.Series(sorted(frame["trade_date"].dropna().unique()))
    if len(unique_dates) < 12:
        raise ValueError("Not enough dated rows to derive fallback walk-forward folds.")

    first_train_end = int(len(unique_dates) * 0.55)
    first_val_end = int(len(unique_dates) * 0.75)
    second_train_end = int(len(unique_dates) * 0.7)
    second_val_end = int(len(unique_dates) * 0.85)

    fallback_specs = [
        FoldSpec(
            fold_id="fallback_1",
            train_start=unique_dates.iloc[0],
            train_end=unique_dates.iloc[first_train_end - 1],
            validation_start=unique_dates.iloc[first_train_end],
            validation_end=unique_dates.iloc[first_val_end - 1],
            test_start=unique_dates.iloc[first_val_end],
            test_end=unique_dates.iloc[min(len(unique_dates) - 1, int(len(unique_dates) * 0.9))],
        ),
        FoldSpec(
            fold_id="fallback_2",
            train_start=unique_dates.iloc[0],
            train_end=unique_dates.iloc[second_train_end - 1],
            validation_start=unique_dates.iloc[second_train_end],
            validation_end=unique_dates.iloc[second_val_end - 1],
            test_start=unique_dates.iloc[second_val_end],
            test_end=unique_dates.iloc[-1],
        ),
    ]
    return fallback_specs


def target_columns(horizon: str) -> tuple[str, str]:
    return f"{TARGET_PREFIX}{horizon}", f"{RETURN_PREFIX}{horizon}"


def split_by_fold(frame: pd.DataFrame, fold: FoldSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = frame["trade_date"].between(fold.train_start, fold.train_end)
    val_mask = frame["trade_date"].between(fold.validation_start, fold.validation_end)
    test_mask = frame["trade_date"].between(fold.test_start, fold.test_end)
    return frame.loc[train_mask].copy(), frame.loc[val_mask].copy(), frame.loc[test_mask].copy()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_prefixes = (TARGET_PREFIX, RETURN_PREFIX)
    excluded_columns = {
        "trade_date",
        "feature_view_version",
        "findf_job_id",
        "timestamp",
        "source",
    }
    output: list[str] = []
    for column in frame.columns:
        if column in excluded_columns:
            continue
        if any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        output.append(column)
    return output


def finite_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if output[column].dtype.kind in {"f", "i"}:
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    return output


def make_run_dir(settings: Settings, model_version: str) -> Path:
    run_dir = settings.models_dir / model_version
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
