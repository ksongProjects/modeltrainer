from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..eval.baselines import baseline_ma_crossover, baseline_majority_probability, baseline_previous_sign
from ..eval.metrics import classification_metrics, simple_backtest, summarize_fold_metrics
from ..registry.store import register_model_metadata
from ..settings import Settings
from .common import (
    PassthroughCalibrator,
    feature_columns,
    finite_frame,
    load_snapshot,
    make_run_dir,
    resolve_fold_specs,
    split_by_fold,
    target_columns,
    timestamp_tag,
)


@dataclass(frozen=True)
class TrainResult:
    model_version: str
    run_dir: str


def _fit_calibrator(val_pred: np.ndarray, y_val: pd.Series):
    if y_val.nunique() < 2:
        return PassthroughCalibrator()
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_pred, y_val.astype(float))
    return calibrator


def _prepare_inputs(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame[columns].copy()
    if "symbol" in output.columns:
        output["symbol"] = output["symbol"].astype("category")
    return output


def _build_model_with_params(family: str, params: dict[str, Any]):
    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params)
    if family == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params)
    raise ValueError(f"Unsupported model family: {family}")


def _fit_family(
    family: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    model = _build_model_with_params(family, params)
    if family == "lightgbm":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], categorical_feature=["symbol"])
        return model

    from catboost import Pool

    cat_features = [X_train.columns.get_loc("symbol")] if "symbol" in X_train.columns else []
    model.fit(
        Pool(X_train, y_train, cat_features=cat_features),
        eval_set=Pool(X_val, y_val, cat_features=cat_features),
        verbose=False,
    )
    return model


def _predict_probability(model, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def train_tabular_expert(
    settings: Settings,
    snapshot_version: str,
    horizon: str,
    family: str,
) -> TrainResult:
    feature_view = settings.load_feature_view()
    buy_threshold = float(feature_view["buy_threshold"])
    sell_threshold = float(feature_view["sell_threshold"])

    model_version = f"{family}_{horizon}_{snapshot_version}_{timestamp_tag()}"
    run_dir = make_run_dir(settings, model_version)

    snapshot = load_snapshot(settings, snapshot_version)
    target_column, return_column = target_columns(horizon)
    snapshot = snapshot[snapshot[target_column].notna()].copy()
    features = feature_columns(snapshot)
    snapshot = finite_frame(snapshot, features)
    model_params = settings.load_model_params().get(family, {})

    fold_metrics: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []

    for fold in resolve_fold_specs(snapshot, settings):
        train_df, val_df, test_df = split_by_fold(snapshot, fold)
        if train_df.empty or val_df.empty or test_df.empty:
            continue

        X_train = _prepare_inputs(train_df, features)
        X_val = _prepare_inputs(val_df, features)
        X_test = _prepare_inputs(test_df, features)
        y_train = train_df[target_column].astype(int)
        y_val = val_df[target_column].astype(int)
        y_test = test_df[target_column].astype(int)

        model = _fit_family(family, model_params, X_train, y_train, X_val, y_val)
        val_raw = _predict_probability(model, X_val)
        calibrator = _fit_calibrator(val_raw, y_val)
        test_prob = calibrator.predict(_predict_probability(model, X_test))

        with (run_dir / f"{fold.fold_id}_model.pkl").open("wb") as handle:
            pickle.dump({"model": model, "calibrator": calibrator, "features": features}, handle)

        test_out = test_df[
            ["symbol", "trade_date", return_column, target_column, "volatility_regime", "news_intensity_regime", "earnings_proximity"]
        ].copy()
        test_out = test_out.rename(columns={target_column: "target_up", return_column: "forward_return"})
        test_out["model_family"] = family
        test_out["horizon"] = horizon
        test_out["fold_id"] = fold.fold_id
        test_out["prob_up"] = test_prob
        oof_rows.append(test_out)

        metrics = classification_metrics(y_test, pd.Series(test_prob, index=test_df.index))
        metrics.update(
            simple_backtest(
                test_out,
                probability_column="prob_up",
                return_column="forward_return",
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
        )

        majority_prob = baseline_majority_probability(train_df, target_column, len(test_df))
        prev_sign_prob = baseline_previous_sign(test_df)
        ma_prob = baseline_ma_crossover(test_df)
        baselines = {
            "majority": classification_metrics(y_test, majority_prob),
            "previous_sign": classification_metrics(y_test, prev_sign_prob),
            "ma_crossover": classification_metrics(y_test, ma_prob),
        }

        fold_metrics.append(
            {
                "fold_id": fold.fold_id,
                **metrics,
                "baseline_majority_balanced_accuracy": baselines["majority"]["balanced_accuracy"],
                "baseline_prev_sign_balanced_accuracy": baselines["previous_sign"]["balanced_accuracy"],
                "baseline_ma_balanced_accuracy": baselines["ma_crossover"]["balanced_accuracy"],
                "baseline_majority_brier": baselines["majority"]["brier_score"],
                "baseline_prev_sign_brier": baselines["previous_sign"]["brier_score"],
                "baseline_ma_brier": baselines["ma_crossover"]["brier_score"],
            }
        )

    if not oof_rows:
        raise ValueError(f"No non-empty folds were available for {family} {horizon} on {snapshot_version}.")

    oof = pd.concat(oof_rows, ignore_index=True)
    oof_path = run_dir / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)

    metrics_frame = pd.DataFrame(fold_metrics)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(metrics_frame.to_json(orient="records", indent=2), encoding="utf-8")

    summary = summarize_fold_metrics(fold_metrics)
    summary["beats_baselines"] = bool(
        summary.get("balanced_accuracy", 0.0)
        > max(
            summary.get("baseline_majority_balanced_accuracy", 0.0),
            summary.get("baseline_prev_sign_balanced_accuracy", 0.0),
            summary.get("baseline_ma_balanced_accuracy", 0.0),
        )
        and summary.get("brier_score", 1.0)
        < min(
            summary.get("baseline_majority_brier", 1.0),
            summary.get("baseline_prev_sign_brier", 1.0),
            summary.get("baseline_ma_brier", 1.0),
        )
    )
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    register_model_metadata(
        settings,
        {
            "model_version": model_version,
            "model_family": family,
            "horizon": horizon,
            "snapshot_version": snapshot_version,
            "artifact_paths": {
                "run_dir": str(run_dir),
                "oof_predictions": str(oof_path),
                "summary": str(summary_path),
                "metrics": str(metrics_path),
            },
            "metrics": summary,
            "params": model_params,
            "promoted": False,
        },
    )
    return TrainResult(model_version=model_version, run_dir=str(run_dir))
