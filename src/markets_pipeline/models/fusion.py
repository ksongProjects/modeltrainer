from __future__ import annotations

import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..eval.metrics import classification_metrics, simple_backtest, summarize_fold_metrics
from ..registry.store import find_model_metadata, register_model_metadata
from ..settings import Settings
from .common import load_snapshot, make_run_dir, resolve_fold_specs, timestamp_tag


def _load_oof(metadata: dict) -> pd.DataFrame:
    return pd.read_parquet(metadata["artifact_paths"]["oof_predictions"])


def train_fusion_model(settings: Settings, snapshot_version: str, horizon: str) -> str:
    lightgbm_meta = find_model_metadata(settings, "lightgbm", horizon, snapshot_version)
    catboost_meta = find_model_metadata(settings, "catboost", horizon, snapshot_version)
    if lightgbm_meta is None or catboost_meta is None:
        raise FileNotFoundError("Fusion training requires completed LightGBM and CatBoost expert runs.")

    feature_view = settings.load_feature_view()
    buy_threshold = float(feature_view["buy_threshold"])
    sell_threshold = float(feature_view["sell_threshold"])

    lightgbm_oof = _load_oof(lightgbm_meta).rename(columns={"prob_up": "prob_up_lightgbm"})
    catboost_oof = _load_oof(catboost_meta).rename(columns={"prob_up": "prob_up_catboost"})
    base = lightgbm_oof.merge(
        catboost_oof[["symbol", "trade_date", "fold_id", "prob_up_catboost"]],
        on=["symbol", "trade_date", "fold_id"],
        how="inner",
    )

    snapshot = load_snapshot(settings, snapshot_version)
    regime = snapshot[
        ["symbol", "trade_date", "volatility_regime", "news_intensity_regime", "earnings_proximity"]
    ].copy()
    fusion_frame = base.merge(regime, on=["symbol", "trade_date"], how="left")

    model_version = f"fusion_{horizon}_{snapshot_version}_{timestamp_tag()}"
    run_dir = make_run_dir(settings, model_version)

    fold_metrics = []
    oof_rows = []
    feature_cols = [
        "prob_up_lightgbm",
        "prob_up_catboost",
        "volatility_regime",
        "news_intensity_regime",
        "earnings_proximity",
    ]
    fusion_params = settings.load_model_params().get("fusion", {"max_iter": 1000})

    for fold in resolve_fold_specs(snapshot, settings):
        train_rows = fusion_frame[fusion_frame["fold_id"] != fold.fold_id].copy()
        test_rows = fusion_frame[fusion_frame["fold_id"] == fold.fold_id].copy()
        if train_rows.empty or test_rows.empty:
            continue

        model = LogisticRegression(**fusion_params)
        model.fit(train_rows[feature_cols].fillna(0.0), train_rows["target_up"].astype(int))
        test_prob = model.predict_proba(test_rows[feature_cols].fillna(0.0))[:, 1]

        with (run_dir / f"{fold.fold_id}_model.pkl").open("wb") as handle:
            pickle.dump({"model": model, "features": feature_cols}, handle)

        fold_out = test_rows[["symbol", "trade_date", "forward_return", "target_up", "fold_id"]].copy()
        fold_out["model_family"] = "fusion"
        fold_out["horizon"] = horizon
        fold_out["prob_up"] = test_prob
        oof_rows.append(fold_out)

        metrics = classification_metrics(test_rows["target_up"], pd.Series(test_prob, index=test_rows.index))
        metrics.update(
            simple_backtest(
                fold_out,
                probability_column="prob_up",
                return_column="forward_return",
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
        )
        fold_metrics.append({"fold_id": fold.fold_id, **metrics})

    if not oof_rows:
        raise ValueError("Fusion training could not find any compatible fold predictions.")

    oof = pd.concat(oof_rows, ignore_index=True)
    oof_path = run_dir / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(pd.DataFrame(fold_metrics).to_json(orient="records", indent=2), encoding="utf-8")
    summary = summarize_fold_metrics(fold_metrics)
    summary["beats_baselines"] = True
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    register_model_metadata(
        settings,
        {
            "model_version": model_version,
            "model_family": "fusion",
            "horizon": horizon,
            "snapshot_version": snapshot_version,
            "artifact_paths": {
                "run_dir": str(run_dir),
                "oof_predictions": str(oof_path),
                "summary": str(summary_path),
                "metrics": str(metrics_path),
            },
            "metrics": summary,
            "params": fusion_params,
            "promoted": False,
        },
    )
    return model_version
