from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPORT_DIR, ensure_directories
from ..research_layers import LAYER_FUSION_DECISION
from .execution import simulate_execution
from .features import load_feature_frame
from .risk import annualized_metrics, bootstrap_drawdown_distribution, cholesky_stress, compute_var_cvar, student_t_tail_simulation
from .training import load_predictor_bundle


@dataclass
class TestingResult:
    report_dir: Path
    metrics: dict[str, float]
    summary: dict[str, object]
    artifacts: dict[str, str]
    traces: list[dict[str, object]]


def run_testing_suite(
    testing_run_id: str,
    model_artifact_dir: str | Path,
    feature_set_id: str,
    stress_iterations: int,
    rebalance_decile: float,
    execution_mode: str,
    decision_top_k: int = 10,
) -> TestingResult:
    ensure_directories()
    frame = load_feature_frame(feature_set_id)
    predictor, layer_predictors, predictor_metadata = load_predictor_bundle(model_artifact_dir)
    scored_frame = frame.copy()
    for layer_id, layer_predictor in layer_predictors.items():
        layer_name = str(layer_id).replace("layer_", "")
        scored_frame[f"{layer_name}_score"] = layer_predictor(scored_frame)
    scored_frame["predicted_return"] = predictor(scored_frame)
    test_df = scored_frame[scored_frame["split"] == "test"].copy()

    portfolio_returns, trade_frame = _backtest_long_short(test_df, rebalance_decile)
    benchmark = trade_frame.groupby("effective_at")["forward_return"].mean()
    perf_metrics = annualized_metrics(portfolio_returns, benchmark)
    var_95, cvar_95 = compute_var_cvar(portfolio_returns)
    drawdown_distribution = bootstrap_drawdown_distribution(portfolio_returns, iterations=stress_iterations)
    sector_returns = trade_frame.pivot_table(
        index="effective_at",
        columns="sector",
        values="forward_return",
        aggfunc="mean",
    ).fillna(0.0)
    cholesky_metrics = cholesky_stress(sector_returns)
    tail_metrics = student_t_tail_simulation(portfolio_returns, iterations=stress_iterations)
    execution = simulate_execution(trade_frame, urgency=0.55, mode=execution_mode)

    metrics = {
        **perf_metrics,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "drawdown_p05": float(np.quantile(drawdown_distribution, 0.05)),
        "drawdown_p50": float(np.quantile(drawdown_distribution, 0.5)),
        "drawdown_p95": float(np.quantile(drawdown_distribution, 0.95)),
        **cholesky_metrics,
        **tail_metrics,
        "avg_slippage_bps": float(execution["summary"]["avg_slippage_bps"]),
        "implementation_shortfall_bps": float(execution["summary"]["implementation_shortfall_bps"]),
        "avg_vpin": float(execution["summary"]["avg_vpin"]),
        "turnover": float(_estimate_turnover(trade_frame)),
        "sector_neutrality_gap": float(_sector_neutrality_gap(trade_frame)),
    }

    report_dir = REPORT_DIR / testing_run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    equity_curve = (1 + portfolio_returns).cumprod().reset_index()
    equity_curve.columns = ["effective_at", "equity_curve"]
    equity_curve["effective_at"] = equity_curve["effective_at"].astype(str)
    backtest_path = report_dir / "equity_curve.json"
    backtest_path.write_text(equity_curve.to_json(orient="records"), encoding="utf-8")
    execution_path = report_dir / "execution_timeline.json"
    execution_path.write_text(json.dumps(execution["timeline"], indent=2), encoding="utf-8")
    decision_report_path = report_dir / "final_decision_report.json"
    decision_report = _build_final_decision_report(
        test_df=test_df,
        metrics=metrics,
        model_artifact_dir=model_artifact_dir,
        predictor_metadata=predictor_metadata,
        top_k=decision_top_k,
    )
    decision_report_path.write_text(json.dumps(decision_report, indent=2), encoding="utf-8")
    summary_path = report_dir / "summary.json"
    summary = {
        "rows": int(len(test_df)),
        "trade_rows": int(len(trade_frame)),
        "execution_mode": execution_mode,
        "decile": rebalance_decile,
        "artifacts": {
            "equity_curve": str(backtest_path),
            "execution_timeline": str(execution_path),
            "final_decision_report": str(decision_report_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps({"metrics": metrics, **summary}, indent=2), encoding="utf-8")

    traces = [
        {
            "formula_id": "phase6.backtest.signal_rank",
            "label": "Top/Bottom decile long-short",
            "inputs": {
                "rebalance_decile": rebalance_decile,
                "rows": len(test_df),
            },
            "transformed_inputs": {
                "selected_trade_rows": len(trade_frame),
                "avg_prediction": float(test_df["predicted_return"].mean()),
            },
            "output": {
                "annualized_return": metrics["annualized_return"],
                "sharpe": metrics["sharpe"],
            },
            "units": "portfolio_metrics",
            "provenance": {
                "feature_set_id": feature_set_id,
                "model_artifact_dir": str(model_artifact_dir),
            },
        },
        {
            "formula_id": "phase7.execution.impact",
            "label": "Paper execution impact model",
            "inputs": execution["summary"],
            "transformed_inputs": {
                "mode": execution_mode,
                "rebalance_decile": rebalance_decile,
            },
            "output": {
                "avg_slippage_bps": metrics["avg_slippage_bps"],
                "implementation_shortfall_bps": metrics["implementation_shortfall_bps"],
            },
            "units": "bps",
            "provenance": {"execution_path": str(execution_path)},
        },
    ]
    if layer_predictors:
        traces.append(
            {
                "formula_id": "phase6.fusion.decision",
                "label": "Layered fusion decision report",
                "inputs": {
                    "layer_count": len(layer_predictors),
                    "decision_top_k": decision_top_k,
                },
                "transformed_inputs": {
                    "decision_date": decision_report["decision_date"],
                    "score_columns": decision_report["score_columns"],
                },
                "output": {
                    "top_long_ticker": decision_report["top_longs"][0]["ticker"] if decision_report["top_longs"] else None,
                    "top_short_ticker": decision_report["top_shorts"][0]["ticker"] if decision_report["top_shorts"] else None,
                },
                "units": "decision_report",
                "provenance": {
                    "layer_id": LAYER_FUSION_DECISION,
                    "decision_report_path": str(decision_report_path),
                },
            }
        )
    return TestingResult(
        report_dir=report_dir,
        metrics=metrics,
        summary=summary,
        artifacts={
            "equity_curve": str(backtest_path),
            "execution_timeline": str(execution_path),
            "final_decision_report": str(decision_report_path),
            "summary": str(summary_path),
        },
        traces=traces,
    )


def _backtest_long_short(test_df: pd.DataFrame, rebalance_decile: float):
    daily_returns: dict[pd.Timestamp, float] = {}
    trades: list[pd.DataFrame] = []
    for effective_at, group in test_df.groupby("effective_at"):
        ranked = group.sort_values("predicted_return", ascending=False).copy()
        bucket_size = max(1, int(len(ranked) * rebalance_decile))
        long_bucket = ranked.head(bucket_size).copy()
        short_bucket = ranked.tail(bucket_size).copy()
        long_bucket["target_weight"] = 1.0 / bucket_size
        short_bucket["target_weight"] = -1.0 / bucket_size
        long_return = long_bucket["forward_return"].mean()
        short_return = short_bucket["forward_return"].mean()
        portfolio_return = float(long_return - short_return)
        daily_returns[pd.Timestamp(effective_at)] = portfolio_return
        trades.append(pd.concat([long_bucket, short_bucket], axis=0))
    trade_frame = pd.concat(trades, axis=0).reset_index(drop=True) if trades else test_df.head(0).copy()
    returns_series = pd.Series(daily_returns).sort_index()
    return returns_series, trade_frame


def _estimate_turnover(trades: pd.DataFrame) -> float:
    turnover = []
    previous_positions: set[str] = set()
    for _, group in trades.groupby("effective_at"):
        positions = set(group["ticker"].tolist())
        if previous_positions:
            changed = len(positions.symmetric_difference(previous_positions))
            turnover.append(changed / max(1, len(positions.union(previous_positions))))
        previous_positions = positions
    return float(np.mean(turnover) if turnover else 0.0)


def _sector_neutrality_gap(trades: pd.DataFrame) -> float:
    sector_weights = trades.groupby("sector")["target_weight"].sum()
    return float(np.abs(sector_weights).mean())


def _build_final_decision_report(
    test_df: pd.DataFrame,
    metrics: dict[str, float],
    model_artifact_dir: str | Path,
    predictor_metadata: dict[str, object],
    top_k: int,
) -> dict[str, object]:
    score_columns = sorted(column for column in test_df.columns if column.endswith("_score"))
    latest_effective_at = pd.to_datetime(test_df["effective_at"]).max()
    latest_slice = test_df[pd.to_datetime(test_df["effective_at"]) == latest_effective_at].copy()
    latest_slice = latest_slice.sort_values("predicted_return", ascending=False)
    return {
        "decision_date": str(latest_effective_at),
        "model_kind": predictor_metadata.get("model_kind"),
        "requested_model_kind": predictor_metadata.get("requested_model_kind"),
        "model_artifact_dir": str(model_artifact_dir),
        "score_columns": score_columns,
        "layer_registry": predictor_metadata.get("layer_registry"),
        "metrics": metrics,
        "top_longs": _serialize_decision_rows(latest_slice.head(top_k), score_columns),
        "top_shorts": _serialize_decision_rows(
            latest_slice.tail(top_k).sort_values("predicted_return", ascending=True),
            score_columns,
        ),
    }


def _serialize_decision_rows(frame: pd.DataFrame, score_columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        payload: dict[str, object] = {
            "ticker": str(row["ticker"]),
            "sector": str(row["sector"]),
            "predicted_return": float(row["predicted_return"]),
            "forward_return": float(row["forward_return"]),
        }
        for column in score_columns:
            payload[column] = float(row[column])
        rows.append(payload)
    return rows
