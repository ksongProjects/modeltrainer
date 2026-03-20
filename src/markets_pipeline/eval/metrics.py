from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


def probability_to_action(probability: pd.Series, buy_threshold: float, sell_threshold: float) -> pd.Series:
    action = pd.Series(0, index=probability.index, dtype=float)
    action = action.mask(probability >= buy_threshold, 1.0)
    action = action.mask(probability <= sell_threshold, -1.0)
    return action


def classification_metrics(y_true: pd.Series, probability: pd.Series) -> dict[str, float]:
    truth = y_true.astype(int).to_numpy()
    prob = pd.Series(probability).clip(0.0, 1.0).to_numpy()
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "brier_score": float(brier_score_loss(truth, prob)),
        "directional_accuracy": float((truth == pred).mean()),
    }
    if len(set(truth.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(truth, prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def simple_backtest(
    frame: pd.DataFrame,
    probability_column: str,
    return_column: str,
    buy_threshold: float,
    sell_threshold: float,
) -> dict[str, float]:
    prob = frame[probability_column].fillna(0.5)
    action = probability_to_action(prob, buy_threshold, sell_threshold)
    strategy_return = action * frame[return_column].fillna(0.0)
    equity_curve = (1.0 + strategy_return).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    active_mask = action != 0
    return {
        "mean_strategy_return": float(strategy_return.mean()),
        "cumulative_return": float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else 0.0,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "active_rate": float(active_mask.mean()),
    }


def summarize_fold_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    frame = pd.DataFrame(records)
    numeric_cols = frame.select_dtypes(include=["number"]).columns
    return {column: float(frame[column].mean()) for column in numeric_cols}
