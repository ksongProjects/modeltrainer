from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_execution(
    trades: pd.DataFrame,
    urgency: float = 0.5,
    mode: str = "paper",
) -> dict[str, object]:
    if trades.empty:
        return {
            "summary": {
                "avg_slippage_bps": 0.0,
                "implementation_shortfall_bps": 0.0,
                "avg_vpin": 0.0,
                "mode": mode,
            },
            "timeline": [],
        }

    frame = trades.copy()
    frame["abs_weight"] = frame["target_weight"].abs()
    frame["dollar_volume_proxy"] = frame["close"] * frame["volume"]
    frame["participation_rate"] = np.clip(frame["abs_weight"] * (0.3 + urgency), 0.01, 0.25)
    frame["twap_slice_count"] = np.maximum(4, np.ceil(12 * frame["participation_rate"])).astype(int)
    frame["vwap_bucket_share"] = np.clip(np.sqrt(frame["participation_rate"]), 0.05, 0.5)
    frame["order_imbalance"] = (frame["predicted_return"].abs() * frame["volume"]).rolling(5, min_periods=1).mean()
    volume_bucket = frame["volume"].rolling(5, min_periods=1).mean().replace(0, np.nan)
    frame["vpin"] = np.clip((frame["order_imbalance"] / volume_bucket).fillna(0.0), 0.0, 1.0)
    volatility_proxy = frame["close"].pct_change().rolling(10, min_periods=1).std().fillna(0.01)
    impact = 0.1 * volatility_proxy * np.sqrt(frame["abs_weight"] / frame["participation_rate"].clip(lower=0.01))
    toxicity_penalty = np.where(frame["vpin"] > 0.7, 0.0025, 0.0008)
    frame["slippage_bps"] = (impact + toxicity_penalty + urgency * 0.0015) * 10_000
    frame["implementation_shortfall_bps"] = frame["slippage_bps"] + (frame["vpin"] * 12)

    summary = {
        "avg_slippage_bps": float(frame["slippage_bps"].mean()),
        "implementation_shortfall_bps": float(frame["implementation_shortfall_bps"].mean()),
        "avg_vpin": float(frame["vpin"].mean()),
        "max_vpin": float(frame["vpin"].max()),
        "paused_for_toxicity_count": int((frame["vpin"] > 0.7).sum()),
        "mode": mode,
    }
    timeline = frame[
        [
            "effective_at",
            "ticker",
            "target_weight",
            "predicted_return",
            "slippage_bps",
            "implementation_shortfall_bps",
            "vpin",
            "twap_slice_count",
            "vwap_bucket_share",
        ]
    ].head(100)
    timeline["effective_at"] = timeline["effective_at"].astype(str)
    return {"summary": summary, "timeline": timeline.to_dict(orient="records")}
