from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    out = left / right.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_qcut(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    non_null = ranked.dropna()
    if len(non_null) < 3 or non_null.nunique() < 3:
        return pd.Series(1.0, index=values.index, dtype=float)
    buckets = pd.qcut(ranked, q=3, labels=[0, 1, 2], duplicates="drop")
    return buckets.astype(float)


def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)

    frame["return_1d"] = grouped["adj_close"].pct_change(1)
    frame["return_5d"] = grouped["adj_close"].pct_change(5)
    frame["return_10d"] = grouped["adj_close"].pct_change(10)
    frame["return_21d"] = grouped["adj_close"].pct_change(21)
    frame["log_return_1d"] = np.log1p(frame["return_1d"])

    for window in (5, 10, 20, 60):
        rolling_close = grouped["adj_close"].transform(lambda s: s.rolling(window).mean())
        rolling_volume = grouped["volume"].transform(lambda s: s.rolling(window).mean())
        rolling_vol = grouped["log_return_1d"].transform(lambda s: s.rolling(window).std())
        frame[f"ma_{window}"] = rolling_close
        frame[f"price_to_ma_{window}"] = _safe_divide(frame["adj_close"], rolling_close) - 1.0
        frame[f"volume_to_avg_{window}"] = _safe_divide(frame["volume"], rolling_volume)
        frame[f"realized_vol_{window}"] = rolling_vol

    frame["range_pct"] = _safe_divide(frame["high"] - frame["low"], frame["close"])
    frame["gap_pct"] = grouped["open"].transform(lambda s: _safe_divide(s - s.shift(1), s.shift(1)))
    frame["drawdown_60d"] = grouped["adj_close"].transform(
        lambda s: _safe_divide(s, s.rolling(60).max()) - 1.0
    )

    delta = grouped["adj_close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.groupby(frame["symbol"]).transform(lambda s: s.rolling(14).mean())
    avg_loss = loss.groupby(frame["symbol"]).transform(lambda s: s.rolling(14).mean())
    rs = _safe_divide(avg_gain, avg_loss)
    frame["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    frame["volatility_regime"] = _safe_qcut(frame["realized_vol_20"])
    return frame
