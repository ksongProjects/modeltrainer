from __future__ import annotations

import pandas as pd


def build_event_features(prices_index: pd.DataFrame) -> pd.DataFrame:
    frame = prices_index[["symbol", "trade_date"]].copy()
    frame["day_of_week"] = frame["trade_date"].dt.dayofweek.astype(float)
    frame["month"] = frame["trade_date"].dt.month.astype(float)
    frame["quarter"] = frame["trade_date"].dt.quarter.astype(float)
    frame["month_end_flag"] = frame["trade_date"].dt.is_month_end.astype(float)
    frame["earnings_proximity"] = 0.0
    frame["event_week_flag"] = 0.0
    return frame
