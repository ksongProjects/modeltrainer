from __future__ import annotations

import numpy as np
import pandas as pd


def add_labels(snapshot: pd.DataFrame, horizons: list[dict[str, int | str]]) -> pd.DataFrame:
    frame = snapshot.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    for horizon in horizons:
        horizon_name = str(horizon["name"])
        days = int(horizon["days"])
        future_price = grouped["adj_close"].shift(-days)
        forward_return = (future_price / frame["adj_close"]) - 1.0
        frame[f"forward_return_{horizon_name}"] = forward_return
        frame[f"target_up_{horizon_name}"] = np.where(forward_return.notna(), (forward_return > 0).astype(int), np.nan)
    return frame
