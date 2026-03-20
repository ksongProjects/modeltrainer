from __future__ import annotations

import pandas as pd


def build_macro_features(prices_index: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        return prices_index[["trade_date"]].drop_duplicates().assign(days_since_macro_release=0.0)

    frame = macro.copy()
    frame["release_date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.sort_values("release_date")

    wide = (
        frame.pivot_table(index="release_date", columns="series_id", values="value", aggfunc="last")
        .sort_index()
        .reset_index()
    )
    wide.columns = [str(col).lower() if isinstance(col, str) else col for col in wide.columns]

    daily = prices_index[["trade_date"]].drop_duplicates().sort_values("trade_date")
    merged = pd.merge_asof(
        daily,
        wide,
        left_on="trade_date",
        right_on="release_date",
        direction="backward",
    )
    merged["days_since_macro_release"] = (
        merged["trade_date"] - merged["release_date"]
    ).dt.days.astype(float)
    merged = merged.drop(columns=["release_date"])

    for column in merged.columns:
        if column == "trade_date":
            continue
        merged[column] = merged[column].ffill()
    return merged
