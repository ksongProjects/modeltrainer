from __future__ import annotations

import pandas as pd


def baseline_majority_probability(train_df: pd.DataFrame, target_column: str, size: int) -> pd.Series:
    positive_rate = float(train_df[target_column].mean())
    return pd.Series([positive_rate] * size, dtype=float)


def baseline_previous_sign(test_df: pd.DataFrame) -> pd.Series:
    return test_df["return_1d"].fillna(0.0).gt(0).astype(float)


def baseline_ma_crossover(test_df: pd.DataFrame) -> pd.Series:
    return test_df["ma_5"].gt(test_df["ma_20"]).astype(float)
