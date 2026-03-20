from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def parse_tickers(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip().upper() for item in value.split(",")]
        return [item for item in parts if item]
    if isinstance(value, Iterable):
        return [str(item).strip().upper() for item in value if str(item).strip()]
    return []


def explode_scored_news(scored_news: pd.DataFrame) -> pd.DataFrame:
    frame = scored_news.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True).dt.tz_localize(None)
    frame["trade_date"] = frame["published_at"].dt.normalize()
    frame["parsed_tickers"] = frame["tickers"].apply(parse_tickers)
    frame["is_market_wide"] = frame["parsed_tickers"].map(lambda items: 1.0 if not items else 0.0)
    frame["parsed_tickers"] = frame["parsed_tickers"].map(lambda items: items or ["__MARKET__"])
    exploded = frame.explode("parsed_tickers").rename(columns={"parsed_tickers": "symbol"})
    return exploded


def build_sentiment_features(
    prices_index: pd.DataFrame,
    scored_news: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    base = prices_index[["symbol", "trade_date"]].drop_duplicates().sort_values(["symbol", "trade_date"])
    if scored_news.empty:
        for window in windows:
            base[f"news_count_{window}d"] = 0.0
            base[f"sentiment_mean_{window}d"] = 0.0
            base[f"sentiment_weighted_{window}d"] = 0.0
        base["last_article_age_days"] = np.nan
        base["market_news_count_7d"] = 0.0
        base["market_sentiment_mean_7d"] = 0.0
        return base

    exploded = explode_scored_news(scored_news)
    hours_to_close = (16 - exploded["published_at"].dt.hour).clip(lower=0)
    exploded["recency_weight"] = 1.0 / (1.0 + hours_to_close)
    exploded["weighted_score"] = exploded["sentiment_score"] * exploded["recency_weight"]

    daily = (
        exploded.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            news_count=("id", "count"),
            sentiment_sum=("sentiment_score", "sum"),
            weighted_sum=("weighted_score", "sum"),
            last_published_at=("published_at", "max"),
        )
        .sort_values(["symbol", "trade_date"])
    )
    daily["sentiment_mean"] = daily["sentiment_sum"] / daily["news_count"].replace(0, np.nan)
    daily["sentiment_weighted"] = daily["weighted_sum"] / daily["news_count"].replace(0, np.nan)

    merged = base.merge(daily, on=["symbol", "trade_date"], how="left")
    merged["news_count"] = merged["news_count"].fillna(0.0)
    merged["sentiment_sum"] = merged["sentiment_sum"].fillna(0.0)
    merged["weighted_sum"] = merged["weighted_sum"].fillna(0.0)
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
    merged["sentiment_weighted"] = merged["sentiment_weighted"].fillna(0.0)

    grouped = merged.groupby("symbol", group_keys=False)
    for window in windows:
        merged[f"news_count_{window}d"] = grouped["news_count"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )
        sentiment_sum = grouped["sentiment_sum"].transform(lambda s: s.rolling(window, min_periods=1).sum())
        weighted_sum = grouped["weighted_sum"].transform(lambda s: s.rolling(window, min_periods=1).sum())
        counts = merged[f"news_count_{window}d"].replace(0, np.nan)
        merged[f"sentiment_mean_{window}d"] = (sentiment_sum / counts).fillna(0.0)
        merged[f"sentiment_weighted_{window}d"] = (weighted_sum / counts).fillna(0.0)

    merged["last_published_at"] = grouped["last_published_at"].ffill()
    merged["last_article_age_days"] = (
        merged["trade_date"] - merged["last_published_at"].dt.normalize()
    ).dt.days.astype(float)

    market_daily = (
        exploded[exploded["symbol"] == "__MARKET__"]
        .groupby("trade_date", as_index=False)
        .agg(
            market_news_count=("id", "count"),
            market_sentiment_sum=("sentiment_score", "sum"),
        )
        .sort_values("trade_date")
    )
    if market_daily.empty:
        merged["market_news_count_7d"] = 0.0
        merged["market_sentiment_mean_7d"] = 0.0
    else:
        market_daily["market_news_count_7d"] = market_daily["market_news_count"].rolling(7, min_periods=1).sum()
        market_daily["market_sentiment_mean_7d"] = (
            market_daily["market_sentiment_sum"].rolling(7, min_periods=1).sum()
            / market_daily["market_news_count_7d"].replace(0, np.nan)
        ).fillna(0.0)
        merged = merged.merge(
            market_daily[["trade_date", "market_news_count_7d", "market_sentiment_mean_7d"]],
            on="trade_date",
            how="left",
        )
        merged["market_news_count_7d"] = merged["market_news_count_7d"].fillna(0.0)
        merged["market_sentiment_mean_7d"] = merged["market_sentiment_mean_7d"].fillna(0.0)

    keep_columns = ["symbol", "trade_date", "last_article_age_days", "market_news_count_7d", "market_sentiment_mean_7d"]
    keep_columns.extend(
        [
            f"news_count_{window}d"
            for window in windows
        ]
    )
    keep_columns.extend(
        [
            f"sentiment_mean_{window}d"
            for window in windows
        ]
    )
    keep_columns.extend(
        [
            f"sentiment_weighted_{window}d"
            for window in windows
        ]
    )
    return merged[keep_columns]
