from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..config import DATASET_DIR, RAW_DATA_DIR, WAREHOUSE_PATH, ensure_directories

SECTORS = [
    "Technology",
    "Financials",
    "Health Care",
    "Industrials",
    "Energy",
    "Consumer Discretionary",
    "Utilities",
    "Materials",
]


@dataclass
class DatasetBuildResult:
    pit_path: Path
    raw_path: Path
    summary: dict[str, object]


def build_synthetic_dataset(
    dataset_id: str,
    name: str,
    num_tickers: int,
    num_days: int,
    seed: int,
) -> DatasetBuildResult:
    ensure_directories()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=num_days)
    sector_factor = {sector: rng.normal(0, 0.01, size=len(dates)) for sector in SECTORS}
    market_factor = rng.normal(0.0005, 0.009, size=len(dates))
    macro_rate = 4.75 + np.sin(np.linspace(0, 6.28, len(dates))) * 0.4
    macro_surprise = rng.normal(0, 0.15, size=len(dates))
    records: list[dict[str, object]] = []

    for ticker_index in range(num_tickers):
        ticker = f"Q{ticker_index:03d}"
        entity_id = f"eq_{ticker_index:04d}"
        sector = SECTORS[ticker_index % len(SECTORS)]
        base_price = rng.uniform(20, 250)
        close_prices = []
        volumes = []
        ev_ebitda = []
        roic = []
        news_shock = rng.normal(0, 0.03, size=len(dates))
        earnings_signal = np.zeros(len(dates))

        for i, current_date in enumerate(dates):
            if i % 63 == 0:
                earnings_signal[i : min(i + 5, len(dates))] += rng.normal(0.0, 0.06)
            ret = (
                market_factor[i] * rng.uniform(0.7, 1.3)
                + sector_factor[sector][i] * rng.uniform(0.8, 1.2)
                + news_shock[i] * 0.3
                + earnings_signal[i]
                + rng.normal(0, 0.012)
            )
            base_price = max(5.0, base_price * (1 + ret))
            volume = max(50_000, int(rng.lognormal(mean=12.3, sigma=0.35)))
            close_prices.append(base_price)
            volumes.append(volume)
            ev_ebitda.append(max(2.0, rng.normal(12.0 - ret * 90, 2.5)))
            roic.append(np.clip(rng.normal(0.11 + ret * 2, 0.03), -0.1, 0.4))

        close_series = pd.Series(close_prices, index=dates)
        open_series = close_series.shift(1).fillna(close_series.iloc[0] * (1 - rng.normal(0, 0.01)))
        high_series = np.maximum(open_series, close_series) * (1 + rng.uniform(0.0, 0.02, size=len(dates)))
        low_series = np.minimum(open_series, close_series) * (1 - rng.uniform(0.0, 0.02, size=len(dates)))
        returns_20d = close_series.pct_change(20).fillna(0.0)
        returns_60d = close_series.pct_change(60).fillna(0.0)
        sentiment_1d = pd.Series(news_shock, index=dates).rolling(1).mean().fillna(0.0)
        sentiment_5d = pd.Series(news_shock, index=dates).rolling(5).mean().fillna(0.0)
        event_intensity = pd.Series(earnings_signal, index=dates).rolling(3).sum().fillna(0.0)

        for i, current_date in enumerate(dates):
            records.append(
                {
                    "entity_id": entity_id,
                    "ticker": ticker,
                    "sector": sector,
                    "effective_at": current_date.isoformat(),
                    "known_at": (current_date + pd.Timedelta(hours=16, minutes=5)).isoformat(),
                    "ingested_at": (current_date + pd.Timedelta(hours=18)).isoformat(),
                    "source_version": name,
                    "open": float(open_series.iloc[i]),
                    "high": float(high_series.iloc[i]),
                    "low": float(low_series.iloc[i]),
                    "close": float(close_series.iloc[i]),
                    "volume": int(volumes[i]),
                    "ev_ebitda": float(ev_ebitda[i]),
                    "roic": float(roic[i]),
                    "momentum_20d": float(returns_20d.iloc[i]),
                    "momentum_60d": float(returns_60d.iloc[i]),
                    "sentiment_1d": float(sentiment_1d.iloc[i]),
                    "sentiment_5d": float(sentiment_5d.iloc[i]),
                    "macro_rate": float(macro_rate[i]),
                    "macro_surprise": float(macro_surprise[i]),
                    "earnings_signal": float(event_intensity.iloc[i]),
                }
            )

    frame = pd.DataFrame(records).sort_values(["effective_at", "ticker"]).reset_index(drop=True)
    dataset_folder = DATASET_DIR / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / f"{dataset_id}_raw.csv"
    pit_path = dataset_folder / "pit_daily.parquet"
    frame.to_csv(raw_path, index=False)
    frame.to_parquet(pit_path, index=False)
    _register_dataset_in_warehouse(dataset_id, pit_path)
    summary = {
        "rows": int(len(frame)),
        "tickers": int(frame["ticker"].nunique()),
        "sectors": sorted(frame["sector"].unique().tolist()),
        "date_range": [str(frame["effective_at"].min()), str(frame["effective_at"].max())],
        "artifacts": {
            "raw_path": str(raw_path),
            "pit_path": str(pit_path),
        },
    }
    (dataset_folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return DatasetBuildResult(pit_path=pit_path, raw_path=raw_path, summary=summary)


def _register_dataset_in_warehouse(dataset_id: str, pit_path: Path) -> None:
    ensure_directories()
    connection = duckdb.connect(str(WAREHOUSE_PATH))
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE pit_{dataset_id} AS
            SELECT * FROM read_parquet('{pit_path.as_posix()}')
            """
        )
    finally:
        connection.close()


def load_dataset_frame(dataset_id: str) -> pd.DataFrame:
    pit_path = DATASET_DIR / dataset_id / "pit_daily.parquet"
    return pd.read_parquet(pit_path)
