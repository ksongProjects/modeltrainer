from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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

REQUIRED_PARQUET_COLUMNS = [
    "entity_id",
    "ticker",
    "sector",
    "effective_at",
    "known_at",
    "ingested_at",
    "source_version",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ev_ebitda",
    "roic",
    "momentum_20d",
    "momentum_60d",
    "sentiment_1d",
    "sentiment_5d",
    "macro_surprise",
    "earnings_signal",
]

OPTIONAL_PARQUET_COLUMNS = [
    "macro_rate",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ev_ebitda",
    "roic",
    "momentum_20d",
    "momentum_60d",
    "sentiment_1d",
    "sentiment_5d",
    "macro_surprise",
    "earnings_signal",
]

TIMESTAMP_COLUMNS = [
    "effective_at",
    "known_at",
    "ingested_at",
]


@dataclass
class DatasetBuildResult:
    pit_path: Path
    raw_path: Path | None
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
    dates = pd.bdate_range(end=pd.Timestamp.now("UTC").normalize(), periods=num_days)
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
        "sample_tickers": sorted(frame["ticker"].astype(str).unique().tolist())[:5],
        "sectors": sorted(frame["sector"].unique().tolist()),
        "date_range": [str(frame["effective_at"].min()), str(frame["effective_at"].max())],
        "schema": {
            "required_columns": REQUIRED_PARQUET_COLUMNS,
            "optional_columns_present": OPTIONAL_PARQUET_COLUMNS,
            "extra_columns": [],
        },
        "artifacts": {
            "raw_path": str(raw_path),
            "pit_path": str(pit_path),
        },
    }
    (dataset_folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return DatasetBuildResult(pit_path=pit_path, raw_path=raw_path, summary=summary)


def import_parquet_dataset(
    dataset_id: str,
    source_path: str,
    name: str | None = None,
) -> DatasetBuildResult:
    ensure_directories()
    resolved_source = Path(source_path).expanduser().resolve()
    if not resolved_source.exists():
        raise FileNotFoundError(f"Parquet source path does not exist: {resolved_source}")

    dataset = _open_parquet_dataset(resolved_source)
    schema_names = set(dataset.schema.names)
    missing_columns = [column for column in REQUIRED_PARQUET_COLUMNS if column not in schema_names]
    if missing_columns:
        raise ValueError(
            "Imported Parquet dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    table = dataset.to_table()
    _validate_import_table(table)

    dataset_folder = DATASET_DIR / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=True)
    pit_path = dataset_folder / "pit_daily.parquet"
    pq.write_table(table, pit_path)
    _register_dataset_in_warehouse(dataset_id, pit_path)

    source_manifest_path = dataset_folder / "source_manifest.json"
    source_manifest = {
        "source_path": str(resolved_source),
        "source_kind": "parquet_file" if resolved_source.is_file() else "parquet_directory",
        "imported_at": pd.Timestamp.now("UTC").isoformat(),
        "dataset_name": name or resolved_source.stem,
    }
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")

    summary = _build_import_summary(table, pit_path, resolved_source, source_manifest_path)
    (dataset_folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return DatasetBuildResult(pit_path=pit_path, raw_path=None, summary=summary)


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


def _open_parquet_dataset(source_path: Path) -> ds.Dataset:
    if source_path.is_file():
        if source_path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file, got: {source_path.name}")
        return ds.dataset(str(source_path), format="parquet")

    parquet_files = list(source_path.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under: {source_path}")
    return ds.dataset(str(source_path), format="parquet")


def _validate_import_table(table: Any) -> None:
    sample_size = min(1000, table.num_rows)
    sample = table.slice(0, sample_size).to_pandas()

    for column in TIMESTAMP_COLUMNS:
        parsed = pd.to_datetime(sample[column], errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError(f"Column {column} contains non-datetime values in the sample window.")

    for column in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(sample[column], errors="coerce")
        if coerced.isna().any():
            raise ValueError(f"Column {column} contains non-numeric values in the sample window.")

    if sample["ticker"].astype(str).str.len().eq(0).any():
        raise ValueError("Column ticker contains empty values in the sample window.")
    if sample["sector"].astype(str).str.len().eq(0).any():
        raise ValueError("Column sector contains empty values in the sample window.")


def _build_import_summary(
    table: Any,
    pit_path: Path,
    source_path: Path,
    source_manifest_path: Path,
) -> dict[str, object]:
    effective_values = pc.cast(table["effective_at"], "timestamp[us]")
    sample_tickers = sorted(pc.unique(pc.cast(table["ticker"], "string")).to_pylist())[:5]
    sectors = sorted(pc.unique(pc.cast(table["sector"], "string")).to_pylist())
    optional_present = [column for column in OPTIONAL_PARQUET_COLUMNS if column in table.column_names]
    return {
        "rows": int(table.num_rows),
        "tickers": int(pc.count_distinct(pc.cast(table["ticker"], "string")).as_py()),
        "sample_tickers": sample_tickers,
        "sectors": sectors,
        "date_range": [
            str(pc.min(effective_values).as_py()),
            str(pc.max(effective_values).as_py()),
        ],
        "schema": {
            "required_columns": REQUIRED_PARQUET_COLUMNS,
            "optional_columns_present": optional_present,
            "extra_columns": sorted(
                column
                for column in table.column_names
                if column not in REQUIRED_PARQUET_COLUMNS and column not in OPTIONAL_PARQUET_COLUMNS
            ),
        },
        "artifacts": {
            "pit_path": str(pit_path),
            "source_path": str(source_path),
            "source_manifest": str(source_manifest_path),
        },
    }
