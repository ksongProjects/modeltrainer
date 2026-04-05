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

REQUIRED_NEWS_EVENT_COLUMNS = [
    "event_id",
    "event_scope",
    "event_type",
    "published_at",
    "known_at",
    "source",
    "source_weight",
    "entity_confidence",
    "novelty_score",
    "headline",
    "body",
]

OPTIONAL_NEWS_EVENT_COLUMNS = [
    "entity_id",
    "ticker",
    "sector",
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
    news_events: list[dict[str, object]] = []

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
            shock = float(news_shock[i])
            earnings_value = float(event_intensity.iloc[i])
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
                    "earnings_signal": earnings_value,
                }
            )
            if abs(shock) > 0.012 or abs(earnings_value) > 0.02 or rng.random() < 0.12:
                sentiment_label = "bullish" if shock >= 0 else "bearish"
                event_type = "earnings" if abs(earnings_value) > 0.02 else "company_news"
                headline = (
                    f"{ticker} {event_type.replace('_', ' ')} drives {sentiment_label} tone in {sector.lower()}"
                )
                body = (
                    f"{ticker} reported {event_type.replace('_', ' ')} developments with "
                    f"{'stronger' if shock >= 0 else 'weaker'} demand signals, "
                    f"management commentary, and margin expectations."
                )
                published_at = current_date + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(minutes=int(rng.integers(0, 360)))
                known_at = published_at + pd.Timedelta(minutes=int(rng.integers(2, 35)))
                news_events.append(
                    {
                        "event_id": f"{dataset_id}_{ticker}_{i:04d}",
                        "entity_id": entity_id,
                        "ticker": ticker,
                        "sector": sector,
                        "event_scope": "ticker",
                        "event_type": event_type,
                        "published_at": published_at.isoformat(),
                        "known_at": known_at.isoformat(),
                        "source": "synthetic_newswire",
                        "source_weight": float(1.0 + min(abs(shock) * 10, 1.0)),
                        "entity_confidence": float(np.clip(0.75 + abs(shock) * 4, 0.75, 0.98)),
                        "novelty_score": float(np.clip(0.45 + abs(shock) * 6, 0.45, 0.99)),
                        "headline": headline,
                        "body": body,
                    }
                )

    macro_topics = ["inflation", "rates", "employment", "growth", "oil", "credit"]
    for i, current_date in enumerate(dates):
        if abs(float(macro_surprise[i])) > 0.06 or rng.random() < 0.16:
            topic = macro_topics[i % len(macro_topics)]
            direction = "cooling" if float(macro_surprise[i]) < 0 else "heating"
            published_at = current_date + pd.Timedelta(hours=7, minutes=30) + pd.Timedelta(minutes=int(rng.integers(0, 180)))
            known_at = published_at + pd.Timedelta(minutes=int(rng.integers(1, 25)))
            news_events.append(
                {
                    "event_id": f"{dataset_id}_macro_{i:04d}",
                    "entity_id": None,
                    "ticker": None,
                    "sector": "Macro",
                    "event_scope": "macro",
                    "event_type": f"macro_{topic}",
                    "published_at": published_at.isoformat(),
                    "known_at": known_at.isoformat(),
                    "source": "synthetic_macro_desk",
                    "source_weight": float(1.0 + min(abs(float(macro_surprise[i])) * 8, 1.2)),
                    "entity_confidence": float(np.clip(0.8 + abs(float(macro_surprise[i])) * 2, 0.8, 0.99)),
                    "novelty_score": float(np.clip(0.5 + abs(float(macro_surprise[i])) * 4, 0.5, 0.99)),
                    "headline": f"Macro {topic} update points to {direction} conditions",
                    "body": f"Economic desks observed {direction} {topic} conditions with spillover expectations for broad risk assets.",
                }
            )

    frame = pd.DataFrame(records).sort_values(["effective_at", "ticker"]).reset_index(drop=True)
    news_frame = pd.DataFrame(news_events).sort_values(["known_at", "event_scope", "ticker"], na_position="last").reset_index(drop=True)
    dataset_folder = DATASET_DIR / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / f"{dataset_id}_raw.csv"
    pit_path = dataset_folder / "pit_daily.parquet"
    news_events_path = dataset_folder / "news_events.parquet"
    frame.to_csv(raw_path, index=False)
    frame.to_parquet(pit_path, index=False)
    news_frame.to_parquet(news_events_path, index=False)
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
        "news_events": {
            "rows": int(len(news_frame)),
            "ticker_news_rows": int((news_frame["event_scope"] == "ticker").sum()),
            "macro_news_rows": int((news_frame["event_scope"] == "macro").sum()),
            "required_columns": REQUIRED_NEWS_EVENT_COLUMNS,
            "optional_columns_present": OPTIONAL_NEWS_EVENT_COLUMNS,
        },
        "artifacts": {
            "raw_path": str(raw_path),
            "pit_path": str(pit_path),
            "news_events_path": str(news_events_path),
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
    news_events_summary = _import_optional_news_events(resolved_source, dataset_folder)

    source_manifest_path = dataset_folder / "source_manifest.json"
    source_manifest = {
        "source_path": str(resolved_source),
        "source_kind": "parquet_file" if resolved_source.is_file() else "parquet_directory",
        "imported_at": pd.Timestamp.now("UTC").isoformat(),
        "dataset_name": name or resolved_source.stem,
        "news_events_source_path": news_events_summary.get("source_path") if news_events_summary else None,
    }
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")

    summary = _build_import_summary(table, pit_path, resolved_source, source_manifest_path, news_events_summary)
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


def load_news_event_frame(dataset_id: str) -> pd.DataFrame:
    news_events_path = DATASET_DIR / dataset_id / "news_events.parquet"
    if not news_events_path.exists():
        return pd.DataFrame(columns=[*REQUIRED_NEWS_EVENT_COLUMNS, *OPTIONAL_NEWS_EVENT_COLUMNS])
    return pd.read_parquet(news_events_path)


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


def _import_optional_news_events(source_path: Path, dataset_folder: Path) -> dict[str, object] | None:
    news_source = _resolve_news_events_source(source_path)
    if news_source is None:
        return None

    dataset = _open_parquet_dataset(news_source)
    schema_names = set(dataset.schema.names)
    missing_columns = [column for column in REQUIRED_NEWS_EVENT_COLUMNS if column not in schema_names]
    if missing_columns:
        raise ValueError(
            "Imported news_events parquet is missing required columns: "
            + ", ".join(missing_columns)
        )

    table = dataset.to_table()
    news_events_path = dataset_folder / "news_events.parquet"
    pq.write_table(table, news_events_path)
    event_scope_column = pc.cast(table["event_scope"], "string")
    return {
        "rows": int(table.num_rows),
        "ticker_news_rows": int(pc.sum(pc.equal(event_scope_column, "ticker")).as_py() or 0),
        "macro_news_rows": int(pc.sum(pc.equal(event_scope_column, "macro")).as_py() or 0),
        "required_columns": REQUIRED_NEWS_EVENT_COLUMNS,
        "optional_columns_present": [column for column in OPTIONAL_NEWS_EVENT_COLUMNS if column in table.column_names],
        "artifacts": {"news_events_path": str(news_events_path)},
        "source_path": str(news_source),
    }


def _resolve_news_events_source(source_path: Path) -> Path | None:
    candidates: list[Path] = []
    if source_path.is_file():
        candidates.extend(
            [
                source_path.parent / "news_events.parquet",
                source_path.parent / f"{source_path.stem}_news_events.parquet",
            ]
        )
    else:
        candidates.extend(
            [
                source_path / "news_events.parquet",
                source_path.parent / "news_events.parquet",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_import_summary(
    table: Any,
    pit_path: Path,
    source_path: Path,
    source_manifest_path: Path,
    news_events_summary: dict[str, object] | None,
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
        "news_events": news_events_summary or {"rows": 0},
        "artifacts": {
            "pit_path": str(pit_path),
            "source_path": str(source_path),
            "source_manifest": str(source_manifest_path),
            **(news_events_summary.get("artifacts", {}) if news_events_summary else {}),
        },
    }
