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

ASSESSMENT_ANALYSIS_COLUMNS = [
    "entity_id",
    "ticker",
    "effective_at",
    "known_at",
    "ingested_at",
    "open",
    "high",
    "low",
    "close",
    "volume",
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
    assessment = _build_dataset_assessment_from_frame(frame)
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
        "assessment": assessment,
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
    assessment = _build_dataset_assessment_from_table(table)
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
        "assessment": assessment,
        "news_events": news_events_summary or {"rows": 0},
        "artifacts": {
            "pit_path": str(pit_path),
            "source_path": str(source_path),
            "source_manifest": str(source_manifest_path),
            **(news_events_summary.get("artifacts", {}) if news_events_summary else {}),
        },
    }


def _build_dataset_assessment_from_table(table: Any) -> dict[str, object]:
    column_names = list(table.column_names)
    null_counts = {column: int(table[column].null_count) for column in column_names}
    analysis_columns = [column for column in ASSESSMENT_ANALYSIS_COLUMNS if column in column_names]
    analysis_frame = table.select(analysis_columns).to_pandas() if analysis_columns else pd.DataFrame(index=range(int(table.num_rows)))
    return _build_dataset_assessment(
        row_count=int(table.num_rows),
        observed_columns=column_names,
        column_null_counts=null_counts,
        analysis_frame=analysis_frame,
    )


def _build_dataset_assessment_from_frame(frame: pd.DataFrame) -> dict[str, object]:
    column_names = [str(column) for column in frame.columns]
    null_counts = {column: int(frame[column].isna().sum()) for column in frame.columns}
    analysis_columns = [column for column in ASSESSMENT_ANALYSIS_COLUMNS if column in frame.columns]
    analysis_frame = frame.loc[:, analysis_columns].copy() if analysis_columns else pd.DataFrame(index=frame.index)
    return _build_dataset_assessment(
        row_count=int(len(frame)),
        observed_columns=column_names,
        column_null_counts=null_counts,
        analysis_frame=analysis_frame,
    )


def _build_dataset_assessment(
    row_count: int,
    observed_columns: list[str],
    column_null_counts: dict[str, int],
    analysis_frame: pd.DataFrame,
) -> dict[str, object]:
    present_required = [column for column in REQUIRED_PARQUET_COLUMNS if column in observed_columns]
    present_optional = [column for column in OPTIONAL_PARQUET_COLUMNS if column in observed_columns]
    column_completeness: dict[str, dict[str, object]] = {}
    required_missing_cells = 0

    for column in [*present_required, *present_optional]:
        missing_values = int(column_null_counts.get(column, 0))
        non_null_pct = 100.0 if row_count == 0 else round(((row_count - missing_values) / row_count) * 100.0, 2)
        column_completeness[column] = {
            "non_null_pct": non_null_pct,
            "missing_values": missing_values,
            "required": column in REQUIRED_PARQUET_COLUMNS,
        }
        if column in REQUIRED_PARQUET_COLUMNS:
            required_missing_cells += missing_values

    required_cell_total = row_count * len(REQUIRED_PARQUET_COLUMNS)
    completeness_pct = 100.0 if required_cell_total == 0 else round(((required_cell_total - required_missing_cells) / required_cell_total) * 100.0, 2)

    entity_id_raw = _analysis_series(analysis_frame, "entity_id").astype("string").str.strip()
    ticker_raw = _analysis_series(analysis_frame, "ticker").astype("string").str.strip()
    effective_raw = _analysis_series(analysis_frame, "effective_at")
    known_raw = _analysis_series(analysis_frame, "known_at")
    ingested_raw = _analysis_series(analysis_frame, "ingested_at")
    open_raw = _analysis_series(analysis_frame, "open")
    high_raw = _analysis_series(analysis_frame, "high")
    low_raw = _analysis_series(analysis_frame, "low")
    close_raw = _analysis_series(analysis_frame, "close")
    volume_raw = _analysis_series(analysis_frame, "volume")

    effective_ts = pd.to_datetime(effective_raw, errors="coerce", utc=True)
    known_ts = pd.to_datetime(known_raw, errors="coerce", utc=True)
    ingested_ts = pd.to_datetime(ingested_raw, errors="coerce", utc=True)

    invalid_effective_mask = effective_ts.isna() & effective_raw.notna()
    invalid_known_mask = known_ts.isna() & known_raw.notna()
    invalid_ingested_mask = ingested_ts.isna() & ingested_raw.notna()
    invalid_timestamp_mask = invalid_effective_mask | invalid_known_mask | invalid_ingested_mask
    invalid_timestamp_rows = int(invalid_timestamp_mask.sum())

    open_num = pd.to_numeric(open_raw, errors="coerce")
    high_num = pd.to_numeric(high_raw, errors="coerce")
    low_num = pd.to_numeric(low_raw, errors="coerce")
    close_num = pd.to_numeric(close_raw, errors="coerce")
    volume_num = pd.to_numeric(volume_raw, errors="coerce")

    invalid_numeric_mask = (
        ((open_num.isna() & open_raw.notna()))
        | ((high_num.isna() & high_raw.notna()))
        | ((low_num.isna() & low_raw.notna()))
        | ((close_num.isna() & close_raw.notna()))
        | ((volume_num.isna() & volume_raw.notna()))
    )
    invalid_numeric_rows = int(invalid_numeric_mask.sum())

    ordered_rows = effective_ts.notna() & known_ts.notna() & ingested_ts.notna()
    timestamp_order_mask = ordered_rows & (
        (known_ts < effective_ts)
        | (ingested_ts < known_ts)
        | (ingested_ts < effective_ts)
    )
    timestamp_order_violations = int(timestamp_order_mask.sum())

    price_rows = open_num.notna() & high_num.notna() & low_num.notna() & close_num.notna()
    ohlc_violation_mask = price_rows & (
        (high_num < low_num)
        | (high_num < open_num)
        | (high_num < close_num)
        | (low_num > open_num)
        | (low_num > close_num)
    )
    ohlc_violations = int(ohlc_violation_mask.sum())

    non_positive_price_mask = price_rows & (
        (open_num <= 0)
        | (high_num <= 0)
        | (low_num <= 0)
        | (close_num <= 0)
    )
    non_positive_price_rows = int(non_positive_price_mask.sum())

    non_positive_volume_mask = volume_num.notna() & (volume_num <= 0)
    non_positive_volume_rows = int(non_positive_volume_mask.sum())

    quality_issue_mask = (
        invalid_timestamp_mask
        | invalid_numeric_mask
        | timestamp_order_mask
        | ohlc_violation_mask
        | non_positive_price_mask
        | non_positive_volume_mask
    )
    quality_issue_rows = int(quality_issue_mask.sum())
    quality_pct = 100.0 if row_count == 0 else round(((row_count - quality_issue_rows) / row_count) * 100.0, 2)

    gaps = _build_gap_summary(entity_id_raw, ticker_raw, effective_ts)
    continuity_pct = float(gaps["continuity_pct"])

    score_pct = round((0.45 * completeness_pct) + (0.35 * continuity_pct) + (0.20 * quality_pct), 2)
    if int(gaps["missing_sessions"]) > 0 or int(gaps["duplicate_key_rows"]) > 0:
        score_pct -= 3.0
    if int(gaps["instruments_with_gaps"]) > 0:
        score_pct -= min(6.0, (int(gaps["instruments_with_gaps"]) / max(int(gaps["instrument_count"]), 1)) * 10.0)
    if quality_issue_rows > 0:
        score_pct -= min(18.0, (quality_issue_rows / max(row_count, 1)) * 300.0)
    score_pct = round(float(np.clip(score_pct, 0.0, 100.0)), 2)

    issues: list[dict[str, object]] = []
    if row_count == 0:
        issues.append(
            _issue(
                "critical",
                "Empty dataset",
                "No rows were available to inspect, so completeness and quality checks could not validate the snapshot.",
                0,
                "Re-run the source export before promoting this dataset into feature generation.",
            )
        )
    if required_missing_cells > 0:
        issues.append(
            _issue(
                "critical" if completeness_pct < 95.0 else "warning",
                "Missing required values",
                f"{required_missing_cells} required PIT cells are blank across the imported snapshot.",
                required_missing_cells,
                "Backfill or drop rows with missing required PIT fields before materializing features.",
            )
        )
    if int(gaps["missing_sessions"]) > 0:
        issues.append(
            _issue(
                "critical" if continuity_pct < 95.0 else "warning",
                "Missing trading sessions",
                f"{gaps['missing_sessions']} interior sessions are missing across {gaps['instruments_with_gaps']} instruments; the largest gap spans {gaps['largest_gap_sessions']} sessions.",
                int(gaps["missing_sessions"]),
                "Re-run the vendor extract or backfill the missing sessions before downstream modeling.",
            )
        )
    if int(gaps["duplicate_key_rows"]) > 0:
        issues.append(
            _issue(
                "warning",
                "Duplicate entity-date keys",
                f"{gaps['duplicate_key_rows']} duplicate rows share the same entity/date key and should be deduplicated before training.",
                int(gaps["duplicate_key_rows"]),
                "Deduplicate on entity_id + effective_at or confirm why multiple rows are expected for the same session.",
            )
        )
    if invalid_timestamp_rows > 0:
        issues.append(
            _issue(
                "critical",
                "Unparseable timestamps",
                f"{invalid_timestamp_rows} rows contain timestamp values that could not be parsed cleanly.",
                invalid_timestamp_rows,
                "Normalize effective_at, known_at, and ingested_at into valid ISO timestamps before using the dataset.",
            )
        )
    if timestamp_order_violations > 0:
        issues.append(
            _issue(
                "critical",
                "Timestamp ordering violations",
                f"{timestamp_order_violations} rows violate the expected PIT ordering of effective_at <= known_at <= ingested_at.",
                timestamp_order_violations,
                "Repair PIT lineage ordering so downstream features cannot leak future information.",
            )
        )
    if invalid_numeric_rows > 0:
        issues.append(
            _issue(
                "critical",
                "Non-numeric market values",
                f"{invalid_numeric_rows} rows contain OHLCV fields that could not be parsed as numbers.",
                invalid_numeric_rows,
                "Clean or coerce invalid OHLCV values before passing the snapshot into feature engineering.",
            )
        )
    if ohlc_violations > 0:
        issues.append(
            _issue(
                "critical",
                "OHLC consistency failures",
                f"{ohlc_violations} rows contain price bars where high/low/open/close are internally inconsistent.",
                ohlc_violations,
                "Repair malformed bars or remove affected rows before using the dataset in model training.",
            )
        )
    if non_positive_price_rows > 0:
        issues.append(
            _issue(
                "critical",
                "Non-positive prices",
                f"{non_positive_price_rows} rows contain zero or negative open/high/low/close values.",
                non_positive_price_rows,
                "Filter or repair rows with impossible price values before feature generation.",
            )
        )
    if non_positive_volume_rows > 0:
        issues.append(
            _issue(
                "critical",
                "Non-positive volume",
                f"{non_positive_volume_rows} rows contain zero or negative volume values.",
                non_positive_volume_rows,
                "Repair or remove rows with impossible volume values before training.",
            )
        )

    critical_issue_count = sum(1 for issue in issues if issue["severity"] == "critical")
    warning_issue_count = sum(1 for issue in issues if issue["severity"] == "warning")

    if row_count == 0 or critical_issue_count > 0 or quality_pct < 95.0 or continuity_pct < 95.0:
        status = "critical"
        data_level = "low"
    elif warning_issue_count > 0 or completeness_pct < 99.0 or score_pct < 99.0:
        status = "warning"
        data_level = "medium"
    else:
        status = "healthy"
        data_level = "high"

    return {
        "data_level": data_level,
        "status": status,
        "score_pct": score_pct,
        "completeness_pct": completeness_pct,
        "continuity_pct": continuity_pct,
        "quality_pct": quality_pct,
        "inspected_rows": row_count,
        "required_missing_cells": required_missing_cells,
        "column_completeness": column_completeness,
        "gaps": gaps,
        "quality_checks": {
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "invalid_numeric_rows": invalid_numeric_rows,
            "timestamp_order_violations": timestamp_order_violations,
            "ohlc_violations": ohlc_violations,
            "non_positive_price_rows": non_positive_price_rows,
            "non_positive_volume_rows": non_positive_volume_rows,
        },
        "issue_counts": {
            "critical": critical_issue_count,
            "warning": warning_issue_count,
        },
        "issues": issues,
    }


def _build_gap_summary(
    entity_ids: pd.Series,
    tickers: pd.Series,
    effective_ts: pd.Series,
) -> dict[str, object]:
    instrument_keys = entity_ids.fillna("").astype("string").str.strip()
    instrument_labels = tickers.fillna("").astype("string").str.strip()
    effective_dates = effective_ts.dt.normalize()

    coverage_frame = pd.DataFrame(
        {
            "instrument_key": instrument_keys,
            "instrument_label": instrument_labels,
            "effective_date": effective_dates,
        }
    )
    valid = coverage_frame[
        coverage_frame["instrument_key"].fillna("").str.len().gt(0)
        & coverage_frame["effective_date"].notna()
    ].copy()
    if valid.empty:
        return {
            "instrument_count": 0,
            "dataset_sessions": 0,
            "expected_rows": 0,
            "observed_unique_keys": 0,
            "duplicate_key_rows": 0,
            "instruments_with_gaps": 0,
            "gap_free_instruments_pct": 100.0,
            "missing_sessions": 0,
            "largest_gap_sessions": 0,
            "continuity_pct": 100.0,
            "gap_samples": [],
        }

    valid.loc[valid["instrument_label"].fillna("").str.len().eq(0), "instrument_label"] = valid["instrument_key"]
    unique_keys = valid.drop_duplicates(subset=["instrument_key", "effective_date"])
    global_dates = pd.DatetimeIndex(sorted(unique_keys["effective_date"].unique()))
    date_positions = {date_value: index for index, date_value in enumerate(global_dates)}

    expected_rows = 0
    missing_sessions = 0
    instruments_with_gaps = 0
    gap_free_instruments = 0
    largest_gap_sessions = 0
    gap_samples: list[dict[str, object]] = []

    for instrument_key, group in unique_keys.groupby("instrument_key", sort=True):
        present_dates = pd.DatetimeIndex(sorted(group["effective_date"].unique()))
        positions = np.array([date_positions[date_value] for date_value in present_dates], dtype=int)
        if positions.size == 0:
            continue

        expected_for_instrument = int(positions[-1] - positions[0] + 1)
        expected_rows += expected_for_instrument
        missing_for_instrument = int(expected_for_instrument - positions.size)

        if missing_for_instrument <= 0:
            gap_free_instruments += 1
            continue

        instruments_with_gaps += 1
        missing_sessions += missing_for_instrument
        gap_sizes = (np.diff(positions) - 1) if positions.size > 1 else np.array([], dtype=int)
        largest_gap_sessions = max(largest_gap_sessions, int(gap_sizes.max()) if gap_sizes.size else 0)

        if len(gap_samples) < 5:
            missing_dates = global_dates[positions[0] : positions[-1] + 1].difference(present_dates)
            label = str(group["instrument_label"].iloc[0] or instrument_key)
            gap_samples.append(
                {
                    "instrument": label,
                    "missing_sessions": missing_for_instrument,
                    "sample_dates": [str(date_value.date()) for date_value in missing_dates[:3]],
                }
            )

    observed_unique_keys = int(len(unique_keys))
    instrument_count = int(unique_keys["instrument_key"].nunique())
    duplicate_key_rows = int(len(valid) - observed_unique_keys)
    continuity_pct = 100.0 if expected_rows == 0 else round((observed_unique_keys / expected_rows) * 100.0, 2)
    gap_free_instruments_pct = 100.0 if instrument_count == 0 else round((gap_free_instruments / instrument_count) * 100.0, 2)

    return {
        "instrument_count": instrument_count,
        "dataset_sessions": int(len(global_dates)),
        "expected_rows": int(expected_rows),
        "observed_unique_keys": observed_unique_keys,
        "duplicate_key_rows": duplicate_key_rows,
        "instruments_with_gaps": instruments_with_gaps,
        "gap_free_instruments_pct": gap_free_instruments_pct,
        "missing_sessions": missing_sessions,
        "largest_gap_sessions": largest_gap_sessions,
        "continuity_pct": continuity_pct,
        "gap_samples": gap_samples,
    }


def _analysis_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(index=frame.index, dtype="object")


def _issue(
    severity: str,
    title: str,
    detail: str,
    count: int,
    recommendation: str,
) -> dict[str, object]:
    return {
        "severity": severity,
        "title": title,
        "detail": detail,
        "count": int(count),
        "recommendation": recommendation,
    }
