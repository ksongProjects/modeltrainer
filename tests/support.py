from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_realistic_parquet(path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=140)
    tickers = [
        ("AAA", "eq_0001", "Technology", 80.0),
        ("BBB", "eq_0002", "Financials", 42.0),
        ("CCC", "eq_0003", "Health Care", 61.0),
        ("DDD", "eq_0004", "Industrials", 55.0),
    ]
    rows: list[dict[str, object]] = []
    for ticker_index, (ticker, entity_id, sector, base_price) in enumerate(tickers):
        for day_index, effective_at in enumerate(dates):
            close = base_price + day_index * (0.14 + ticker_index * 0.02)
            rows.append(
                {
                    "entity_id": entity_id,
                    "ticker": ticker,
                    "sector": sector,
                    "effective_at": effective_at.isoformat(),
                    "known_at": (effective_at + pd.Timedelta(hours=16, minutes=5)).isoformat(),
                    "ingested_at": (effective_at + pd.Timedelta(hours=18)).isoformat(),
                    "source_version": "findf_v1",
                    "open": close - 0.35,
                    "high": close + 0.7,
                    "low": close - 0.9,
                    "close": close,
                    "volume": 120_000 + (day_index * 900) + (ticker_index * 1_500),
                    "ev_ebitda": 8.0 + ticker_index + (day_index % 5) * 0.15,
                    "roic": 0.08 + ticker_index * 0.01 + (day_index % 7) * 0.001,
                    "momentum_20d": (day_index % 20) * 0.004,
                    "momentum_60d": (day_index % 60) * 0.002,
                    "sentiment_1d": ((day_index % 5) - 2) * 0.05,
                    "sentiment_5d": ((day_index % 7) - 3) * 0.04,
                    "macro_surprise": ((day_index % 9) - 4) * 0.02,
                    "earnings_signal": 0.06 if day_index % 30 == 0 else 0.0,
                    "macro_rate": 4.5,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_news_events_parquet(path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=140)
    rows: list[dict[str, object]] = []
    for day_index, effective_at in enumerate(dates[:24]):
        published_at = effective_at + pd.Timedelta(hours=9, minutes=30)
        known_at = published_at + pd.Timedelta(minutes=10)
        rows.append(
            {
                "event_id": f"aaa_{day_index:03d}",
                "event_scope": "ticker",
                "event_type": "company_news",
                "published_at": published_at.isoformat(),
                "known_at": known_at.isoformat(),
                "source": "newswire",
                "source_weight": 1.2,
                "entity_confidence": 0.94,
                "novelty_score": 0.81,
                "headline": f"AAA event {day_index}",
                "body": "AAA headline catalyst affecting price behavior.",
                "entity_id": "eq_0001",
                "ticker": "AAA",
                "sector": "Technology",
            }
        )
        rows.append(
            {
                "event_id": f"macro_{day_index:03d}",
                "event_scope": "macro",
                "event_type": "macro_rates",
                "published_at": (published_at - pd.Timedelta(hours=1)).isoformat(),
                "known_at": (known_at - pd.Timedelta(hours=1)).isoformat(),
                "source": "macro_desk",
                "source_weight": 1.0,
                "entity_confidence": 0.9,
                "novelty_score": 0.73,
                "headline": f"Macro rates update {day_index}",
                "body": "Macro rate expectations shifted before the open.",
                "entity_id": None,
                "ticker": None,
                "sector": "Macro",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
