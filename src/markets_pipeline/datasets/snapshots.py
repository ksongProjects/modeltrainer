from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts.findf import FindfRunManifest
from ..features.events import build_event_features
from ..features.macro import build_macro_features
from ..features.sentiment import build_sentiment_features
from ..features.technical import add_technical_features
from ..settings import Settings
from .labels import add_labels


@dataclass(frozen=True)
class SnapshotBuildResult:
    snapshot_version: str
    snapshot_path: Path
    metadata_path: Path


def _load_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["trade_date"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize(None).dt.normalize()
    frame = frame.rename(columns={"ticker": "symbol"})
    frame = frame.sort_values(["symbol", "trade_date"])
    return frame


def _load_macro(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


def _load_news_scores(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "id",
                "published_at",
                "tickers",
                "sentiment_score",
                "prob_negative",
                "prob_neutral",
                "prob_positive",
                "embedding_norm",
                "embedding_mean",
                "embedding_std",
            ]
        )
    return pd.read_parquet(path).copy()


def _load_fundamentals(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["symbol", "trade_date"])
    frame = pd.read_parquet(path).copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "trade_date"])
    frame["trade_date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.rename(columns={"ticker": "symbol"})
    pivoted = (
        frame.pivot_table(index=["symbol", "trade_date"], columns="metric", values="value", aggfunc="last")
        .reset_index()
    )
    pivoted.columns = [str(col).lower() if isinstance(col, str) else col for col in pivoted.columns]
    return pivoted


def _add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for source, target in (
        ("news_count_7d", "news_intensity_regime"),
        ("days_since_macro_release", "macro_staleness_regime"),
    ):
        ranked = output[source].rank(method="first")
        valid = ranked.dropna()
        if len(valid) < 3 or valid.nunique() < 3:
            output[target] = 1.0
        else:
            output[target] = pd.qcut(ranked, q=3, labels=[0, 1, 2], duplicates="drop").astype(float)
    return output


def build_snapshot_daily(
    settings: Settings,
    manifest: FindfRunManifest,
    feature_view: dict[str, Any],
) -> SnapshotBuildResult:
    snapshot_version = f"{manifest.job_id}_{feature_view['version']}"
    output_dir = settings.datasets_dir / snapshot_version
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = _load_prices(manifest.artifact_paths.prices_silver)
    prices = add_technical_features(prices)
    price_index = prices[["symbol", "trade_date"]].drop_duplicates().sort_values(["symbol", "trade_date"])

    macro = _load_macro(manifest.artifact_paths.macro_silver)
    macro_features = build_macro_features(price_index, macro)
    snapshot = prices.merge(macro_features, on="trade_date", how="left")

    news_scores_path = output_dir / "news_scores.parquet"
    scored_news = _load_news_scores(news_scores_path)
    sentiment_features = build_sentiment_features(
        price_index,
        scored_news,
        windows=list(feature_view["sentiment_windows"]),
    )
    snapshot = snapshot.merge(sentiment_features, on=["symbol", "trade_date"], how="left")

    event_features = build_event_features(price_index)
    snapshot = snapshot.merge(event_features, on=["symbol", "trade_date"], how="left")

    fundamentals = _load_fundamentals(manifest.artifact_paths.fundamentals_silver)
    if not fundamentals.empty:
        snapshot = snapshot.merge(fundamentals, on=["symbol", "trade_date"], how="left")

    snapshot = _add_regime_features(snapshot)
    snapshot = add_labels(snapshot, settings.load_horizons())
    snapshot["horizon_ready"] = snapshot["target_up_10d"].notna().astype(int)
    snapshot["feature_view_version"] = feature_view["version"]
    snapshot["findf_job_id"] = manifest.job_id

    snapshot = snapshot.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    snapshot_path = output_dir / "snapshot_daily.parquet"
    snapshot.to_parquet(snapshot_path, index=False)

    metadata = {
        "snapshot_version": snapshot_version,
        "feature_view_version": feature_view["version"],
        "findf_job_id": manifest.job_id,
        "row_count": int(len(snapshot)),
        "columns": list(snapshot.columns),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SnapshotBuildResult(
        snapshot_version=snapshot_version,
        snapshot_path=snapshot_path,
        metadata_path=metadata_path,
    )
