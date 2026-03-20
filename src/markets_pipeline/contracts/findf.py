from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..settings import Settings

EXPECTED_COLUMNS = {
    "prices": {"ticker", "timestamp", "open", "high", "low", "close", "volume", "adj_close"},
    "news": {"id", "title", "summary", "tickers", "published_at", "source", "url"},
    "macro": {"series_id", "date", "value", "source"},
    "fundamentals": {"ticker", "date", "metric", "value", "source"},
}


@dataclass(frozen=True)
class ArtifactPaths:
    prices_silver: Path
    news_silver: Path
    macro_silver: Path
    fundamentals_silver: Path | None = None

    def to_json(self) -> dict[str, str | None]:
        return {
            "prices_silver": str(self.prices_silver),
            "news_silver": str(self.news_silver),
            "macro_silver": str(self.macro_silver),
            "fundamentals_silver": str(self.fundamentals_silver) if self.fundamentals_silver else None,
        }


@dataclass(frozen=True)
class FindfRunManifest:
    job_id: str
    tickers: list[str]
    start_date: str
    end_date: str
    providers: dict[str, list[str]]
    artifact_paths: ArtifactPaths
    schema_version: str = "findf_silver_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_paths"] = self.artifact_paths.to_json()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FindfRunManifest":
        artifact_paths = ArtifactPaths(
            prices_silver=Path(payload["artifact_paths"]["prices_silver"]),
            news_silver=Path(payload["artifact_paths"]["news_silver"]),
            macro_silver=Path(payload["artifact_paths"]["macro_silver"]),
            fundamentals_silver=(
                Path(payload["artifact_paths"]["fundamentals_silver"])
                if payload["artifact_paths"].get("fundamentals_silver")
                else None
            ),
        )
        return cls(
            job_id=payload["job_id"],
            tickers=list(payload["tickers"]),
            start_date=payload["start_date"],
            end_date=payload["end_date"],
            providers=dict(payload.get("providers", {})),
            artifact_paths=artifact_paths,
            schema_version=payload.get("schema_version", "findf_silver_v1"),
        )


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _validate_columns(name: str, frame: pd.DataFrame) -> None:
    required = EXPECTED_COLUMNS[name]
    missing = required.difference(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{name} dataset is missing required columns: {missing_cols}")


def _discover_artifact(settings: Settings, job_id: str, suffix: str) -> Path | None:
    candidate = settings.silver_dir / f"{job_id}_{suffix}.parquet"
    if candidate.exists():
        return candidate
    return None


def discover_manifest(settings: Settings, job_id: str) -> FindfRunManifest:
    prices = _discover_artifact(settings, job_id, "prices")
    news = _discover_artifact(settings, job_id, "news")
    macro = _discover_artifact(settings, job_id, "macro")
    fundamentals = _discover_artifact(settings, job_id, "fundamentals")
    if prices is None or news is None or macro is None:
        raise FileNotFoundError(
            f"Could not discover all required silver artifacts for job_id={job_id} under {settings.silver_dir}"
        )

    prices_df = _read_parquet(prices)
    news_df = _read_parquet(news)
    macro_df = _read_parquet(macro)
    _validate_columns("prices", prices_df)
    _validate_columns("news", news_df)
    _validate_columns("macro", macro_df)

    if fundamentals and fundamentals.exists():
        fundamentals_df = _read_parquet(fundamentals)
        if not fundamentals_df.empty:
            _validate_columns("fundamentals", fundamentals_df)
    else:
        fundamentals = None

    prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"]).dt.tz_localize(None)
    tickers = sorted(prices_df["ticker"].dropna().astype(str).unique().tolist())
    start_date = prices_df["timestamp"].min().date().isoformat()
    end_date = prices_df["timestamp"].max().date().isoformat()

    providers = {
        "prices": sorted(prices_df.get("source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "news": sorted(news_df.get("source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "macro": sorted(macro_df.get("source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
    }

    return FindfRunManifest(
        job_id=job_id,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        providers=providers,
        artifact_paths=ArtifactPaths(
            prices_silver=prices,
            news_silver=news,
            macro_silver=macro,
            fundamentals_silver=fundamentals,
        ),
    )


def persist_manifest(settings: Settings, manifest: FindfRunManifest) -> Path:
    path = settings.manifests_dir / f"{manifest.job_id}.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def load_manifest(settings: Settings, job_id: str) -> FindfRunManifest:
    path = settings.manifests_dir / f"{job_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FindfRunManifest.from_dict(payload)
