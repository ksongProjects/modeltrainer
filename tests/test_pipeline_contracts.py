from __future__ import annotations

from pathlib import Path

import pandas as pd

from markets_pipeline.contracts.findf import discover_manifest, persist_manifest
from markets_pipeline.settings import Settings


def make_settings(project_root: Path, findf_root: Path) -> Settings:
    artifacts_dir = project_root / "artifacts"
    settings = Settings(
        project_root=project_root,
        findf_root=findf_root,
        artifacts_dir=artifacts_dir,
        manifests_dir=artifacts_dir / "manifests",
        datasets_dir=artifacts_dir / "datasets",
        reports_dir=artifacts_dir / "reports",
        models_dir=artifacts_dir / "models",
        registry_dir=artifacts_dir / "registry",
        configs_dir=project_root / "configs",
    )
    settings.ensure_directories()
    return settings


def test_discover_manifest_from_findf_silver(tmp_path: Path) -> None:
    project_root = tmp_path / "trainer"
    findf_root = tmp_path / "findf"
    silver_dir = findf_root / "data" / "silver"
    silver_dir.mkdir(parents=True)
    (project_root / "configs").mkdir(parents=True)

    job_id = "job123"
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "timestamp": "2024-01-02",
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "volume": 1000.0,
                "adj_close": 10.5,
                "source": "polygon",
            }
        ]
    ).to_parquet(silver_dir / f"{job_id}_prices.parquet", index=False)
    pd.DataFrame(
        [
            {
                "id": "n1",
                "title": "Headline",
                "summary": "Summary",
                "tickers": "AAPL",
                "published_at": "2024-01-02T10:00:00Z",
                "source": "marketaux",
                "url": "https://example.com",
            }
        ]
    ).to_parquet(silver_dir / f"{job_id}_news.parquet", index=False)
    pd.DataFrame(
        [
            {
                "series_id": "FEDFUNDS",
                "date": "2024-01-01",
                "value": 5.25,
                "source": "fred",
            }
        ]
    ).to_parquet(silver_dir / f"{job_id}_macro.parquet", index=False)

    settings = make_settings(project_root, findf_root)
    manifest = discover_manifest(settings, job_id)
    manifest_path = persist_manifest(settings, manifest)

    assert manifest.job_id == job_id
    assert manifest.tickers == ["AAPL"]
    assert manifest.providers["prices"] == ["polygon"]
    assert manifest_path.exists()
