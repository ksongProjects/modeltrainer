from __future__ import annotations

import json

from .contracts.findf import discover_manifest, load_manifest, persist_manifest
from .datasets.snapshots import SnapshotBuildResult, build_snapshot_daily
from .models.finbert_features import score_news_with_finbert
from .models.fusion import train_fusion_model
from .models.tabular import train_tabular_expert
from .registry.store import load_model_metadata, promote_model as promote_registry_model
from .settings import Settings


def register_findf_run(settings: Settings, job_id: str) -> str:
    manifest = discover_manifest(settings, job_id)
    path = persist_manifest(settings, manifest)
    return str(path)


def build_snapshots(settings: Settings, job_id: str) -> SnapshotBuildResult:
    manifest = load_manifest(settings, job_id)
    feature_view = settings.load_feature_view()
    snapshot_version = f"{job_id}_{feature_view['version']}"
    score_news_with_finbert(settings, manifest, snapshot_version)
    return build_snapshot_daily(settings, manifest, feature_view)


def train_experts(settings: Settings, snapshot_version: str, horizon: str) -> dict[str, str]:
    lightgbm_result = train_tabular_expert(settings, snapshot_version, horizon, "lightgbm")
    catboost_result = train_tabular_expert(settings, snapshot_version, horizon, "catboost")
    return {
        "lightgbm": lightgbm_result.model_version,
        "catboost": catboost_result.model_version,
    }


def train_fusion(settings: Settings, snapshot_version: str, horizon: str) -> str:
    return train_fusion_model(settings, snapshot_version, horizon)


def evaluate_model(settings: Settings, model_version: str) -> str:
    metadata = load_model_metadata(settings, model_version)
    report_path = settings.reports_dir / f"{model_version}.json"
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return str(report_path)


def promote_model(settings: Settings, model_version: str) -> str:
    metadata = load_model_metadata(settings, model_version)
    if not metadata.get("metrics", {}).get("beats_baselines", False):
        raise ValueError(f"Model {model_version} did not beat baselines and cannot be promoted.")
    promoted = promote_registry_model(settings, model_version)
    return json.dumps(promoted, indent=2)
