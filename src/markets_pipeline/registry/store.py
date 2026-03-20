from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..settings import Settings


def _metadata_path(settings: Settings, model_version: str) -> Path:
    return settings.registry_dir / f"{model_version}.json"


def register_model_metadata(settings: Settings, metadata: dict[str, Any]) -> Path:
    path = _metadata_path(settings, metadata["model_version"])
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_model_metadata(settings: Settings, model_version: str) -> dict[str, Any]:
    return json.loads(_metadata_path(settings, model_version).read_text(encoding="utf-8"))


def find_model_metadata(
    settings: Settings,
    model_family: str,
    horizon: str,
    snapshot_version: str,
) -> dict[str, Any] | None:
    for path in sorted(settings.registry_dir.glob("*.json"), reverse=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("model_family") == model_family
            and payload.get("horizon") == horizon
            and payload.get("snapshot_version") == snapshot_version
        ):
            return payload
    return None


def promote_model(settings: Settings, model_version: str) -> dict[str, Any]:
    payload = load_model_metadata(settings, model_version)
    payload["promoted"] = True
    register_model_metadata(settings, payload)
    active_path = settings.registry_dir / "active_models.json"
    active = {}
    if active_path.exists():
        active = json.loads(active_path.read_text(encoding="utf-8"))
    active[f"{payload['model_family']}::{payload['horizon']}"] = model_version
    active_path.write_text(json.dumps(active, indent=2), encoding="utf-8")
    return payload
