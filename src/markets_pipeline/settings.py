from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Settings:
    project_root: Path
    findf_root: Path
    artifacts_dir: Path
    manifests_dir: Path
    datasets_dir: Path
    reports_dir: Path
    models_dir: Path
    registry_dir: Path
    configs_dir: Path

    @classmethod
    def load(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        findf_root = Path(
            os.getenv("MARKETS_FINDF_ROOT", project_root.parent / "findf")
        ).resolve()
        artifacts_dir = Path(os.getenv("MARKETS_ARTIFACTS_DIR", project_root / "artifacts")).resolve()
        settings = cls(
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

    def ensure_directories(self) -> None:
        for path in (
            self.artifacts_dir,
            self.manifests_dir,
            self.datasets_dir,
            self.reports_dir,
            self.models_dir,
            self.registry_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def findf_data_dir(self) -> Path:
        return self.findf_root / "data"

    @property
    def silver_dir(self) -> Path:
        return self.findf_data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.findf_data_dir / "gold"

    def load_feature_view(self) -> dict[str, Any]:
        return _load_json(self.configs_dir / "feature_view.json")

    def load_horizons(self) -> list[dict[str, Any]]:
        payload = _load_json(self.configs_dir / "horizons.json")
        return payload["horizons"]

    def load_folds(self) -> dict[str, Any]:
        return _load_json(self.configs_dir / "folds.json")

    def load_model_params(self) -> dict[str, Any]:
        return _load_json(self.configs_dir / "model_params.json")
