from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainArtifactPaths:
    run_dir: Path
    metrics_path: Path
    oof_path: Path
    summary_path: Path
