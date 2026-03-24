from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
WAREHOUSE_DIR = ARTIFACTS_DIR / "warehouse"
DATASET_DIR = ARTIFACTS_DIR / "datasets"
FEATURE_DIR = ARTIFACTS_DIR / "features"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"
CONTROL_DB_PATH = ARTIFACTS_DIR / "control_plane.sqlite3"
WAREHOUSE_PATH = WAREHOUSE_DIR / "quant.duckdb"


def ensure_directories() -> None:
    for directory in (
        ARTIFACTS_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        WAREHOUSE_DIR,
        DATASET_DIR,
        FEATURE_DIR,
        MODEL_DIR,
        REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
