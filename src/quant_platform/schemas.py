from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetCreateRequest(BaseModel):
    name: str = "Synthetic Daily + Events"
    source_id: str = "source_synthetic"
    num_tickers: int = Field(default=48, ge=10, le=500)
    num_days: int = Field(default=320, ge=120, le=2000)
    seed: int = 7
    tags: list[str] = Field(default_factory=list)


class DatasetImportRequest(BaseModel):
    path: str
    name: str | None = None
    source_id: str = "source_findf_parquet"
    tags: list[str] = Field(default_factory=list)


class SavedDatasetTagRequest(BaseModel):
    name: str


class FeatureMaterializationRequest(BaseModel):
    dataset_version_id: str
    name: str = "Core Multi-Factor Feature Set"
    winsor_limit: float = 3.0
    forecast_horizon_days: int = Field(default=5, ge=1, le=30)


class TrainingRunRequest(BaseModel):
    dataset_version_id: str | None = None
    feature_set_version_id: str | None = None
    model_spec_id: str | None = None
    model_kind: str = "lightgbm"
    name: str = "Baseline Snapshot Trainer"
    epochs: int = Field(default=8, ge=1, le=200)
    learning_rate: float = Field(default=0.01, gt=0, le=1.0)
    hidden_dim: int = Field(default=64, ge=8, le=1024)
    checkpoint_frequency: int = Field(default=1, ge=1, le=10)
    horizon_days: int = Field(default=5, ge=1, le=30)


class TestingRunRequest(BaseModel):
    model_version_id: str | None = None
    feature_set_version_id: str | None = None
    name: str = "Out-of-Sample Backtest"
    execution_mode: str = "paper"
    rebalance_decile: float = Field(default=0.1, gt=0, le=0.4)
    stress_iterations: int = Field(default=300, ge=50, le=5000)


class RunOverrideRequest(BaseModel):
    overrides: dict[str, Any]


class RunStateResponse(BaseModel):
    id: str
    state: str
    current_stage: str
    updated_at: str
