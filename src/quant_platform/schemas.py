from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RuntimeSettingsRequest(BaseModel):
    compute_target: str | None = None
    precision_mode: str | None = None
    batch_size: int | None = Field(default=None, ge=8, le=4096)
    sequence_length: int | None = Field(default=None, ge=5, le=252)
    gradient_clip_norm: float | None = Field(default=None, ge=0.1, le=20.0)


class RuntimeSelfCheckRequest(BaseModel):
    compute_target: str = "auto"
    precision_mode: str = "auto"
    batch_size: int = Field(default=32, ge=8, le=1024)
    sequence_length: int = Field(default=20, ge=5, le=252)
    gradient_clip_norm: float = Field(default=1.0, ge=0.1, le=20.0)
    model_kind: str = "pytorch_mlp"
    input_dim: int = Field(default=8, ge=4, le=128)


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
    layer_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    runtime_settings: RuntimeSettingsRequest = Field(default_factory=RuntimeSettingsRequest)


class TestingRunRequest(BaseModel):
    model_version_id: str | None = None
    feature_set_version_id: str | None = None
    name: str = "Out-of-Sample Backtest"
    execution_mode: str = "paper"
    rebalance_decile: float = Field(default=0.1, gt=0, le=0.4)
    stress_iterations: int = Field(default=300, ge=50, le=5000)
    decision_top_k: int = Field(default=10, ge=1, le=50)


class ResearchLayerControlRequest(BaseModel):
    preferred_model_kind: str | None = None
    candidate_model_kinds: list[str] | None = None
    process_step_state: dict[str, bool] = Field(default_factory=dict)
    selection_metric: str | None = None
    runtime_settings: RuntimeSettingsRequest | None = None


class RunOverrideRequest(BaseModel):
    overrides: dict[str, Any]


class RunStateResponse(BaseModel):
    id: str
    state: str
    current_stage: str
    updated_at: str
