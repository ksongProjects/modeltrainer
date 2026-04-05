from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..config import MODEL_DIR, ensure_directories
from ..runtime_profiles import autocast_context, normalize_runtime_settings, resolve_runtime
from .features import FEATURE_COLUMNS, load_feature_frame

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


CheckpointHook = Callable[[int, dict[str, Any]], dict[str, Any]]


@dataclass
class TrainingResult:
    artifact_dir: Path
    metrics: dict[str, float]
    summary: dict[str, object]
    checkpoint_paths: list[str]
    feature_importance: list[dict[str, float]]
    warnings: list[str]
    layer_metrics: dict[str, dict[str, float]] | None = None
    layer_artifacts: dict[str, dict[str, str]] | None = None
    layer_comparisons: dict[str, dict[str, Any]] | None = None


class ConstantProbabilityModel:
    def __init__(self, positive_probability: float):
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        rows = len(inputs)
        positive = np.full(rows, self.positive_probability, dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


def _finite_metric(value: object, default: float = 0.0) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else default


def train_model(
    model_version_id: str,
    feature_set_id: str,
    model_kind: str,
    config: dict[str, Any],
    checkpoint_hook: CheckpointHook | None = None,
) -> TrainingResult:
    ensure_directories()
    if model_kind == "layered_decision":
        from .layered_models import train_layer_stack

        artifact_dir = MODEL_DIR / model_version_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        frame = load_feature_frame(feature_set_id)
        layered_result = train_layer_stack(artifact_dir=artifact_dir, frame=frame, config=config)
        with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(layered_result["metadata"], handle, indent=2)
        return TrainingResult(
            artifact_dir=artifact_dir,
            metrics=layered_result["metrics"],
            summary={"feature_set_id": feature_set_id, **layered_result["summary"]},
            checkpoint_paths=layered_result["checkpoint_paths"],
            feature_importance=layered_result["feature_importance"],
            warnings=layered_result["warnings"],
            layer_metrics=layered_result["layer_metrics"],
            layer_artifacts=layered_result["layer_artifacts"],
            layer_comparisons=layered_result.get("layer_comparisons"),
        )

    frame = load_feature_frame(feature_set_id)
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "validation"].copy()
    X_train = train_df[FEATURE_COLUMNS].fillna(0.0)
    y_train = train_df["forward_return"]
    X_val = val_df[FEATURE_COLUMNS].fillna(0.0)
    y_val = val_df["forward_return"]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    artifact_dir = MODEL_DIR / model_version_id
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    runtime_settings = normalize_runtime_settings(config.get("runtime_settings"), _default_runtime_settings())
    runtime_summary = _cpu_runtime_summary(runtime_settings)

    resolved_kind = model_kind
    if model_kind in {"gru", "temporal_cnn"}:
        warnings.append(f"{model_kind} is scaffolded through the snapshot MLP path in v1.")
        resolved_kind = "pytorch_mlp"

    if resolved_kind == "lightgbm":
        model, warning = _train_lightgbm_like(X_train, y_train, config)
        if warning:
            warnings.append(warning)
        if runtime_settings["compute_target"] not in {"auto", "cpu"}:
            warnings.append(f"{resolved_kind} currently trains on CPU; requested compute target `{runtime_settings['compute_target']}` was ignored.")
        checkpoint_paths = _save_pickle_checkpoint(model, scaler, checkpoint_dir / "checkpoint_epoch_1.pkl")
        predictions = model.predict(X_val)
    elif resolved_kind == "logistic_fusion":
        y_train_cls = (y_train > 0).astype(int)
        if pd.Series(y_train_cls).nunique() < 2:
            warnings.append("logistic_fusion received a single-class target; using a constant probability baseline.")
            model = ConstantProbabilityModel(float(pd.Series(y_train_cls).mean()))
        else:
            model = LogisticRegression(max_iter=200)
            model.fit(X_train_scaled, y_train_cls)
        if runtime_settings["compute_target"] not in {"auto", "cpu"}:
            warnings.append(f"{resolved_kind} currently trains on CPU; requested compute target `{runtime_settings['compute_target']}` was ignored.")
        checkpoint_paths = _save_pickle_checkpoint(model, scaler, checkpoint_dir / "checkpoint_epoch_1.pkl")
        predictions = model.predict_proba(X_val_scaled)[:, 1] - 0.5
    else:
        model, predictions, checkpoint_paths, torch_warning, runtime_summary = _train_checkpointable_mlp(
            X_train_scaled,
            y_train.to_numpy(),
            X_val_scaled,
            config,
            checkpoint_dir,
            checkpoint_hook,
            runtime_settings,
        )
        if torch_warning:
            warnings.append(torch_warning)

    metrics = _compute_metrics(y_val.to_numpy(), np.asarray(predictions))
    importance = _feature_importance(model, resolved_kind)
    model_metadata = {
        "model_kind": resolved_kind,
        "requested_model_kind": model_kind,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "warnings": warnings,
        "scaler_path": str(artifact_dir / "scaler.pkl"),
        "runtime_settings": runtime_settings,
        "runtime_summary": runtime_summary,
    }
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(model_metadata, handle, indent=2)
    with (artifact_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    if resolved_kind in {"lightgbm", "logistic_fusion"}:
        with (artifact_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

    summary = {
        "model_kind": resolved_kind,
        "feature_set_id": feature_set_id,
        "rows": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
        },
        "checkpoints": checkpoint_paths,
        "feature_importance_top5": importance[:5],
        "runtime_settings": runtime_settings,
        "runtime_summary": runtime_summary,
    }
    return TrainingResult(
        artifact_dir=artifact_dir,
        metrics=metrics,
        summary=summary,
        checkpoint_paths=checkpoint_paths,
        feature_importance=importance,
        warnings=warnings,
        layer_metrics={},
        layer_artifacts={},
        layer_comparisons={},
    )


def load_predictor(artifact_dir: str | Path) -> Callable[[pd.DataFrame], np.ndarray]:
    return load_predictor_bundle(artifact_dir)[0]


def load_predictor_bundle(
    artifact_dir: str | Path,
) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict[str, Callable[[pd.DataFrame], np.ndarray]], dict[str, Any]]:
    artifact_path = Path(artifact_dir)
    metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
    model_kind = metadata["model_kind"]

    if model_kind == "layered_decision":
        from .layered_models import load_layered_bundle

        return load_layered_bundle(artifact_path)

    with (artifact_path / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)

    if model_kind in {"lightgbm", "logistic_fusion"}:
        with (artifact_path / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)

        def predict_fn(frame: pd.DataFrame) -> np.ndarray:
            scaled = scaler.transform(frame[FEATURE_COLUMNS].fillna(0.0))
            if model_kind == "logistic_fusion":
                return model.predict_proba(scaled)[:, 1] - 0.5
            return model.predict(frame[FEATURE_COLUMNS].fillna(0.0))

        return predict_fn, {}, metadata

    if model_kind == "pytorch_mlp" and torch is not None:
        checkpoint_files = sorted((artifact_path / "checkpoints").glob("checkpoint_epoch_*.pt"))
        latest = checkpoint_files[-1]
        payload = torch.load(latest, map_location="cpu")
        network = _TorchMLP(input_dim=len(FEATURE_COLUMNS), hidden_dim=payload["hidden_dim"])
        network.load_state_dict(payload["state_dict"])
        runtime = resolve_runtime(metadata.get("runtime_settings"), _default_runtime_settings())
        if runtime.device is not None:
            network.to(runtime.device)
        network.eval()

        def predict_fn(frame: pd.DataFrame) -> np.ndarray:
            scaled = scaler.transform(frame[FEATURE_COLUMNS].fillna(0.0))
            with torch.no_grad():
                tensor = torch.tensor(scaled, dtype=torch.float32, device=runtime.device)
                with autocast_context(runtime):
                    outputs = network(tensor).squeeze(-1)
                return outputs.detach().cpu().numpy()

        return predict_fn, {}, metadata

    with (artifact_path / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        scaled = scaler.transform(frame[FEATURE_COLUMNS].fillna(0.0))
        return model.predict(scaled)

    return predict_fn, {}, metadata


def _train_lightgbm_like(X_train: pd.DataFrame, y_train: pd.Series, config: dict[str, Any]):
    if lgb is not None:  # pragma: no branch - straightforward optional dependency
        model = lgb.LGBMRegressor(
            n_estimators=int(config.get("epochs", 100)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            random_state=7,
        )
        model.fit(X_train, y_train)
        return model, None
    model = HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.05)),
        max_depth=6,
        random_state=7,
    )
    model.fit(X_train, y_train)
    return model, "LightGBM not installed; using sklearn HistGradientBoostingRegressor fallback."


def _save_pickle_checkpoint(model: object, scaler: StandardScaler, checkpoint_path: Path) -> list[str]:
    with checkpoint_path.open("wb") as handle:
        pickle.dump({"model": model, "scaler": scaler}, handle)
    return [str(checkpoint_path)]


if nn is not None:
    class _TorchMLP(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            mid_dim = max(8, hidden_dim // 2)
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, 1),
            )

        def forward(self, inputs):  # pragma: no cover - tiny wrapper
            return self.network(inputs)
else:
    class _TorchMLP:  # pragma: no cover - used only to satisfy type checker when torch is absent
        pass


def _train_checkpointable_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    config: dict[str, Any],
    checkpoint_dir: Path,
    checkpoint_hook: CheckpointHook | None,
    runtime_settings: dict[str, Any],
):
    epochs = int(config.get("epochs", 8))
    hidden_dim = int(config.get("hidden_dim", 64))
    checkpoint_paths: list[str] = []

    if torch is not None:
        runtime = resolve_runtime(runtime_settings, _default_runtime_settings())
        network = _TorchMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
        if runtime.device is not None:
            network.to(runtime.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=float(config.get("learning_rate", 0.01)))
        loss_fn = nn.MSELoss()
        batch_size = int(runtime_settings.get("batch_size", 128))
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        scaler = torch.amp.GradScaler("cuda", enabled=runtime.amp_enabled and runtime.autocast_device_type == "cuda")

        for epoch in range(1, epochs + 1):
            if checkpoint_hook:
                config = checkpoint_hook(epoch, dict(config))
                for group in optimizer.param_groups:
                    group["lr"] = float(config.get("learning_rate", 0.01))
            network.train()
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(runtime.device)
                batch_targets = batch_targets.to(runtime.device)
                optimizer.zero_grad()
                with autocast_context(runtime):
                    outputs = network(batch_inputs)
                    loss = loss_fn(outputs, batch_targets)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=float(runtime_settings.get("gradient_clip_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=float(runtime_settings.get("gradient_clip_norm", 1.0)))
                    optimizer.step()
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "state_dict": network.state_dict(),
                    "hidden_dim": hidden_dim,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            checkpoint_paths.append(str(checkpoint_path))
        network.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_val, dtype=torch.float32, device=runtime.device)
            with autocast_context(runtime):
                predictions = network(val_inputs).squeeze(-1).detach().cpu().numpy()
        return network, predictions, checkpoint_paths, None, runtime.to_summary()

    model = MLPRegressor(
        hidden_layer_sizes=(hidden_dim, max(8, hidden_dim // 2)),
        learning_rate_init=float(config.get("learning_rate", 0.01)),
        warm_start=True,
        max_iter=1,
        random_state=7,
    )
    for epoch in range(1, epochs + 1):
        if checkpoint_hook:
            config = checkpoint_hook(epoch, dict(config))
            model.set_params(learning_rate_init=float(config.get("learning_rate", 0.01)))
        model.fit(X_train, y_train)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        with checkpoint_path.open("wb") as handle:
            pickle.dump(model, handle)
        checkpoint_paths.append(str(checkpoint_path))
    predictions = model.predict(X_val)
    return (
        model,
        predictions,
        checkpoint_paths,
        "PyTorch not installed; using sklearn MLPRegressor fallback.",
        _cpu_runtime_summary(runtime_settings, "PyTorch is unavailable in the current environment."),
    )


def _default_runtime_settings() -> dict[str, Any]:
    return {
        "compute_target": "auto",
        "precision_mode": "auto",
        "batch_size": 128,
        "sequence_length": 20,
        "gradient_clip_norm": 1.0,
    }


def _cpu_runtime_summary(runtime_settings: dict[str, Any], extra_note: str | None = None) -> dict[str, Any]:
    normalized = normalize_runtime_settings(runtime_settings, _default_runtime_settings())
    notes: list[str] = []
    if normalized["compute_target"] not in {"auto", "cpu"}:
        notes.append(f"Requested compute target `{normalized['compute_target']}` is not used by the current CPU model path.")
    if extra_note:
        notes.append(extra_note)
    return {
        "requested_compute_target": normalized["compute_target"],
        "resolved_compute_target": "cpu",
        "provider": "native",
        "backend": "cpu",
        "requested_precision_mode": normalized["precision_mode"],
        "precision_mode": "fp32",
        "amp_enabled": False,
        "notes": notes,
    }


def _compute_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    rmse = _finite_metric(np.sqrt(mean_squared_error(y_true, predictions)))
    mae = _finite_metric(mean_absolute_error(y_true, predictions))
    directional_accuracy = _finite_metric((np.sign(predictions) == np.sign(y_true)).mean())
    prediction_series = pd.Series(predictions)
    truth_series = pd.Series(y_true)
    rank_ic = _finite_metric(prediction_series.rank().corr(truth_series.rank(), method="pearson"))
    spearman = _finite_metric(prediction_series.corr(truth_series, method="spearman"))
    calibration_gap = _finite_metric(abs(prediction_series.mean() - truth_series.mean()))
    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
        "rank_ic": rank_ic,
        "spearman": spearman,
        "calibration_gap": calibration_gap,
    }


def _feature_importance(model: object, model_kind: str) -> list[dict[str, float]]:
    if model_kind == "logistic_fusion" and hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coefs_"):
        importances = np.abs(model.coefs_[0]).mean(axis=1)
    elif hasattr(model, "network"):
        first_layer = model.network[0].weight.detach().cpu().numpy()
        importances = np.abs(first_layer).mean(axis=0)
    else:
        importances = np.zeros(len(FEATURE_COLUMNS))
    pairs = [
        {"feature": feature, "importance": float(score)}
        for feature, score in zip(FEATURE_COLUMNS, importances)
    ]
    return sorted(pairs, key=lambda row: row["importance"], reverse=True)
