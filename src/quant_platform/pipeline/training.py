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
from .features import FEATURE_COLUMNS, load_feature_frame

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


CheckpointHook = Callable[[int, dict[str, Any]], dict[str, Any]]


@dataclass
class TrainingResult:
    artifact_dir: Path
    metrics: dict[str, float]
    summary: dict[str, object]
    checkpoint_paths: list[str]
    feature_importance: list[dict[str, float]]
    warnings: list[str]


def train_model(
    model_version_id: str,
    feature_set_id: str,
    model_kind: str,
    config: dict[str, Any],
    checkpoint_hook: CheckpointHook | None = None,
) -> TrainingResult:
    ensure_directories()
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

    resolved_kind = model_kind
    if model_kind in {"gru", "temporal_cnn"}:
        warnings.append(f"{model_kind} is scaffolded through the snapshot MLP path in v1.")
        resolved_kind = "pytorch_mlp"

    if resolved_kind == "lightgbm":
        model, warning = _train_lightgbm_like(X_train, y_train, config)
        if warning:
            warnings.append(warning)
        checkpoint_paths = _save_pickle_checkpoint(model, scaler, checkpoint_dir / "checkpoint_epoch_1.pkl")
        predictions = model.predict(X_val)
    elif resolved_kind == "logistic_fusion":
        y_train_cls = (y_train > 0).astype(int)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train_scaled, y_train_cls)
        checkpoint_paths = _save_pickle_checkpoint(model, scaler, checkpoint_dir / "checkpoint_epoch_1.pkl")
        predictions = model.predict_proba(X_val_scaled)[:, 1] - 0.5
    else:
        model, predictions, checkpoint_paths, torch_warning = _train_checkpointable_mlp(
            X_train_scaled,
            y_train.to_numpy(),
            X_val_scaled,
            config,
            checkpoint_dir,
            checkpoint_hook,
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
    }
    return TrainingResult(
        artifact_dir=artifact_dir,
        metrics=metrics,
        summary=summary,
        checkpoint_paths=checkpoint_paths,
        feature_importance=importance,
        warnings=warnings,
    )


def load_predictor(artifact_dir: str | Path) -> Callable[[pd.DataFrame], np.ndarray]:
    artifact_path = Path(artifact_dir)
    metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
    with (artifact_path / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    model_kind = metadata["model_kind"]

    if model_kind in {"lightgbm", "logistic_fusion"}:
        with (artifact_path / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)

        def predict_fn(frame: pd.DataFrame) -> np.ndarray:
            scaled = scaler.transform(frame[FEATURE_COLUMNS].fillna(0.0))
            if model_kind == "logistic_fusion":
                return model.predict_proba(scaled)[:, 1] - 0.5
            return model.predict(frame[FEATURE_COLUMNS].fillna(0.0))

        return predict_fn

    if model_kind == "pytorch_mlp" and torch is not None:
        checkpoint_files = sorted((artifact_path / "checkpoints").glob("checkpoint_epoch_*.pt"))
        latest = checkpoint_files[-1]
        payload = torch.load(latest, map_location="cpu")
        network = _TorchMLP(input_dim=len(FEATURE_COLUMNS), hidden_dim=payload["hidden_dim"])
        network.load_state_dict(payload["state_dict"])
        network.eval()

        def predict_fn(frame: pd.DataFrame) -> np.ndarray:
            scaled = scaler.transform(frame[FEATURE_COLUMNS].fillna(0.0))
            with torch.no_grad():
                tensor = torch.tensor(scaled, dtype=torch.float32)
                return network(tensor).squeeze(-1).numpy()

        return predict_fn

    with (artifact_path / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        scaled = scaler.transform(frame[FEATURE_COLUMNS].fillna(0.0))
        return model.predict(scaled)

    return predict_fn


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
):
    epochs = int(config.get("epochs", 8))
    hidden_dim = int(config.get("hidden_dim", 64))
    checkpoint_paths: list[str] = []

    if torch is not None:
        network = _TorchMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
        optimizer = torch.optim.Adam(network.parameters(), lr=float(config.get("learning_rate", 0.01)))
        loss_fn = nn.MSELoss()
        train_inputs = torch.tensor(X_train, dtype=torch.float32)
        train_targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        val_inputs = torch.tensor(X_val, dtype=torch.float32)

        for epoch in range(1, epochs + 1):
            if checkpoint_hook:
                config = checkpoint_hook(epoch, dict(config))
                for group in optimizer.param_groups:
                    group["lr"] = float(config.get("learning_rate", 0.01))
            network.train()
            optimizer.zero_grad()
            outputs = network(train_inputs)
            loss = loss_fn(outputs, train_targets)
            loss.backward()
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
            predictions = network(val_inputs).squeeze(-1).numpy()
        return network, predictions, checkpoint_paths, None

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
    return model, predictions, checkpoint_paths, "PyTorch not installed; using sklearn MLPRegressor fallback."


def _compute_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    mae = float(mean_absolute_error(y_true, predictions))
    directional_accuracy = float((np.sign(predictions) == np.sign(y_true)).mean())
    prediction_series = pd.Series(predictions)
    truth_series = pd.Series(y_true)
    rank_ic = float(prediction_series.rank().corr(truth_series.rank(), method="pearson") or 0.0)
    spearman = float(prediction_series.corr(truth_series, method="spearman") or 0.0)
    calibration_gap = float(abs(prediction_series.mean() - truth_series.mean()))
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
        first_layer = model.network[0].weight.detach().numpy()
        importances = np.abs(first_layer).mean(axis=0)
    else:
        importances = np.zeros(len(FEATURE_COLUMNS))
    pairs = [
        {"feature": feature, "importance": float(score)}
        for feature, score in zip(FEATURE_COLUMNS, importances)
    ]
    return sorted(pairs, key=lambda row: row["importance"], reverse=True)
