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

from .features import MACRO_TEXT_FEATURE_COLUMNS, TEXT_FEATURE_COLUMNS
from ..layer_controls import research_layer_runtime_defaults
from ..research_layers import (
    LAYER_FUNDAMENTAL_SIGNAL,
    LAYER_FUSION_DECISION,
    LAYER_MACRO_REGIME,
    LAYER_PRICE_SIGNAL,
    LAYER_SENTIMENT_SIGNAL,
)
from ..runtime_profiles import autocast_context, normalize_runtime_settings, resolve_runtime

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


LayerPredictor = Callable[[pd.DataFrame], np.ndarray]


class IdentityScaler:
    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(frame, dtype=float)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(frame, dtype=float)


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

LAYER_FEATURES: dict[str, list[str]] = {
    LAYER_PRICE_SIGNAL: [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "momentum_20d",
        "momentum_60d",
        "momentum_z",
        "volume_z",
        "composite_score",
    ],
    LAYER_FUNDAMENTAL_SIGNAL: [
        "ev_ebitda",
        "roic",
        "value_z",
        "quality_sector_z",
        "composite_score",
    ],
    LAYER_SENTIMENT_SIGNAL: [
        "sentiment_1d",
        "sentiment_5d",
        "sentiment_z",
        "earnings_signal",
        "earnings_z",
        "composite_score",
        *TEXT_FEATURE_COLUMNS,
    ],
    LAYER_MACRO_REGIME: [
        "macro_surprise",
        "macro_z",
        "macro_rate",
        "sentiment_z",
        "momentum_z",
        "composite_score",
        *MACRO_TEXT_FEATURE_COLUMNS,
    ],
}

LAYER_ALIASES = {
    LAYER_PRICE_SIGNAL: "price_signal",
    LAYER_FUNDAMENTAL_SIGNAL: "fundamental_signal",
    LAYER_SENTIMENT_SIGNAL: "sentiment_signal",
    LAYER_MACRO_REGIME: "macro_regime",
    LAYER_FUSION_DECISION: "fusion_decision",
}


@dataclass
class LayerArtifact:
    layer_id: str
    layer_name: str
    resolved_model_kind: str
    requested_model_kind: str
    feature_columns: list[str]
    metrics: dict[str, float]
    warnings: list[str]
    model_dir: Path
    predictor: LayerPredictor
    feature_importance: list[dict[str, float]]
    report_path: Path
    comparison_report_path: Path
    candidate_metrics: dict[str, dict[str, float]]
    selection_metric: str
    process_step_state: dict[str, bool]
    runtime_settings: dict[str, Any]
    runtime_summary: dict[str, Any]
    primary_model_path: Path


def train_layer_stack(
    artifact_dir: Path,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "validation"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Layered decision training requires non-empty train and validation splits.")

    val_dates = sorted(pd.to_datetime(val_df["effective_at"]).dt.normalize().unique())
    if len(val_dates) < 2:
        meta_train_df = val_df.copy()
        meta_eval_df = val_df.copy()
    else:
        split_index = max(1, len(val_dates) // 2)
        meta_train_dates = set(val_dates[:split_index])
        meta_eval_dates = set(val_dates[split_index:])
        meta_train_df = val_df[pd.to_datetime(val_df["effective_at"]).dt.normalize().isin(meta_train_dates)].copy()
        meta_eval_df = val_df[pd.to_datetime(val_df["effective_at"]).dt.normalize().isin(meta_eval_dates)].copy()
        if meta_eval_df.empty:
            meta_eval_df = meta_train_df.copy()

    layers_dir = artifact_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    layer_configs = config.get("layer_configs") if isinstance(config.get("layer_configs"), dict) else {}
    layer_artifacts: dict[str, LayerArtifact] = {}
    layer_metrics: dict[str, dict[str, float]] = {}
    layer_reports: dict[str, str] = {}
    layer_comparisons: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for layer_id in (LAYER_PRICE_SIGNAL, LAYER_FUNDAMENTAL_SIGNAL, LAYER_SENTIMENT_SIGNAL, LAYER_MACRO_REGIME):
        layer_config = _merge_layer_config(config, layer_configs.get(layer_id, {}), layer_id)
        columns = _resolve_feature_columns(frame, _apply_layer_feature_steps(layer_id, LAYER_FEATURES[layer_id], layer_config))
        artifact = _train_layer_candidates(
            layer_id=layer_id,
            layer_name=LAYER_ALIASES[layer_id],
            model_dir=layers_dir / LAYER_ALIASES[layer_id],
            train_df=train_df,
            eval_df=val_df,
            feature_columns=columns,
            config=layer_config,
        )
        layer_artifacts[layer_id] = artifact
        layer_metrics[layer_id] = artifact.metrics
        layer_reports[layer_id] = str(artifact.report_path)
        layer_comparisons[layer_id] = {
            "selection_metric": artifact.selection_metric,
            "selected_model_kind": artifact.resolved_model_kind,
            "requested_model_kind": artifact.requested_model_kind,
            "candidate_metrics": artifact.candidate_metrics,
            "comparison_report_path": str(artifact.comparison_report_path),
            "feature_columns": artifact.feature_columns,
            "process_step_state": artifact.process_step_state,
            "runtime_settings": artifact.runtime_settings,
            "runtime_summary": artifact.runtime_summary,
        }
        warnings.extend(artifact.warnings)
        validation_scores = pd.Series(artifact.predictor(val_df), index=val_df.index)
        meta_train_df[f"{LAYER_ALIASES[layer_id]}_score"] = validation_scores.reindex(meta_train_df.index).to_numpy()
        meta_eval_df[f"{LAYER_ALIASES[layer_id]}_score"] = validation_scores.reindex(meta_eval_df.index).to_numpy()

    fusion_config = _merge_layer_config(config, layer_configs.get(LAYER_FUSION_DECISION, {}), LAYER_FUSION_DECISION)
    fusion_feature_columns = _resolve_feature_columns(
        meta_train_df,
        _apply_fusion_feature_steps(
            [f"{LAYER_ALIASES[layer_id]}_score" for layer_id in layer_artifacts],
            fusion_config,
        ),
    )
    fusion_artifact = _train_layer_candidates(
        layer_id=LAYER_FUSION_DECISION,
        layer_name=LAYER_ALIASES[LAYER_FUSION_DECISION],
        model_dir=artifact_dir / "fusion",
        train_df=meta_train_df,
        eval_df=meta_eval_df,
        feature_columns=fusion_feature_columns,
        config=fusion_config,
    )
    layer_artifacts[LAYER_FUSION_DECISION] = fusion_artifact
    layer_metrics[LAYER_FUSION_DECISION] = fusion_artifact.metrics
    layer_reports[LAYER_FUSION_DECISION] = str(fusion_artifact.report_path)
    layer_comparisons[LAYER_FUSION_DECISION] = {
        "selection_metric": fusion_artifact.selection_metric,
        "selected_model_kind": fusion_artifact.resolved_model_kind,
        "requested_model_kind": fusion_artifact.requested_model_kind,
        "candidate_metrics": fusion_artifact.candidate_metrics,
        "comparison_report_path": str(fusion_artifact.comparison_report_path),
        "feature_columns": fusion_artifact.feature_columns,
        "process_step_state": fusion_artifact.process_step_state,
        "runtime_settings": fusion_artifact.runtime_settings,
        "runtime_summary": fusion_artifact.runtime_summary,
    }
    warnings.extend(fusion_artifact.warnings)

    layer_registry = {
        "layers": [
            {
                "layer_id": artifact.layer_id,
                "layer_name": artifact.layer_name,
                "requested_model_kind": artifact.requested_model_kind,
                "resolved_model_kind": artifact.resolved_model_kind,
                "feature_columns": artifact.feature_columns,
                "metrics": artifact.metrics,
                "warnings": artifact.warnings,
                "model_dir": str(artifact.model_dir),
                "report_path": str(artifact.report_path),
                "comparison_report_path": str(artifact.comparison_report_path),
                "candidate_metrics": artifact.candidate_metrics,
                "selection_metric": artifact.selection_metric,
                "process_step_state": artifact.process_step_state,
                "runtime_settings": artifact.runtime_settings,
                "runtime_summary": artifact.runtime_summary,
            }
            for artifact in layer_artifacts.values()
        ],
        "fusion_feature_columns": fusion_feature_columns,
    }
    layer_registry_path = artifact_dir / "layer_registry.json"
    layer_registry_path.write_text(json.dumps(layer_registry, indent=2), encoding="utf-8")

    validation_report_path = artifact_dir / "validation_decision_report.json"
    validation_report = _build_validation_decision_report(
        meta_eval_df=meta_eval_df,
        fusion_predictor=fusion_artifact.predictor,
        layer_metrics=layer_metrics,
        warnings=warnings,
    )
    validation_report_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    return {
        "metrics": fusion_artifact.metrics,
        "warnings": warnings,
        "feature_importance": fusion_artifact.feature_importance,
        "layer_metrics": layer_metrics,
        "layer_artifacts": {
            layer_id: {
                "selected_model_dir": str(artifact.model_dir),
                "layer_report": str(artifact.report_path),
                "comparison_report": str(artifact.comparison_report_path),
            }
            for layer_id, artifact in layer_artifacts.items()
        }
        | {
            LAYER_FUSION_DECISION: {
                "selected_model_dir": str(fusion_artifact.model_dir),
                "layer_report": str(fusion_artifact.report_path),
                "comparison_report": str(fusion_artifact.comparison_report_path),
                "validation_decision_report": str(validation_report_path),
            }
        },
        "summary": {
            "model_kind": "layered_decision",
            "rows": {
                "train": int(len(train_df)),
                "validation": int(len(val_df)),
                "meta_train": int(len(meta_train_df)),
                "meta_eval": int(len(meta_eval_df)),
            },
            "layers": {
                layer_id: {
                    "model_kind": artifact.resolved_model_kind,
                    "feature_columns": artifact.feature_columns,
                    "metrics": artifact.metrics,
                    "candidate_metrics": artifact.candidate_metrics,
                    "selection_metric": artifact.selection_metric,
                    "process_step_state": artifact.process_step_state,
                    "runtime_settings": artifact.runtime_settings,
                    "runtime_summary": artifact.runtime_summary,
                }
                for layer_id, artifact in layer_artifacts.items()
            },
            "artifacts": {
                "layer_registry": str(layer_registry_path),
                "validation_decision_report": str(validation_report_path),
            },
        },
        "checkpoint_paths": [str(fusion_artifact.primary_model_path)],
        "metadata": {
            "model_kind": "layered_decision",
            "requested_model_kind": "layered_decision",
            "feature_columns": fusion_feature_columns,
            "metrics": fusion_artifact.metrics,
            "warnings": warnings,
            "layer_registry_path": str(layer_registry_path),
            "validation_decision_report": str(validation_report_path),
            "runtime_summary": fusion_artifact.runtime_summary,
        },
        "layer_comparisons": layer_comparisons,
    }


def load_layered_bundle(artifact_dir: str | Path) -> tuple[LayerPredictor, dict[str, LayerPredictor], dict[str, Any]]:
    artifact_path = Path(artifact_dir)
    metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
    layer_registry_path = Path(metadata["layer_registry_path"])
    layer_registry = json.loads(layer_registry_path.read_text(encoding="utf-8"))
    layer_predictors: dict[str, LayerPredictor] = {}
    fusion_model_dir = artifact_path / "fusion"
    for layer in layer_registry["layers"]:
        layer_id = str(layer["layer_id"])
        if layer_id == LAYER_FUSION_DECISION:
            fusion_model_dir = Path(layer["model_dir"])
            continue
        layer_predictors[layer_id] = _load_layer_predictor(Path(layer["model_dir"]))

    fusion_predictor = _load_layer_predictor(fusion_model_dir)

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        fusion_frame = pd.DataFrame(index=frame.index)
        for layer_id, predictor in layer_predictors.items():
            fusion_frame[f"{LAYER_ALIASES[layer_id]}_score"] = predictor(frame)
        return fusion_predictor(fusion_frame)

    return predict_fn, layer_predictors, {**metadata, "layer_registry": layer_registry}


def _merge_layer_config(base_config: dict[str, Any], layer_override: dict[str, Any], layer_id: str) -> dict[str, Any]:
    defaults = {
        "epochs": int(base_config.get("epochs", 8)),
        "learning_rate": float(base_config.get("learning_rate", 0.01)),
        "hidden_dim": int(base_config.get("hidden_dim", 64)),
        "selection_metric": "rank_ic",
        "process_step_state": {},
        "runtime_settings": normalize_runtime_settings(
            (base_config.get("runtime_settings") if isinstance(base_config.get("runtime_settings"), dict) else {}),
            research_layer_runtime_defaults(layer_id),
        ),
    }
    if layer_id == LAYER_FUSION_DECISION:
        defaults["model_kind"] = "logistic_fusion"
    elif layer_id == LAYER_PRICE_SIGNAL:
        defaults["model_kind"] = "gru"
    elif layer_id == LAYER_FUNDAMENTAL_SIGNAL:
        defaults["model_kind"] = "lightgbm"
    elif layer_id == LAYER_SENTIMENT_SIGNAL:
        defaults["model_kind"] = "pytorch_mlp"
    else:
        defaults["model_kind"] = "logistic_fusion"
    defaults.update(layer_override)
    defaults["runtime_settings"] = normalize_runtime_settings(
        defaults.get("runtime_settings") if isinstance(defaults.get("runtime_settings"), dict) else {},
        research_layer_runtime_defaults(layer_id),
    )
    return defaults


def _resolve_feature_columns(frame: pd.DataFrame, requested_columns: list[str]) -> list[str]:
    available = [column for column in requested_columns if column in frame.columns]
    if not available:
        raise ValueError(f"No requested feature columns are available: {requested_columns}")
    return available


def _apply_layer_feature_steps(layer_id: str, feature_columns: list[str], config: dict[str, Any]) -> list[str]:
    process_state = config.get("process_step_state") if isinstance(config.get("process_step_state"), dict) else {}
    filtered = list(feature_columns)
    if not process_state.get("include_cross_signal_composite", True) and "composite_score" in filtered:
        filtered.remove("composite_score")
    if layer_id == LAYER_PRICE_SIGNAL and not process_state.get("include_volume_context", True):
        filtered = [column for column in filtered if column not in {"volume", "volume_z"}]
    if layer_id == LAYER_SENTIMENT_SIGNAL and not process_state.get("include_earnings_context", True):
        filtered = [column for column in filtered if column not in {"earnings_signal", "earnings_z"}]
    if layer_id == LAYER_SENTIMENT_SIGNAL and not process_state.get("include_text_embeddings", True):
        filtered = [column for column in filtered if column not in set(TEXT_FEATURE_COLUMNS)]
    if layer_id == LAYER_MACRO_REGIME and not process_state.get("include_sentiment_context", True):
        filtered = [column for column in filtered if column != "sentiment_z"]
    if layer_id == LAYER_MACRO_REGIME and not process_state.get("include_momentum_context", True):
        filtered = [column for column in filtered if column != "momentum_z"]
    if layer_id == LAYER_MACRO_REGIME and not process_state.get("include_macro_text_embeddings", True):
        filtered = [column for column in filtered if column not in set(MACRO_TEXT_FEATURE_COLUMNS)]
    return filtered


def _apply_fusion_feature_steps(feature_columns: list[str], config: dict[str, Any]) -> list[str]:
    process_state = config.get("process_step_state") if isinstance(config.get("process_step_state"), dict) else {}
    filtered = list(feature_columns)
    exclusions = {
        "include_price_signal": "price_signal_score",
        "include_fundamental_signal": "fundamental_signal_score",
        "include_sentiment_signal": "sentiment_signal_score",
        "include_macro_signal": "macro_regime_score",
    }
    for step_id, column_name in exclusions.items():
        if not process_state.get(step_id, True):
            filtered = [column for column in filtered if column != column_name]
    return filtered


def _train_layer_candidates(
    layer_id: str,
    layer_name: str,
    model_dir: Path,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> LayerArtifact:
    model_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = model_dir / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    candidate_model_kinds = list(dict.fromkeys(config.get("candidate_model_kinds") or [config.get("model_kind", "lightgbm")]))
    selection_metric = str(config.get("selection_metric", "rank_ic"))
    process_step_state = {
        str(key): bool(value)
        for key, value in (config.get("process_step_state") if isinstance(config.get("process_step_state"), dict) else {}).items()
    }
    runtime_settings = normalize_runtime_settings(
        config.get("runtime_settings") if isinstance(config.get("runtime_settings"), dict) else {},
        research_layer_runtime_defaults(layer_id),
    )
    candidate_artifacts: list[LayerArtifact] = []
    for model_kind in candidate_model_kinds:
        candidate_config = dict(config)
        candidate_config["model_kind"] = model_kind
        candidate_config["runtime_settings"] = runtime_settings
        candidate_artifacts.append(
            _train_single_layer_model(
                layer_id=layer_id,
                layer_name=layer_name,
                model_dir=candidate_root / str(model_kind),
                train_df=train_df,
                eval_df=eval_df,
                feature_columns=feature_columns,
                config=candidate_config,
                process_step_state=process_step_state,
            )
        )
    selected_artifact = max(
        candidate_artifacts,
        key=lambda artifact: _selection_score(artifact.metrics, selection_metric),
    )
    comparison_report_path = model_dir / "model_comparison.json"
    comparison_report = {
        "layer_id": layer_id,
        "selection_metric": selection_metric,
        "selected_model_kind": selected_artifact.resolved_model_kind,
        "requested_model_kind": selected_artifact.requested_model_kind,
        "feature_columns": feature_columns,
        "process_step_state": process_step_state,
        "runtime_settings": runtime_settings,
        "candidates": {
            artifact.requested_model_kind: {
                "resolved_model_kind": artifact.resolved_model_kind,
                "metrics": artifact.metrics,
                "warnings": artifact.warnings,
                "model_dir": str(artifact.model_dir),
                "runtime_summary": artifact.runtime_summary,
            }
            for artifact in candidate_artifacts
        },
    }
    comparison_report_path.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
    return LayerArtifact(
        layer_id=selected_artifact.layer_id,
        layer_name=selected_artifact.layer_name,
        resolved_model_kind=selected_artifact.resolved_model_kind,
        requested_model_kind=selected_artifact.requested_model_kind,
        feature_columns=selected_artifact.feature_columns,
        metrics=selected_artifact.metrics,
        warnings=selected_artifact.warnings,
        model_dir=selected_artifact.model_dir,
        predictor=selected_artifact.predictor,
        feature_importance=selected_artifact.feature_importance,
        report_path=selected_artifact.report_path,
        comparison_report_path=comparison_report_path,
        candidate_metrics={artifact.requested_model_kind: artifact.metrics for artifact in candidate_artifacts},
        selection_metric=selection_metric,
        process_step_state=process_step_state,
        runtime_settings=selected_artifact.runtime_settings,
        runtime_summary=selected_artifact.runtime_summary,
        primary_model_path=selected_artifact.primary_model_path,
    )


def _train_single_layer_model(
    layer_id: str,
    layer_name: str,
    model_dir: Path,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    process_step_state: dict[str, bool],
) -> LayerArtifact:
    model_dir.mkdir(parents=True, exist_ok=True)
    X_train = train_df[feature_columns].fillna(0.0)
    y_train = train_df["forward_return"]
    X_eval = eval_df[feature_columns].fillna(0.0)
    y_eval = eval_df["forward_return"]
    warnings: list[str] = []
    runtime_settings = normalize_runtime_settings(
        config.get("runtime_settings") if isinstance(config.get("runtime_settings"), dict) else {},
        research_layer_runtime_defaults(layer_id),
    )
    runtime_summary = _cpu_runtime_summary(runtime_settings)
    primary_model_path = model_dir / "model.pkl"

    requested_model_kind = str(config.get("model_kind", "lightgbm"))
    resolved_model_kind = requested_model_kind
    if requested_model_kind in {"gru", "temporal_cnn"} and torch is None:
        resolved_model_kind = "pytorch_mlp"
        warnings.append(f"{requested_model_kind} requires torch; falling back to the layer MLP baseline.")

    use_scaler = bool(process_step_state.get("standard_scale_inputs", resolved_model_kind != "lightgbm"))
    scaler: StandardScaler | IdentityScaler = StandardScaler() if use_scaler else IdentityScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    if resolved_model_kind == "lightgbm":
        model, warning = _train_lightgbm_like(pd.DataFrame(X_train_scaled, columns=feature_columns), y_train, config)
        predictions = model.predict(pd.DataFrame(X_eval_scaled, columns=feature_columns))
        if warning:
            warnings.append(warning)
        if runtime_settings["compute_target"] not in {"auto", "cpu"}:
            warnings.append(f"{resolved_model_kind} currently trains on CPU; requested compute target `{runtime_settings['compute_target']}` was ignored.")
    elif resolved_model_kind == "logistic_fusion":
        y_train_cls = (y_train > 0).astype(int)
        if pd.Series(y_train_cls).nunique() < 2:
            warnings.append(f"{layer_name} received a single-class target; using a constant probability baseline.")
            model = ConstantProbabilityModel(float(pd.Series(y_train_cls).mean()))
        else:
            model = LogisticRegression(max_iter=200)
            model.fit(X_train_scaled, y_train_cls)
        predictions = model.predict_proba(X_eval_scaled)[:, 1] - 0.5
        if runtime_settings["compute_target"] not in {"auto", "cpu"}:
            warnings.append(f"{resolved_model_kind} currently trains on CPU; requested compute target `{runtime_settings['compute_target']}` was ignored.")
    else:
        try:
            if resolved_model_kind in {"gru", "temporal_cnn"}:
                train_scaled_df = train_df.copy()
                train_scaled_df[feature_columns] = X_train_scaled
                eval_reference_df = pd.concat([train_df, eval_df], axis=0).copy()
                eval_reference_inputs = eval_reference_df[feature_columns].fillna(0.0)
                eval_reference_df[feature_columns] = scaler.transform(eval_reference_inputs)
                train_sequences, train_targets = _build_sequence_dataset(
                    reference_df=train_scaled_df,
                    target_df=train_df,
                    feature_columns=feature_columns,
                    sequence_length=int(runtime_settings["sequence_length"]),
                )
                eval_sequences, eval_targets = _build_sequence_dataset(
                    reference_df=eval_reference_df,
                    target_df=eval_df,
                    feature_columns=feature_columns,
                    sequence_length=int(runtime_settings["sequence_length"]),
                )
                model, predictions, runtime_summary = _train_torch_regressor(
                    model_kind=resolved_model_kind,
                    train_inputs=train_sequences,
                    train_targets=train_targets,
                    eval_inputs=eval_sequences,
                    config=config,
                    runtime_settings=runtime_settings,
                    input_dim=len(feature_columns),
                )
                y_eval = pd.Series(eval_targets)
                primary_model_path = model_dir / "model.pt"
            elif torch is not None:
                model, predictions, runtime_summary = _train_torch_regressor(
                    model_kind=resolved_model_kind,
                    train_inputs=np.asarray(X_train_scaled, dtype=np.float32),
                    train_targets=y_train.to_numpy(dtype=np.float32),
                    eval_inputs=np.asarray(X_eval_scaled, dtype=np.float32),
                    config=config,
                    runtime_settings=runtime_settings,
                    input_dim=len(feature_columns),
                )
                primary_model_path = model_dir / "model.pt"
            else:
                raise RuntimeError("torch_unavailable")
        except Exception as exc:
            fallback_reason = "PyTorch is unavailable in the current environment." if str(exc) == "torch_unavailable" else str(exc)
            warnings.append(f"Torch training path failed for {requested_model_kind}; using sklearn MLP fallback. Detail: {fallback_reason}")
            resolved_model_kind = "pytorch_mlp"
            model = MLPRegressor(
                hidden_layer_sizes=(int(config.get("hidden_dim", 64)), max(8, int(config.get("hidden_dim", 64)) // 2)),
                learning_rate_init=float(config.get("learning_rate", 0.01)),
                max_iter=max(50, int(config.get("epochs", 8)) * 20),
                random_state=7,
            )
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_eval_scaled)
            runtime_summary = _cpu_runtime_summary(runtime_settings, fallback_reason)
            primary_model_path = model_dir / "model.pkl"

    metrics = _compute_metrics(y_eval.to_numpy(), np.asarray(predictions))
    importance = _feature_importance(model, resolved_model_kind, feature_columns)

    with (model_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    if resolved_model_kind in {"gru", "temporal_cnn", "pytorch_mlp"} and hasattr(model, "state_dict"):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": len(feature_columns),
                "hidden_dim": int(config.get("hidden_dim", 64)),
                "sequence_length": int(runtime_settings["sequence_length"]),
                "model_kind": resolved_model_kind,
            },
            model_dir / "model.pt",
        )
    else:
        with (model_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

    metadata = {
        "layer_id": layer_id,
        "layer_name": layer_name,
        "model_kind": resolved_model_kind,
        "requested_model_kind": requested_model_kind,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "warnings": warnings,
        "use_scaler": use_scaler,
        "process_step_state": process_step_state,
        "runtime_settings": runtime_settings,
        "runtime_summary": runtime_summary,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_path = model_dir / "layer_report.json"
    report_path.write_text(
        json.dumps(
            {
                "layer_id": layer_id,
                "metrics": metrics,
                "feature_importance_top5": importance[:5],
                "model_kind": resolved_model_kind,
                "rows": {"train": int(len(train_df)), "eval": int(len(eval_df))},
                "runtime_summary": runtime_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    predictor = _load_layer_predictor(model_dir)
    return LayerArtifact(
        layer_id=layer_id,
        layer_name=layer_name,
        resolved_model_kind=resolved_model_kind,
        requested_model_kind=requested_model_kind,
        feature_columns=feature_columns,
        metrics=metrics,
        warnings=warnings,
        model_dir=model_dir,
        predictor=predictor,
        feature_importance=importance,
        report_path=report_path,
        comparison_report_path=model_dir / "model_comparison.json",
        candidate_metrics={requested_model_kind: metrics},
        selection_metric=str(config.get("selection_metric", "rank_ic")),
        process_step_state=process_step_state,
        runtime_settings=runtime_settings,
        runtime_summary=runtime_summary,
        primary_model_path=primary_model_path,
    )


def _load_layer_predictor(model_dir: Path) -> LayerPredictor:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_columns = list(metadata["feature_columns"])
    model_kind = str(metadata["model_kind"])
    with (model_dir / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    runtime = resolve_runtime(metadata.get("runtime_settings"), research_layer_runtime_defaults(str(metadata["layer_id"])))

    if model_kind in {"gru", "temporal_cnn", "pytorch_mlp"} and torch is not None and (model_dir / "model.pt").exists():
        payload = torch.load(model_dir / "model.pt", map_location="cpu")
        model = _build_torch_model(
            model_kind=model_kind,
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
        )
        model.load_state_dict(payload["state_dict"])
        if runtime.device is not None:
            model.to(runtime.device)
        model.eval()
        sequence_length = int(payload.get("sequence_length", metadata.get("runtime_settings", {}).get("sequence_length", 20)))

        def predict_fn(frame: pd.DataFrame) -> np.ndarray:
            inputs = frame[feature_columns].fillna(0.0)
            scaled = scaler.transform(inputs)
            if model_kind in {"gru", "temporal_cnn"}:
                scaled_frame = frame.copy()
                scaled_frame[feature_columns] = scaled
                sequence_inputs, _ = _build_sequence_dataset(
                    reference_df=scaled_frame,
                    target_df=scaled_frame,
                    feature_columns=feature_columns,
                    sequence_length=sequence_length,
                )
                return _predict_torch_model(model, sequence_inputs, runtime)
            return _predict_torch_model(model, np.asarray(scaled, dtype=np.float32), runtime)

        return predict_fn

    with (model_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        inputs = frame[feature_columns].fillna(0.0)
        scaled = scaler.transform(inputs)
        if model_kind == "logistic_fusion":
            return model.predict_proba(scaled)[:, 1] - 0.5
        return model.predict(scaled if model_kind != "lightgbm" else pd.DataFrame(scaled, columns=feature_columns))

    return predict_fn


if nn is not None:
    class _TorchTabularMLP(nn.Module):  # type: ignore[misc]
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

        def forward(self, inputs):
            return self.network(inputs)


    class _TorchGRURegressor(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                nn.GELU(),
                nn.Linear(max(8, hidden_dim // 2), 1),
            )

        def forward(self, inputs):
            outputs, _ = self.gru(inputs)
            return self.head(outputs[:, -1, :])


    class _TorchTemporalCNN(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            reduced_dim = max(8, hidden_dim // 2)
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, reduced_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(reduced_dim, 1)

        def forward(self, inputs):
            channels_first = inputs.transpose(1, 2)
            encoded = self.encoder(channels_first).squeeze(-1)
            return self.head(encoded)
else:
    class _TorchTabularMLP:  # pragma: no cover - type fallback only
        pass


    class _TorchGRURegressor:  # pragma: no cover - type fallback only
        pass


    class _TorchTemporalCNN:  # pragma: no cover - type fallback only
        pass


def _build_torch_model(model_kind: str, input_dim: int, hidden_dim: int):
    if model_kind == "gru":
        return _TorchGRURegressor(input_dim=input_dim, hidden_dim=hidden_dim)
    if model_kind == "temporal_cnn":
        return _TorchTemporalCNN(input_dim=input_dim, hidden_dim=hidden_dim)
    return _TorchTabularMLP(input_dim=input_dim, hidden_dim=hidden_dim)


def _train_torch_regressor(
    model_kind: str,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    eval_inputs: np.ndarray,
    config: dict[str, Any],
    runtime_settings: dict[str, Any],
    input_dim: int,
) -> tuple[object, np.ndarray, dict[str, Any]]:
    if torch is None or TensorDataset is None or DataLoader is None:
        raise RuntimeError("torch_unavailable")
    runtime = resolve_runtime(runtime_settings, _default_runtime_settings())
    model = _build_torch_model(model_kind=model_kind, input_dim=input_dim, hidden_dim=int(config.get("hidden_dim", 64)))
    if runtime.device is not None:
        model.to(runtime.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.01)))
    loss_fn = nn.MSELoss()
    batch_size = int(runtime_settings.get("batch_size", 128))
    gradient_clip_norm = float(runtime_settings.get("gradient_clip_norm", 1.0))
    epochs = max(1, int(config.get("epochs", 8)))
    train_dataset = TensorDataset(
        torch.tensor(train_inputs, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32).unsqueeze(-1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.amp_enabled and runtime.autocast_device_type == "cuda")

    for _ in range(epochs):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(runtime.device)
            batch_targets = batch_targets.to(runtime.device)
            optimizer.zero_grad()
            with autocast_context(runtime):
                outputs = model(batch_inputs)
                loss = loss_fn(outputs, batch_targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()

    predictions = _predict_torch_model(model, eval_inputs, runtime)
    return model, predictions, runtime.to_summary()


def _predict_torch_model(model: object, inputs: np.ndarray, runtime) -> np.ndarray:
    if torch is None:
        return np.zeros(len(inputs), dtype=float)
    tensor = torch.tensor(inputs, dtype=torch.float32, device=runtime.device)
    with torch.no_grad():
        with autocast_context(runtime):
            outputs = model(tensor).squeeze(-1)
    return outputs.detach().cpu().numpy()


def _build_sequence_dataset(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if target_df.empty:
        return (
            np.zeros((0, sequence_length, len(feature_columns)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    reference = reference_df[["ticker", "effective_at", *feature_columns]].copy()
    reference["effective_at"] = pd.to_datetime(reference["effective_at"], utc=True).dt.tz_localize(None)
    reference = reference.sort_values(["ticker", "effective_at"]).reset_index(drop=True)
    target = target_df[["ticker", "effective_at", "forward_return"]].copy()
    target["effective_at"] = pd.to_datetime(target["effective_at"], utc=True).dt.tz_localize(None)

    reference_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ticker, group in reference.groupby("ticker", sort=False):
        dates = pd.to_datetime(group["effective_at"]).to_numpy(dtype="datetime64[ns]")
        matrix = group[feature_columns].to_numpy(dtype=np.float32)
        reference_groups[str(ticker)] = (dates, matrix)

    sequence_rows: list[np.ndarray] = []
    target_values: list[float] = []
    for _, row in target.iterrows():
        ticker = str(row["ticker"])
        dates, matrix = reference_groups.get(
            ticker,
            (
                np.asarray([], dtype="datetime64[ns]"),
                np.zeros((0, len(feature_columns)), dtype=np.float32),
            ),
        )
        cutoff = np.searchsorted(dates, np.datetime64(row["effective_at"]), side="right")
        window = matrix[max(0, cutoff - sequence_length):cutoff]
        if len(window) < sequence_length:
            pad = np.zeros((sequence_length - len(window), len(feature_columns)), dtype=np.float32)
            window = np.vstack([pad, window])
        sequence_rows.append(window.astype(np.float32, copy=False))
        target_values.append(float(row["forward_return"]))
    return np.stack(sequence_rows), np.asarray(target_values, dtype=np.float32)


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


def _selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    value = float(metrics.get(selection_metric, metrics.get("rank_ic", 0.0)))
    if selection_metric in {"rmse", "mae", "calibration_gap"}:
        return -value
    return value


def _train_lightgbm_like(X_train: pd.DataFrame, y_train: pd.Series, config: dict[str, Any]) -> tuple[object, str | None]:
    if lgb is not None:  # pragma: no branch
        model = lgb.LGBMRegressor(
            n_estimators=max(8, int(config.get("epochs", 8)) * 4),
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


def _feature_importance(model: object, model_kind: str, feature_columns: list[str]) -> list[dict[str, float]]:
    if model_kind == "logistic_fusion" and hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coefs_"):
        importances = np.abs(model.coefs_[0]).mean(axis=1)
    elif model_kind == "gru" and hasattr(model, "gru"):
        importances = np.abs(model.gru.weight_ih_l0.detach().cpu().numpy()).mean(axis=0)
    elif model_kind == "temporal_cnn" and hasattr(model, "encoder"):
        importances = np.abs(model.encoder[0].weight.detach().cpu().numpy()).mean(axis=(0, 2))
    elif hasattr(model, "network"):
        importances = np.abs(model.network[0].weight.detach().cpu().numpy()).mean(axis=0)
    else:
        importances = np.zeros(len(feature_columns))
    return sorted(
        [
            {"feature": feature, "importance": float(score)}
            for feature, score in zip(feature_columns, importances)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )


def _build_validation_decision_report(
    meta_eval_df: pd.DataFrame,
    fusion_predictor: LayerPredictor,
    layer_metrics: dict[str, dict[str, float]],
    warnings: list[str],
) -> dict[str, Any]:
    report_frame = meta_eval_df.copy()
    report_frame["fusion_decision_score"] = fusion_predictor(report_frame)
    latest_effective_at = pd.to_datetime(report_frame["effective_at"]).max()
    latest_slice = report_frame[pd.to_datetime(report_frame["effective_at"]) == latest_effective_at].copy()
    latest_slice = latest_slice.sort_values("fusion_decision_score", ascending=False)
    score_columns = [column for column in report_frame.columns if column.endswith("_score")]

    return {
        "decision_date": str(latest_effective_at),
        "rows": int(len(report_frame)),
        "layer_metrics": layer_metrics,
        "warnings": warnings,
        "top_longs": _serialize_decision_rows(latest_slice.head(5), score_columns),
        "top_shorts": _serialize_decision_rows(latest_slice.tail(5).sort_values("fusion_decision_score", ascending=True), score_columns),
    }


def _serialize_decision_rows(frame: pd.DataFrame, score_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        payload = {
            "ticker": str(row["ticker"]),
            "sector": str(row["sector"]),
            "fusion_decision_score": float(row["fusion_decision_score"]),
            "forward_return": float(row["forward_return"]),
        }
        for column in score_columns:
            payload[column] = float(row[column])
        rows.append(payload)
    return rows
