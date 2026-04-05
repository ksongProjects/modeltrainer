from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..database import connect, init_db, utcnow
from ..layer_controls import (
    research_layer_control_defaults,
    research_layer_model_catalog,
    research_layer_process_steps,
    research_layer_runtime_catalog,
    sanitize_runtime_settings,
)
from ..pipeline.data import import_parquet_dataset, load_dataset_frame, load_news_event_frame
from ..pipeline.features import load_feature_frame, materialize_features
from ..pipeline.testing import run_testing_suite
from ..pipeline.training import load_predictor_bundle, train_model
from ..runtime_profiles import runtime_capabilities, runtime_self_check
from ..research_layers import (
    LAYER_FUNDAMENTAL_SIGNAL,
    LAYER_EXECUTION_POLICY,
    LAYER_FEATURE_STORE,
    LAYER_FUSION_DECISION,
    LAYER_MACRO_REGIME,
    LAYER_PORTFOLIO_CONSTRUCTION,
    LAYER_PRICE_SIGNAL,
    LAYER_SNAPSHOT_SIGNAL,
    LAYER_SENTIMENT_SIGNAL,
    LAYER_DATA_FOUNDATION,
    RESEARCH_LAYER_ORDER,
    research_architecture_manifest,
)
from ..schemas import (
    DatasetCreateRequest,
    DatasetImportRequest,
    FeatureMaterializationRequest,
    ResearchLayerControlRequest,
    SavedDatasetTagRequest,
    TestingRunRequest,
    TrainingRunRequest,
)
from ..seed import seed_defaults


class RunStopped(Exception):
    pass


@dataclass
class RunController:
    pause_requested: threading.Event = field(default_factory=threading.Event)
    stop_requested: threading.Event = field(default_factory=threading.Event)


class ControlPlane:
    def __init__(self) -> None:
        init_db()
        seed_defaults()
        self._controllers: dict[tuple[str, str], RunController] = {}

    def overview(self) -> dict[str, Any]:
        counts = {}
        with connect() as connection:
            for table in ("dataset_versions", "feature_set_versions", "model_versions", "training_runs", "testing_runs", "research_layers"):
                counts[table] = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
        return {
            "counts": counts,
            "training_runs": self.list_training_runs(limit=8),
            "testing_runs": self.list_testing_runs(limit=8),
            "model_versions": self.list_model_versions(limit=8),
            "monitoring": self.monitoring_summary(),
        }

    def catalog(self) -> dict[str, Any]:
        return {
            "data_sources": self._list_table("data_sources"),
            "universes": self._list_table("universe_definitions"),
            "factors": self._list_table("factor_definitions"),
            "research_layers": self.list_research_layers(),
            "model_specs": self._list_table("model_specs"),
            "acceptance_policies": self._list_table("acceptance_policies"),
        }

    def runtime_capabilities(self) -> dict[str, Any]:
        return runtime_capabilities()

    def runtime_self_check(self, request: dict[str, Any]) -> dict[str, Any]:
        return runtime_self_check(
            request,
            model_kind=str(request.get("model_kind", "pytorch_mlp")),
            input_dim=int(request.get("input_dim", 8)),
        )

    def list_datasets(self) -> list[dict[str, Any]]:
        return [self._enrich_dataset_record(dataset) for dataset in self._list_table("dataset_versions")]

    def dataset_visualization(
        self,
        dataset_version_id: str,
        ticker: str | None = None,
        feature_set_version_id: str | None = None,
        model_version_id: str | None = None,
    ) -> dict[str, Any]:
        dataset = self._enrich_dataset_record(self._get_table_row("dataset_versions", dataset_version_id))
        frame = load_dataset_frame(dataset_version_id).copy()
        if frame.empty:
            raise ValueError("Selected dataset has no rows to visualize.")

        frame["ticker"] = frame["ticker"].astype(str)
        frame["effective_at"] = pd.to_datetime(frame["effective_at"], utc=True).dt.normalize()
        frame = frame.sort_values(["ticker", "effective_at"]).reset_index(drop=True)
        tickers = sorted(frame["ticker"].dropna().unique().tolist())
        if not tickers:
            raise ValueError("Selected dataset does not contain any tickers.")

        selected_ticker = ticker if ticker in tickers else tickers[0]
        ticker_frame = frame[frame["ticker"] == selected_ticker].copy()
        prediction_payload = self._build_dataset_prediction_payload(
            dataset_version_id=dataset_version_id,
            ticker=selected_ticker,
            feature_set_version_id=feature_set_version_id,
            model_version_id=model_version_id,
        )
        news_events = self._build_dataset_news_payload(dataset_version_id=dataset_version_id, ticker=selected_ticker)
        event_markers = self._build_dataset_event_markers(ticker_frame)

        return {
            "dataset_version_id": dataset_version_id,
            "dataset_name": dataset["name"],
            "ticker": selected_ticker,
            "tickers": tickers,
            "feature_set_version_id": prediction_payload["feature_set_version_id"],
            "model_version_id": prediction_payload["model_version_id"],
            "price_series": [
                {
                    "effective_at": str(row["effective_at"].isoformat()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "macro_surprise": float(row["macro_surprise"]),
                    "earnings_signal": float(row["earnings_signal"]),
                    "sentiment_1d": float(row["sentiment_1d"]),
                    "sentiment_5d": float(row["sentiment_5d"]),
                }
                for _, row in ticker_frame.iterrows()
            ],
            "prediction_series": prediction_payload["prediction_series"],
            "layer_score_columns": prediction_payload["layer_score_columns"],
            "news_events": news_events,
            "event_markers": event_markers,
            "assessment": dataset.get("summary", {}).get("assessment", {}),
        }

    def list_research_layers(self) -> list[dict[str, Any]]:
        layers = self._list_table("research_layers", limit=200)
        order = {layer_id: index for index, layer_id in enumerate(RESEARCH_LAYER_ORDER)}
        layers.sort(key=lambda layer: order.get(layer["id"], len(order)))
        return [self._enrich_research_layer(layer) for layer in layers]

    def get_research_layer(self, layer_id: str) -> dict[str, Any]:
        return self._enrich_research_layer(self._get_table_row("research_layers", layer_id))

    def research_architecture(self) -> dict[str, Any]:
        layers = self.list_research_layers()
        implemented = sum(1 for layer in layers if str(layer.get("status", "")).startswith("implemented"))
        partial = sum(1 for layer in layers if "partial" in str(layer.get("status", "")))
        planned = sum(1 for layer in layers if layer.get("status") == "planned")
        return {
            "layers": layers,
            "orchestration": research_architecture_manifest(),
            "summary": {
                "implemented_layers": implemented,
                "partially_implemented_layers": partial,
                "planned_layers": planned,
            },
        }

    def research_layer_observability(self, layer_id: str) -> dict[str, Any]:
        self._get_table_row("research_layers", layer_id)
        return self._layer_observability(layer_id)

    def update_research_layer_controls(self, layer_id: str, request: ResearchLayerControlRequest) -> dict[str, Any]:
        self._get_table_row("research_layers", layer_id)
        self._upsert_layer_control_overrides(layer_id, request.model_dump(exclude_none=True))
        return self.get_research_layer(layer_id)

    def list_dataset_tags(self) -> list[dict[str, Any]]:
        with connect() as connection:
            rows = connection.execute(
                "SELECT * FROM saved_dataset_tags ORDER BY name COLLATE NOCASE ASC"
            ).fetchall()
        return [self._deserialize_row(row) for row in rows]

    def list_feature_sets(self) -> list[dict[str, Any]]:
        return self._list_table("feature_set_versions")

    def list_model_versions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._list_table("model_versions", limit=limit)

    def list_training_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._list_table("training_runs", limit=limit)

    def list_testing_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._list_table("testing_runs", limit=limit)

    def create_dataset_version(self, request: DatasetCreateRequest) -> dict[str, Any]:
        raise ValueError("Synthetic dataset creation is disabled. Import a real parquet dataset instead.")

    def import_dataset_version(self, request: DatasetImportRequest) -> dict[str, Any]:
        dataset_id = self._new_id("dataset")
        tags = self._normalize_tags(request.tags)
        result = import_parquet_dataset(
            dataset_id=dataset_id,
            source_path=request.path,
            name=request.name,
        )
        record = {
            "id": dataset_id,
            "name": request.name or Path(request.path).stem,
            "source_id": request.source_id,
            "status": "completed",
            "tags_json": json.dumps(tags),
            "summary_json": json.dumps(result.summary),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        self._insert("dataset_versions", record)
        return self._enrich_dataset_record(self._deserialize_record(record))

    def create_dataset_tag(self, request: SavedDatasetTagRequest) -> dict[str, Any]:
        tag_name = self._normalize_tag_name(request.name)
        normalized_name = tag_name.casefold()
        with connect() as connection:
            existing = connection.execute(
                "SELECT * FROM saved_dataset_tags WHERE normalized_name = ?",
                (normalized_name,),
            ).fetchone()
            if existing is not None:
                return self._deserialize_row(existing)

            record = {
                "id": self._new_id("tag"),
                "name": tag_name,
                "normalized_name": normalized_name,
                "created_at": utcnow(),
                "updated_at": utcnow(),
            }
            columns = ", ".join(record.keys())
            placeholders = ", ".join("?" for _ in record)
            connection.execute(
                f"INSERT INTO saved_dataset_tags ({columns}) VALUES ({placeholders})",
                tuple(record.values()),
            )
            connection.commit()
        return self._deserialize_record(record)

    def delete_dataset_tag(self, tag_id: str) -> dict[str, Any]:
        deleted = self._get_table_row("saved_dataset_tags", tag_id)
        with connect() as connection:
            connection.execute("DELETE FROM saved_dataset_tags WHERE id = ?", (tag_id,))
            connection.commit()
        return deleted

    def create_feature_set_version(self, request: FeatureMaterializationRequest) -> dict[str, Any]:
        feature_set_id = self._new_id("features")
        feature_process_state = self._resolved_layer_process_state(LAYER_FEATURE_STORE)
        result = materialize_features(
            feature_set_id=feature_set_id,
            dataset_version_id=request.dataset_version_id,
            name=request.name,
            winsor_limit=request.winsor_limit,
            forecast_horizon_days=request.forecast_horizon_days,
            process_step_state=feature_process_state,
        )
        record = {
            "id": feature_set_id,
            "name": request.name,
            "dataset_version_id": request.dataset_version_id,
            "status": "completed",
            "summary_json": json.dumps(result.summary),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        self._insert("feature_set_versions", record)
        return self._deserialize_record(record)

    def start_training_run(self, request: TrainingRunRequest) -> dict[str, Any]:
        if not request.dataset_version_id and not request.feature_set_version_id:
            raise ValueError("Training requires a real dataset or an existing feature set. Synthetic fallback is disabled.")
        if request.feature_set_version_id and request.dataset_version_id:
            feature_set = self._get_table_row("feature_set_versions", request.feature_set_version_id)
            if str(feature_set["dataset_version_id"]) != str(request.dataset_version_id):
                raise ValueError("Selected feature set does not belong to the selected dataset.")
        run_id = self._new_id("train")
        record = {
            "id": run_id,
            "kind": "training",
            "state": "queued",
            "phase": "phase0",
            "current_stage": "queued",
            "config_json": request.model_dump_json(),
            "dataset_version_id": request.dataset_version_id,
            "feature_set_version_id": request.feature_set_version_id,
            "model_spec_id": request.model_spec_id or self._resolve_model_spec_id(request.model_kind),
            "model_version_id": None,
            "pending_overrides_json": json.dumps({}),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        self._insert("training_runs", record)
        self._controllers[("training", run_id)] = RunController()
        self._log_event(run_id, "training", "phase0", "queued", "status", "info", "Training run queued.", 0.0, {})
        threading.Thread(target=self._execute_training_run, args=(run_id,), daemon=True).start()
        return self.get_run("training", run_id)

    def start_testing_run(self, request: TestingRunRequest) -> dict[str, Any]:
        run_id = self._new_id("test")
        model_version_id = request.model_version_id or self._latest_model_version_id()
        feature_set_version_id = request.feature_set_version_id or self._feature_set_for_model(model_version_id)
        record = {
            "id": run_id,
            "kind": "testing",
            "state": "queued",
            "phase": "phase6",
            "current_stage": "queued",
            "config_json": request.model_dump_json(),
            "model_version_id": model_version_id,
            "feature_set_version_id": feature_set_version_id,
            "baseline_model_version_id": None,
            "pending_overrides_json": json.dumps({}),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        self._insert("testing_runs", record)
        self._controllers[("testing", run_id)] = RunController()
        self._log_event(run_id, "testing", "phase6", "queued", "status", "info", "Testing run queued.", 0.0, {})
        threading.Thread(target=self._execute_testing_run, args=(run_id,), daemon=True).start()
        return self.get_run("testing", run_id)

    def get_run(self, run_kind: str, run_id: str) -> dict[str, Any]:
        table = self._run_table(run_kind)
        with connect() as connection:
            row = connection.execute(f"SELECT * FROM {table} WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown {run_kind} run: {run_id}")
        return self._deserialize_row(row)

    def pause_run(self, run_kind: str, run_id: str) -> dict[str, Any]:
        controller = self._controllers.setdefault((run_kind, run_id), RunController())
        controller.pause_requested.set()
        self._update_run_state(run_kind, run_id, "paused")
        self._log_event(run_id, run_kind, self.get_run(run_kind, run_id)["phase"], self.get_run(run_kind, run_id)["current_stage"], "status", "warning", "Pause requested.", 0.0, {})
        return self.get_run(run_kind, run_id)

    def resume_run(self, run_kind: str, run_id: str) -> dict[str, Any]:
        controller = self._controllers.setdefault((run_kind, run_id), RunController())
        controller.pause_requested.clear()
        self._update_run_state(run_kind, run_id, "running")
        self._log_event(run_id, run_kind, self.get_run(run_kind, run_id)["phase"], self.get_run(run_kind, run_id)["current_stage"], "status", "info", "Run resumed.", 0.0, {})
        return self.get_run(run_kind, run_id)

    def stop_run(self, run_kind: str, run_id: str) -> dict[str, Any]:
        controller = self._controllers.setdefault((run_kind, run_id), RunController())
        controller.stop_requested.set()
        controller.pause_requested.clear()
        self._update_run_state(run_kind, run_id, "stopped")
        self._log_event(run_id, run_kind, self.get_run(run_kind, run_id)["phase"], self.get_run(run_kind, run_id)["current_stage"], "status", "warning", "Stop requested.", 0.0, {})
        return self.get_run(run_kind, run_id)

    def apply_run_overrides(self, run_kind: str, run_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        table = self._run_table(run_kind)
        with connect() as connection:
            row = connection.execute(f"SELECT pending_overrides_json FROM {table} WHERE id = ?", (run_id,)).fetchone()
            pending = json.loads(row["pending_overrides_json"] or "{}")
            pending.update(overrides)
            connection.execute(
                f"UPDATE {table} SET pending_overrides_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(pending), utcnow(), run_id),
            )
            connection.commit()
        self._log_event(run_id, run_kind, self.get_run(run_kind, run_id)["phase"], self.get_run(run_kind, run_id)["current_stage"], "override", "info", "Checkpoint override queued.", 0.0, overrides)
        return self.get_run(run_kind, run_id)

    def list_run_events(self, run_kind: str, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        with connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_events WHERE run_id = ? AND run_kind = ? AND id > ? ORDER BY id ASC",
                (run_id, run_kind, after_id),
            ).fetchall()
        return [self._deserialize_row(row) for row in rows]

    def list_run_metrics(self, run_kind: str, run_id: str) -> list[dict[str, Any]]:
        with connect() as connection:
            rows = connection.execute(
                "SELECT * FROM metric_records WHERE run_id = ? AND run_kind = ? ORDER BY id ASC",
                (run_id, run_kind),
            ).fetchall()
        return [self._deserialize_row(row) for row in rows]

    def list_run_traces(self, run_kind: str, run_id: str) -> list[dict[str, Any]]:
        with connect() as connection:
            rows = connection.execute(
                "SELECT * FROM calculation_traces WHERE run_id = ? AND run_kind = ? ORDER BY id ASC",
                (run_id, run_kind),
            ).fetchall()
        return [self._deserialize_row(row) for row in rows]

    def list_run_artifacts(self, run_kind: str, run_id: str) -> list[dict[str, Any]]:
        with connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifact_manifests WHERE run_id = ? AND run_kind = ? ORDER BY created_at ASC",
                (run_id, run_kind),
            ).fetchall()
        return [self._deserialize_row(row) for row in rows]

    def promote_model_version(self, model_version_id: str) -> dict[str, Any]:
        self._update_table("model_versions", model_version_id, {"status": "promoted", "updated_at": utcnow()})
        return self._get_table_row("model_versions", model_version_id)

    def reject_model_version(self, model_version_id: str) -> dict[str, Any]:
        self._update_table("model_versions", model_version_id, {"status": "rejected", "updated_at": utcnow()})
        return self._get_table_row("model_versions", model_version_id)

    def monitoring_summary(self) -> dict[str, Any]:
        with connect() as connection:
            total_training = connection.execute("SELECT COUNT(*) AS count FROM training_runs").fetchone()["count"]
            total_testing = connection.execute("SELECT COUNT(*) AS count FROM testing_runs").fetchone()["count"]
            completed = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM training_runs WHERE state = 'completed') +
                    (SELECT COUNT(*) FROM testing_runs WHERE state = 'completed') AS count
                """
            ).fetchone()["count"]
            latest_dataset = connection.execute("SELECT * FROM dataset_versions ORDER BY updated_at DESC LIMIT 1").fetchone()
            metric_rows = connection.execute(
                """
                SELECT name, value FROM metric_records
                WHERE name IN ('avg_slippage_bps', 'avg_vpin', 'sharpe', 'rank_ic')
                ORDER BY id DESC LIMIT 12
                """
            ).fetchall()
        metrics = {}
        for row in metric_rows:
            metrics.setdefault(row["name"], row["value"])
        total_runs = total_training + total_testing
        return {
            "run_success_rate": float(completed / total_runs) if total_runs else 0.0,
            "data_freshness_updated_at": latest_dataset["updated_at"] if latest_dataset else None,
            "phase_latency_seconds": {"training": 8.0, "testing": 4.0},
            "latest_metrics": metrics,
            "retrain_trigger_count": int(metrics.get("avg_vpin", 0) > 0.6) + int(metrics.get("rank_ic", 0) < 0.01),
        }

    def _execute_training_run(self, run_id: str) -> None:
        try:
            self._set_stage("training", run_id, phase="phase0", stage="bootstrap", state="running")
            run = self.get_run("training", run_id)
            config = json.loads(run["config_json"])
            if str(config.get("model_kind")) == "layered_decision":
                config = self._apply_master_layer_controls(config)
                self._update_table("training_runs", run_id, {"config_json": json.dumps(config), "updated_at": utcnow()})
            self._log_event(run_id, "training", "phase0", "bootstrap", "status", "info", "Bootstrapping training pipeline.", 2.5, config)
            dataset_version_id = run["dataset_version_id"]
            feature_set_version_id = run["feature_set_version_id"]
            if not dataset_version_id and feature_set_version_id:
                feature_set = self._get_table_row("feature_set_versions", feature_set_version_id)
                dataset_version_id = feature_set["dataset_version_id"]
                self._update_table("training_runs", run_id, {"dataset_version_id": dataset_version_id, "updated_at": utcnow()})
            if not dataset_version_id:
                raise ValueError("Training requires a real imported dataset. Synthetic fallback is disabled.")

            self._set_stage("training", run_id, phase="phase2", stage="feature-materialization", state="running")
            if not feature_set_version_id:
                self._wait_for_permission("training", run_id)
                feature_request = FeatureMaterializationRequest(
                    dataset_version_id=dataset_version_id,
                    name=f"{config['name']} Feature Set",
                    forecast_horizon_days=int(config["horizon_days"]),
                )
                feature_result = self.create_feature_set_version(feature_request)
                feature_set_version_id = feature_result["id"]
                self._update_table("training_runs", run_id, {"feature_set_version_id": feature_set_version_id, "updated_at": utcnow()})
                self._log_event(run_id, "training", "phase2", "feature-materialization", "features_created", "info", "Feature store version materialized.", 34.0, feature_result)
                trace_result = materialize_features(
                    feature_set_id=feature_set_version_id,
                    dataset_version_id=dataset_version_id,
                    name=f"{config['name']} Feature Set",
                    winsor_limit=3.0,
                    forecast_horizon_days=int(config["horizon_days"]),
                    process_step_state=self._resolved_layer_process_state(LAYER_FEATURE_STORE),
                )
                for trace in trace_result.traces[:6]:
                    trace.setdefault("provenance", {})["layer_id"] = LAYER_FEATURE_STORE
                    self._log_trace(run_id, "training", "phase2", "feature-materialization", trace)

            model_version_id = self._new_id("model")
            self._update_table("training_runs", run_id, {"model_version_id": model_version_id, "updated_at": utcnow()})
            self._set_stage("training", run_id, phase="phase3", stage="model-training", state="running")

            def checkpoint_hook(epoch: int, current_config: dict[str, Any]) -> dict[str, Any]:
                self._wait_for_permission("training", run_id)
                merged = self._pop_pending_overrides("training", run_id, current_config)
                self._log_event(
                    run_id,
                    "training",
                    "phase3",
                    "model-training",
                    "checkpoint",
                    "info",
                    f"Reached checkpoint epoch {epoch}.",
                    min(35.0 + (epoch / max(1, int(config["epochs"]))) * 40.0, 75.0),
                    {"epoch": epoch, "config": merged},
                )
                return merged

            training_result = train_model(
                model_version_id=model_version_id,
                feature_set_id=feature_set_version_id,
                model_kind=str(config["model_kind"]),
                config=config,
                checkpoint_hook=checkpoint_hook,
            )
            primary_layer_id = LAYER_FUSION_DECISION if str(config["model_kind"]) == "layered_decision" else LAYER_SNAPSHOT_SIGNAL
            for warning in training_result.warnings:
                self._log_event(run_id, "training", "phase3", "model-training", "warning", "warning", warning, 70.0, {})
            for metric_name, metric_value in training_result.metrics.items():
                self._log_metric(
                    run_id,
                    "training",
                    "phase3",
                    "model-training",
                    "training",
                    metric_name,
                    metric_value,
                    0,
                    {"layer_id": primary_layer_id, "model_kind": str(config["model_kind"])},
                )
            for layer_id, layer_metric_map in (training_result.layer_metrics or {}).items():
                for metric_name, metric_value in layer_metric_map.items():
                    self._log_metric(
                        run_id,
                        "training",
                        "phase3",
                        "model-training",
                        "training",
                        metric_name,
                        metric_value,
                        0,
                        {"layer_id": layer_id, "model_kind": str(config["model_kind"])},
                    )
            for layer_id, comparison in (training_result.layer_comparisons or {}).items():
                for model_kind, metric_map in comparison.get("candidate_metrics", {}).items():
                    for metric_name, metric_value in metric_map.items():
                        self._log_metric(
                            run_id,
                            "training",
                            "phase3",
                            "model-training",
                            "comparison",
                            metric_name,
                            metric_value,
                            0,
                            {
                                "layer_id": layer_id,
                                "compared_model_kind": model_kind,
                                "selected_model_kind": comparison.get("selected_model_kind"),
                            },
                        )
                self._log_trace(
                    run_id,
                    "training",
                    "phase3",
                    "model-training",
                    {
                        "formula_id": "phase3.layer_comparison.selection",
                        "label": layer_id,
                        "inputs": {
                            "selection_metric": comparison.get("selection_metric"),
                            "feature_columns": comparison.get("feature_columns"),
                        },
                        "transformed_inputs": {
                            "process_step_state": comparison.get("process_step_state"),
                            "candidate_models": list(comparison.get("candidate_metrics", {}).keys()),
                            "runtime_settings": comparison.get("runtime_settings"),
                        },
                        "output": {
                            "selected_model_kind": comparison.get("selected_model_kind"),
                            "selected_metrics": comparison.get("candidate_metrics", {}).get(
                                comparison.get("requested_model_kind"),
                                comparison.get("candidate_metrics", {}).get(comparison.get("selected_model_kind"), {}),
                            ),
                            "runtime_summary": comparison.get("runtime_summary"),
                        },
                        "units": "model_selection",
                        "provenance": {
                            "layer_id": layer_id,
                            "comparison_report_path": comparison.get("comparison_report_path"),
                        },
                    },
                )
            self._set_stage("training", run_id, phase="phase3", stage="validation", state="running")
            self._log_event(run_id, "training", "phase3", "validation", "status", "info", "Validation complete; writing registry artifact.", 86.0, training_result.summary)
            model_record = {
                "id": model_version_id,
                "name": f"{config['name']} Model",
                "model_spec_id": run["model_spec_id"],
                "feature_set_version_id": feature_set_version_id,
                "status": "completed",
                "artifact_uri": str(training_result.artifact_dir),
                "metrics_json": json.dumps(training_result.metrics),
                "summary_json": json.dumps(training_result.summary),
                "created_at": utcnow(),
                "updated_at": utcnow(),
            }
            self._insert("model_versions", model_record)
            self._log_artifact(
                run_id,
                "training",
                "model_dir",
                str(training_result.artifact_dir),
                {"layer_id": primary_layer_id, **training_result.summary},
            )
            self._log_artifact(
                run_id,
                "training",
                "checkpoint_dir",
                str(training_result.artifact_dir / "checkpoints"),
                {"layer_id": primary_layer_id, "checkpoints": training_result.checkpoint_paths},
            )
            for layer_id, artifact_map in (training_result.layer_artifacts or {}).items():
                for artifact_type, artifact_path in artifact_map.items():
                    self._log_artifact(
                        run_id,
                        "training",
                        artifact_type,
                        artifact_path,
                        {"layer_id": layer_id},
                    )
            self._set_stage("training", run_id, phase="phase3", stage="registry", state="completed")
            self._log_event(run_id, "training", "phase3", "registry", "status", "info", "Training run completed and model version frozen.", 100.0, {"model_version_id": model_version_id})
        except RunStopped:
            current = self.get_run("training", run_id)
            self._set_stage("training", run_id, phase=current["phase"], stage=current["current_stage"], state="stopped")
            self._log_event(run_id, "training", current["phase"], current["current_stage"], "status", "warning", "Training run stopped.", 100.0, {})
        except Exception as exc:  # pragma: no cover
            current = self.get_run("training", run_id)
            self._set_stage("training", run_id, phase=current["phase"], stage=current["current_stage"], state="failed")
            self._log_event(run_id, "training", current["phase"], current["current_stage"], "error", "error", str(exc), 100.0, {})

    def _execute_testing_run(self, run_id: str) -> None:
        try:
            run = self.get_run("testing", run_id)
            config = json.loads(run["config_json"])
            self._set_stage("testing", run_id, phase="phase6", stage="backtest", state="running")
            self._wait_for_permission("testing", run_id)
            model_version = self._get_table_row("model_versions", run["model_version_id"])
            result = run_testing_suite(
                testing_run_id=run_id,
                model_artifact_dir=model_version["artifact_uri"],
                feature_set_id=run["feature_set_version_id"],
                stress_iterations=int(config["stress_iterations"]),
                rebalance_decile=float(config["rebalance_decile"]),
                execution_mode=str(config["execution_mode"]),
                decision_top_k=int(config.get("decision_top_k", 10)),
            )
            for metric_name, metric_value in result.metrics.items():
                group = "risk" if "var" in metric_name or "drawdown" in metric_name or "ruin" in metric_name else "testing"
                if "slippage" in metric_name or "shortfall" in metric_name or "vpin" in metric_name:
                    group = "execution"
                layer_id = LAYER_EXECUTION_POLICY if group == "execution" else LAYER_PORTFOLIO_CONSTRUCTION
                self._log_metric(run_id, "testing", "phase6", "backtest", group, metric_name, metric_value, 0, {"layer_id": layer_id})
            for trace in result.traces:
                stage = "execution-simulation" if "execution" in trace["formula_id"] else "backtest"
                trace.setdefault("provenance", {})["layer_id"] = (
                    LAYER_EXECUTION_POLICY if stage == "execution-simulation" else LAYER_PORTFOLIO_CONSTRUCTION
                )
                self._log_trace(run_id, "testing", "phase6", stage, trace)
            self._set_stage("testing", run_id, phase="phase7", stage="execution-simulation", state="running")
            self._log_event(run_id, "testing", "phase7", "execution-simulation", "status", "info", "Execution simulation complete.", 78.0, result.summary)
            self._set_stage("testing", run_id, phase="phase8", stage="monitoring", state="running")
            self._log_event(run_id, "testing", "phase8", "monitoring", "status", "info", "Monitoring metrics refreshed.", 92.0, self.monitoring_summary())
            for artifact_type, artifact_path in result.artifacts.items():
                if artifact_type == "execution_timeline":
                    layer_id = LAYER_EXECUTION_POLICY
                elif artifact_type == "final_decision_report":
                    layer_id = LAYER_FUSION_DECISION
                else:
                    layer_id = LAYER_PORTFOLIO_CONSTRUCTION
                self._log_artifact(run_id, "testing", artifact_type, artifact_path, {"layer_id": layer_id})
            self._set_stage("testing", run_id, phase="phase8", stage="monitoring", state="completed")
            self._log_event(run_id, "testing", "phase8", "monitoring", "status", "info", "Testing run completed.", 100.0, {"model_version_id": run["model_version_id"]})
        except RunStopped:
            current = self.get_run("testing", run_id)
            self._set_stage("testing", run_id, phase=current["phase"], stage=current["current_stage"], state="stopped")
            self._log_event(run_id, "testing", current["phase"], current["current_stage"], "status", "warning", "Testing run stopped.", 100.0, {})
        except Exception as exc:  # pragma: no cover
            current = self.get_run("testing", run_id)
            self._set_stage("testing", run_id, phase=current["phase"], stage=current["current_stage"], state="failed")
            self._log_event(run_id, "testing", current["phase"], current["current_stage"], "error", "error", str(exc), 100.0, {})

    def _wait_for_permission(self, run_kind: str, run_id: str) -> None:
        controller = self._controllers.setdefault((run_kind, run_id), RunController())
        if controller.stop_requested.is_set():
            raise RunStopped()
        while controller.pause_requested.is_set():
            time.sleep(0.25)
            if controller.stop_requested.is_set():
                raise RunStopped()

    def _pop_pending_overrides(self, run_kind: str, run_id: str, current_config: dict[str, Any]) -> dict[str, Any]:
        table = self._run_table(run_kind)
        with connect() as connection:
            row = connection.execute(f"SELECT pending_overrides_json FROM {table} WHERE id = ?", (run_id,)).fetchone()
            overrides = json.loads(row["pending_overrides_json"] or "{}")
            if overrides:
                current_config.update(overrides)
                connection.execute(
                    f"UPDATE {table} SET config_json = ?, pending_overrides_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(current_config), json.dumps({}), utcnow(), run_id),
                )
                connection.commit()
        return current_config

    def _run_table(self, run_kind: str) -> str:
        return "training_runs" if run_kind == "training" else "testing_runs"

    def _resolve_model_spec_id(self, model_kind: str) -> str:
        with connect() as connection:
            row = connection.execute("SELECT id FROM model_specs WHERE kind = ? ORDER BY created_at LIMIT 1", (model_kind,)).fetchone()
        return row["id"] if row else "spec_lightgbm"

    def _latest_model_version_id(self) -> str:
        with connect() as connection:
            row = connection.execute("SELECT id FROM model_versions ORDER BY created_at DESC LIMIT 1").fetchone()
        if row is None:
            raise KeyError("No model versions available. Run training first.")
        return row["id"]

    def _feature_set_for_model(self, model_version_id: str) -> str:
        with connect() as connection:
            row = connection.execute("SELECT feature_set_version_id FROM model_versions WHERE id = ?", (model_version_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown model version: {model_version_id}")
        return row["feature_set_version_id"]

    def _set_stage(self, run_kind: str, run_id: str, phase: str, stage: str, state: str) -> None:
        self._update_table(self._run_table(run_kind), run_id, {"phase": phase, "current_stage": stage, "state": state, "updated_at": utcnow()})

    def _update_run_state(self, run_kind: str, run_id: str, state: str) -> None:
        self._update_table(self._run_table(run_kind), run_id, {"state": state, "updated_at": utcnow()})

    def _log_event(self, run_id: str, run_kind: str, phase: str, stage: str, event_type: str, severity: str, message: str, progress_pct: float, payload: dict[str, Any]) -> None:
        self._insert(
            "run_events",
            {
                "run_id": run_id,
                "run_kind": run_kind,
                "phase": phase,
                "stage": stage,
                "event_type": event_type,
                "severity": severity,
                "message": message,
                "progress_pct": progress_pct,
                "payload_json": json.dumps(payload),
                "timestamp": utcnow(),
            },
        )

    def _log_metric(self, run_id: str, run_kind: str, phase: str, stage: str, group_name: str, name: str, value: float, step: int, metadata: dict[str, Any]) -> None:
        self._insert(
            "metric_records",
            {
                "run_id": run_id,
                "run_kind": run_kind,
                "phase": phase,
                "stage": stage,
                "group_name": group_name,
                "name": name,
                "value": value,
                "step": step,
                "metadata_json": json.dumps(metadata),
                "timestamp": utcnow(),
            },
        )

    def _log_trace(self, run_id: str, run_kind: str, phase: str, stage: str, trace: dict[str, Any]) -> None:
        self._insert(
            "calculation_traces",
            {
                "run_id": run_id,
                "run_kind": run_kind,
                "phase": phase,
                "stage": stage,
                "formula_id": trace["formula_id"],
                "label": trace["label"],
                "inputs_json": json.dumps(trace["inputs"]),
                "transformed_inputs_json": json.dumps(trace["transformed_inputs"]),
                "output_json": json.dumps(trace["output"]),
                "units": trace["units"],
                "provenance_json": json.dumps(trace["provenance"]),
                "timestamp": utcnow(),
            },
        )

    def _log_artifact(self, run_id: str, run_kind: str, artifact_type: str, path: str, metadata: dict[str, Any]) -> None:
        self._insert(
            "artifact_manifests",
            {
                "id": self._new_id("artifact"),
                "run_id": run_id,
                "run_kind": run_kind,
                "artifact_type": artifact_type,
                "path": path,
                "metadata_json": json.dumps(metadata),
                "created_at": utcnow(),
            },
        )

    def _insert(self, table: str, record: dict[str, Any]) -> None:
        with connect() as connection:
            columns = ", ".join(record.keys())
            placeholders = ", ".join("?" for _ in record)
            connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(record.values()))
            connection.commit()

    def _update_table(self, table: str, row_id: str, values: dict[str, Any]) -> None:
        assignments = ", ".join(f"{column} = ?" for column in values)
        with connect() as connection:
            connection.execute(f"UPDATE {table} SET {assignments} WHERE id = ?", (*values.values(), row_id))
            connection.commit()

    def _get_table_row(self, table: str, row_id: str) -> dict[str, Any]:
        with connect() as connection:
            row = connection.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown row in {table}: {row_id}")
        return self._deserialize_row(row)

    def _list_table(self, table: str, limit: int = 100) -> list[dict[str, Any]]:
        order_column = "created_at" if table != "run_events" else "id"
        with connect() as connection:
            rows = connection.execute(f"SELECT * FROM {table} ORDER BY {order_column} DESC LIMIT ?", (limit,)).fetchall()
        return [self._deserialize_row(row) for row in rows]

    def _deserialize_row(self, row: Any) -> dict[str, Any]:
        return self._deserialize_record(dict(row))

    def _deserialize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(record)
        for key, value in list(parsed.items()):
            if key.endswith("_json") and isinstance(value, str):
                parsed[key[:-5]] = json.loads(value)
        return parsed

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"

    def _enrich_dataset_record(self, dataset: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(dataset)
        manual_source = enriched.get("manual_tags", enriched.get("tags"))
        manual_tags = self._normalize_tags(manual_source)
        auto_tags = self._derive_dataset_auto_tags(enriched)
        enriched["manual_tags"] = manual_tags
        enriched["auto_tags"] = auto_tags
        enriched["tags"] = self._normalize_tags([*manual_tags, *auto_tags])
        return enriched

    def _enrich_research_layer(self, layer: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(layer)
        config = enriched.get("config") if isinstance(enriched.get("config"), dict) else {}
        layer_id = str(enriched["id"])
        enriched["depends_on"] = config.get("depends_on", [])
        enriched["implementation_mode"] = config.get("implementation_mode")
        enriched["data_contract"] = config.get("data_contract", {})
        enriched["control_surface"] = config.get("control_surface", {})
        enriched["observability_contract"] = config.get("observability_contract", {})
        enriched["model_catalog"] = self._resolved_model_catalog(layer_id)
        enriched["runtime_catalog"] = research_layer_runtime_catalog(layer_id)
        enriched["process_steps"] = self._resolved_process_steps(layer_id)
        enriched["control_state"] = {
            "preferred_model_kind": self._resolved_preferred_model_kind(layer_id),
            "candidate_model_kinds": self._resolved_candidate_model_kinds(layer_id),
            "selection_metric": self._resolved_selection_metric(layer_id),
            "process_step_state": self._resolved_layer_process_state(layer_id),
            "runtime_settings": self._resolved_layer_runtime_settings(layer_id),
        }
        enriched["latest_observability"] = self._layer_observability(layer_id)
        enriched["latest_comparison"] = self._latest_layer_comparison(layer_id)
        return enriched

    def _get_layer_control_overrides(self, layer_id: str) -> dict[str, Any]:
        with connect() as connection:
            row = connection.execute(
                "SELECT overrides_json FROM research_layer_controls WHERE layer_id = ?",
                (layer_id,),
            ).fetchone()
        return json.loads(row["overrides_json"]) if row and row["overrides_json"] else {}

    def _upsert_layer_control_overrides(self, layer_id: str, updates: dict[str, Any]) -> None:
        current = self._get_layer_control_overrides(layer_id)
        current.update({key: value for key, value in updates.items() if value is not None})
        valid_models = {candidate["kind"] for candidate in research_layer_model_catalog(layer_id).get("candidates", [])}
        if current.get("preferred_model_kind") and valid_models and current["preferred_model_kind"] not in valid_models:
            raise ValueError(f"Unsupported model kind for {layer_id}: {current['preferred_model_kind']}")
        if "candidate_model_kinds" in current and valid_models:
            current["candidate_model_kinds"] = [model_kind for model_kind in current.get("candidate_model_kinds", []) if model_kind in valid_models]
        step_defaults = research_layer_control_defaults(layer_id).get("process_step_state", {})
        non_disableable_steps = {
            step["id"]
            for step in research_layer_process_steps(layer_id)
            if not step.get("can_disable", True)
        }
        if "process_step_state" in current:
            current["process_step_state"] = {
                step_id: True if step_id in non_disableable_steps else bool(enabled)
                for step_id, enabled in current.get("process_step_state", {}).items()
                if step_id in step_defaults
            }
        if "runtime_settings" in current:
            current["runtime_settings"] = sanitize_runtime_settings(layer_id, current.get("runtime_settings"))
        with connect() as connection:
            existing = connection.execute(
                "SELECT layer_id FROM research_layer_controls WHERE layer_id = ?",
                (layer_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO research_layer_controls (layer_id, overrides_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (layer_id, json.dumps(current), utcnow(), utcnow()),
                )
            else:
                connection.execute(
                    "UPDATE research_layer_controls SET overrides_json = ?, updated_at = ? WHERE layer_id = ?",
                    (json.dumps(current), utcnow(), layer_id),
                )
            connection.commit()

    def _resolved_model_catalog(self, layer_id: str) -> dict[str, Any]:
        catalog = dict(research_layer_model_catalog(layer_id))
        if not catalog:
            return {}
        candidate_model_kinds = set(self._resolved_candidate_model_kinds(layer_id))
        preferred_model_kind = self._resolved_preferred_model_kind(layer_id)
        catalog["candidates"] = [
            {
                **candidate,
                "enabled_for_comparison": candidate["kind"] in candidate_model_kinds,
                "selected": candidate["kind"] == preferred_model_kind,
            }
            for candidate in catalog.get("candidates", [])
        ]
        catalog["selected_model_kind"] = preferred_model_kind
        catalog["candidate_model_kinds"] = list(candidate_model_kinds)
        catalog["selection_metric"] = self._resolved_selection_metric(layer_id)
        return catalog

    def _resolved_process_steps(self, layer_id: str) -> list[dict[str, Any]]:
        state = self._resolved_layer_process_state(layer_id)
        return [
            {**step, "enabled": bool(state.get(step["id"], step.get("enabled_by_default", True)))}
            for step in research_layer_process_steps(layer_id)
        ]

    def _resolved_preferred_model_kind(self, layer_id: str) -> str | None:
        defaults = research_layer_control_defaults(layer_id)
        overrides = self._get_layer_control_overrides(layer_id)
        return overrides.get("preferred_model_kind", defaults.get("preferred_model_kind"))

    def _resolved_candidate_model_kinds(self, layer_id: str) -> list[str]:
        defaults = research_layer_control_defaults(layer_id)
        overrides = self._get_layer_control_overrides(layer_id)
        return list(overrides.get("candidate_model_kinds", defaults.get("candidate_model_kinds", [])))

    def _resolved_selection_metric(self, layer_id: str) -> str | None:
        defaults = research_layer_control_defaults(layer_id)
        overrides = self._get_layer_control_overrides(layer_id)
        return overrides.get("selection_metric", defaults.get("selection_metric"))

    def _resolved_layer_process_state(self, layer_id: str) -> dict[str, bool]:
        defaults = research_layer_control_defaults(layer_id).get("process_step_state", {})
        overrides = self._get_layer_control_overrides(layer_id).get("process_step_state", {})
        resolved = dict(defaults)
        resolved.update({step_id: bool(enabled) for step_id, enabled in overrides.items() if step_id in defaults})
        return resolved

    def _resolved_layer_runtime_settings(self, layer_id: str) -> dict[str, Any]:
        defaults = research_layer_control_defaults(layer_id).get("runtime_settings", {})
        overrides = self._get_layer_control_overrides(layer_id).get("runtime_settings", {})
        return sanitize_runtime_settings(layer_id, {**defaults, **overrides})

    def _apply_master_layer_controls(self, config: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(config)
        layer_configs = dict(enriched.get("layer_configs") or {})
        global_runtime_settings = enriched.get("runtime_settings") if isinstance(enriched.get("runtime_settings"), dict) else {}
        for layer_id in (
            LAYER_PRICE_SIGNAL,
            LAYER_FUNDAMENTAL_SIGNAL,
            LAYER_SENTIMENT_SIGNAL,
            LAYER_MACRO_REGIME,
            LAYER_FUSION_DECISION,
        ):
            defaults = {
                "model_kind": self._resolved_preferred_model_kind(layer_id),
                "candidate_model_kinds": self._resolved_candidate_model_kinds(layer_id),
                "selection_metric": self._resolved_selection_metric(layer_id),
                "process_step_state": self._resolved_layer_process_state(layer_id),
                "runtime_settings": {**global_runtime_settings, **self._resolved_layer_runtime_settings(layer_id)},
            }
            existing = dict(layer_configs.get(layer_id, {}))
            existing_process_state = defaults["process_step_state"]
            existing_process_state.update(existing.get("process_step_state", {}))
            existing_runtime_settings = defaults["runtime_settings"]
            existing_runtime_settings.update(existing.get("runtime_settings", {}))
            layer_configs[layer_id] = {
                **defaults,
                **existing,
                "process_step_state": existing_process_state,
                "runtime_settings": sanitize_runtime_settings(layer_id, existing_runtime_settings),
            }
        enriched["layer_configs"] = layer_configs
        return enriched

    def _layer_observability(self, layer_id: str) -> dict[str, Any]:
        if layer_id == LAYER_DATA_FOUNDATION:
            dataset = self._list_table("dataset_versions", limit=1)
            latest_dataset = self._enrich_dataset_record(dataset[0]) if dataset else None
            return {
                "latest_run": None,
                "latest_metrics": latest_dataset.get("summary", {}) if latest_dataset else {},
                "latest_artifacts": latest_dataset.get("summary", {}).get("artifacts", {}) if latest_dataset else {},
                "latest_status": latest_dataset.get("status") if latest_dataset else "not_started",
            }
        if layer_id == LAYER_FEATURE_STORE:
            feature_sets = self._list_table("feature_set_versions", limit=1)
            latest_feature_set = feature_sets[0] if feature_sets else None
            return {
                "latest_run": None,
                "latest_metrics": latest_feature_set.get("summary", {}) if latest_feature_set else {},
                "latest_artifacts": latest_feature_set.get("summary", {}).get("artifacts", {}) if latest_feature_set else {},
                "latest_status": latest_feature_set.get("status") if latest_feature_set else "not_started",
            }
        if layer_id == LAYER_SNAPSHOT_SIGNAL:
            latest_model_versions = self._list_table("model_versions", limit=1)
            latest_training_runs = self._list_table("training_runs", limit=1)
            latest_model = latest_model_versions[0] if latest_model_versions else None
            return {
                "latest_run": latest_training_runs[0] if latest_training_runs else None,
                "latest_metrics": self._latest_layer_metric_map(layer_id) or (latest_model.get("metrics", {}) if latest_model else {}),
                "latest_artifacts": self._latest_layer_artifact_map(layer_id),
                "latest_status": latest_training_runs[0]["state"] if latest_training_runs else (latest_model.get("status") if latest_model else "not_started"),
            }
        if layer_id in {LAYER_PRICE_SIGNAL, LAYER_FUNDAMENTAL_SIGNAL, LAYER_SENTIMENT_SIGNAL, LAYER_MACRO_REGIME, LAYER_FUSION_DECISION}:
            latest_training_runs = self._list_table("training_runs", limit=1)
            return {
                "latest_run": latest_training_runs[0] if latest_training_runs else None,
                "latest_metrics": self._latest_layer_metric_map(layer_id),
                "latest_artifacts": self._latest_layer_artifact_map(layer_id),
                "latest_status": latest_training_runs[0]["state"] if latest_training_runs else "not_started",
            }
        if layer_id in {LAYER_PORTFOLIO_CONSTRUCTION, LAYER_EXECUTION_POLICY}:
            latest_testing_runs = self._list_table("testing_runs", limit=1)
            return {
                "latest_run": latest_testing_runs[0] if latest_testing_runs else None,
                "latest_metrics": self._latest_layer_metric_map(layer_id),
                "latest_artifacts": self._latest_layer_artifact_map(layer_id),
                "latest_status": latest_testing_runs[0]["state"] if latest_testing_runs else "not_started",
            }
        return {
            "latest_run": None,
            "latest_metrics": {},
            "latest_artifacts": {},
            "latest_status": "planned",
        }

    def _latest_layer_metric_map(self, layer_id: str) -> dict[str, float]:
        with connect() as connection:
            rows = connection.execute("SELECT * FROM metric_records ORDER BY id DESC LIMIT 500").fetchall()
        metric_map: dict[str, float] = {}
        for row in rows:
            parsed = self._deserialize_row(row)
            metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
            if metadata.get("layer_id") != layer_id:
                continue
            metric_map.setdefault(str(parsed["name"]), float(parsed["value"]))
        return metric_map

    def _latest_layer_artifact_map(self, layer_id: str) -> dict[str, str]:
        with connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifact_manifests ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
        artifacts: dict[str, str] = {}
        for row in rows:
            parsed = self._deserialize_row(row)
            metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
            if metadata.get("layer_id") != layer_id:
                continue
            artifacts.setdefault(str(parsed["artifact_type"]), str(parsed["path"]))
        return artifacts

    def _latest_layer_comparison(self, layer_id: str) -> dict[str, Any] | None:
        comparison_path = self._latest_layer_artifact_map(layer_id).get("comparison_report")
        if not comparison_path:
            return None
        path = Path(comparison_path)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _build_dataset_prediction_payload(
        self,
        dataset_version_id: str,
        ticker: str,
        feature_set_version_id: str | None,
        model_version_id: str | None,
    ) -> dict[str, Any]:
        resolved_feature_set_id = feature_set_version_id
        model_record: dict[str, Any] | None = None
        if model_version_id:
            model_record = self._get_table_row("model_versions", model_version_id)
            resolved_feature_set_id = resolved_feature_set_id or str(model_record["feature_set_version_id"])

        if not resolved_feature_set_id:
            return {
                "feature_set_version_id": None,
                "model_version_id": model_version_id,
                "prediction_series": [],
                "layer_score_columns": [],
            }

        feature_set = self._get_table_row("feature_set_versions", resolved_feature_set_id)
        if str(feature_set["dataset_version_id"]) != str(dataset_version_id):
            raise ValueError("Selected feature set does not belong to the selected dataset.")
        if model_record and str(model_record["feature_set_version_id"]) != str(resolved_feature_set_id):
            raise ValueError("Selected model version does not belong to the selected feature set.")

        feature_frame = load_feature_frame(resolved_feature_set_id).copy()
        feature_frame["ticker"] = feature_frame["ticker"].astype(str)
        feature_frame["effective_at"] = pd.to_datetime(feature_frame["effective_at"], utc=True).dt.normalize()
        ticker_frame = feature_frame[feature_frame["ticker"] == ticker].copy()
        if ticker_frame.empty:
            return {
                "feature_set_version_id": resolved_feature_set_id,
                "model_version_id": model_version_id,
                "prediction_series": [],
                "layer_score_columns": [],
            }

        layer_score_columns: list[dict[str, str]] = []
        if model_record:
            predictor, layer_predictors, _ = load_predictor_bundle(model_record["artifact_uri"])
            ticker_frame["predicted_return"] = predictor(ticker_frame)
            for layer_id, layer_predictor in layer_predictors.items():
                column_key = f"{layer_id}_score"
                ticker_frame[column_key] = layer_predictor(ticker_frame)
                layer_score_columns.append(
                    {
                        "key": column_key,
                        "label": str(layer_id).replace("layer_", "").replace("_", " ").title(),
                    }
                )

        ticker_frame = ticker_frame.sort_values("effective_at").reset_index(drop=True)
        prediction_series: list[dict[str, object]] = []
        for _, row in ticker_frame.iterrows():
            payload: dict[str, object] = {
                "effective_at": str(row["effective_at"].isoformat()),
                "split": str(row["split"]),
                "forward_return": float(row["forward_return"]),
            }
            if "predicted_return" in ticker_frame.columns and pd.notna(row.get("predicted_return")):
                payload["predicted_return"] = float(row["predicted_return"])
            for layer_column in layer_score_columns:
                column_key = layer_column["key"]
                if pd.notna(row.get(column_key)):
                    payload[column_key] = float(row[column_key])
            prediction_series.append(payload)

        return {
            "feature_set_version_id": resolved_feature_set_id,
            "model_version_id": model_version_id,
            "prediction_series": prediction_series,
            "layer_score_columns": layer_score_columns,
        }

    def _build_dataset_news_payload(self, dataset_version_id: str, ticker: str) -> list[dict[str, object]]:
        news_frame = load_news_event_frame(dataset_version_id).copy()
        if news_frame.empty:
            return []

        news_frame["event_scope"] = news_frame["event_scope"].astype(str)
        news_frame["ticker"] = news_frame.get("ticker", pd.Series(index=news_frame.index, dtype="object")).astype("string")
        news_frame["known_at"] = pd.to_datetime(news_frame["known_at"], utc=True, errors="coerce")
        news_frame = news_frame[
            (news_frame["event_scope"] == "macro")
            | ((news_frame["event_scope"] == "ticker") & news_frame["ticker"].fillna("").astype(str).eq(ticker))
        ].copy()
        news_frame = news_frame.dropna(subset=["known_at"]).sort_values("known_at").reset_index(drop=True)

        return [
            {
                "event_id": str(row["event_id"]),
                "known_at": str(row["known_at"].isoformat()),
                "event_scope": str(row["event_scope"]),
                "event_type": str(row["event_type"]),
                "headline": str(row["headline"]),
                "body": str(row["body"])[:220],
                "source": str(row["source"]),
                "source_weight": float(row["source_weight"]),
                "novelty_score": float(row["novelty_score"]),
            }
            for _, row in news_frame.iterrows()
        ]

    def _build_dataset_event_markers(self, ticker_frame: pd.DataFrame) -> list[dict[str, object]]:
        event_markers: list[dict[str, object]] = []
        if ticker_frame.empty:
            return event_markers

        ticker_frame = ticker_frame.sort_values("effective_at").reset_index(drop=True)
        sentiment_threshold = float(ticker_frame["sentiment_1d"].abs().quantile(0.8)) if len(ticker_frame) > 1 else 0.0

        for _, row in ticker_frame.iterrows():
            effective_at = str(pd.Timestamp(row["effective_at"]).isoformat())
            if abs(float(row["earnings_signal"])) > 0:
                event_markers.append(
                    {
                        "effective_at": effective_at,
                        "category": "earnings",
                        "label": f"Earnings signal {float(row['earnings_signal']):.4f}",
                        "value": float(row["earnings_signal"]),
                    }
                )
            if abs(float(row["macro_surprise"])) >= 0.05:
                event_markers.append(
                    {
                        "effective_at": effective_at,
                        "category": "macro_surprise",
                        "label": f"Macro surprise {float(row['macro_surprise']):.4f}",
                        "value": float(row["macro_surprise"]),
                    }
                )
            if sentiment_threshold > 0 and abs(float(row["sentiment_1d"])) >= sentiment_threshold:
                event_markers.append(
                    {
                        "effective_at": effective_at,
                        "category": "sentiment_shock",
                        "label": f"Sentiment shock {float(row['sentiment_1d']):.4f}",
                        "value": float(row["sentiment_1d"]),
                    }
                )

        return event_markers

    def _derive_dataset_auto_tags(self, dataset: dict[str, Any]) -> list[str]:
        summary = dataset.get("summary") if isinstance(dataset.get("summary"), dict) else {}
        schema = summary.get("schema") if isinstance(summary, dict) and isinstance(summary.get("schema"), dict) else {}
        required_columns = schema.get("required_columns") if isinstance(schema.get("required_columns"), list) else []
        optional_columns = schema.get("optional_columns_present") if isinstance(schema.get("optional_columns_present"), list) else []
        extra_columns = schema.get("extra_columns") if isinstance(schema.get("extra_columns"), list) else []
        column_names = {str(column) for column in [*required_columns, *optional_columns, *extra_columns]}

        if not column_names:
            source_id = str(dataset.get("source_id") or "")
            if source_id in {"source_synthetic", "source_findf_parquet"}:
                column_names = {
                    "entity_id",
                    "ticker",
                    "sector",
                    "effective_at",
                    "known_at",
                    "ingested_at",
                    "source_version",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "ev_ebitda",
                    "roic",
                    "momentum_20d",
                    "momentum_60d",
                    "sentiment_1d",
                    "sentiment_5d",
                    "macro_surprise",
                    "earnings_signal",
                    "macro_rate",
                }

        auto_tags: list[str] = []
        tag_columns = {
            "pit": {"effective_at", "known_at", "ingested_at"},
            "ohlcv": {"open", "high", "low", "close", "volume"},
            "fundamental": {"ev_ebitda", "roic"},
            "momentum": {"momentum_20d", "momentum_60d"},
            "sentiment": {"sentiment_1d", "sentiment_5d"},
            "macro": {"macro_surprise", "macro_rate"},
            "event": {"earnings_signal"},
        }
        for tag_name, required in tag_columns.items():
            if required.intersection(column_names) == required or (tag_name in {"fundamental", "momentum", "sentiment", "macro", "event"} and required.intersection(column_names)):
                auto_tags.append(tag_name)

        ticker_count = summary.get("tickers")
        if isinstance(ticker_count, (int, float)):
            ticker_count_int = int(ticker_count)
            if ticker_count_int <= 1:
                auto_tags.append("single-ticker")
            else:
                auto_tags.append("multi-ticker")
                if ticker_count_int <= 10:
                    auto_tags.append("micro-universe")
                elif ticker_count_int <= 50:
                    auto_tags.append("small-universe")
                elif ticker_count_int <= 200:
                    auto_tags.append("mid-universe")
                else:
                    auto_tags.append("broad-universe")

        sample_tickers = summary.get("sample_tickers") if isinstance(summary.get("sample_tickers"), list) else []
        if isinstance(ticker_count, (int, float)) and int(ticker_count) <= 5:
            auto_tags.extend(f"ticker:{str(ticker).upper()}" for ticker in sample_tickers)

        sectors = summary.get("sectors") if isinstance(summary.get("sectors"), list) else []
        if len(sectors) == 1:
            sector_name = str(sectors[0]).strip().lower().replace(" ", "-")
            if sector_name:
                auto_tags.append(f"sector:{sector_name}")
        elif len(sectors) > 1:
            auto_tags.append("multi-sector")

        source_id = str(dataset.get("source_id") or "")
        source_tags = {
            "source_synthetic": "synthetic",
            "source_findf_parquet": "parquet-import",
        }
        if source_id in source_tags:
            auto_tags.append(source_tags[source_id])

        news_events = summary.get("news_events") if isinstance(summary.get("news_events"), dict) else {}
        if isinstance(news_events.get("rows"), (int, float)) and int(news_events.get("rows", 0)) > 0:
            auto_tags.append("raw-text-news")
            if int(news_events.get("macro_news_rows", 0) or 0) > 0:
                auto_tags.append("macro-news")

        return self._normalize_tags(auto_tags)

    def _normalize_tags(self, tags: list[str] | None) -> list[str]:
        if not tags:
            return []

        normalized_tags: list[str] = []
        seen: set[str] = set()
        for raw_tag in tags:
            tag_name = " ".join(str(raw_tag).split())
            if not tag_name:
                continue
            tag_name = tag_name[:64]
            tag_key = tag_name.casefold()
            if tag_key in seen:
                continue
            seen.add(tag_key)
            normalized_tags.append(tag_name)
        return normalized_tags

    def _normalize_tag_name(self, tag: str) -> str:
        normalized = " ".join(str(tag).split())
        if not normalized:
            raise ValueError("Tag name cannot be empty.")
        return normalized[:64]
