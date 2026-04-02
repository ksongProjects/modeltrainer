from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..database import connect, init_db, utcnow
from ..pipeline.data import build_synthetic_dataset, import_parquet_dataset
from ..pipeline.features import materialize_features
from ..pipeline.testing import run_testing_suite
from ..pipeline.training import train_model
from ..schemas import (
    DatasetCreateRequest,
    DatasetImportRequest,
    FeatureMaterializationRequest,
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
            for table in ("dataset_versions", "feature_set_versions", "model_versions", "training_runs", "testing_runs"):
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
            "model_specs": self._list_table("model_specs"),
            "acceptance_policies": self._list_table("acceptance_policies"),
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        return [self._enrich_dataset_record(dataset) for dataset in self._list_table("dataset_versions")]

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
        dataset_id = self._new_id("dataset")
        tags = self._normalize_tags(request.tags)
        result = build_synthetic_dataset(
            dataset_id=dataset_id,
            name=request.name,
            num_tickers=request.num_tickers,
            num_days=request.num_days,
            seed=request.seed,
        )
        record = {
            "id": dataset_id,
            "name": request.name,
            "source_id": request.source_id,
            "status": "completed",
            "tags_json": json.dumps(tags),
            "summary_json": json.dumps(result.summary),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        self._insert("dataset_versions", record)
        return self._enrich_dataset_record(self._deserialize_record(record))

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
        result = materialize_features(
            feature_set_id=feature_set_id,
            dataset_version_id=request.dataset_version_id,
            name=request.name,
            winsor_limit=request.winsor_limit,
            forecast_horizon_days=request.forecast_horizon_days,
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
            self._log_event(run_id, "training", "phase0", "bootstrap", "status", "info", "Bootstrapping training pipeline.", 2.5, config)
            dataset_version_id = run["dataset_version_id"]
            if not dataset_version_id:
                self._wait_for_permission("training", run_id)
                dataset = self.create_dataset_version(DatasetCreateRequest(name=f"{config['name']} Dataset", num_tickers=48, num_days=320, seed=11))
                dataset_version_id = dataset["id"]
                self._update_table("training_runs", run_id, {"dataset_version_id": dataset_version_id, "updated_at": utcnow()})
                self._log_event(run_id, "training", "phase0", "data-foundation", "dataset_created", "info", "Synthetic PIT dataset built.", 14.0, dataset)

            self._set_stage("training", run_id, phase="phase2", stage="feature-materialization", state="running")
            feature_set_version_id = run["feature_set_version_id"]
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
                )
                for trace in trace_result.traces[:6]:
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
            for warning in training_result.warnings:
                self._log_event(run_id, "training", "phase3", "model-training", "warning", "warning", warning, 70.0, {})
            for metric_name, metric_value in training_result.metrics.items():
                self._log_metric(run_id, "training", "phase3", "model-training", "training", metric_name, metric_value, 0, {})
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
            self._log_artifact(run_id, "training", "model_dir", str(training_result.artifact_dir), training_result.summary)
            self._log_artifact(run_id, "training", "checkpoint_dir", str(training_result.artifact_dir / "checkpoints"), {"checkpoints": training_result.checkpoint_paths})
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
            )
            for metric_name, metric_value in result.metrics.items():
                group = "risk" if "var" in metric_name or "drawdown" in metric_name or "ruin" in metric_name else "testing"
                if "slippage" in metric_name or "shortfall" in metric_name or "vpin" in metric_name:
                    group = "execution"
                self._log_metric(run_id, "testing", "phase6", "backtest", group, metric_name, metric_value, 0, {})
            for trace in result.traces:
                stage = "execution-simulation" if "execution" in trace["formula_id"] else "backtest"
                self._log_trace(run_id, "testing", "phase6", stage, trace)
            self._set_stage("testing", run_id, phase="phase7", stage="execution-simulation", state="running")
            self._log_event(run_id, "testing", "phase7", "execution-simulation", "status", "info", "Execution simulation complete.", 78.0, result.summary)
            self._set_stage("testing", run_id, phase="phase8", stage="monitoring", state="running")
            self._log_event(run_id, "testing", "phase8", "monitoring", "status", "info", "Monitoring metrics refreshed.", 92.0, self.monitoring_summary())
            for artifact_type, artifact_path in result.artifacts.items():
                self._log_artifact(run_id, "testing", artifact_type, artifact_path, {})
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
