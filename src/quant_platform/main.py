from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .schemas import (
    DatasetCreateRequest,
    DatasetImportRequest,
    FeatureMaterializationRequest,
    RunOverrideRequest,
    SavedDatasetTagRequest,
    TestingRunRequest,
    TrainingRunRequest,
)
from .services.control_plane import ControlPlane

app = FastAPI(title="Quant Research Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
control_plane = ControlPlane()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview")
def overview():
    return control_plane.overview()


@app.get("/api/catalog")
def catalog():
    return control_plane.catalog()


@app.get("/api/datasets")
def list_datasets():
    return control_plane.list_datasets()


@app.get("/api/dataset-tags")
def list_dataset_tags():
    return control_plane.list_dataset_tags()


@app.post("/api/dataset-tags")
def create_dataset_tag(request: SavedDatasetTagRequest):
    try:
        return control_plane.create_dataset_tag(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/dataset-tags/{tag_id}")
def delete_dataset_tag(tag_id: str):
    try:
        return control_plane.delete_dataset_tag(tag_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/datasets")
def create_dataset(request: DatasetCreateRequest):
    return control_plane.create_dataset_version(request)


@app.post("/api/datasets/import-parquet")
def import_dataset(request: DatasetImportRequest):
    try:
        return control_plane.import_dataset_version(request)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/features")
def list_features():
    return control_plane.list_feature_sets()


@app.post("/api/features")
def create_features(request: FeatureMaterializationRequest):
    return control_plane.create_feature_set_version(request)


@app.get("/api/model-versions")
def list_model_versions():
    return control_plane.list_model_versions()


@app.post("/api/model-versions/{model_version_id}/promote")
def promote_model_version(model_version_id: str):
    try:
        return control_plane.promote_model_version(model_version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/model-versions/{model_version_id}/reject")
def reject_model_version(model_version_id: str):
    try:
        return control_plane.reject_model_version(model_version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/training-runs")
def list_training_runs():
    return control_plane.list_training_runs()


@app.post("/api/training-runs")
def start_training_run(request: TrainingRunRequest):
    return control_plane.start_training_run(request)


@app.get("/api/training-runs/{run_id}")
def get_training_run(run_id: str):
    try:
        return control_plane.get_run("training", run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/training-runs/{run_id}/pause")
def pause_training_run(run_id: str):
    return control_plane.pause_run("training", run_id)


@app.post("/api/training-runs/{run_id}/resume")
def resume_training_run(run_id: str):
    return control_plane.resume_run("training", run_id)


@app.post("/api/training-runs/{run_id}/stop")
def stop_training_run(run_id: str):
    return control_plane.stop_run("training", run_id)


@app.post("/api/training-runs/{run_id}/overrides")
def override_training_run(run_id: str, request: RunOverrideRequest):
    return control_plane.apply_run_overrides("training", run_id, request.overrides)


@app.get("/api/testing-runs")
def list_testing_runs():
    return control_plane.list_testing_runs()


@app.post("/api/testing-runs")
def start_testing_run(request: TestingRunRequest):
    try:
        return control_plane.start_testing_run(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/testing-runs/{run_id}")
def get_testing_run(run_id: str):
    try:
        return control_plane.get_run("testing", run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/testing-runs/{run_id}/pause")
def pause_testing_run(run_id: str):
    return control_plane.pause_run("testing", run_id)


@app.post("/api/testing-runs/{run_id}/resume")
def resume_testing_run(run_id: str):
    return control_plane.resume_run("testing", run_id)


@app.post("/api/testing-runs/{run_id}/stop")
def stop_testing_run(run_id: str):
    return control_plane.stop_run("testing", run_id)


@app.post("/api/testing-runs/{run_id}/overrides")
def override_testing_run(run_id: str, request: RunOverrideRequest):
    return control_plane.apply_run_overrides("testing", run_id, request.overrides)


@app.get("/api/runs/{run_kind}/{run_id}/events")
def run_events(run_kind: str, run_id: str, after_id: int = 0):
    return control_plane.list_run_events(run_kind, run_id, after_id)


@app.get("/api/runs/{run_kind}/{run_id}/metrics")
def run_metrics(run_kind: str, run_id: str):
    return control_plane.list_run_metrics(run_kind, run_id)


@app.get("/api/runs/{run_kind}/{run_id}/traces")
def run_traces(run_kind: str, run_id: str):
    return control_plane.list_run_traces(run_kind, run_id)


@app.get("/api/runs/{run_kind}/{run_id}/artifacts")
def run_artifacts(run_kind: str, run_id: str):
    return control_plane.list_run_artifacts(run_kind, run_id)


@app.get("/api/monitoring")
def monitoring():
    return control_plane.monitoring_summary()


@app.get("/api/stream/{run_kind}/{run_id}")
async def stream_run(run_kind: str, run_id: str) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        last_id = 0
        while True:
            events = control_plane.list_run_events(run_kind, run_id, after_id=last_id)
            for event in events:
                last_id = max(last_id, int(event["id"]))
                yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def run() -> None:
    uvicorn.run("quant_platform.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
