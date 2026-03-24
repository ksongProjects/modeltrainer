# Quant Research Platform

Local-first quant research platform with separate training and testing flows, a Python control plane, and a React master control panel UI.

## What This Repo Contains

- Point-in-time synthetic daily market data generation with immutable raw files and Parquet feature sets
- Versioned datasets, feature sets, model specs, model versions, run events, metrics, traces, and artifacts in SQLite
- Factor standardization and composite scoring with winsorization, z-scores, sector-neutralization, and calculation traces
- Training flow with LightGBM-style tree baseline, logistic fusion, and checkpointable MLP scaffolding
- Testing flow with out-of-sample backtesting, risk metrics, stress testing, and paper-only execution simulation
- Browser control panel for data, factors, training, testing, registry, and monitoring

## Project Layout

- `src/quant_platform/` backend API, control plane, pipeline modules, and persistence
- `frontend/` React + Vite control panel UI
- `specs/` planning and quantitative reference documents
- `tests/` backend math and API integration tests

## Quick Start

### 1. Install the backend

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Optional ML extras:

```powershell
pip install -e .[dev,ml]
```

`ml` enables LightGBM, PyTorch, and Hugging Face integrations when you want to move beyond the built-in fallbacks.

### 2. Start the API

```powershell
python -m quant_platform.main
```

Or use the script entry point:

```powershell
quant-platform-api
```

The backend listens on `http://127.0.0.1:8000`.

### 3. Start the UI

```powershell
cd frontend
npm install
npm run dev
```

The Vite UI runs on `http://127.0.0.1:5173` and talks to the backend at `http://127.0.0.1:8000` by default.

If you need to point the UI elsewhere:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

## API Highlights

- `POST /api/datasets` builds a PIT dataset
- `POST /api/features` materializes a feature store version
- `POST /api/training-runs` starts training
- `POST /api/testing-runs` starts testing
- `GET /api/runs/{kind}/{id}/events` lists structured run events
- `GET /api/runs/{kind}/{id}/metrics` lists metrics
- `GET /api/runs/{kind}/{id}/traces` lists formula and calculation traces
- `GET /api/stream/{kind}/{id}` streams live events over SSE

## Notes

- The current implementation is optimized for daily OHLCV plus timestamped news, macro, and event factors.
- Testing is paper-only and uses execution simulation instead of live order routing.
- The repo seeds a deterministic synthetic datasource so the full workflow is runnable without external vendors.
