# Quant Research Platform

Local-first quant research platform with a Python control plane, a React master control panel UI, versioned datasets and features, model training and testing flows, and persistent metrics, traces, and artifacts.

## What The App Does

The app supports an end-to-end research workflow:

- build a synthetic point-in-time market dataset locally or import a parquet dataset
- materialize a feature store version from that dataset
- launch a training run from the UI or API
- monitor training status, progress, checkpoints, metrics, traces, and artifacts
- register a frozen model version
- launch a testing run against that model
- review backtest, risk, stress, and execution metrics in the UI

The default synthetic data path makes the full workflow runnable without external vendors.

## Main Components

### Backend

The backend lives in `src/quant_platform/`.

- `main.py`: FastAPI app and HTTP endpoints
- `services/control_plane.py`: orchestration for dataset creation, feature materialization, training, testing, pause/resume/stop, metrics, traces, and artifacts
- `pipeline/data.py`: synthetic data generation and parquet import
- `pipeline/features.py`: feature engineering and factor standardization
- `pipeline/training.py`: model training and artifact export
- `pipeline/testing.py`: backtesting, stress testing, and execution simulation
- `database.py`: SQLite schema and DB initialization
- `seed.py`: seeds built-in data sources, factors, model specs, and acceptance policies

### Frontend

The UI lives in `frontend/`.

- React 18 + TypeScript
- Vite dev/build tooling
- single-page control panel in `frontend/src/App.tsx`
- fetch-based API client in `frontend/src/api.ts`
- live run event streaming over server-sent events

### UI Sections

The control panel includes tabs for:

- `Data Pipeline`
- `Factor Lab`
- `Training Studio`
- `Testing Console`
- `Risk Lab`
- `Portfolio/Backtest`
- `Execution Simulator`
- `Model Registry`
- `Monitoring`

## How Models Are Trained

Training is orchestrated by the control plane.

Typical flow:

1. create or select a dataset version
2. create or select a feature set version
3. start a training run
4. monitor run events and progress
5. inspect resulting metrics and artifacts
6. register the completed model version

Supported model kinds:

- `lightgbm`
- `logistic_fusion`
- `pytorch_mlp`
- `gru`
- `temporal_cnn`

Notes:

- `lightgbm` uses LightGBM if installed, otherwise falls back to sklearn `HistGradientBoostingRegressor`
- `pytorch_mlp` uses PyTorch if installed, otherwise falls back to sklearn `MLPRegressor`
- `gru` and `temporal_cnn` are scaffolded through the checkpointable MLP path in the current version

Training metrics currently include:

- `rmse`
- `mae`
- `directional_accuracy`
- `rank_ic`
- `spearman`
- `calibration_gap`

Testing metrics include:

- annualized return and volatility metrics
- `sharpe`
- `var_95`
- `cvar_95`
- drawdown distribution metrics
- execution metrics such as `avg_slippage_bps`, `implementation_shortfall_bps`, and `avg_vpin`

## Where Data Is Stored

Storage paths are created automatically under the repo root.

- `artifacts/control_plane.sqlite3`: SQLite control-plane database
- `artifacts/warehouse/quant.duckdb`: DuckDB warehouse tables for dataset and feature snapshots
- `artifacts/datasets/`: versioned PIT dataset parquet files and summaries
- `artifacts/features/`: versioned feature parquet files and summaries
- `artifacts/models/`: trained model directories, metadata, checkpoints, pickles, and scalers
- `artifacts/reports/`: testing outputs such as equity curves and execution timelines
- `data/raw/`: raw synthetic CSV exports

SQLite stores:

- catalog metadata
- dataset, feature, model, and run records
- run events
- metrics
- calculation traces
- artifact manifests

Files on disk store:

- imported or generated parquet data
- feature snapshots
- model artifacts
- backtest and execution reports

## Project Layout

- `src/quant_platform/`: backend API, control plane, pipeline modules, persistence
- `frontend/`: React + Vite control panel UI
- `specs/`: planning and quant reference docs
- `tests/`: backend API and pipeline tests

## Requirements

### Backend

- Python `3.10`, `3.11`, or `3.12`

Core install:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Optional ML extras:

```powershell
pip install -e .[dev,ml]
```

The `ml` extras enable LightGBM and PyTorch so training can use those libraries directly instead of the built-in sklearn fallbacks.

### Frontend

- Node.js
- npm

Install frontend dependencies:

```powershell
cd frontend
npm install
```

## Running The App

### 1. Start the backend API

From the repo root:

```powershell
python -m quant_platform.main
```

Or use the installed script:

```powershell
quant-platform-api
```

The backend listens on `http://127.0.0.1:8000`.

### 2. Start the frontend UI

In a second terminal:

```powershell
cd frontend
npm run dev
```

The frontend runs on `http://127.0.0.1:5173`.

By default it talks to `http://127.0.0.1:8000`.

If you need a different backend URL:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

## Using The App

### Fastest path

The easiest way to exercise the app is:

1. open the `Data Pipeline` tab
2. click `Build Synthetic PIT`
3. in `Feature Store`, select the dataset and click `Materialize Features`
4. open `Training Studio`
5. select a model kind and training parameters, then click `Start Training`
6. watch progress, events, metrics, traces, and artifacts in the side rail
7. when training completes, open `Testing Console`
8. select the model and test parameters, then click `Start Testing`
9. review results in `Testing Console`, `Risk Lab`, `Portfolio/Backtest`, `Execution Simulator`, and `Monitoring`

### Running with real parquet data

You can also import a local point-in-time parquet dataset in the `Data Pipeline` tab.

The importer accepts:

- a single `.parquet` file
- a directory containing parquet parts

Required columns:

- `entity_id`
- `ticker`
- `sector`
- `effective_at`
- `known_at`
- `ingested_at`
- `source_version`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `ev_ebitda`
- `roic`
- `momentum_20d`
- `momentum_60d`
- `sentiment_1d`
- `sentiment_5d`
- `macro_surprise`
- `earnings_signal`

Optional columns currently recognized:

- `macro_rate`

## UI Features For Operating Runs

The UI supports:

- dataset creation and parquet import
- feature materialization from a selected dataset
- explicit training launch forms with dataset, feature set, model kind, and hyperparameters
- explicit testing launch forms with model version, feature set, execution mode, and stress settings
- run selection and live event streaming
- pause, resume, and stop controls for runs
- checkpoint learning-rate overrides for training runs
- model promotion and rejection in the registry

## API Highlights

- `GET /health`
- `GET /api/overview`
- `GET /api/catalog`
- `GET /api/datasets`
- `POST /api/datasets`
- `POST /api/datasets/import-parquet`
- `GET /api/features`
- `POST /api/features`
- `GET /api/model-versions`
- `POST /api/training-runs`
- `GET /api/training-runs/{run_id}`
- `POST /api/training-runs/{run_id}/pause`
- `POST /api/training-runs/{run_id}/resume`
- `POST /api/training-runs/{run_id}/stop`
- `POST /api/training-runs/{run_id}/overrides`
- `POST /api/testing-runs`
- `GET /api/testing-runs/{run_id}`
- `POST /api/testing-runs/{run_id}/pause`
- `POST /api/testing-runs/{run_id}/resume`
- `POST /api/testing-runs/{run_id}/stop`
- `GET /api/runs/{kind}/{id}/events`
- `GET /api/runs/{kind}/{id}/metrics`
- `GET /api/runs/{kind}/{id}/traces`
- `GET /api/runs/{kind}/{id}/artifacts`
- `GET /api/stream/{kind}/{id}`

## Example API Usage

### Start a training run with auto-created synthetic data

```powershell
curl -X POST http://127.0.0.1:8000/api/training-runs `
  -H "Content-Type: application/json" `
  -d "{}"
```

This is the minimum input required to see model-training output through the API. The backend will auto-create a synthetic dataset and feature set if they are not provided.

### Import parquet data

```powershell
curl -X POST http://127.0.0.1:8000/api/datasets/import-parquet `
  -H "Content-Type: application/json" `
  -d "{\"path\":\"C:\\\\path\\\\to\\\\pit_snapshot.parquet\",\"name\":\"findf PIT import\"}"
```

## Testing The App

### Backend tests

Run all backend tests from the repo root:

```powershell
pytest
```

Current automated tests cover:

- API integration flow in `tests/test_api.py`
- feature math behavior in `tests/test_pipeline_math.py`

### Frontend verification

The frontend does not currently include dedicated unit or integration test files.

Use the build as a verification step:

```powershell
cd frontend
npm run build
```

### Manual end-to-end verification

To verify the full app:

1. start the backend
2. start the frontend
3. create a synthetic dataset
4. materialize features
5. launch a training run
6. wait for training to complete
7. confirm metrics, events, traces, and artifacts appear
8. launch a testing run
9. confirm risk, portfolio, execution, and monitoring views populate

## Notes

- The implementation is optimized for daily OHLCV plus timestamped news, macro, and event-style factors
- testing is paper-only and uses execution simulation rather than live order routing
- the repository seeds deterministic local defaults so the workflow is usable without external data vendors
