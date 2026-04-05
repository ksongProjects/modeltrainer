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

Optional ML extras for a standard local environment:

```powershell
pip install -e .[dev,ml]
```

The `ml` extras enable LightGBM and PyTorch so training can use those libraries directly instead of the built-in sklearn fallbacks.

For the WSL + ROCm setup below, do not use `.[dev,ml]`. Install the base app first, then install ROCm PyTorch and the optional WSL-safe extras separately.

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

The backend bind address and port can also be controlled with:

```powershell
$env:QUANT_PLATFORM_HOST="127.0.0.1"
$env:QUANT_PLATFORM_PORT="8000"
python -m quant_platform.main
```

### 2. Start the frontend UI

In a second terminal:

```powershell
cd frontend
npm run dev
```

The frontend runs on `http://127.0.0.1:5173`.

By default it talks to `http://127.0.0.1:8000`.

If you want to keep the backend URL in a frontend env file, create `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

There is also an example file at `frontend/.env.example`.

If you prefer to set it only for one shell session:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

### WSL + ROCm Path For Radeon RX 7900 XT

If you want the training backend to run in a dedicated Linux ROCm environment while keeping the frontend on Windows, use the WSL path.

High-level flow:

1. install WSL with Ubuntu 22.04 or 24.04
2. install or confirm a Windows Adrenalin driver supported by AMD's current WSL guide
3. install the Windows SDK on Windows
4. sync the repo into the WSL Linux filesystem
5. install the ROCm + ROCDXG WSL stack inside Ubuntu
6. create the app env inside WSL
7. install ROCm PyTorch wheels in that env
8. optionally install LightGBM and Transformers into that env
9. launch the backend in WSL and the frontend on Windows

Repo helpers:

```powershell
# From Windows, after Ubuntu WSL is installed
scripts\start-wsl-rocm.ps1 -Distro Ubuntu-24.04
```

Recommended WSL repo sync:

```bash
sudo apt update
sudo apt install -y rsync

rsync -a --info=progress2 \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.pytest_cache' \
  --exclude 'artifacts' \
  /mnt/c/ksong/Projects/modeltrainer/ \
  ~/modeltrainer/

cd ~/modeltrainer
```

Inside WSL:

```bash
./scripts/wsl_rocm/install_rocm_stack.sh
./scripts/wsl_rocm/setup_app_env.sh
./scripts/wsl_rocm/install_pytorch_rocm.sh
./scripts/wsl_rocm/install_optional_ml.sh
./scripts/wsl_rocm/verify_rocm.sh
./scripts/wsl_rocm/start_backend.sh
```

Important notes:

- Use `pip install -e .[dev]` for the app env inside WSL, not `.[dev,ml]`, because the generic `ml` extras would replace ROCm wheels with the wrong torch build.
- To add optional model libraries after ROCm PyTorch is installed, use `./scripts/wsl_rocm/install_optional_ml.sh` or `pip install -e .[ml-runtime]`. Both paths avoid reinstalling torch.
- Keep the repo in the WSL Linux filesystem such as `~/modeltrainer` for better performance. Use `rsync` to refresh it after changes on the Windows side.
- The WSL launcher script starts the backend with `QUANT_PLATFORM_HOST=0.0.0.0` so the Windows frontend can reach it through `http://127.0.0.1:<port>`.
- The current helper now follows AMD's ROCDXG-based WSL flow. It installs base ROCm userspace packages in WSL, builds `librocdxg`, and exports `HSA_ENABLE_DXG_DETECTION=1` for verification and backend startup.
- The helper expects a Windows SDK include path visible from WSL, usually `/mnt/c/Program Files (x86)/Windows Kits/10/Include/<version>`. You can override detection with `WIN_SDK_PATH=/mnt/c/.../Include/<version>`.
- Create `frontend/.env.local` with `VITE_API_BASE_URL=http://127.0.0.1:8000` if you want the frontend to remember the backend URL without setting a shell variable every time.

Official references:

- Microsoft WSL install guide: https://learn.microsoft.com/en-us/windows/wsl/install
- AMD WSL ROCm guide (current ROCDXG path): https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- AMD WSL ROCm guide (legacy package-install reference): https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-radeon.html
- AMD WSL PyTorch install guide: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html

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
