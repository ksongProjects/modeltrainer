# markets-trainer

Offline training pipeline for specialist-expert equity models built on top of `findf` output artifacts.

## What It Does

- Imports completed `findf` jobs by `job_id`
- Validates the `findf` `silver` contract for prices, news, and macro data
- Scores news with pretrained FinBERT and aggregates ticker-day sentiment features
- Builds canonical point-in-time daily snapshots and labels
- Trains LightGBM and CatBoost tabular experts
- Trains a logistic-regression fusion model from expert out-of-fold predictions
- Saves evaluation reports, backtests, and model registry metadata locally

## Quick Start

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -e .[dev]
markets-trainer import-findf-run --job-id <job_id>
markets-trainer build-snapshots --job-id <job_id>
markets-trainer train-experts --snapshot-version <snapshot_version> --horizon 5d
markets-trainer train-fusion --snapshot-version <snapshot_version> --horizon 5d
```

By default, the pipeline looks for the sibling repo at `../findf`. Override with `MARKETS_FINDF_ROOT` if needed.

Use Python `3.10` or `3.11`. The dependency set is pinned to reduce `pip` backtracking during install.

The base install does not require PyTorch. If you want pretrained FinBERT instead of the built-in fallback sentiment scorer, install the NLP extra:

```powershell
pip install -e .[dev,nlp]
```

If you want the local experiment dashboard UI, install the UI extra:

```powershell
pip install -e .[dev,ui]
```

If you want both the dashboard and pretrained FinBERT:

```powershell
pip install -e .[dev,ui,nlp]
```

## Run the Pipeline

Use an existing `findf` job id:

```powershell
markets-trainer import-findf-run --job-id a04c733e-d589-4b7d-a2dd-07a21d1430a7
markets-trainer build-snapshots --job-id a04c733e-d589-4b7d-a2dd-07a21d1430a7
markets-trainer train-experts --snapshot-version a04c733e-d589-4b7d-a2dd-07a21d1430a7_v1_daily_specialists --horizon 5d
markets-trainer train-fusion --snapshot-version a04c733e-d589-4b7d-a2dd-07a21d1430a7_v1_daily_specialists --horizon 5d
```

## Run the Dashboard

After training has produced artifacts, start the dashboard:

```powershell
markets-dashboard
```

The dashboard shows:

- imported `findf` runs
- snapshot metadata
- model registry entries
- out-of-fold predictions
- simple equity curves
- promoted model state

## Tune Model Iterations

Tabular and fusion model parameters live in:

- `configs/model_params.json`

Update that file, rerun training, and compare runs in the dashboard.

## Layout

- `configs/` pipeline defaults and walk-forward fold definitions
- `src/markets_pipeline/` trainer implementation
- `artifacts/` manifests, datasets, model outputs, reports, and registry metadata
- `tests/` contract, snapshot, and label tests
