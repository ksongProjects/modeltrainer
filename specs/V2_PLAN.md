# V2 Plan: PyTorch Sequence Expert + Stronger Experiment Loop

## Goals

- Add a real PyTorch forecasting expert to the current specialist-fusion pipeline
- Keep `findf` as the only ingestion and cleaning system
- Keep the daily snapshot as the canonical point-in-time dataset
- Improve experiment iteration with config-driven hyperparameters and dashboard visibility

## Architecture

- Keep V1 experts:
  - LightGBM tabular expert
  - CatBoost tabular expert
  - Fusion meta-model
- Add one PyTorch sequence expert:
  - GRU first
  - optional LSTM variant after GRU is stable
- Feed the fusion layer with:
  - `prob_up_lightgbm`
  - `prob_up_catboost`
  - `prob_up_gru`
  - regime features such as volatility, news intensity, and earnings proximity

## Data Path

- Reuse `snapshot_daily.parquet` as the source
- Build rolling windows per symbol from snapshot features
- Support `20d` and `60d` lookbacks
- Keep labels identical to V1:
  - `target_up_1d`
  - `target_up_5d`
  - `target_up_10d`
- Do not feed raw text into the sequence expert in V2
- Continue using FinBERT or fallback sentiment as an upstream feature generator

## Implementation Changes

- Add `src/markets_pipeline/models/sequence_dataset.py`
  - convert snapshot rows into `(X, y, metadata)` tensors
- Add `src/markets_pipeline/models/sequence.py`
  - GRU model
  - training loop
  - early stopping
  - checkpoint save/load
  - validation scoring
- Add `configs/sequence_params.json`
  - lookback
  - hidden size
  - layer count
  - dropout
  - batch size
  - learning rate
  - epochs
- Extend `src/markets_pipeline/models/fusion.py`
  - consume GRU out-of-fold predictions when present
- Extend dashboard
  - sequence run page
  - training loss curves
  - checkpoint comparison

## Training Defaults

- Model:
  - `GRU(input_size=n_features, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)`
- Head:
  - `Linear -> ReLU -> Dropout -> Linear`
- Loss:
  - `BCEWithLogitsLoss`
- Optimizer:
  - `AdamW`
- Selection:
  - best validation Brier score
- Calibration:
  - isotonic calibration on validation predictions after training

## CLI Additions

- `build-sequence-dataset --snapshot-version <id> --lookback <20|60> --horizon <1d|5d|10d>`
- `train-sequence --snapshot-version <id> --lookback <20|60> --horizon <1d|5d|10d>`

## Validation and Acceptance

- Compare GRU vs:
  - LightGBM
  - CatBoost
  - V1 fusion
  - V2 fusion with GRU included
- Require:
  - end-to-end training on at least one real `findf` job
  - saved checkpoints per fold
  - OOF predictions available to fusion
  - measurable gain in balanced accuracy or Brier score on holdout data

## Iteration Workflow

- Keep hyperparameters config-driven rather than code-driven
- Track every run in the registry with metrics and params
- Use the dashboard to compare runs before promotion
- Add checkpoint warm-starting for sequence experiments only after base GRU training is stable

## Out of Scope

- TFT
- Transformer encoder
- true gated MoE
- intraday sequence training
- live serving changes
