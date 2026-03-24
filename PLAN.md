# Quant Research Platform v1

## Summary
- Ground the build in [quant.md](C:/ksong/Projects/modeltrainer/specs/quant.md), [training.md](C:/ksong/Projects/modeltrainer/specs/training.md), [models.md](C:/ksong/Projects/modeltrainer/specs/models.md), [models2.md](C:/ksong/Projects/modeltrainer/specs/models2.md), and [research.md](C:/ksong/Projects/modeltrainer/specs/research.md); the repo is specs-only today, so this plan assumes a greenfield implementation.
- Build a local-first web platform with a Python control plane and a browser UI: FastAPI API + local subprocess worker supervisor, SQLite for run/event metadata, DuckDB/Parquet for PIT data and features, filesystem artifact storage, MLflow for experiment tracking, and a React/TypeScript master control panel with WebSocket/SSE live updates.
- Hard-separate two top-level flows. Training flow: dataset selection -> feature materialization -> model training -> validation -> checkpointing -> model registry. Testing flow: frozen model selection -> backtest -> stress/risk -> execution simulation -> compare/approve/reject. Testing never mutates training artifacts.

## Implementation Changes
- Cross-cutting control plane: define persisted entities `DataSource`, `DatasetVersion`, `FeatureSetVersion`, `FactorDefinition`, `UniverseDefinition`, `ModelSpec`, `ModelVersion`, `TrainingRun`, `TestingRun`, `RunEvent`, `MetricRecord`, `CalculationTrace`, and `ArtifactManifest`. Every stage emits structured events, metrics, warnings, and formula traces.
- Phase 0: implement the raw lake, PIT warehouse, and feature store. Store immutable raw files, normalize into Parquet/DuckDB tables with `entity_id`, `effective_at`, `known_at`, `ingested_at`, `source_version`, and corporate-action/symbol-mapping fields, then materialize reusable factor features with lineage and freshness metadata.
- Phase 1: implement universe and factor selection. Ship built-in universes for S&P 500, Russell 1000, and custom watchlists; create a factor registry for value, quality, momentum, alt-data, macro, sentiment, and event factors; expose factor formulas, weights, and eligibility rules in the UI.
- Phase 2: implement cleaning and standardization as versioned transforms: missing-data policy, winsorization, cross-sectional z-score, sector-neutral z-score, directionality normalization, and outlier flags. Log raw input -> transformed value -> final feature for audit and factual checks.
- Phase 3: implement signal generation and model training. Start with professional baselines first: LightGBM/CatBoost on snapshot features, logistic/XGBoost fusion, `statsmodels` for spread/cointegration, and pretrained FinBERT for sentiment scoring. Add PyTorch models on the same interfaces: MLP snapshot baseline, GRU/LSTM, Temporal CNN, and unlock TFT/Transformer after the daily+event data path is stable.
- Phase 4: implement risk modeling and stress testing with covariance/cholesky stress, Student-t tail simulations, VaR/CVaR, drawdown distributions, shocked-correlation scenarios, and pluggable copula support. Expose scenario definitions, assumptions, and resulting losses in the UI.
- Phase 5: implement portfolio construction with CVXPY optimization, sector/beta/turnover constraints, benchmark-relative allocations, long-top-decile/short-bottom-decile rules, and portfolio attribution outputs.
- Phase 6: implement the testing engine as an event-driven backtester that consumes only frozen `ModelVersion` + `FeatureSetVersion`, enforces OOS guardrails, applies slippage/cost models, and logs signal creation, rebalance decisions, fills, PnL, exposures, benchmark deltas, and constraint hits.
- Phase 7: implement paper-only execution simulation for TWAP, VWAP, Implementation Shortfall, and VPIN-aware urgency/pausing. Because v1 is `Daily + Events`, simulate order slicing from historical volume curves and event windows rather than sending live orders.
- Phase 8: implement monitoring and feedback loops: alpha-decay tracking, feature/model drift, factor sensitivity, TCA, run history, failed-experiment graveyard, and checkpoint-based retraining controls from the UI.
- Master control panel: ship pages for Data Pipeline, Factor Lab, Training Studio, Testing Console, Risk Lab, Portfolio/Backtest, Execution Simulator, Model Registry, and Monitoring. Each run page must show live progress, step timeline, logs, formulas, calculation drill-down, artifacts, and pause/stop/restart controls.
- Safe tuning rule: “fine tune on the fly” means edits are accepted at checkpoint boundaries only. Active runs can be paused or stopped, configs can be changed in the UI, and training resumes from the latest checkpoint; no mid-batch mutation of optimizer/model state.
- Delivery order: milestone 1 phases 0-2 plus the UI shell and event model; milestone 2 phase 3 training flow and model registry; milestone 3 phases 4-6 testing/backtest flow; milestone 4 phases 7-8 execution sim and monitoring.

## Public APIs / Interfaces
- Training API: create dataset versions, materialize features, start/pause/stop training runs, stream epoch/batch events, upload or select model specs, apply next-checkpoint config edits, and promote/reject `ModelVersion`s.
- Testing API: start backtests, stress tests, and execution sims against frozen versions; stream `RunEvent`s; fetch `CalculationTrace`s, metrics, and artifacts; compare runs against baselines and benchmarks.
- UI contracts: every `RunEvent` includes `run_id`, `phase`, `stage`, `event_type`, `timestamp`, `severity`, `progress_pct`, and `payload`. Every `CalculationTrace` includes formula id, raw inputs, transformed inputs, output, units, and provenance.
- Run state machine: `draft`, `queued`, `running`, `paused`, `stopped`, `completed`, `failed`, `promoted`, `rejected`. The control plane owns transitions and prevents testing from starting on mutable or unapproved artifacts.
- Acceptance policy interface: approval gates are configurable templates, not hard-coded per model. v1 ships presets for rank quality, OOS portfolio performance, drawdown budget, PIT integrity, and drift tolerance.

## Metrics
- Training metrics: train/val loss, RMSE/MAE on forward returns, directional accuracy, rank IC/Spearman, horizon-by-horizon calibration, regime-sliced performance, feature importance or attention summaries, and checkpoint throughput.
- Testing metrics: annualized return, volatility, Sharpe, Sortino, Information Ratio, max drawdown, hit rate, turnover, exposure drift, beta neutrality, sector neutrality, slippage, implementation shortfall, and benchmark-relative alpha.
- Risk metrics: VaR, CVaR, probability of ruin, drawdown distribution, stress loss under shocked correlations, Student-t tail sensitivity, and copula tail dependence when enabled.
- Monitoring metrics: data freshness, PIT violations, run success rate, phase latency, model/feature drift, alpha half-life, TCA variance, and retrain trigger counts.

## Test Plan
- Data integrity tests for PIT timestamp rules, restatements, symbol remaps, dead tickers, corporate actions, and strict train/validation/test leakage prevention.
- Formula correctness tests for winsorization, z-scores, sector neutralization, composite scores, spread calculations, optimizer constraints, risk math, and execution-cost formulas using known fixtures.
- Flow/state tests for pause/stop/resume, checkpoint restart, model freezing before testing, OOS guardrails, and live event streaming to the UI.
- Evaluation tests comparing tree baselines vs PyTorch models across multiple horizons, plus ablations for price-only, price+news, and full multi-modal features.
- End-to-end acceptance tests covering import -> PIT build -> feature materialization -> train -> register -> backtest -> stress test -> execution sim -> monitoring update, all visible from the control panel.

## Assumptions
- v1 is single-tenant, local-first, paper-only, and optimized for daily OHLCV plus timestamped news, macro, and event data.
- Open/public-friendly connectors are used first for macro and filings, with pluggable provider adapters for OHLCV, news, and alt-data feeds.
- Pretrained models are allowed for NLP/embeddings; custom training is prioritized for tabular and time-series models.
