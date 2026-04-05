from __future__ import annotations

import json

from .pipeline.data import (
    OPTIONAL_NEWS_EVENT_COLUMNS,
    OPTIONAL_PARQUET_COLUMNS,
    REQUIRED_NEWS_EVENT_COLUMNS,
    REQUIRED_PARQUET_COLUMNS,
)
from .pipeline.features import FEATURE_COLUMNS, MACRO_TEXT_FEATURE_COLUMNS, TEXT_FEATURE_COLUMNS

LAYER_DATA_FOUNDATION = "layer_data_foundation"
LAYER_FEATURE_STORE = "layer_feature_store"
LAYER_SNAPSHOT_SIGNAL = "layer_snapshot_signal"
LAYER_PRICE_SIGNAL = "layer_price_signal"
LAYER_FUNDAMENTAL_SIGNAL = "layer_fundamental_signal"
LAYER_SENTIMENT_SIGNAL = "layer_sentiment_signal"
LAYER_MACRO_REGIME = "layer_macro_regime"
LAYER_FUSION_DECISION = "layer_fusion_decision"
LAYER_PORTFOLIO_CONSTRUCTION = "layer_portfolio_construction"
LAYER_EXECUTION_POLICY = "layer_execution_policy"

RESEARCH_LAYER_ORDER = [
    LAYER_DATA_FOUNDATION,
    LAYER_FEATURE_STORE,
    LAYER_SNAPSHOT_SIGNAL,
    LAYER_PRICE_SIGNAL,
    LAYER_FUNDAMENTAL_SIGNAL,
    LAYER_SENTIMENT_SIGNAL,
    LAYER_MACRO_REGIME,
    LAYER_FUSION_DECISION,
    LAYER_PORTFOLIO_CONSTRUCTION,
    LAYER_EXECUTION_POLICY,
]

RESEARCH_LAYER_DEPENDENCIES = {
    LAYER_DATA_FOUNDATION: [],
    LAYER_FEATURE_STORE: [LAYER_DATA_FOUNDATION],
    LAYER_SNAPSHOT_SIGNAL: [LAYER_FEATURE_STORE],
    LAYER_PRICE_SIGNAL: [LAYER_DATA_FOUNDATION],
    LAYER_FUNDAMENTAL_SIGNAL: [LAYER_FEATURE_STORE],
    LAYER_SENTIMENT_SIGNAL: [LAYER_DATA_FOUNDATION],
    LAYER_MACRO_REGIME: [LAYER_DATA_FOUNDATION],
    LAYER_FUSION_DECISION: [
        LAYER_PRICE_SIGNAL,
        LAYER_FUNDAMENTAL_SIGNAL,
        LAYER_SENTIMENT_SIGNAL,
        LAYER_MACRO_REGIME,
    ],
    LAYER_PORTFOLIO_CONSTRUCTION: [LAYER_FUSION_DECISION],
    LAYER_EXECUTION_POLICY: [LAYER_PORTFOLIO_CONSTRUCTION],
}


def research_architecture_manifest() -> dict[str, object]:
    return {
        "layer_order": RESEARCH_LAYER_ORDER,
        "dependencies": RESEARCH_LAYER_DEPENDENCIES,
        "baseline_path": [
            LAYER_DATA_FOUNDATION,
            LAYER_FEATURE_STORE,
            LAYER_SNAPSHOT_SIGNAL,
            LAYER_PORTFOLIO_CONSTRUCTION,
            LAYER_EXECUTION_POLICY,
        ],
        "target_path": [
            LAYER_DATA_FOUNDATION,
            LAYER_FEATURE_STORE,
            LAYER_PRICE_SIGNAL,
            LAYER_FUNDAMENTAL_SIGNAL,
            LAYER_SENTIMENT_SIGNAL,
            LAYER_MACRO_REGIME,
            LAYER_FUSION_DECISION,
            LAYER_PORTFOLIO_CONSTRUCTION,
            LAYER_EXECUTION_POLICY,
        ],
        "decision_layers": [
            LAYER_PRICE_SIGNAL,
            LAYER_FUNDAMENTAL_SIGNAL,
            LAYER_SENTIMENT_SIGNAL,
            LAYER_MACRO_REGIME,
            LAYER_FUSION_DECISION,
        ],
    }


def default_research_layer_rows(now: str) -> list[dict[str, str]]:
    return [
        _row(
            layer_id=LAYER_DATA_FOUNDATION,
            name="Data Foundation",
            stage="data",
            status="implemented",
            description="Point-in-time dataset ingestion, integrity controls, and source lineage.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_DATA_FOUNDATION],
                "implementation_mode": "active",
                "data_contract": {
                    "input_kind": "source_bundle",
                    "output_kind": "pit_dataset",
                    "entity_keys": ["entity_id", "ticker", "sector"],
                    "time_keys": ["effective_at", "known_at", "ingested_at"],
                    "required_columns": REQUIRED_PARQUET_COLUMNS,
                    "optional_columns": OPTIONAL_PARQUET_COLUMNS,
                    "optional_sidecar_contracts": {
                        "news_events": {
                            "required_columns": REQUIRED_NEWS_EVENT_COLUMNS,
                            "optional_columns": OPTIONAL_NEWS_EVENT_COLUMNS,
                        }
                    },
                    "granularity": "daily",
                },
                "control_surface": {
                    "actions": ["view", "build", "import", "tag", "promote"],
                    "access_controls": {
                        "view": True,
                        "build": True,
                        "train": False,
                        "test": False,
                        "tune": False,
                        "promote": True,
                    },
                    "tunable_parameters": [
                        {"name": "num_tickers", "type": "int", "default": 48, "min": 10, "max": 500},
                        {"name": "num_days", "type": "int", "default": 320, "min": 120, "max": 2000},
                        {"name": "seed", "type": "int", "default": 7},
                    ],
                    "promotion_gates": [
                        "pit_integrity",
                        "schema_validity",
                        "source_lineage_complete",
                    ],
                },
                "observability_contract": {
                    "metrics": ["rows", "tickers", "data_freshness_seconds", "news_event_rows"],
                    "artifacts": ["pit_daily.parquet", "news_events.parquet", "summary.json", "source_manifest.json"],
                    "events": ["dataset_created", "dataset_imported", "pit_violation_detected", "news_events_attached"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_FEATURE_STORE,
            name="Feature Store",
            stage="research",
            status="implemented",
            description="Deterministic factor engineering, target creation, and train/validation/test slicing.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_FEATURE_STORE],
                "implementation_mode": "active",
                "data_contract": {
                    "input_kind": "pit_dataset",
                    "output_kind": "feature_snapshot",
                    "entity_keys": ["entity_id", "ticker", "sector"],
                    "time_keys": ["effective_at", "known_at", "ingested_at"],
                    "feature_columns": FEATURE_COLUMNS,
                    "text_feature_columns": TEXT_FEATURE_COLUMNS,
                    "macro_text_feature_columns": MACRO_TEXT_FEATURE_COLUMNS,
                    "target_columns": ["forward_return", "direction_label"],
                    "split_column": "split",
                    "supported_horizons_days": [1, 5, 10, 20, 30],
                },
                "control_surface": {
                    "actions": ["view", "build", "tune", "promote"],
                    "access_controls": {
                        "view": True,
                        "build": True,
                        "train": False,
                        "test": False,
                        "tune": True,
                        "promote": True,
                    },
                    "tunable_parameters": [
                        {"name": "winsor_limit", "type": "float", "default": 3.0, "min": 1.0, "max": 5.0},
                        {"name": "forecast_horizon_days", "type": "int", "default": 5, "min": 1, "max": 30},
                        {"name": "text_embedding_dim", "type": "int", "default": 8, "min": 4, "max": 64},
                        {"name": "text_lookback_days", "type": "int", "default": 5, "min": 1, "max": 30},
                        {"name": "text_half_life_days", "type": "float", "default": 2.0, "min": 0.5, "max": 10.0},
                        {"name": "composite_weights", "type": "mapping", "default": {"value": 0.30, "quality": 0.25, "momentum": 0.20, "sentiment": 0.15, "macro": 0.05, "earnings": 0.05}},
                    ],
                    "promotion_gates": [
                        "feature_lineage_complete",
                        "zero_lookahead_leaks",
                        "split_coverage_valid",
                    ],
                },
                "observability_contract": {
                    "metrics": ["rows", "missing_flags", "outlier_flags", "split_balance", "ticker_text_coverage", "macro_text_coverage"],
                    "artifacts": ["features.parquet", "summary.json"],
                    "events": ["features_created", "feature_contract_changed"],
                    "traces": ["phase2.standardize.composite", "phase2.text_embeddings.decayed_pooling"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_SNAPSHOT_SIGNAL,
            name="Snapshot Alpha",
            stage="signal",
            status="implemented",
            description="Current baseline single-model tabular predictor over engineered multi-factor snapshots.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_SNAPSHOT_SIGNAL],
                "implementation_mode": "active_baseline",
                "data_contract": {
                    "input_kind": "tabular_snapshot",
                    "input_columns": FEATURE_COLUMNS,
                    "target_column": "forward_return",
                    "supported_model_families": ["lightgbm", "logistic_fusion", "pytorch_mlp", "gru", "temporal_cnn"],
                    "prediction_columns": ["predicted_return"],
                },
                "control_surface": {
                    "actions": ["view", "train", "test", "tune", "promote", "override"],
                    "access_controls": {
                        "view": True,
                        "build": False,
                        "train": True,
                        "test": True,
                        "tune": True,
                        "promote": True,
                        "override": True,
                    },
                    "tunable_parameters": [
                        {"name": "model_kind", "type": "enum", "default": "lightgbm", "choices": ["lightgbm", "logistic_fusion", "pytorch_mlp", "gru", "temporal_cnn"]},
                        {"name": "epochs", "type": "int", "default": 8, "min": 1, "max": 200},
                        {"name": "learning_rate", "type": "float", "default": 0.01, "min": 0.0001, "max": 1.0},
                        {"name": "hidden_dim", "type": "int", "default": 64, "min": 8, "max": 1024},
                        {"name": "checkpoint_frequency", "type": "int", "default": 1, "min": 1, "max": 10},
                    ],
                    "iteration_policy": {
                        "checkpoint_overrides_supported": True,
                        "optimization_targets": ["rank_ic", "directional_accuracy", "rmse"],
                    },
                    "promotion_gates": [
                        "rank_ic_positive",
                        "directional_accuracy_above_baseline",
                        "calibration_gap_within_limit",
                    ],
                },
                "observability_contract": {
                    "metrics": ["rmse", "mae", "directional_accuracy", "rank_ic", "spearman", "calibration_gap"],
                    "artifacts": ["metadata.json", "model.pkl", "scaler.pkl", "checkpoints/*"],
                    "events": ["checkpoint", "warning", "validation_complete"],
                    "traces": ["feature_importance_top5"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_PRICE_SIGNAL,
            name="Price Sequence Signal",
            stage="signal",
            status="implemented_partial",
            description="Dedicated price-focused signal trainer. Current implementation uses a layer-specific baseline model over price and momentum features.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_PRICE_SIGNAL],
                "implementation_mode": "active_baseline",
                "data_contract": {
                    "input_kind": "sequence_tensor",
                    "tensor_shape": ["batch", "lookback_window", "price_feature_count"],
                    "required_columns": ["open", "high", "low", "close", "volume", "momentum_20d", "momentum_60d"],
                    "prediction_columns": ["price_alpha_1d", "price_alpha_5d", "price_alpha_30d"],
                },
                "control_surface": {
                    "actions": ["view", "train", "test", "tune", "promote"],
                    "access_controls": {"view": True, "train": True, "test": True, "tune": True, "promote": True},
                    "tunable_parameters": [
                        {"name": "model_family", "type": "enum", "default": "gru", "choices": ["gru", "temporal_cnn", "transformer"]},
                        {"name": "lookback_window", "type": "int", "default": 60, "min": 5, "max": 252},
                        {"name": "hidden_dim", "type": "int", "default": 128, "min": 16, "max": 1024},
                        {"name": "learning_rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.1},
                    ],
                    "promotion_gates": ["multi_horizon_rank_ic_positive", "alpha_decay_stable"],
                },
                "observability_contract": {
                    "metrics": ["rank_ic_1d", "rank_ic_5d", "rank_ic_30d", "directional_accuracy_5d"],
                    "artifacts": ["sequence_config.json", "encoder.pt", "checkpoints/*"],
                    "events": ["sequence_window_built", "checkpoint", "drift_alert"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_FUNDAMENTAL_SIGNAL,
            name="Fundamental Quality Signal",
            stage="signal",
            status="implemented_partial",
            description="Dedicated model for slower-moving valuation and business quality inputs.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_FUNDAMENTAL_SIGNAL],
                "implementation_mode": "active_baseline",
                "data_contract": {
                    "input_kind": "tabular_snapshot",
                    "required_columns": ["ev_ebitda", "roic", "value_z", "quality_sector_z"],
                    "prediction_columns": ["fundamental_alpha_30d", "quality_confidence"],
                },
                "control_surface": {
                    "actions": ["view", "train", "test", "tune", "promote"],
                    "access_controls": {"view": True, "train": True, "test": True, "tune": True, "promote": True},
                    "tunable_parameters": [
                        {"name": "model_family", "type": "enum", "default": "lightgbm", "choices": ["lightgbm", "xgboost", "mlp"]},
                        {"name": "learning_rate", "type": "float", "default": 0.03, "min": 0.0001, "max": 0.5},
                        {"name": "max_depth", "type": "int", "default": 6, "min": 2, "max": 12},
                    ],
                    "promotion_gates": ["rank_ic_positive", "sector_bias_within_limit"],
                },
                "observability_contract": {
                    "metrics": ["rank_ic_30d", "directional_accuracy_30d", "sector_bias"],
                    "artifacts": ["feature_importance.json", "model.pkl"],
                    "events": ["sector_neutrality_breach", "promotion_ready"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_SENTIMENT_SIGNAL,
            name="Sentiment and Event Signal",
            stage="signal",
            status="implemented_partial",
            description="Dedicated model for sentiment, earnings/event windows, and text-derived numeric sentiment baselines.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_SENTIMENT_SIGNAL],
                "implementation_mode": "active_baseline",
                "data_contract": {
                    "input_kind": "event_stream_or_snapshot",
                    "required_columns": ["sentiment_1d", "sentiment_5d", "earnings_signal"],
                    "text_embedding_columns": TEXT_FEATURE_COLUMNS,
                    "future_extensions": ["headline_scores", "event_time_lags", "transformer_embeddings"],
                    "prediction_columns": ["sentiment_alpha_1d", "sentiment_alpha_5d"],
                },
                "control_surface": {
                    "actions": ["view", "train", "test", "tune", "promote"],
                    "access_controls": {"view": True, "train": True, "test": True, "tune": True, "promote": True},
                    "tunable_parameters": [
                        {"name": "aggregation_window_days", "type": "int", "default": 5, "min": 1, "max": 30},
                        {"name": "embedding_dim", "type": "int", "default": 128, "min": 16, "max": 2048},
                        {"name": "use_text_embeddings", "type": "bool", "default": True},
                        {"name": "learning_rate", "type": "float", "default": 0.0005, "min": 0.00001, "max": 0.1},
                    ],
                    "promotion_gates": ["event_window_precision_above_threshold", "drift_capture_positive"],
                },
                "observability_contract": {
                    "metrics": ["event_precision", "post_news_rank_ic", "drift_capture_5d", "ticker_text_coverage"],
                    "artifacts": ["event_feature_summary.json", "model.pt", "comparison_report.json"],
                    "events": ["news_window_aligned", "embedding_shift_detected", "raw_text_embeddings_materialized"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_MACRO_REGIME,
            name="Macro Regime Context",
            stage="context",
            status="implemented_partial",
            description="Context model for macro shocks, policy state, and market regime gating.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_MACRO_REGIME],
                "implementation_mode": "active_baseline",
                "data_contract": {
                    "input_kind": "macro_snapshot",
                    "required_columns": ["macro_surprise", "macro_rate"],
                    "macro_text_embedding_columns": MACRO_TEXT_FEATURE_COLUMNS,
                    "prediction_columns": ["regime_score", "risk_budget_multiplier"],
                },
                "control_surface": {
                    "actions": ["view", "train", "test", "tune", "promote"],
                    "access_controls": {"view": True, "train": True, "test": True, "tune": True, "promote": True},
                    "tunable_parameters": [
                        {"name": "regime_horizon_days", "type": "int", "default": 20, "min": 5, "max": 252},
                        {"name": "model_family", "type": "enum", "default": "lightgbm", "choices": ["lightgbm", "hmm", "mlp"]},
                        {"name": "use_macro_text_embeddings", "type": "bool", "default": True},
                    ],
                    "promotion_gates": ["regime_stability_positive", "drawdown_reduction_observed"],
                },
                "observability_contract": {
                    "metrics": ["regime_accuracy", "drawdown_reduction", "risk_budget_stability", "macro_text_coverage"],
                    "artifacts": ["regime_map.json", "model.pkl", "comparison_report.json"],
                    "events": ["regime_shift", "risk_budget_override", "macro_text_embeddings_materialized"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_FUSION_DECISION,
            name="Fusion Decision Layer",
            stage="decision",
            status="implemented_partial",
            description="Meta-model that consumes specialized signal layers and outputs calibrated decision scores.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_FUSION_DECISION],
                "implementation_mode": "active_baseline",
                "data_contract": {
                    "input_kind": "signal_bundle",
                    "required_columns": [
                        "price_alpha_1d",
                        "price_alpha_5d",
                        "fundamental_alpha_30d",
                        "sentiment_alpha_5d",
                        "regime_score",
                    ],
                    "prediction_columns": ["decision_score", "expected_return_5d", "expected_return_30d", "confidence_score"],
                },
                "control_surface": {
                    "actions": ["view", "train", "test", "tune", "promote", "override"],
                    "access_controls": {"view": True, "train": True, "test": True, "tune": True, "promote": True, "override": True},
                    "tunable_parameters": [
                        {"name": "meta_model_family", "type": "enum", "default": "logistic_regression", "choices": ["logistic_regression", "lightgbm", "xgboost"]},
                        {"name": "calibration_method", "type": "enum", "default": "isotonic", "choices": ["none", "isotonic", "platt"]},
                        {"name": "decision_threshold", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
                    ],
                    "promotion_gates": ["ensemble_sharpe_above_baseline", "confidence_calibrated", "agreement_stable"],
                },
                "observability_contract": {
                    "metrics": ["ensemble_rank_ic", "ensemble_sharpe", "confidence_calibration", "signal_agreement"],
                    "artifacts": ["meta_model.pkl", "calibration.json"],
                    "events": ["submodel_disagreement_spike", "threshold_override_applied"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_PORTFOLIO_CONSTRUCTION,
            name="Portfolio Construction",
            stage="portfolio",
            status="implemented_partial",
            description="Convert decision scores into long/short target weights and risk-constrained allocations.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_PORTFOLIO_CONSTRUCTION],
                "implementation_mode": "active_partial",
                "data_contract": {
                    "input_kind": "ranked_decision_scores",
                    "required_columns": ["predicted_return"],
                    "current_policy": "top_bottom_decile_equal_weight",
                    "target_output_columns": ["target_weight"],
                },
                "control_surface": {
                    "actions": ["view", "test", "tune", "promote"],
                    "access_controls": {"view": True, "train": False, "test": True, "tune": True, "promote": True},
                    "tunable_parameters": [
                        {"name": "rebalance_decile", "type": "float", "default": 0.1, "min": 0.01, "max": 0.4},
                        {"name": "sector_neutrality_budget", "type": "float", "default": 0.02, "min": 0.0, "max": 0.25},
                        {"name": "turnover_budget", "type": "float", "default": 0.8, "min": 0.0, "max": 1.0},
                    ],
                    "promotion_gates": ["sharpe_above_threshold", "max_drawdown_within_budget", "turnover_within_budget"],
                },
                "observability_contract": {
                    "metrics": ["annualized_return", "sharpe", "sortino", "max_drawdown", "turnover", "sector_neutrality_gap"],
                    "artifacts": ["equity_curve.json", "summary.json"],
                    "events": ["portfolio_generated", "risk_constraint_triggered"],
                },
            },
            now=now,
        ),
        _row(
            layer_id=LAYER_EXECUTION_POLICY,
            name="Execution Policy",
            stage="execution",
            status="implemented_partial",
            description="Paper execution model for slippage, shortfall, toxicity, and order slicing.",
            config={
                "depends_on": RESEARCH_LAYER_DEPENDENCIES[LAYER_EXECUTION_POLICY],
                "implementation_mode": "active_partial",
                "data_contract": {
                    "input_kind": "trade_list",
                    "required_columns": ["effective_at", "ticker", "target_weight", "predicted_return", "close", "volume"],
                    "output_columns": ["slippage_bps", "implementation_shortfall_bps", "vpin", "twap_slice_count", "vwap_bucket_share"],
                },
                "control_surface": {
                    "actions": ["view", "test", "tune", "promote"],
                    "access_controls": {"view": True, "train": False, "test": True, "tune": True, "promote": True},
                    "tunable_parameters": [
                        {"name": "execution_mode", "type": "enum", "default": "paper", "choices": ["paper"]},
                        {"name": "urgency", "type": "float", "default": 0.55, "min": 0.0, "max": 1.0},
                        {"name": "toxicity_pause_threshold", "type": "float", "default": 0.7, "min": 0.0, "max": 1.0},
                    ],
                    "promotion_gates": ["avg_slippage_within_budget", "shortfall_within_budget", "avg_vpin_within_budget"],
                },
                "observability_contract": {
                    "metrics": ["avg_slippage_bps", "implementation_shortfall_bps", "avg_vpin", "max_vpin", "paused_for_toxicity_count"],
                    "artifacts": ["execution_timeline.json", "summary.json"],
                    "events": ["execution_simulated", "toxicity_pause_triggered"],
                },
            },
            now=now,
        ),
    ]


def _row(
    layer_id: str,
    name: str,
    stage: str,
    status: str,
    description: str,
    config: dict[str, object],
    now: str,
) -> dict[str, str]:
    return {
        "id": layer_id,
        "name": name,
        "stage": stage,
        "status": status,
        "description": description,
        "config_json": json.dumps(config),
        "created_at": now,
        "updated_at": now,
    }
