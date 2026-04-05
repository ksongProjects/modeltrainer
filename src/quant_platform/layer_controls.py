from __future__ import annotations

from typing import Any

from .pipeline.features import FEATURE_COLUMNS
from .runtime_profiles import normalize_runtime_settings
from .research_layers import (
    LAYER_DATA_FOUNDATION,
    LAYER_EXECUTION_POLICY,
    LAYER_FEATURE_STORE,
    LAYER_FUNDAMENTAL_SIGNAL,
    LAYER_FUSION_DECISION,
    LAYER_MACRO_REGIME,
    LAYER_PORTFOLIO_CONSTRUCTION,
    LAYER_PRICE_SIGNAL,
    LAYER_SENTIMENT_SIGNAL,
    LAYER_SNAPSHOT_SIGNAL,
)


def research_layer_model_catalog(layer_id: str) -> dict[str, Any]:
    catalogs: dict[str, dict[str, Any]] = {
        LAYER_SNAPSHOT_SIGNAL: _catalog(
            "rank_ic",
            "maximize",
            "lightgbm",
            [
                _candidate("lightgbm", "Gradient Boosted Trees", "tabular_snapshot", "Strong baseline for mixed numeric factor snapshots.", True, True, "native", ["cpu"]),
                _candidate("logistic_fusion", "Logistic Direction Classifier", "tabular_snapshot", "Useful for sign accuracy over return magnitude.", False, True, "native", ["cpu"]),
                _candidate("pytorch_mlp", "MLP Regressor", "tabular_snapshot", "Dense baseline for non-linear factor interactions with optional torch acceleration.", False, True, "native_or_sklearn_fallback", ["cpu", "cuda", "directml"]),
            ],
        ),
        LAYER_PRICE_SIGNAL: _catalog(
            "rank_ic",
            "maximize",
            "gru",
            [
                _candidate("gru", "GRU Sequence", "sequence_tensor", "Preferred family for temporal price paths with torch-backed sequence training.", True, True, "native", ["cpu", "cuda", "directml"]),
                _candidate("temporal_cnn", "Temporal CNN", "sequence_tensor", "Good for local pattern detection over rolling price windows.", False, False, "native", ["cpu", "cuda", "directml"]),
                _candidate("lightgbm", "Tree Baseline", "tabular_price_projection", "Reliable fallback when sequence tensors are projected into structured features.", False, True, "native", ["cpu"]),
                _candidate("pytorch_mlp", "MLP Proxy", "tabular_price_projection", "Dense proxy baseline over projected price and momentum features.", False, True, "native_or_sklearn_fallback", ["cpu", "cuda", "directml"]),
            ],
        ),
        LAYER_FUNDAMENTAL_SIGNAL: _catalog(
            "rank_ic",
            "maximize",
            "lightgbm",
            [
                _candidate("lightgbm", "Tree Regressor", "tabular_snapshot", "Recommended for slower-moving valuation and quality data.", True, True, "native", ["cpu"]),
                _candidate("logistic_fusion", "Logistic Direction Classifier", "tabular_snapshot", "Alternative when the layer should optimize direction instead of magnitude.", False, True, "native", ["cpu"]),
                _candidate("pytorch_mlp", "MLP Regressor", "tabular_snapshot", "Dense baseline for non-linear balance-sheet interactions.", False, True, "native_or_sklearn_fallback", ["cpu", "cuda", "directml"]),
            ],
        ),
        LAYER_SENTIMENT_SIGNAL: _catalog(
            "rank_ic",
            "maximize",
            "pytorch_mlp",
            [
                _candidate("pytorch_mlp", "Event MLP", "event_snapshot", "Best current fit for compact event and sentiment vectors with optional torch acceleration.", True, True, "native_or_sklearn_fallback", ["cpu", "cuda", "directml"]),
                _candidate("lightgbm", "Tree Event Baseline", "event_snapshot", "Strong fallback when event effects are mostly threshold-based.", False, True, "native", ["cpu"]),
                _candidate("logistic_fusion", "Logistic Event Classifier", "event_snapshot", "Useful for short-horizon event direction classification.", False, True, "native", ["cpu"]),
            ],
        ),
        LAYER_MACRO_REGIME: _catalog(
            "rank_ic",
            "maximize",
            "lightgbm",
            [
                _candidate("lightgbm", "Regime Tree", "macro_snapshot", "Recommended baseline for sparse macro indicators and non-linear regime boundaries.", True, True, "native", ["cpu"]),
                _candidate("logistic_fusion", "Macro Classifier", "macro_snapshot", "Useful when the output is closer to a discrete risk-on versus risk-off state.", False, True, "native", ["cpu"]),
                _candidate("pytorch_mlp", "Macro MLP", "macro_snapshot", "Dense alternative for macro interaction effects after scaling.", False, True, "native_or_sklearn_fallback", ["cpu", "cuda", "directml"]),
            ],
        ),
        LAYER_FUSION_DECISION: _catalog(
            "rank_ic",
            "maximize",
            "logistic_fusion",
            [
                _candidate("logistic_fusion", "Linear Meta Model", "signal_bundle", "Recommended fusion baseline because it stays interpretable and stable.", True, True, "native", ["cpu"]),
                _candidate("lightgbm", "Tree Meta Model", "signal_bundle", "Useful when layer outputs interact non-linearly.", False, True, "native", ["cpu"]),
                _candidate("pytorch_mlp", "Dense Meta Model", "signal_bundle", "Alternative smooth non-linear combiner for layer scores.", False, True, "native_or_sklearn_fallback", ["cpu", "cuda", "directml"]),
            ],
        ),
    }
    return catalogs.get(layer_id, {})


def research_layer_runtime_catalog(layer_id: str) -> dict[str, Any]:
    defaults = research_layer_runtime_defaults(layer_id)
    if not defaults:
        return {}
    supports_sequence = layer_id == LAYER_PRICE_SIGNAL
    return {
        "supported_compute_targets": ["auto", "cpu", "cuda", "directml"],
        "supported_precision_modes": ["auto", "fp32", "amp"],
        "supports_sequence_length": supports_sequence,
        "defaults": defaults,
        "notes": [
            "Torch-backed models can use CUDA/ROCm or DirectML when the local environment exposes those backends.",
            "Tree and logistic candidates currently run on CPU even if a GPU target is selected.",
        ],
    }


def research_layer_runtime_defaults(layer_id: str) -> dict[str, Any]:
    if layer_id in {
        LAYER_SNAPSHOT_SIGNAL,
        LAYER_PRICE_SIGNAL,
        LAYER_FUNDAMENTAL_SIGNAL,
        LAYER_SENTIMENT_SIGNAL,
        LAYER_MACRO_REGIME,
        LAYER_FUSION_DECISION,
    }:
        defaults = {
            "compute_target": "auto",
            "precision_mode": "auto",
            "batch_size": 128,
            "sequence_length": 20 if layer_id == LAYER_PRICE_SIGNAL else 20,
            "gradient_clip_norm": 1.0,
        }
        if layer_id == LAYER_PRICE_SIGNAL:
            defaults["batch_size"] = 96
            defaults["sequence_length"] = 24
        elif layer_id == LAYER_FUNDAMENTAL_SIGNAL:
            defaults["batch_size"] = 256
        elif layer_id == LAYER_FUSION_DECISION:
            defaults["batch_size"] = 192
        return defaults
    return {}


def research_layer_process_steps(layer_id: str) -> list[dict[str, Any]]:
    process_catalog: dict[str, list[dict[str, Any]]] = {
        LAYER_DATA_FOUNDATION: [
            _step("schema_validation", "Schema Validation", "required_columns subset observed_columns", "Validate parquet inputs against the PIT schema before ingest.", "required", False, ["raw source bundle"], ["validated PIT dataset"], "Required to prevent malformed datasets from entering research."),
            _step("pit_timestamp_alignment", "PIT Timestamp Alignment", "known_at <= ingested_at and effective_at aligned", "Preserve effective, known, and ingested timestamps per row.", "required", False, ["effective_at", "known_at", "ingested_at"], ["point-in-time lineage"], "Required to avoid look-ahead leakage."),
            _step("source_lineage_manifest", "Lineage Manifest", "manifest = f(source_version, schema, row_count)", "Emit a source manifest with origin, coverage, and schema footprint.", "required", False, ["source_version", "schema", "row_count"], ["source_manifest.json"], "Required for reproducibility."),
        ],
        LAYER_FEATURE_STORE: [
            _step("winsorize_value_factor", "Winsorize Value Factor", "value_z = -clip(z(ev_ebitda), -winsor_limit, winsor_limit)", "Cross-sectionally z-score EV/EBITDA each date and clip outliers when enabled.", "optional", True, ["ev_ebitda"], ["value_z"], "Recommended for noisy accounting outliers but not strictly required."),
            _step("sector_neutralize_quality", "Sector-Neutral Quality", "quality_sector_z = z(roic | effective_at, sector)", "Z-score ROIC inside each date-sector bucket when enabled.", "optional", True, ["roic", "sector"], ["quality_sector_z"], "Useful to remove structural sector bias; optional if sector bets are intentional."),
            _step("blend_momentum_horizons", "Blend Momentum Horizons", "momentum_raw = 0.65 * momentum_20d + 0.35 * momentum_60d", "Create the short/medium horizon momentum blend and z-score it cross-sectionally.", "required", False, ["momentum_20d", "momentum_60d"], ["momentum_z"], "Required by the current factor design."),
            _step("blend_sentiment_horizons", "Blend Sentiment Horizons", "sentiment_raw = 0.35 * sentiment_1d + 0.65 * sentiment_5d", "Blend short and recent sentiment windows before cross-sectional normalization.", "required", False, ["sentiment_1d", "sentiment_5d"], ["sentiment_z"], "Required by the current sentiment factor construction."),
            _step("aggregate_text_embeddings", "Aggregate Text Embeddings", "embedding_t = decay_pool(hash_ngrams(headline + body + event_type))", "Build deterministic pooled text embeddings from the raw news_events sidecar using PIT-safe decay windows.", "optional", True, ["news_events.parquet"], ["text_embedding_*", "macro_text_embedding_*"], "Optional because some datasets may not include raw text events, but recommended when they do."),
            _step("demean_macro_surprise", "Demean Macro Surprise", "macro_raw = macro_surprise - mean_t(macro_surprise)", "Center macro surprise cross-sectionally each date before z-scoring when enabled.", "optional", True, ["macro_surprise"], ["macro_z"], "Recommended when macro columns carry date-level level shifts."),
            _step("log_transform_volume", "Log-Transform Volume", "volume_signal = log(1 + volume)", "Compress heavy-tailed volume before winsorization and normalization when enabled.", "optional", True, ["volume"], ["volume_z"], "Recommended for volume skew but optional for raw volume experiments."),
            _step("weighted_composite_score", "Weighted Composite Score", "0.30*value_z + 0.25*quality_sector_z + 0.20*momentum_z + 0.15*sentiment_z + 0.05*macro_z + 0.05*earnings_z", "Combine normalized factors into the current composite alpha score.", "required", False, ["value_z", "quality_sector_z", "momentum_z", "sentiment_z", "macro_z", "earnings_z"], ["composite_score"], "Required because downstream layers currently expect a composite summary feature."),
            _step("forward_return_target", "Forward Return Target", "forward_return = close[t + horizon] / close[t] - 1", "Create the prediction target at the configured horizon.", "required", False, ["close", "forecast_horizon_days"], ["forward_return", "direction_label"], "Required for supervised learning."),
            _step("chronological_split", "Chronological Split", "dates -> 70% train / 15% validation / 15% test", "Split data by time, not random rows, to preserve realistic out-of-sample evaluation.", "required", False, ["effective_at"], ["split"], "Required for temporal validity."),
        ],
        LAYER_SNAPSHOT_SIGNAL: [
            _step("fill_missing_numeric", "Fill Missing Numeric Inputs", "x_i = 0 if x_i is missing else x_i", "Use zero-fill before model fitting for the current baseline estimators.", "required", False, FEATURE_COLUMNS, FEATURE_COLUMNS, "Required because the current baselines do not accept NaNs."),
            _step("standard_scale_inputs", "Standard Scale Inputs", "x_scaled = (x - mean_train) / std_train", "Apply train-fit standardization for dense and logistic models when enabled.", "conditional", True, FEATURE_COLUMNS, FEATURE_COLUMNS, "Recommended for logistic and MLP families; optional for tree models."),
            _step("include_cross_signal_composite", "Include Composite Context", "feature_set = feature_set union {composite_score}", "Keep the aggregated composite factor available to the model when enabled.", "optional", True, ["composite_score"], ["composite_score"], "Optional if you want pure primitive factors without a handcrafted summary feature."),
        ],
        LAYER_PRICE_SIGNAL: [
            _step("project_price_features", "Project Price Inputs", "X_price = [open, high, low, close, volume, momentum_20d, momentum_60d, momentum_z, volume_z]", "Project current price and momentum columns into the price-layer matrix.", "required", False, ["open", "high", "low", "close", "volume", "momentum_20d", "momentum_60d", "momentum_z", "volume_z"], ["price feature matrix"], "Required because the current price layer is trained from projected structured features."),
            _step("include_volume_context", "Include Volume Context", "X_price = X_price union {volume, volume_z}", "Expose raw and normalized volume context when enabled.", "optional", True, ["volume", "volume_z"], ["price feature matrix"], "Optional if you want a pure price-path experiment."),
            _step("include_cross_signal_composite", "Include Composite Context", "X_price = X_price union {composite_score}", "Provide a cross-factor context feature when enabled.", "optional", True, ["composite_score"], ["price feature matrix"], "Optional when experiments should stay market-only."),
            _step("standard_scale_inputs", "Standard Scale Inputs", "x_scaled = (x - mean_train) / std_train", "Apply standard scaling before dense or logistic baselines when enabled.", "conditional", True, ["price feature matrix"], ["scaled price feature matrix"], "Recommended for dense or logistic baselines; optional for trees."),
        ],
        LAYER_FUNDAMENTAL_SIGNAL: [
            _step("fill_missing_numeric", "Fill Missing Numeric Inputs", "x_i = 0 if x_i is missing else x_i", "Zero-fill slower-moving fundamental inputs before training.", "required", False, ["ev_ebitda", "roic", "value_z", "quality_sector_z"], ["fundamental feature matrix"], "Required by the current baseline estimators."),
            _step("include_cross_signal_composite", "Include Composite Context", "X_fundamental = X_fundamental union {composite_score}", "Expose the composite alpha context to the fundamental layer when enabled.", "optional", True, ["composite_score"], ["fundamental feature matrix"], "Optional for strictly accounting-driven experiments."),
            _step("standard_scale_inputs", "Standard Scale Inputs", "x_scaled = (x - mean_train) / std_train", "Apply train-fit standardization for dense or logistic candidates when enabled.", "conditional", True, ["fundamental feature matrix"], ["scaled fundamental feature matrix"], "Recommended for dense or logistic candidates; optional for trees."),
        ],
        LAYER_SENTIMENT_SIGNAL: [
            _step("event_window_projection", "Project Event Windows", "X_sentiment = [sentiment_1d, sentiment_5d, sentiment_z, earnings_signal, earnings_z]", "Project numeric event and sentiment windows into the layer matrix.", "required", False, ["sentiment_1d", "sentiment_5d", "sentiment_z", "earnings_signal", "earnings_z"], ["sentiment feature matrix"], "Required because the current implementation operates on structured event features."),
            _step("include_earnings_context", "Include Earnings Context", "X_sentiment = X_sentiment union {earnings_signal, earnings_z}", "Keep earnings-event context available when enabled.", "optional", True, ["earnings_signal", "earnings_z"], ["sentiment feature matrix"], "Optional for headline-only or sentiment-only experiments."),
            _step("include_text_embeddings", "Include Text Embeddings", "X_sentiment = X_sentiment union {text_embedding_*, text_event_count, text_event_weight}", "Expose pooled ticker-level text embeddings from raw news events when enabled.", "optional", True, ["text_embedding_*", "text_event_count", "text_event_weight"], ["sentiment feature matrix"], "Optional so the layer can be ablated against structured numeric sentiment only."),
            _step("include_cross_signal_composite", "Include Composite Context", "X_sentiment = X_sentiment union {composite_score}", "Provide a broader market context summary when enabled.", "optional", True, ["composite_score"], ["sentiment feature matrix"], "Optional when sentiment effects should be measured in isolation."),
            _step("standard_scale_inputs", "Standard Scale Inputs", "x_scaled = (x - mean_train) / std_train", "Apply train-fit standardization for dense or logistic candidates when enabled.", "conditional", True, ["sentiment feature matrix"], ["scaled sentiment feature matrix"], "Recommended for dense or logistic candidates; optional for trees."),
        ],
        LAYER_MACRO_REGIME: [
            _step("macro_context_projection", "Project Macro Context", "X_macro = [macro_surprise, macro_z, macro_rate, sentiment_z, momentum_z]", "Project macro and context columns into the regime-layer matrix.", "required", False, ["macro_surprise", "macro_z", "macro_rate", "sentiment_z", "momentum_z"], ["macro feature matrix"], "Required because the current regime layer uses structured macro context columns."),
            _step("include_sentiment_context", "Include Sentiment Context", "X_macro = X_macro union {sentiment_z}", "Expose cross-modal sentiment context when enabled.", "optional", True, ["sentiment_z"], ["macro feature matrix"], "Optional when regime experiments should be macro-only."),
            _step("include_momentum_context", "Include Momentum Context", "X_macro = X_macro union {momentum_z}", "Expose cross-modal momentum context when enabled.", "optional", True, ["momentum_z"], ["macro feature matrix"], "Optional when regime experiments should avoid technical context."),
            _step("include_macro_text_embeddings", "Include Macro Text Embeddings", "X_macro = X_macro union {macro_text_embedding_*, macro_text_event_count, macro_text_event_weight}", "Expose pooled economic-news embeddings from the raw news_events sidecar when enabled.", "optional", True, ["macro_text_embedding_*", "macro_text_event_count", "macro_text_event_weight"], ["macro feature matrix"], "Optional so the regime layer can be ablated against purely structured macro inputs."),
            _step("standard_scale_inputs", "Standard Scale Inputs", "x_scaled = (x - mean_train) / std_train", "Apply train-fit standardization for dense or logistic candidates when enabled.", "conditional", True, ["macro feature matrix"], ["scaled macro feature matrix"], "Recommended for dense or logistic candidates; optional for trees."),
        ],
        LAYER_FUSION_DECISION: [
            _step("stack_layer_scores", "Stack Layer Scores", "X_fusion = [price_score, fundamental_score, sentiment_score, macro_score]", "Collect upstream layer outputs into one fusion feature matrix.", "required", False, ["price signal", "fundamental signal", "sentiment signal", "macro regime"], ["fusion feature matrix"], "Required because the fusion layer only operates on upstream outputs."),
            _step("include_price_signal", "Include Price Signal", "X_fusion = X_fusion union {price_signal_score}", "Use the price-layer score inside the fusion model when enabled.", "optional", True, ["price_signal_score"], ["fusion feature matrix"], "Optional so the fusion layer can be ablated against the price stream."),
            _step("include_fundamental_signal", "Include Fundamental Signal", "X_fusion = X_fusion union {fundamental_signal_score}", "Use the fundamental-layer score when enabled.", "optional", True, ["fundamental_signal_score"], ["fusion feature matrix"], "Optional so the fusion layer can be ablated against the slow-moving business-quality stream."),
            _step("include_sentiment_signal", "Include Sentiment Signal", "X_fusion = X_fusion union {sentiment_signal_score}", "Use the sentiment-layer score when enabled.", "optional", True, ["sentiment_signal_score"], ["fusion feature matrix"], "Optional so the fusion layer can be ablated against the event-driven stream."),
            _step("include_macro_signal", "Include Macro Signal", "X_fusion = X_fusion union {macro_regime_score}", "Use the macro-regime score when enabled.", "optional", True, ["macro_regime_score"], ["fusion feature matrix"], "Optional so the fusion layer can be ablated against the context stream."),
            _step("standard_scale_inputs", "Standard Scale Inputs", "x_scaled = (x - mean_train) / std_train", "Apply train-fit standardization for dense or logistic meta-models when enabled.", "conditional", True, ["fusion feature matrix"], ["scaled fusion feature matrix"], "Recommended for dense or logistic meta-models; optional for trees."),
        ],
        LAYER_PORTFOLIO_CONSTRUCTION: [
            _step("top_bottom_decile_selection", "Top/Bottom Decile Selection", "bucket_size = max(1, floor(N * rebalance_decile))", "Rank names by predicted return and take symmetric long and short buckets.", "required", False, ["predicted_return", "rebalance_decile"], ["trade list"], "Required by the current portfolio policy."),
            _step("equal_weight_allocation", "Equal Weight Allocation", "target_weight = plus_or_minus 1 / bucket_size", "Assign equal magnitude weights to selected long and short buckets.", "required", False, ["ranked trade list"], ["target_weight"], "Required because the current portfolio engine is equal-weight only."),
            _step("sector_neutrality_audit", "Sector Neutrality Audit", "sector_neutrality_gap = mean(abs(sum sector target_weight))", "Compute the ex-post sector neutrality gap for monitoring.", "optional", True, ["sector", "target_weight"], ["sector_neutrality_gap"], "Optional because it is currently a monitoring step rather than a hard optimizer constraint."),
        ],
        LAYER_EXECUTION_POLICY: [
            _step("slippage_impact_model", "Slippage Impact Model", "slippage_bps = f(turnover, urgency, volume, toxicity)", "Simulate paper slippage and shortfall for the generated trade list.", "required", False, ["target_weight", "close", "volume", "urgency"], ["slippage_bps", "implementation_shortfall_bps"], "Required for the current execution simulation."),
            _step("toxicity_pause_rule", "Toxicity Pause Rule", "pause if vpin >= toxicity_pause_threshold", "Pause or flag toxic intervals when the VPIN proxy breaches threshold.", "optional", True, ["vpin", "toxicity_pause_threshold"], ["paused_for_toxicity_count"], "Optional because some experiments may want pure slippage simulation without toxicity gating."),
        ],
    }
    return process_catalog.get(layer_id, [])


def research_layer_control_defaults(layer_id: str) -> dict[str, Any]:
    catalog = research_layer_model_catalog(layer_id)
    process_steps = research_layer_process_steps(layer_id)
    return {
        "preferred_model_kind": catalog.get("default_model_kind"),
        "candidate_model_kinds": [candidate["kind"] for candidate in catalog.get("candidates", []) if candidate.get("enabled_by_default")],
        "selection_metric": catalog.get("selection_metric"),
        "process_step_state": {step["id"]: bool(step.get("enabled_by_default", True)) for step in process_steps},
        "runtime_settings": research_layer_runtime_defaults(layer_id),
    }


def _catalog(selection_metric: str, objective: str, default_model_kind: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selection_metric": selection_metric,
        "objective": objective,
        "default_model_kind": default_model_kind,
        "candidates": candidates,
    }


def _candidate(
    kind: str,
    label: str,
    input_fit: str,
    rationale: str,
    recommended: bool,
    enabled_by_default: bool,
    implementation_mode: str,
    acceleration_modes: list[str],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "input_fit": input_fit,
        "rationale": rationale,
        "recommended": recommended,
        "enabled_by_default": enabled_by_default,
        "implementation_mode": implementation_mode,
        "acceleration_modes": acceleration_modes,
    }


def _step(
    step_id: str,
    name: str,
    formula: str,
    algorithm: str,
    requirement_level: str,
    can_disable: bool,
    inputs: list[str],
    outputs: list[str],
    validity_reason: str,
) -> dict[str, Any]:
    return {
        "id": step_id,
        "name": name,
        "formula": formula,
        "algorithm": algorithm,
        "requirement_level": requirement_level,
        "enabled_by_default": True,
        "can_disable": can_disable,
        "inputs": inputs,
        "outputs": outputs,
        "validity_reason": validity_reason,
    }


def sanitize_runtime_settings(layer_id: str, runtime_settings: dict[str, Any] | None) -> dict[str, Any]:
    defaults = research_layer_runtime_defaults(layer_id)
    if not defaults:
        return {}
    return normalize_runtime_settings(runtime_settings, defaults)
