from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .settings import Settings


def _safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_files_to_frame(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        payload = _safe_read_json(path)
        if isinstance(payload, dict):
            payload["_path"] = str(path)
            rows.append(payload)
    return pd.DataFrame(rows)


def _registry_frame(settings: Settings) -> pd.DataFrame:
    rows = []
    for path in sorted(settings.registry_dir.glob("*.json")):
        if path.name == "active_models.json":
            continue
        payload = _safe_read_json(path)
        if isinstance(payload, dict):
            payload["_path"] = str(path)
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "metrics" in frame.columns:
        metrics = pd.json_normalize(frame["metrics"]).add_prefix("metrics.")
        frame = pd.concat([frame.drop(columns=["metrics"]), metrics], axis=1)
    if "params" in frame.columns:
        params = pd.json_normalize(frame["params"]).add_prefix("params.")
        frame = pd.concat([frame.drop(columns=["params"]), params], axis=1)
    return frame.sort_values("model_version", ascending=False)


def _manifests_frame(settings: Settings) -> pd.DataFrame:
    return _json_files_to_frame(sorted(settings.manifests_dir.glob("*.json")))


def _datasets_frame(settings: Settings) -> pd.DataFrame:
    rows = []
    for path in sorted(settings.datasets_dir.glob("*/metadata.json")):
        payload = _safe_read_json(path)
        if isinstance(payload, dict):
            payload["_path"] = str(path)
            rows.append(payload)
    return pd.DataFrame(rows)


def _load_oof_predictions(settings: Settings, model_version: str) -> pd.DataFrame:
    registry = _safe_read_json(settings.registry_dir / f"{model_version}.json")
    if not isinstance(registry, dict):
        return pd.DataFrame()
    artifact_paths = registry.get("artifact_paths", {})
    oof_path = artifact_paths.get("oof_predictions")
    if not oof_path:
        return pd.DataFrame()
    path = Path(oof_path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _render_overview(settings: Settings) -> None:
    st.subheader("Overview")
    manifests = _manifests_frame(settings)
    datasets = _datasets_frame(settings)
    registry = _registry_frame(settings)

    col1, col2, col3 = st.columns(3)
    col1.metric("Imported findf runs", int(len(manifests)))
    col2.metric("Snapshot versions", int(len(datasets)))
    col3.metric("Registered models", int(len(registry)))

    if not registry.empty and "promoted" in registry.columns:
        st.metric("Promoted models", int(registry["promoted"].fillna(False).sum()))

    st.markdown("Recent model registry entries")
    if registry.empty:
        st.info("No model runs found yet. Train experts to populate the dashboard.")
    else:
        display_cols = [
            col for col in [
                "model_version",
                "model_family",
                "horizon",
                "snapshot_version",
                "promoted",
                "metrics.balanced_accuracy",
                "metrics.brier_score",
                "metrics.cumulative_return",
            ] if col in registry.columns
        ]
        st.dataframe(registry[display_cols], use_container_width=True, hide_index=True)


def _render_runs(settings: Settings) -> None:
    st.subheader("Runs and Snapshots")
    manifests = _manifests_frame(settings)
    datasets = _datasets_frame(settings)

    left, right = st.columns(2)
    with left:
        st.markdown("Imported findf manifests")
        if manifests.empty:
            st.info("No imported manifests yet.")
        else:
            st.dataframe(manifests, use_container_width=True, hide_index=True)

    with right:
        st.markdown("Snapshot metadata")
        if datasets.empty:
            st.info("No snapshot metadata found yet.")
        else:
            st.dataframe(datasets, use_container_width=True, hide_index=True)


def _render_models(settings: Settings) -> None:
    st.subheader("Models")
    registry = _registry_frame(settings)
    if registry.empty:
        st.info("No model runs available.")
        return

    selected_model = st.selectbox("Model version", registry["model_version"].tolist())
    row = registry.loc[registry["model_version"] == selected_model].iloc[0]
    st.json(row.dropna().to_dict())

    oof = _load_oof_predictions(settings, selected_model)
    if oof.empty:
        st.warning("No OOF predictions found for this model.")
        return

    st.markdown("Out-of-fold predictions")
    st.dataframe(oof.head(200), use_container_width=True, hide_index=True)

    if {"trade_date", "forward_return", "prob_up"}.issubset(oof.columns):
        chart_frame = oof.sort_values("trade_date").copy()
        chart_frame["strategy_return"] = chart_frame["forward_return"] * (
            (chart_frame["prob_up"] >= 0.6).astype(float) - (chart_frame["prob_up"] <= 0.4).astype(float)
        )
        chart_frame["equity_curve"] = (1.0 + chart_frame["strategy_return"].fillna(0.0)).cumprod()
        chart_frame = chart_frame.set_index("trade_date")
        st.markdown("Equity curve")
        st.line_chart(chart_frame[["equity_curve"]])

        by_fold = (
            oof.groupby("fold_id", as_index=False)
            .agg(prob_up_mean=("prob_up", "mean"), return_mean=("forward_return", "mean"))
        )
        st.markdown("Fold summary")
        st.dataframe(by_fold, use_container_width=True, hide_index=True)


def _render_registry(settings: Settings) -> None:
    st.subheader("Registry")
    registry = _registry_frame(settings)
    active = _safe_read_json(settings.registry_dir / "active_models.json")

    if registry.empty:
        st.info("No registry entries available.")
    else:
        st.dataframe(registry, use_container_width=True, hide_index=True)

    st.markdown("Active models")
    if active:
        st.json(active)
    else:
        st.info("No active models promoted yet.")


def main() -> None:
    st.set_page_config(page_title="Markets Trainer Dashboard", layout="wide")
    settings = Settings.load()
    st.title("Markets Trainer Dashboard")
    st.caption("Offline experiment dashboard for findf-backed model training runs.")

    overview_tab, runs_tab, models_tab, registry_tab = st.tabs(
        ["Overview", "Runs", "Models", "Registry"]
    )

    with overview_tab:
        _render_overview(settings)
    with runs_tab:
        _render_runs(settings)
    with models_tab:
        _render_models(settings)
    with registry_tab:
        _render_registry(settings)


if __name__ == "__main__":
    main()
