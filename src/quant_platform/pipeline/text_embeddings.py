from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from .data import load_news_event_frame

TEXT_EMBEDDING_DIM = 8
TEXT_EMBEDDING_COLUMNS = [f"text_embedding_{index:02d}" for index in range(TEXT_EMBEDDING_DIM)]
MACRO_TEXT_EMBEDDING_COLUMNS = [f"macro_text_embedding_{index:02d}" for index in range(TEXT_EMBEDDING_DIM)]


@dataclass
class TextEmbeddingBuildResult:
    features: pd.DataFrame
    summary: dict[str, object]
    traces: list[dict[str, object]]


def materialize_text_embedding_features(
    dataset_version_id: str,
    snapshot_frame: pd.DataFrame,
    lookback_days: int = 5,
    half_life_days: float = 2.0,
) -> TextEmbeddingBuildResult:
    news_events = load_news_event_frame(dataset_version_id).copy()
    base = snapshot_frame[["ticker", "effective_at", "known_at"]].copy()
    base["effective_at"] = pd.to_datetime(base["effective_at"], utc=True).dt.normalize()
    base["known_at"] = pd.to_datetime(base["known_at"], utc=True)

    empty_payload = base[["ticker", "effective_at"]].copy()
    for column in [*TEXT_EMBEDDING_COLUMNS, *MACRO_TEXT_EMBEDDING_COLUMNS]:
        empty_payload[column] = 0.0
    empty_payload["text_event_count"] = 0.0
    empty_payload["text_event_weight"] = 0.0
    empty_payload["macro_text_event_count"] = 0.0
    empty_payload["macro_text_event_weight"] = 0.0

    if news_events.empty:
        return TextEmbeddingBuildResult(
            features=empty_payload,
            summary={
                "rows": int(len(base)),
                "news_event_rows": 0,
                "ticker_text_coverage": 0.0,
                "macro_text_coverage": 0.0,
                "text_embedding_columns": TEXT_EMBEDDING_COLUMNS,
                "macro_text_embedding_columns": MACRO_TEXT_EMBEDDING_COLUMNS,
                "lookback_days": lookback_days,
                "half_life_days": half_life_days,
            },
            traces=[],
        )

    news_events["published_at"] = pd.to_datetime(news_events["published_at"], utc=True)
    news_events["known_at"] = pd.to_datetime(news_events["known_at"], utc=True)
    news_events["event_effective_at"] = _event_effective_day(news_events["known_at"])
    news_events["document_text"] = (
        news_events["headline"].fillna("").astype(str)
        + " "
        + news_events["body"].fillna("").astype(str)
        + " "
        + news_events["event_type"].fillna("").astype(str)
    ).str.strip()
    vectorizer = HashingVectorizer(
        n_features=TEXT_EMBEDDING_DIM,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
    )
    embeddings = vectorizer.transform(news_events["document_text"].tolist()).toarray()
    for index, column in enumerate(TEXT_EMBEDDING_COLUMNS):
        news_events[column] = embeddings[:, index]
    news_events["event_weight"] = (
        pd.to_numeric(news_events.get("source_weight", 1.0), errors="coerce").fillna(1.0)
        * pd.to_numeric(news_events.get("entity_confidence", 1.0), errors="coerce").fillna(1.0)
        * pd.to_numeric(news_events.get("novelty_score", 1.0), errors="coerce").fillna(1.0)
    )
    news_events["event_count"] = 1.0

    ticker_events = news_events[news_events["event_scope"].astype(str) == "ticker"].copy()
    ticker_daily = _weighted_group_features(ticker_events, ["ticker", "event_effective_at"], TEXT_EMBEDDING_COLUMNS)
    ticker_rolled = _apply_decay_by_ticker(
        ticker_daily,
        target_frame=base[["ticker", "effective_at"]].drop_duplicates(),
        date_column="event_effective_at",
        key_column="ticker",
        value_columns=[*TEXT_EMBEDDING_COLUMNS, "text_event_count", "text_event_weight"],
        lookback_days=lookback_days,
        half_life_days=half_life_days,
    )

    macro_events = news_events[news_events["event_scope"].astype(str) == "macro"].copy()
    macro_daily = _weighted_group_features(macro_events, ["event_effective_at"], TEXT_EMBEDDING_COLUMNS)
    macro_rolled = _apply_decay_by_date(
        macro_daily,
        target_dates=base[["effective_at"]].drop_duplicates(),
        date_column="event_effective_at",
        value_columns=[*TEXT_EMBEDDING_COLUMNS, "text_event_count", "text_event_weight"],
        lookback_days=lookback_days,
        half_life_days=half_life_days,
        rename_map={
            **{column: macro_column for column, macro_column in zip(TEXT_EMBEDDING_COLUMNS, MACRO_TEXT_EMBEDDING_COLUMNS)},
            "text_event_count": "macro_text_event_count",
            "text_event_weight": "macro_text_event_weight",
        },
    )

    feature_frame = base.merge(
        ticker_rolled.rename(columns={"event_effective_at": "effective_at"}),
        on=["ticker", "effective_at"],
        how="left",
    ).merge(
        macro_rolled.rename(columns={"event_effective_at": "effective_at"}),
        on="effective_at",
        how="left",
    )
    feature_frame = feature_frame.drop(columns=["known_at"])
    for column in [
        *TEXT_EMBEDDING_COLUMNS,
        *MACRO_TEXT_EMBEDDING_COLUMNS,
        "text_event_count",
        "text_event_weight",
        "macro_text_event_count",
        "macro_text_event_weight",
    ]:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(0.0)

    sample_row = feature_frame.iloc[0].to_dict() if not feature_frame.empty else {}
    traces = [
        {
            "formula_id": "phase2.text_embeddings.decayed_pooling",
            "label": "Ticker and macro text embeddings",
            "inputs": {
                "lookback_days": lookback_days,
                "half_life_days": half_life_days,
                "event_rows": int(len(news_events)),
            },
            "transformed_inputs": {
                "ticker_text_rows": int(len(ticker_events)),
                "macro_text_rows": int(len(macro_events)),
                "embedding_dim": TEXT_EMBEDDING_DIM,
            },
            "output": {
                "text_event_count": float(sample_row.get("text_event_count", 0.0)),
                "macro_text_event_count": float(sample_row.get("macro_text_event_count", 0.0)),
                "text_embedding_00": float(sample_row.get(TEXT_EMBEDDING_COLUMNS[0], 0.0)),
            },
            "units": "text_embedding_features",
            "provenance": {"dataset_version_id": dataset_version_id},
        }
    ]
    return TextEmbeddingBuildResult(
        features=feature_frame,
        summary={
            "rows": int(len(feature_frame)),
            "news_event_rows": int(len(news_events)),
            "ticker_text_coverage": float((feature_frame["text_event_count"] > 0).mean() if len(feature_frame) else 0.0),
            "macro_text_coverage": float((feature_frame["macro_text_event_count"] > 0).mean() if len(feature_frame) else 0.0),
            "text_embedding_columns": TEXT_EMBEDDING_COLUMNS,
            "macro_text_embedding_columns": MACRO_TEXT_EMBEDDING_COLUMNS,
            "lookback_days": lookback_days,
            "half_life_days": half_life_days,
        },
        traces=traces,
    )


def _event_effective_day(known_at: pd.Series) -> pd.Series:
    normalized = known_at.dt.normalize()
    after_close = known_at.dt.hour >= 16
    shifted = normalized.where(~after_close, normalized + pd.offsets.BDay(1))
    return pd.to_datetime(shifted, utc=True)


def _weighted_group_features(frame: pd.DataFrame, group_columns: list[str], embedding_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        columns = [*group_columns, *embedding_columns, "text_event_count", "text_event_weight"]
        return pd.DataFrame(columns=columns)
    grouped_rows: list[dict[str, object]] = []
    for group_key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        payload = {column_name: group_value for column_name, group_value in zip(group_columns, group_key)}
        weights = group["event_weight"].to_numpy(dtype=float)
        weight_sum = float(weights.sum())
        for column in embedding_columns:
            values = group[column].to_numpy(dtype=float)
            payload[column] = float(np.average(values, weights=weights)) if weight_sum > 0 else 0.0
        payload["text_event_count"] = float(group["event_count"].sum())
        payload["text_event_weight"] = weight_sum
        grouped_rows.append(payload)
    return pd.DataFrame(grouped_rows)


def _apply_decay_by_ticker(
    frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    date_column: str,
    key_column: str,
    value_columns: list[str],
    lookback_days: int,
    half_life_days: float,
) -> pd.DataFrame:
    if target_frame.empty:
        return pd.DataFrame(columns=[key_column, date_column, *value_columns])
    expanded_rows: list[pd.DataFrame] = []
    decay_lambda = np.log(2.0) / max(half_life_days, 0.5)
    all_tickers = sorted(set(target_frame[key_column].astype(str).tolist()))
    for ticker in all_tickers:
        targets = target_frame[target_frame[key_column].astype(str) == ticker].copy().sort_values("effective_at").reset_index(drop=True)
        group = frame[frame[key_column].astype(str) == ticker].copy().sort_values(date_column).reset_index(drop=True)
        dates = pd.to_datetime(group[date_column], utc=True) if not group.empty else pd.Series(dtype="datetime64[ns, UTC]")
        matrix = group[value_columns].to_numpy(dtype=float) if not group.empty else np.zeros((0, len(value_columns)))
        rolled = []
        for current_date in pd.to_datetime(targets["effective_at"], utc=True):
            age_days = (current_date - dates).dt.days.to_numpy() if len(dates) else np.array([], dtype=int)
            valid_mask = (age_days >= 0) & (age_days <= lookback_days)
            if not valid_mask.any():
                rolled.append(np.zeros(len(value_columns)))
                continue
            decay = np.exp(-decay_lambda * age_days[valid_mask])
            weighted = matrix[valid_mask] * decay[:, None]
            rolled.append(weighted.sum(axis=0))
        rolled_frame = pd.DataFrame(rolled, columns=value_columns)
        rolled_frame[key_column] = ticker
        rolled_frame[date_column] = targets["effective_at"].to_numpy()
        expanded_rows.append(rolled_frame[[key_column, date_column, *value_columns]])
    return pd.concat(expanded_rows, ignore_index=True) if expanded_rows else pd.DataFrame(columns=[key_column, date_column, *value_columns])


def _apply_decay_by_date(
    frame: pd.DataFrame,
    target_dates: pd.DataFrame,
    date_column: str,
    value_columns: list[str],
    lookback_days: int,
    half_life_days: float,
    rename_map: dict[str, str],
) -> pd.DataFrame:
    if target_dates.empty:
        return pd.DataFrame(columns=[date_column, *rename_map.values()])
    frame = frame.sort_values(date_column).reset_index(drop=True)
    dates = pd.to_datetime(frame[date_column], utc=True) if not frame.empty else pd.Series(dtype="datetime64[ns, UTC]")
    matrix = frame[value_columns].to_numpy(dtype=float) if not frame.empty else np.zeros((0, len(value_columns)))
    decay_lambda = np.log(2.0) / max(half_life_days, 0.5)
    rolled = []
    target_index = pd.to_datetime(target_dates["effective_at"], utc=True)
    for current_date in target_index:
        age_days = (current_date - dates).dt.days.to_numpy() if len(dates) else np.array([], dtype=int)
        valid_mask = (age_days >= 0) & (age_days <= lookback_days)
        if not valid_mask.any():
            rolled.append(np.zeros(len(value_columns)))
            continue
        decay = np.exp(-decay_lambda * age_days[valid_mask])
        weighted = matrix[valid_mask] * decay[:, None]
        rolled.append(weighted.sum(axis=0))
    rolled_frame = pd.DataFrame(rolled, columns=value_columns)
    rolled_frame[date_column] = target_dates["effective_at"].to_numpy()
    return rolled_frame.rename(columns=rename_map)
