from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts.findf import FindfRunManifest
from ..settings import Settings


def _lazy_transformers():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return AutoModelForSequenceClassification, AutoTokenizer


def _lazy_torch():
    import torch

    return torch


def _keyword_sentiment_score(text: str) -> float:
    positive_words = {
        "beat",
        "beats",
        "bullish",
        "growth",
        "gain",
        "gains",
        "upside",
        "surge",
        "strong",
        "record",
        "upgrade",
        "profit",
        "profits",
    }
    negative_words = {
        "miss",
        "misses",
        "bearish",
        "decline",
        "drop",
        "drops",
        "downside",
        "weak",
        "downgrade",
        "loss",
        "losses",
        "risk",
        "lawsuit",
        "fall",
    }
    tokens = [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split()]
    if not tokens:
        return 0.0
    pos_hits = sum(token in positive_words for token in tokens)
    neg_hits = sum(token in negative_words for token in tokens)
    return float((pos_hits - neg_hits) / max(len(tokens), 1) * 5.0)


def _fallback_score_news(news: pd.DataFrame, output_path: Path) -> Path:
    headline_text = (
        news["title"].fillna("").astype(str).str.strip()
        + " "
        + news["summary"].fillna("").astype(str).str.strip()
    ).str.strip()
    headline_text = headline_text.replace("", "No text provided.")
    sentiment_score = headline_text.map(_keyword_sentiment_score).clip(-1.0, 1.0)
    neutral = (1.0 - sentiment_score.abs()).clip(lower=0.0)
    positive = sentiment_score.clip(lower=0.0)
    negative = (-sentiment_score).clip(lower=0.0)
    scored = news.copy()
    scored["headline_text"] = headline_text
    scored["sentiment_score"] = sentiment_score
    scored["prob_negative"] = negative
    scored["prob_neutral"] = neutral
    scored["prob_positive"] = positive
    scored["embedding_norm"] = sentiment_score.abs()
    scored["embedding_mean"] = sentiment_score
    scored["embedding_std"] = 0.0
    scored = scored[
        [
            "id",
            "published_at",
            "tickers",
            "source",
            "url",
            "headline_text",
            "sentiment_score",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
            "embedding_norm",
            "embedding_mean",
            "embedding_std",
        ]
    ]
    scored.to_parquet(output_path, index=False)
    return output_path


def score_news_with_finbert(
    settings: Settings,
    manifest: FindfRunManifest,
    snapshot_version: str,
    batch_size: int = 16,
    model_name: str = "ProsusAI/finbert",
) -> Path:
    output_dir = settings.datasets_dir / snapshot_version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "news_scores.parquet"
    if output_path.exists():
        return output_path

    news = pd.read_parquet(manifest.artifact_paths.news_silver).copy()
    if news.empty:
        pd.DataFrame(
            columns=[
                "id",
                "published_at",
                "tickers",
                "sentiment_score",
                "prob_negative",
                "prob_neutral",
                "prob_positive",
                "embedding_norm",
                "embedding_mean",
                "embedding_std",
            ]
        ).to_parquet(output_path, index=False)
        return output_path

    try:
        AutoModelForSequenceClassification, AutoTokenizer = _lazy_transformers()
        torch = _lazy_torch()
    except Exception:
        return _fallback_score_news(news, output_path)

    news["headline_text"] = (
        news["title"].fillna("").astype(str).str.strip()
        + " "
        + news["summary"].fillna("").astype(str).str.strip()
    ).str.strip()
    news["headline_text"] = news["headline_text"].replace("", "No text provided.")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
        model = model.to(device)
        model.eval()
    except Exception:
        return _fallback_score_news(news, output_path)

    probability_rows: list[pd.DataFrame] = []
    texts = news["headline_text"].tolist()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        hidden = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        batch_df = pd.DataFrame(
            {
                "prob_negative": probs[:, 0],
                "prob_neutral": probs[:, 1],
                "prob_positive": probs[:, 2],
                "embedding_norm": np.linalg.norm(hidden, axis=1),
                "embedding_mean": hidden.mean(axis=1),
                "embedding_std": hidden.std(axis=1),
            }
        )
        probability_rows.append(batch_df)

    features = pd.concat(probability_rows, ignore_index=True)
    scored = pd.concat([news.reset_index(drop=True), features], axis=1)
    scored["sentiment_score"] = scored["prob_positive"] - scored["prob_negative"]
    scored = scored[
        [
            "id",
            "published_at",
            "tickers",
            "source",
            "url",
            "headline_text",
            "sentiment_score",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
            "embedding_norm",
            "embedding_mean",
            "embedding_std",
        ]
    ]
    scored.to_parquet(output_path, index=False)
    return output_path
