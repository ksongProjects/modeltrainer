from __future__ import annotations

import pandas as pd

from quant_platform.pipeline.features import sector_zscore, winsorize_series, zscore_series


def test_winsorize_caps_outliers() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 200.0])
    clipped = winsorize_series(series, limit=2.0)
    assert clipped.max() <= 2.0
    assert clipped.min() >= -2.0


def test_zscore_and_sector_zscore_stay_finite() -> None:
    frame = pd.DataFrame(
        {
            "effective_at": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-02"],
            "sector": ["Tech", "Tech", "Energy", "Tech"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    zscores = zscore_series(frame["value"])
    sector_scores = sector_zscore(frame, "value")
    assert zscores.notna().all()
    assert sector_scores.notna().all()
