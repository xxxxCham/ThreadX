from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threadx.indicators.engine import enrich_indicators


def _build_ohlc_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = rng.normal(100.0, 2.0, size=n)
    high = base + rng.normal(1.0, 0.5, size=n)
    low = base - rng.normal(1.0, 0.5, size=n)
    close = base + rng.normal(0.0, 0.2, size=n)
    return pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
    })


def test_enrich_indicators_xatr_cpu():
    df = _build_ohlc_frame()
    specs = [{"name": "xatr", "params": {"period": 14}, "outputs": ["atr14"]}]

    enriched = enrich_indicators(df, specs, backend="cpu")

    assert "atr14" in enriched.columns
    assert len(enriched) == len(df)
    assert enriched["atr14"].isna().sum() < len(df)


@pytest.mark.parametrize("backend", ["auto", "gpu"])
def test_enrich_indicators_backend_selection(backend: str):
    df = _build_ohlc_frame()
    specs = [{"name": "xatr", "params": {"period": 10}, "outputs": ["atr10"]}]

    enriched = enrich_indicators(df, specs, backend=backend)

    assert "atr10" in enriched
    assert enriched["atr10"].shape[0] == df.shape[0]
