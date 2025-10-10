"""Benchmarks légers pour valider les performances indicateurs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.config import load_settings
from threadx.indicators.bank import IndicatorBank
from threadx.utils.determinism import get_rng, set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)
SETTINGS = load_settings()
MAX_WORKERS = max(1, SETTINGS.MAX_WORKERS)
SEED_GLOBAL = SETTINGS.MC_SEED


def generate_benchmark_data(n_points: int) -> pd.DataFrame:
    rng = get_rng(SEED_GLOBAL)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="1min")
    close = 50000 + rng.normal(0, 200, size=n_points).cumsum()
    high = close + np.abs(rng.normal(0, 50, size=n_points))
    low = close - np.abs(rng.normal(0, 50, size=n_points))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = rng.integers(1_000, 100_000, size=n_points)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _run_benchmark_case(bank: IndicatorBank, indicator: str, params: dict[str, float], data: pd.DataFrame) -> dict[str, float | str | int | None]:
    start = time.perf_counter()
    bank.ensure(indicator, params, data if indicator == "atr" else data["close"], symbol="BENCH", timeframe="1m")
    duration = time.perf_counter() - start
    throughput = len(data) / duration if duration > 0 else 0.0
    return {
        "indicator": indicator,
        "params": str(params),
        "n_points": len(data),
        "duration": duration,
        "throughput": throughput,
    }


def _dispatch_cases(indicator: str, param_grid: Iterable[dict[str, float]], data_sizes: Iterable[int]) -> list[dict[str, float | str | int | None]]:
    bank = IndicatorBank()
    cases = []
    for n_points in data_sizes:
        data = generate_benchmark_data(n_points)
        for params in param_grid:
            cases.append((n_points, params, data))

    results: list[dict[str, float | str | int | None]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_run_benchmark_case, bank, indicator, params, data)
            for (_, params, data) in cases
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_comprehensive_benchmarks() -> pd.DataFrame:
    set_global_seed(SEED_GLOBAL)
    bollinger_params = [
        {"period": 20, "std": 2.0},
        {"period": 50, "std": 1.5},
    ]
    atr_params = [
        {"period": 14, "method": "ema"},
        {"period": 21, "method": "sma"},
    ]
    data_sizes = [10_000, 50_000]

    bb_results = _dispatch_cases("bollinger", bollinger_params, data_sizes)
    atr_results = _dispatch_cases("atr", atr_params, data_sizes)

    results = pd.DataFrame(bb_results + atr_results)
    logger.info("Benchmark terminé (%d scénarios)", len(results))
    return results


if __name__ == "__main__":
    df_results = run_comprehensive_benchmarks()
    print(df_results.head())
