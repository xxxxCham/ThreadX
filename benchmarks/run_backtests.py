"""Benchmarks légers pour le moteur de backtest."""

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
from threadx.utils.determinism import get_rng, set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)
SETTINGS = load_settings()
MAX_WORKERS = max(1, SETTINGS.MAX_WORKERS)
SEED_GLOBAL = SETTINGS.MC_SEED


def simulate_backtest_results(n_scenarios: int) -> list[dict]:
    rng = get_rng(SEED_GLOBAL)
    results = []
    for i in range(n_scenarios):
        total_return = float(rng.normal(0.08, 0.20))
        volatility = float(abs(rng.normal(0.15, 0.05)))
        sharpe = total_return / volatility if volatility > 0 else 0.0
        drawdown = min(0.0, -abs(rng.normal(0.2, 0.1)))
        results.append(
            {
                "scenario_id": i,
                "metrics": {
                    "total_return": total_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": drawdown,
                },
            }
        )
    return results


def _benchmark_task(n_scenarios: int) -> dict[str, float | int]:
    start = time.perf_counter()
    results = simulate_backtest_results(n_scenarios)
    duration = time.perf_counter() - start
    throughput = n_scenarios / duration if duration > 0 else 0.0
    return {
        "scenarios": n_scenarios,
        "duration": duration,
        "throughput": throughput,
        "mean_return": float(np.mean([r["metrics"]["total_return"] for r in results])),
    }


def run_backtest_benchmarks(counts: Iterable[int]) -> pd.DataFrame:
    set_global_seed(SEED_GLOBAL)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_benchmark_task, n): n for n in counts}
        completed = []
        for future in as_completed(futures):
            completed.append(future.result())
    df = pd.DataFrame(completed)
    logger.info("Backtest benchmark terminé (%d tâches)", len(df))
    return df.sort_values("scenarios")


if __name__ == "__main__":
    counts = [100, 1000, 5000]
    results = run_backtest_benchmarks(counts)
    print(results)
