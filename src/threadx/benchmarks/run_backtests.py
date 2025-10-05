"""Backtest benchmark runner for ThreadX."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.utils import env_snapshot, gpu_timer, hash_series, now_tag, perf_ns
from threadx.config import ConfigurationError, load_config_dict
from threadx.utils.determinism import set_global_seed
from threadx.utils.log import get_logger
from threadx.utils.xp import CUPY_AVAILABLE

logger = get_logger(__name__)

DEFAULT_CONFIG = ROOT_DIR / "configs" / "sweeps" / "plan.toml"


def _ensure_positive_int_list(values: Any, config_path: str) -> Iterable[int]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        raise ConfigurationError(config_path, "Invalid `sizes`: expected sequence of positive integers")
    parsed: list[int] = []
    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise ConfigurationError(
                config_path, "Invalid `sizes`: expected positive integers", details=str(value)
            )
        parsed.append(value)
    return parsed


def _ensure_strategy_block(strategies: Any, config_path: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(strategies, dict):
        raise ConfigurationError(config_path, "Invalid `strategies`: expected a mapping")

    validated: Dict[str, Dict[str, Any]] = {}
    for name, block in strategies.items():
        if not isinstance(block, dict):
            raise ConfigurationError(config_path, f"Strategy `{name}` must be a mapping")
        params = block.get("params", {})
        if not isinstance(params, dict):
            raise ConfigurationError(
                config_path, f"Strategy `{name}`: `params` must be a mapping of hyper-parameters"
            )
        validated[name] = block
    return validated


def validate_benchmark_config(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    sizes = _ensure_positive_int_list(config.get("sizes", [10000, 100000]), config_path)
    strategies = _ensure_strategy_block(config.get("strategies", {}), config_path)
    config.setdefault("sizes", list(sizes))
    config.setdefault("strategies", strategies)
    return config


def load_config(path: Path) -> Dict[str, Any]:
    config = load_config_dict(path)
    return validate_benchmark_config(config, str(path))


def _generate_synthetic_data(length: int) -> pd.DataFrame:
    index = pd.date_range(start=datetime(2020, 1, 1), periods=length, freq="1min")
    df = pd.DataFrame(index=index)
    price = 100.0
    prices = []
    rng = np.random.default_rng(42)
    for _ in range(length):
        change = rng.normal(0, 0.01)
        price *= 1 + change
        prices.append(price)
    df["open"] = prices
    df["high"] = df["open"] * (1 + rng.uniform(0, 0.005, size=length))
    df["low"] = df["open"] * (1 - rng.uniform(0, 0.005, size=length))
    df["close"] = df["open"] * (1 + rng.normal(0, 0.002, size=length))
    df["volume"] = rng.uniform(100, 1000, size=length)
    return df


def run_backtest_benchmark(config_path: Path, output_dir: Optional[Path] = None) -> None:
    output_dir = output_dir or ROOT_DIR / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = now_tag()
    logger.info("🚀 Lancement benchmark backtest - %s", tag)
    logger.info("📊 Configuration: %s", config_path)

    config = load_config(config_path)

    results: list[Dict[str, Any]] = []
    env_info = env_snapshot()

    from threadx.backtest import create_engine
    from threadx.indicators import get_gpu_accelerated_bank

    engine = create_engine()
    _ = get_gpu_accelerated_bank()  # Pré-initialise les indicateurs GPU le cas échéant

    for size in config["sizes"]:
        for strategy_name, strategy_config in config["strategies"].items():
            set_global_seed(42)
            np.random.seed(42)

            df = _generate_synthetic_data(size)
            params = strategy_config.get("params", {})

            logger.info("📈 Benchmark %s - %d candles", strategy_name, size)

            cpu_start = perf_ns()()
            try:
                cpu_result = engine.run(data=df, strategy=strategy_name, params=params, use_gpu=False)
                cpu_time_ns = perf_ns()() - cpu_start
                cpu_time_ms = cpu_time_ns / 1_000_000
                cpu_equity_hash = hash_series(cpu_result.equity)
                cpu_success = True
            except Exception as exc:  # pragma: no cover - diagnostic path
                logger.error("❌ Erreur CPU: %s", exc)
                cpu_time_ms = 0.0
                cpu_equity_hash = "error"
                cpu_success = False

            gpu_time_ms = 0.0
            gpu_equity_hash = "not_available"
            gpu_success = False

            if CUPY_AVAILABLE:
                logger.info("🔄 Exécution GPU...")
                try:
                    with gpu_timer() as timer:
                        gpu_result = engine.run(
                            data=df, strategy=strategy_name, params=params, use_gpu=True
                        )
                    gpu_time_ms = timer() * 1000
                    gpu_equity_hash = hash_series(gpu_result.equity)
                    gpu_success = True
                except Exception as exc:  # pragma: no cover - diagnostic path
                    logger.error("❌ Erreur GPU: %s", exc)

            speedup = cpu_time_ms / gpu_time_ms if gpu_time_ms > 0 else 0.0
            deterministic = (cpu_equity_hash == gpu_equity_hash) if gpu_success else None

            results.append(
                {
                    "timestamp": tag,
                    "strategy": strategy_name,
                    "size": size,
                    "cpu_time_ms": cpu_time_ms,
                    "gpu_time_ms": gpu_time_ms,
                    "speedup": speedup,
                    "cpu_success": cpu_success,
                    "gpu_success": gpu_success,
                    "deterministic": deterministic,
                    "cpu_hash": cpu_equity_hash,
                    "gpu_hash": gpu_equity_hash,
                    **env_info,
                }
            )

            logger.info(
                "⏱️ CPU: %.2fms, GPU: %.2fms, Speedup: %.2fx",
                cpu_time_ms,
                gpu_time_ms,
                speedup,
            )

    results_df = pd.DataFrame(results)
    csv_path = output_dir / f"benchmark_backtests_{tag}.csv"
    results_df.to_csv(csv_path, index=False)

    summary = {
        "tag": tag,
        "config": str(config_path),
        "results": len(results_df),
        "cpu_success_rate": float(results_df["cpu_success"].mean() if not results_df.empty else 0.0),
        "gpu_success_rate": float(results_df["gpu_success"].mean() if not results_df.empty else 0.0),
        "avg_speedup": float(results_df["speedup"].mean() if not results_df.empty else 0.0),
        "env": env_info,
    }

    json_path = output_dir / f"benchmark_backtests_{tag}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("✅ Résultats sauvegardés: %s", csv_path)
    logger.info("📄 Résumé: %s", json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ThreadX backtest benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        run_backtest_benchmark(args.config, args.output)
    except ConfigurationError as exc:
        logger.error(exc.user_message)
        logger.debug("Configuration error", exc_info=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
