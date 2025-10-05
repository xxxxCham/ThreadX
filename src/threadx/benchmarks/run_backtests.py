"""
ThreadX Benchmark Runner - Backtests
==================================

Exécute des benchmarks sur le moteur de backtest ThreadX:
- Compare CPU vs GPU pour différentes stratégies
- Mesure performance avec différentes tailles de données
- Vérifie le déterminisme des résultats
- Génère rapport CSV et badges Markdown

Exécution:
    python -m threadx.benchmarks.run_backtests [--config file.toml]
"""

import argparse
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Ajouter la racine du projet au sys.path pour les imports
import sys

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from benchmarks.utils import (
    now_tag,
    perf_ns,
    gpu_timer,
    stable_hash,
    hash_series,
    env_snapshot,
)
from threadx.config import ConfigurationError, load_config_dict
from threadx.utils.xp import CUPY_AVAILABLE
from threadx.utils.determinism import set_global_seed
from threadx.backtest import create_engine
from threadx.indicators import get_gpu_accelerated_bank
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_CONFIG = ROOT_DIR / "configs" / "sweeps" / "plan.toml"


def load_config(config_path: str) -> dict:
    """Charge la configuration de benchmark à partir d'un fichier TOML."""

    try:
        config = load_config_dict(config_path)
    except ConfigurationError as exc:
        logger.error(f"❌ Erreur lors du chargement de la configuration: {exc}")
        raise
    except Exception as exc:
        logger.error(f"❌ Erreur inattendue lors du chargement de la configuration: {exc}")
        raise

    if not isinstance(config, dict):
        message = (
            f"❌ Configuration invalide ({config_path}) - contenu inattendu de type "
            f"{type(config).__name__}"
        )
        logger.error(message)
        raise ConfigurationError(message)

    logger.info(f"Configuration chargée: {config_path}")
    return config


def run_backtest_benchmark(config_path: str, output_dir: str = None):
    """
    Exécute le benchmark de backtest selon la configuration.

    Args:
        config_path: Chemin vers fichier de config TOML
        output_dir: Répertoire de sortie pour rapports (optionnel)
    """
    # Répertoire de sortie par défaut
    if output_dir is None:
        output_dir = str(ROOT_DIR / "benchmarks" / "results")
    os.makedirs(output_dir, exist_ok=True)

    # Tag pour cette exécution
    tag = now_tag()

    logger.info(f"🚀 Lancement benchmark backtest - {tag}")
    logger.info(f"📊 Configuration: {config_path}")

    # Charger configuration
    config = load_config(config_path)

    # Préparer résultats
    results = []

    # Mesurer environnement
    env_info = env_snapshot()

    # Créer moteur de backtest
    engine = create_engine()

    # Pour chaque taille de données et stratégie
    for size in config.get("sizes", [10000, 100000]):
        for strategy_name, strategy_config in config.get("strategies", {}).items():
            # Générer données synthétiques
            set_global_seed(42)  # Seed fixe pour reproductibilité
            np.random.seed(42)

            # Données synthétiques
            data_length = size
            df = pd.DataFrame(
                index=pd.date_range(
                    start=datetime(2020, 1, 1), periods=data_length, freq="1min"
                )
            )

            # Simuler prix avec random walk
            price = 100.0
            prices = []
            for _ in range(data_length):
                change = np.random.normal(0, 0.01)
                price *= 1 + change
                prices.append(price)

            # Créer OHLCV
            df["open"] = prices
            df["high"] = df["open"] * (
                1 + np.random.uniform(0, 0.005, size=data_length)
            )
            df["low"] = df["open"] * (1 - np.random.uniform(0, 0.005, size=data_length))
            df["close"] = df["open"] * (
                1 + np.random.normal(0, 0.002, size=data_length)
            )
            df["volume"] = np.random.uniform(100, 1000, size=data_length)

            logger.info(f"📈 Benchmark {strategy_name} - {data_length} candles")

            # Préparer paramètres
            params = strategy_config.get("params", {})

            # Exécution CPU
            logger.info("🔄 Exécution CPU...")
            cpu_start = perf_ns()()
            try:
                cpu_result = engine.run(
                    data=df, strategy=strategy_name, params=params, use_gpu=False
                )
                cpu_time_ns = perf_ns()() - cpu_start
                cpu_time_ms = cpu_time_ns / 1_000_000
                cpu_equity_hash = hash_series(cpu_result.equity)
                cpu_success = True
            except Exception as e:
                logger.error(f"❌ Erreur CPU: {e}")
                cpu_time_ms = 0
                cpu_equity_hash = "error"
                cpu_success = False

            # Exécution GPU si disponible
            gpu_time_ms = 0
            gpu_equity_hash = "not_available"
            gpu_success = False

            if CUPY_AVAILABLE:
                logger.info("🔄 Exécution GPU...")
                try:
                    with gpu_timer() as timer:
                        gpu_result = engine.run(
                            data=df, strategy=strategy_name, params=params, use_gpu=True
                        )
                    gpu_time_ms = timer() * 1000  # ms
                    gpu_equity_hash = hash_series(gpu_result.equity)
                    gpu_success = True
                except Exception as e:
                    logger.error(f"❌ Erreur GPU: {e}")

            # Calcul speedup
            speedup = cpu_time_ms / gpu_time_ms if gpu_time_ms > 0 else 0

            # Test déterminisme
            deterministic = (
                (cpu_equity_hash == gpu_equity_hash) if gpu_success else None
            )

            # Ajouter aux résultats
            result = {
                "timestamp": tag,
                "strategy": strategy_name,
                "size": data_length,
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
            results.append(result)

            logger.info(
                f"⏱️ CPU: {cpu_time_ms:.2f}ms, GPU: {gpu_time_ms:.2f}ms, "
                f"Speedup: {speedup:.2f}x"
            )

    # Sauvegarder résultats en CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"benchmark_backtests_{tag}.csv")
    results_df.to_csv(csv_path, index=False)

    # Résumé en JSON
    summary = {
        "timestamp": tag,
        "strategies_tested": len(config.get("strategies", {})),
        "sizes_tested": config.get("sizes", []),
        "avg_speedup": results_df["speedup"].mean() if not results_df.empty else 0,
        "max_speedup": results_df["speedup"].max() if not results_df.empty else 0,
        "all_deterministic": (
            all(results_df["deterministic"]) if not results_df.empty else False
        ),
        "env": env_info,
    }

    json_path = os.path.join(output_dir, f"benchmark_backtests_summary_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Générer rapport Markdown
    md_path = os.path.join(output_dir, f"benchmark_backtests_{tag}.md")
    with open(md_path, "w") as f:
        f.write(f"# ThreadX Benchmark - Backtests ({tag})\n\n")

        # Badge KPI
        kpi_ok = summary["avg_speedup"] >= 3.0 and summary["all_deterministic"]
        badge = "✅ OK" if kpi_ok else "❌ KO"
        f.write(f"**KPI Status**: {badge}\n\n")

        # Tableau des résultats
        f.write("## Résultats\n\n")
        f.write(
            "| Stratégie | Taille | CPU (ms) | GPU (ms) | Speedup | Déterministe |\n"
        )
        f.write(
            "|-----------|--------|----------|----------|---------|---------------|\n"
        )

        for _, row in results_df.iterrows():
            determ = "✅" if row["deterministic"] else "❌"
            if row["deterministic"] is None:
                determ = "N/A"

            f.write(
                f"| {row['strategy']} | {row['size']} | "
                f"{row['cpu_time_ms']:.2f} | {row['gpu_time_ms']:.2f} | "
                f"{row['speedup']:.2f}x | {determ} |\n"
            )

        # Environnement
        f.write("\n## Environnement\n\n")
        f.write(f"- **OS**: {env_info['os']}\n")
        f.write(f"- **CPU**: {env_info['cpu']}\n")
        f.write(f"- **RAM**: {env_info['ram_gb']} GB\n")
        f.write(f"- **GPU**: {env_info['gpu']}\n")
        f.write(f"- **CUDA**: {env_info['cuda_version']}\n")

    logger.info(f"✅ Benchmark terminé - Rapport: {md_path}")
    return results_df, summary


def main():
    """Point d'entrée principal pour benchmark des backtests."""
    parser = argparse.ArgumentParser(description="ThreadX Benchmark Runner - Backtests")

    # Options de benchmark
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Fichier de configuration TOML (défaut: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "benchmarks" / "results"),
        help="Répertoire de sortie (défaut: benchmarks/results)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbose")

    args = parser.parse_args()

    # Exécuter le benchmark
    run_backtest_benchmark(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
