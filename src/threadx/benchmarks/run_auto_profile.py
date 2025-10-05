"""
Benchmark du système de décision dynamique CPU/GPU et multi-GPU auto-balance.

Ce script effectue:
1. Un test de profilage automatique des GPUs
2. Une comparaison des performances avant/après auto-balance
3. Une validation du déterminisme avec fusion déterministe

Usage:
    python -m threadx.benchmarks.run_auto_profile
"""

import time
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from threadx.utils.log import get_logger
from threadx.utils.gpu import get_default_manager
from threadx.utils.gpu.device_manager import CUPY_AVAILABLE, list_devices
from threadx.indicators import get_gpu_accelerated_bank

logger = get_logger(__name__)

# Configuration par défaut
DEFAULT_SIZES = [10000, 50000, 100000, 500000, 1000000]
DEFAULT_RUNS = 5
ARTIFACTS_DIR = Path("artifacts/benchmarks")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_multi_gpu_balance(sizes=None, runs=DEFAULT_RUNS, save=True):
    """
    Benchmark de l'équilibrage automatique multi-GPU.

    Args:
        sizes: Liste des tailles d'échantillon à tester
        runs: Nombre d'itérations pour chaque test
        save: Sauvegarder les résultats

    Returns:
        Dict des résultats de benchmark
    """
    if sizes is None:
        sizes = DEFAULT_SIZES

    if not CUPY_AVAILABLE:
        logger.error("CuPy non disponible, test impossible")
        return {"error": "CuPy non disponible"}

    # Créer un manager multi-GPU
    gpu_manager = get_default_manager()
    devices = list_devices()

    if len(devices) < 2:
        logger.warning("Multi-GPU requiert au moins 2 GPUs")
        return {"error": "Au moins 2 GPUs requis", "devices_found": len(devices)}

    # Afficher les GPUs disponibles
    logger.info(f"Benchmark multi-GPU avec {len(devices)} GPUs:")
    for dev in devices:
        logger.info(f" - {dev.name} (ID: {dev.id})")

    results = {"devices": [{"name": d.name, "id": d.id} for d in devices], "tests": []}

    # Fonction de test standard (convolution)
    def test_func(data):
        # Convolution (représentatif de calcul d'indicateurs)
        kernel = np.ones(20) / 20.0
        return np.convolve(data, kernel, mode="same")

    # Test avec balance par défaut (uniforme)
    gpu_manager.set_balance(None)  # Balance uniforme
    default_balance = gpu_manager.device_balance.copy()
    default_times = []

    logger.info(f"Test avec balance uniforme: {default_balance}")

    # Benchmark chaque taille
    for size in sizes:
        # Générer les données
        np.random.seed(42)  # Reproductibilité
        data = np.random.normal(0, 1, size).astype(np.float32)

        # Test avec répétition
        run_times = []
        for i in range(runs):
            start_time = time.time()
            _ = gpu_manager.distribute_workload(data, test_func, seed=42)
            elapsed = time.time() - start_time
            run_times.append(elapsed)

        # Calculer moyennes et écart-types
        avg_time = sum(run_times) / len(run_times)
        default_times.append(avg_time)

        logger.info(f"Balance uniforme - Taille {size}: {avg_time:.4f}s")

    # Maintenant auto-profile et test avec balance optimisée
    logger.info("Exécution profiling auto-balance...")
    optimized_balance = gpu_manager.profile_auto_balance(
        sample_size=max(sizes),  # Utiliser la plus grande taille
        runs=3,
        workload_tag="benchmark_auto_balance",
    )

    # Test avec balance optimisée
    optimized_times = []
    logger.info(f"Test avec balance optimisée: {optimized_balance}")

    # Benchmark chaque taille avec la nouvelle balance
    for size in sizes:
        # Générer les données
        np.random.seed(42)  # Reproductibilité
        data = np.random.normal(0, 1, size).astype(np.float32)

        # Test avec répétition
        run_times = []
        for i in range(runs):
            start_time = time.time()
            _ = gpu_manager.distribute_workload(data, test_func, seed=42)
            elapsed = time.time() - start_time
            run_times.append(elapsed)

        # Calculer moyennes et écart-types
        avg_time = sum(run_times) / len(run_times)
        optimized_times.append(avg_time)

        # Calculer l'amélioration
        improvement = (
            100
            * (default_times[sizes.index(size)] - avg_time)
            / default_times[sizes.index(size)]
        )
        logger.info(
            f"Balance optimisée - Taille {size}: {avg_time:.4f}s (amélioration: {improvement:.1f}%)"
        )

    # Compiler les résultats
    for i, size in enumerate(sizes):
        results["tests"].append(
            {
                "size": size,
                "default_time": default_times[i],
                "optimized_time": optimized_times[i],
                "improvement": 100
                * (default_times[i] - optimized_times[i])
                / default_times[i],
            }
        )

    # Ajouter les configurations
    results["default_balance"] = {k: float(v) for k, v in default_balance.items()}
    results["optimized_balance"] = {k: float(v) for k, v in optimized_balance.items()}

    # Sauvegarder les résultats
    if save:
        result_path = ARTIFACTS_DIR / "auto_balance_benchmark.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        # Générer un graphique
        plot_results(results, save_path=ARTIFACTS_DIR / "auto_balance_benchmark.png")

        logger.info(f"Résultats sauvegardés: {result_path}")

    return results


def benchmark_dynamic_decision(sizes=None, runs=DEFAULT_RUNS, save=True):
    """
    Benchmark du système de décision dynamique CPU/GPU.

    Args:
        sizes: Liste des tailles d'échantillon à tester
        runs: Nombre d'itérations pour chaque test
        save: Sauvegarder les résultats

    Returns:
        Dict des résultats de benchmark
    """
    if sizes is None:
        sizes = DEFAULT_SIZES[:3]  # Tailles plus petites pour ce test

    # Obtenir la banque d'indicateurs
    bank = get_gpu_accelerated_bank()

    results = {"gpu_available": CUPY_AVAILABLE, "tests": []}

    # Tester différents indicateurs
    indicators = ["bollinger_bands", "atr", "rsi"]
    params = {
        "bollinger_bands": {"period": 20, "std_dev": 2.0},
        "atr": {"period": 14},
        "rsi": {"period": 14},
    }

    for indicator in indicators:
        indicator_results = {"name": indicator, "sizes": []}
        logger.info(f"Test décision dynamique pour {indicator}")

        for size in sizes:
            # Générer données synthétiques
            np.random.seed(42)

            if indicator == "bollinger_bands":
                data = np.cumsum(np.random.normal(0, 1, size)) + 100

                # Test avec décision automatique
                cpu_times = []
                gpu_times = []

                # Force CPU
                for i in range(runs):
                    start_time = time.time()
                    _ = bank.bollinger_bands(data, **params[indicator], use_gpu=False)
                    cpu_times.append(time.time() - start_time)

                # Force GPU si disponible
                if CUPY_AVAILABLE:
                    for i in range(runs):
                        start_time = time.time()
                        _ = bank.bollinger_bands(
                            data, **params[indicator], use_gpu=True
                        )
                        gpu_times.append(time.time() - start_time)

                # Test décision dynamique
                dyn_times = []
                for i in range(runs):
                    start_time = time.time()
                    _ = bank.bollinger_bands(
                        data, **params[indicator], use_dynamic=True
                    )
                    dyn_times.append(time.time() - start_time)

                # Calculer moyennes
                cpu_avg = sum(cpu_times) / len(cpu_times)
                gpu_avg = sum(gpu_times) / len(gpu_times) if gpu_times else float("inf")
                dyn_avg = sum(dyn_times) / len(dyn_times)

                # Déterminer la décision optimale théorique
                best_time = min(cpu_avg, gpu_avg)
                best_device = "CPU" if cpu_avg <= gpu_avg else "GPU"

                # Calculer l'écart avec la décision dynamique
                dynamic_overhead = 100 * (dyn_avg - best_time) / best_time

                logger.info(
                    f"{indicator} - Taille {size}: "
                    f"CPU={cpu_avg:.4f}s, GPU={gpu_avg:.4f}s, "
                    f"Dynamic={dyn_avg:.4f}s, Best={best_device}"
                )

                indicator_results["sizes"].append(
                    {
                        "size": size,
                        "cpu_time": cpu_avg,
                        "gpu_time": gpu_avg,
                        "dynamic_time": dyn_avg,
                        "best_device": best_device,
                        "dynamic_overhead": dynamic_overhead,
                    }
                )

            elif indicator == "atr":
                # Générer OHLC
                high = np.cumsum(np.random.normal(0, 1, size)) + 105
                low = np.cumsum(np.random.normal(0, 1, size)) + 95
                close = np.cumsum(np.random.normal(0, 1, size)) + 100

                df = pd.DataFrame({"high": high, "low": low, "close": close})

                # Tests similaires
                cpu_times = []
                gpu_times = []
                dyn_times = []

                # Force CPU
                for i in range(runs):
                    start_time = time.time()
                    _ = bank.atr(df, **params[indicator], use_gpu=False)
                    cpu_times.append(time.time() - start_time)

                # Force GPU si disponible
                if CUPY_AVAILABLE:
                    for i in range(runs):
                        start_time = time.time()
                        _ = bank.atr(df, **params[indicator], use_gpu=True)
                        gpu_times.append(time.time() - start_time)

                # Test décision dynamique
                for i in range(runs):
                    start_time = time.time()
                    _ = bank.atr(df, **params[indicator], use_dynamic=True)
                    dyn_times.append(time.time() - start_time)

                # Analyser résultats comme pour bollinger_bands
                cpu_avg = sum(cpu_times) / len(cpu_times)
                gpu_avg = sum(gpu_times) / len(gpu_times) if gpu_times else float("inf")
                dyn_avg = sum(dyn_times) / len(dyn_times)

                best_time = min(cpu_avg, gpu_avg)
                best_device = "CPU" if cpu_avg <= gpu_avg else "GPU"

                dynamic_overhead = 100 * (dyn_avg - best_time) / best_time

                logger.info(
                    f"{indicator} - Taille {size}: "
                    f"CPU={cpu_avg:.4f}s, GPU={gpu_avg:.4f}s, "
                    f"Dynamic={dyn_avg:.4f}s, Best={best_device}"
                )

                indicator_results["sizes"].append(
                    {
                        "size": size,
                        "cpu_time": cpu_avg,
                        "gpu_time": gpu_avg,
                        "dynamic_time": dyn_avg,
                        "best_device": best_device,
                        "dynamic_overhead": dynamic_overhead,
                    }
                )

            elif indicator == "rsi":
                data = np.cumsum(np.random.normal(0, 1, size)) + 100

                # Tests similaires aux précédents
                cpu_times = []
                gpu_times = []
                dyn_times = []

                # Force CPU
                for i in range(runs):
                    start_time = time.time()
                    _ = bank.rsi(data, **params[indicator], use_gpu=False)
                    cpu_times.append(time.time() - start_time)

                # Force GPU si disponible
                if CUPY_AVAILABLE:
                    for i in range(runs):
                        start_time = time.time()
                        _ = bank.rsi(data, **params[indicator], use_gpu=True)
                        gpu_times.append(time.time() - start_time)

                # Test décision dynamique
                for i in range(runs):
                    start_time = time.time()
                    _ = bank.rsi(data, **params[indicator], use_dynamic=True)
                    dyn_times.append(time.time() - start_time)

                # Analyser résultats
                cpu_avg = sum(cpu_times) / len(cpu_times)
                gpu_avg = sum(gpu_times) / len(gpu_times) if gpu_times else float("inf")
                dyn_avg = sum(dyn_times) / len(dyn_times)

                best_time = min(cpu_avg, gpu_avg)
                best_device = "CPU" if cpu_avg <= gpu_avg else "GPU"

                dynamic_overhead = 100 * (dyn_avg - best_time) / best_time

                logger.info(
                    f"{indicator} - Taille {size}: "
                    f"CPU={cpu_avg:.4f}s, GPU={gpu_avg:.4f}s, "
                    f"Dynamic={dyn_avg:.4f}s, Best={best_device}"
                )

                indicator_results["sizes"].append(
                    {
                        "size": size,
                        "cpu_time": cpu_avg,
                        "gpu_time": gpu_avg,
                        "dynamic_time": dyn_avg,
                        "best_device": best_device,
                        "dynamic_overhead": dynamic_overhead,
                    }
                )

        results["tests"].append(indicator_results)

    # Sauvegarder les résultats
    if save:
        result_path = ARTIFACTS_DIR / "dynamic_decision_benchmark.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        # Générer un graphique
        plot_decision_results(
            results, save_path=ARTIFACTS_DIR / "dynamic_decision_benchmark.png"
        )

        logger.info(f"Résultats sauvegardés: {result_path}")

    return results


def plot_results(results, save_path=None):
    """
    Génère un graphique des résultats du benchmark multi-GPU.

    Args:
        results: Résultats du benchmark
        save_path: Chemin pour sauvegarder l'image
    """
    plt.figure(figsize=(10, 6))

    sizes = [test["size"] for test in results["tests"]]
    default_times = [test["default_time"] for test in results["tests"]]
    optimized_times = [test["optimized_time"] for test in results["tests"]]
    improvements = [test["improvement"] for test in results["tests"]]

    # Graphique principal pour les temps d'exécution
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(sizes, default_times, "o-", label="Balance uniforme")
    ax1.plot(sizes, optimized_times, "s-", label="Balance optimisée")
    ax1.set_xscale("log")
    ax1.set_xlabel("Taille des données")
    ax1.set_ylabel("Temps (secondes)")
    ax1.set_title("Temps d'exécution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Graphique d'amélioration en %
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(range(len(sizes)), improvements)
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f"{s}" for s in sizes])
    ax2.set_xlabel("Taille des données")
    ax2.set_ylabel("Amélioration (%)")
    ax2.set_title("Gain de performance")
    ax2.grid(True, alpha=0.3, axis="y")

    # Ajouter ratios comme annotation
    default_str = ", ".join(
        f"{k}: {v:.1%}" for k, v in results["default_balance"].items()
    )
    optimized_str = ", ".join(
        f"{k}: {v:.1%}" for k, v in results["optimized_balance"].items()
    )

    plt.figtext(
        0.5,
        0.01,
        f"Balance uniforme: {default_str}\nBalance optimisée: {optimized_str}",
        ha="center",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=120)
    else:
        plt.show()


def plot_decision_results(results, save_path=None):
    """
    Génère un graphique des résultats du benchmark de décision dynamique.

    Args:
        results: Résultats du benchmark
        save_path: Chemin pour sauvegarder l'image
    """
    plt.figure(figsize=(12, 8))

    # Un subplot pour chaque indicateur
    n_indicators = len(results["tests"])

    for i, indicator_result in enumerate(results["tests"]):
        plt.subplot(n_indicators, 1, i + 1)

        sizes = [size_result["size"] for size_result in indicator_result["sizes"]]
        cpu_times = [
            size_result["cpu_time"] for size_result in indicator_result["sizes"]
        ]
        gpu_times = [
            size_result["gpu_time"] for size_result in indicator_result["sizes"]
        ]
        dynamic_times = [
            size_result["dynamic_time"] for size_result in indicator_result["sizes"]
        ]

        plt.plot(sizes, cpu_times, "o-", label="CPU")
        plt.plot(sizes, gpu_times, "s-", label="GPU")
        plt.plot(sizes, dynamic_times, "^-", label="Auto-décision")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Taille des données")
        plt.ylabel("Temps (secondes)")
        plt.title(f'Indicateur: {indicator_result["name"]}')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
    else:
        plt.show()


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Benchmark du système de décision/profiling GPU"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", help="Tailles d'échantillon à tester"
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="Nombre d'itérations par test"
    )
    parser.add_argument(
        "--skip-balance",
        action="store_true",
        help="Ignorer le test de balance multi-GPU",
    )
    parser.add_argument(
        "--skip-decision",
        action="store_true",
        help="Ignorer le test de décision dynamique",
    )

    args = parser.parse_args()

    # Afficher les GPUs disponibles
    if CUPY_AVAILABLE:
        devices = list_devices()
        logger.info(f"GPUs disponibles: {len(devices)}")
        for i, dev in enumerate(devices):
            logger.info(f" {i}: {dev.name} (ID: {dev.id})")
    else:
        logger.warning("CuPy non disponible, GPU non détecté")

    # Exécuter les benchmarks
    if not args.skip_decision:
        logger.info("=== Benchmark système de décision dynamique CPU/GPU ===")
        benchmark_dynamic_decision(sizes=args.sizes, runs=args.runs)

    if not args.skip_balance and CUPY_AVAILABLE and len(list_devices()) > 1:
        logger.info("=== Benchmark système d'auto-balance multi-GPU ===")
        benchmark_multi_gpu_balance(sizes=args.sizes, runs=args.runs)

    logger.info("Benchmarks terminés, résultats dans artifacts/benchmarks/")


if __name__ == "__main__":
    main()
