#!/usr/bin/env python3
"""
ThreadX CPU vs GPU Benchmark - Phase C
======================================

Script de benchmark pour comparer les performances CPU vs GPU:
- Mesures CPU vs GPU pour indicateurs techniques
- Cache hit-rate sur sweeps
- Déterminisme (hash stable)
- Pareto early-stop non-régressif

Génère:
- CSV: benchmarks/results/bench_cpu_gpu_<TIMESTAMP>.csv
- Markdown: benchmarks/reports/REPORT_<TIMESTAMP>.md
- Badges KPI: OK/KO

Usage:
    python -m tools.benchmarks_cpu_gpu --indicators bollinger,atr --sizes 10000,100000,1000000 --repeats 5 --seed 191159

Author: ThreadX Framework
Version: Phase C - Benchmarks & KPI Gates
"""

import argparse
import csv
import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime

# Ajouter chemin du projet
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Imports ThreadX
from threadx.utils.determinism import set_global_seed, hash_df
from threadx.utils.log import get_logger, configure_logging
from threadx.utils.xp import CUPY_AVAILABLE, get_xp
from threadx.indicators import get_gpu_accelerated_bank
from threadx.optimization.pareto import optimize_pareto, ParetoResult
from benchmarks.utils import (
    now_tag,
    perf_ns,
    gpu_timer,
    stable_hash,
    hash_series,
    write_csv,
    write_md,
    env_snapshot,
    KPI_SPEEDUP_THRESHOLD,
    KPI_CACHE_HIT_THRESHOLD,
    KPI_PARETO_TOLERANCE,
)

# Configuration du logger
logger = get_logger(__name__)
configure_logging(level="INFO")

# Chemins constants
RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"
REPORTS_DIR = ROOT_DIR / "benchmarks" / "reports"
BASELINES_DIR = ROOT_DIR / "benchmarks" / "baselines"


def generate_synthetic_data(size: int, seed: int = 42) -> np.ndarray:
    """
    Génère des données synthétiques pour les benchmarks.

    Args:
        size: Nombre de points de données
        seed: Seed pour reproductibilité

    Returns:
        np.ndarray: Série temporelle synthétique
    """
    np.random.seed(seed)

    # Simuler une série de prix réaliste
    price = 100.0
    prices = np.zeros(size)

    for i in range(size):
        # Random walk avec drift
        change_pct = np.random.normal(0, 0.01)
        price *= 1 + change_pct
        prices[i] = price

    return prices


def benchmark_indicators(
    indicators: List[str],
    sizes: List[int],
    repeats: int = 5,
    warmup: int = 1,
    seed: int = 191159,
) -> List[Dict[str, Any]]:
    """
    Benchmark des indicateurs techniques sur CPU et GPU.

    Args:
        indicators: Liste des indicateurs à tester
        sizes: Liste des tailles de données à tester
        repeats: Nombre de répétitions pour chaque test
        warmup: Nombre d'exécutions de warmup (ignorées dans les résultats)
        seed: Seed pour reproductibilité

    Returns:
        List[Dict[str, Any]]: Résultats des benchmarks
    """
    set_global_seed(seed)

    # Obtenir la banque d'indicateurs
    bank = get_gpu_accelerated_bank()

    # Préparer les résultats
    results = []
    indicator_funcs = {
        "bollinger": bank.bollinger_bands,
        "atr": bank.atr,
        "macd": bank.macd,
        "rsi": bank.rsi,
    }

    # Paramètres par défaut pour chaque indicateur
    default_params = {
        "bollinger": {"period": 20, "std_dev": 2.0},
        "atr": {"period": 14},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "rsi": {"period": 14},
    }

    # Date de benchmark
    benchmark_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Pour chaque indicateur
    for indicator in indicators:
        if indicator not in indicator_funcs:
            logger.warning(f"Indicateur non supporté: {indicator}")
            continue

        func = indicator_funcs[indicator]
        params = default_params[indicator]

        logger.info(f"Benchmarking {indicator}...")

        # Pour chaque taille
        for size in sizes:
            logger.info(f"  Taille: {size:,}")

            # Générer données de test
            data = generate_synthetic_data(size, seed)

            # Mesurer CPU
            cpu_times = []

            # Warmup CPU
            for _ in range(warmup):
                _ = func(data, use_gpu=False, **params)

            # Mesures CPU
            for r in range(repeats):
                timer = perf_ns()
                start = timer()
                result_cpu = func(data, use_gpu=False, **params)
                end = timer()
                elapsed_ns = end - start
                elapsed_ms = elapsed_ns / 1_000_000  # ns to ms
                cpu_times.append(elapsed_ms)

            # Stats CPU
            cpu_mean = np.mean(cpu_times)
            cpu_std = np.std(cpu_times)

            # Ajouter résultat CPU
            results.append(
                {
                    "date": benchmark_date,
                    "indicator": indicator,
                    "N": size,
                    "device": "cpu",
                    "repeats": repeats,
                    "mean_ms": cpu_mean,
                    "std_ms": cpu_std,
                    "gain_vs_cpu": float("nan"),
                    "gpu_kernel_ratio": float("nan"),
                }
            )

            # Si GPU disponible
            if CUPY_AVAILABLE:
                gpu_times = []
                kernel_times = []

                # Warmup GPU
                for _ in range(warmup):
                    _ = func(data, use_gpu=True, **params)

                # Mesures GPU
                for r in range(repeats):
                    # Temps total avec perf_counter
                    timer = perf_ns()
                    start = timer()

                    # Temps kernels avec gpu_timer
                    with gpu_timer() as get_elapsed:
                        result_gpu = func(data, use_gpu=True, **params)
                        # Synchronize pour s'assurer que tous les kernels ont terminé
                        xp = get_xp()
                        if hasattr(xp, "cuda"):
                            xp.cuda.Stream.null.synchronize()

                    # Récupérer les temps
                    end = timer()
                    elapsed_ns = end - start
                    elapsed_ms = elapsed_ns / 1_000_000  # ns to ms
                    kernel_ms = get_elapsed()

                    gpu_times.append(elapsed_ms)
                    kernel_times.append(kernel_ms)

                # Stats GPU
                gpu_mean = np.mean(gpu_times)
                gpu_std = np.std(gpu_times)
                kernel_mean = np.mean(kernel_times)

                # GPU/CPU ratio
                gain_vs_cpu = cpu_mean / gpu_mean if gpu_mean > 0 else float("inf")
                gpu_kernel_ratio = kernel_mean / gpu_mean if gpu_mean > 0 else 0.0

                # Ajouter résultat GPU
                results.append(
                    {
                        "date": benchmark_date,
                        "indicator": indicator,
                        "N": size,
                        "device": "gpu",
                        "repeats": repeats,
                        "mean_ms": gpu_mean,
                        "std_ms": gpu_std,
                        "gain_vs_cpu": gain_vs_cpu,
                        "gpu_kernel_ratio": gpu_kernel_ratio,
                    }
                )

                logger.info(
                    f"  CPU: {cpu_mean:.2f}ms (±{cpu_std:.2f}) | "
                    f"GPU: {gpu_mean:.2f}ms (±{gpu_std:.2f}) | "
                    f"Gain: {gain_vs_cpu:.2f}x"
                )
            else:
                logger.info(f"  CPU: {cpu_mean:.2f}ms (±{cpu_std:.2f}) | GPU: N/A")

    return results


def benchmark_cache_hit_rate(
    indicators: List[str], size: int = 50000, n_params: int = 200
) -> Dict[str, float]:
    """
    Benchmark du taux de hit du cache sur un sweep de paramètres.

    Args:
        indicators: Liste des indicateurs à tester
        size: Taille des données
        n_params: Nombre de jeux de paramètres à tester

    Returns:
        Dict[str, float]: Taux de hit par indicateur
    """
    bank = get_gpu_accelerated_bank()

    # Générer données de test
    data = generate_synthetic_data(size)

    # Résultats
    hit_rates = {}

    # Pour chaque indicateur
    for indicator in indicators:
        logger.info(f"Test cache pour {indicator} avec {n_params} paramètres...")

        # Générer des paramètres variés
        if indicator == "bollinger":
            params_list = [
                {"period": period, "std_dev": std}
                for period in range(10, 30)
                for std in [1.5, 2.0, 2.5, 3.0, 3.5]
            ]
        elif indicator == "atr":
            params_list = [{"period": period} for period in range(7, 30)]
        elif indicator == "macd":
            params_list = [
                {"fast_period": fast, "slow_period": slow, "signal_period": 9}
                for fast in range(8, 16)
                for slow in range(20, 30)
            ]
        elif indicator == "rsi":
            params_list = [{"period": period} for period in range(7, 30)]
        else:
            continue

        # Limiter le nombre de paramètres
        params_list = params_list[:n_params]

        # Reset compteurs de cache
        bank._reset_cache_stats()

        # Appeler batch_ensure pour tous les paramètres
        _ = bank.batch_ensure(indicator, params_list, data)

        # Récupérer statistiques de cache
        stats = bank._get_cache_stats()
        hits = stats.get("hits", 0)
        misses = stats.get("misses", 0)

        # Calculer hit rate
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        hit_rates[indicator] = hit_rate
        logger.info(f"  Hit rate: {hit_rate:.2%} ({hits}/{total})")

    return hit_rates


def test_determinism(
    indicators: List[str], size: int = 100000, runs: int = 3
) -> Dict[str, bool]:
    """
    Teste le déterminisme des indicateurs avec seed fixe.

    Args:
        indicators: Liste des indicateurs à tester
        size: Taille des données
        runs: Nombre d'exécutions à comparer

    Returns:
        Dict[str, bool]: Résultat du test pour chaque indicateur
    """
    bank = get_gpu_accelerated_bank()
    results = {}

    # Paramètres par défaut
    default_params = {
        "bollinger": {"period": 20, "std_dev": 2.0},
        "atr": {"period": 14},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "rsi": {"period": 14},
    }

    logger.info(f"Test de déterminisme sur {runs} exécutions...")

    for indicator in indicators:
        logger.info(f"  Indicateur: {indicator}")
        params = default_params.get(indicator, {})

        hashes = []

        for run in range(runs):
            # Reset le seed global pour chaque run
            set_global_seed(191159)

            # Générer données (identiques à chaque run grâce au seed)
            data = generate_synthetic_data(size, seed=191159)

            # Calculer l'indicateur
            if indicator in ["bollinger", "macd"]:
                # Pour les indicateurs qui retournent plusieurs arrays
                result = bank.ensure(indicator, params, data)
                if isinstance(result, tuple):
                    # Hash de chaque composant
                    run_hash = [hash_series(arr) for arr in result]
                else:
                    run_hash = hash_series(result)
            else:
                # Pour les indicateurs qui retournent un seul array
                result = bank.ensure(indicator, params, data)
                run_hash = hash_series(result)

            hashes.append(stable_hash(run_hash))

        # Vérifier que tous les hashes sont identiques
        is_deterministic = all(h == hashes[0] for h in hashes)
        results[indicator] = is_deterministic

        logger.info(f"  Déterminisme: {'✓' if is_deterministic else '✗'}")

    return results


def test_pareto_non_regression(
    baseline_file: str = "pareto_baseline.json",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Teste la non-régression de l'algorithme Pareto early-stop.

    Args:
        baseline_file: Nom du fichier de baseline

    Returns:
        Tuple[bool, Dict[str, Any]]: (test réussi, métriques)
    """
    # Chemin complet du fichier baseline
    baseline_path = BASELINES_DIR / baseline_file

    # Générer un petit jeu de test Pareto
    np.random.seed(191159)

    # Créer des points dans l'espace 2D (on veut minimiser les deux dimensions)
    n_points = 200
    points = []

    # Générer une frontière de Pareto et des points dominés
    for i in range(n_points):
        if i < 20:
            # Points sur la frontière de Pareto
            x = np.random.uniform(0.1, 1.0)
            y = 1.0 / x + np.random.normal(0, 0.01)
        else:
            # Points dominés
            x = np.random.uniform(0.1, 1.0)
            y = 1.0 / x + np.random.uniform(0.5, 2.0)

        points.append({"id": i, "metrics": {"x": x, "y": y}})

    # Paramètres de l'optimisation
    params = {
        "metrics": ["x", "y"],
        "objectives": ["min", "min"],
        "early_stop": True,
        "early_stop_generations": 3,
        "early_stop_threshold": 0.01,
    }

    # Exécuter l'algorithme
    logger.info("Test de non-régression Pareto...")
    start_time = time.time()
    result: ParetoResult = optimize_pareto(points, **params)
    elapsed = time.time() - start_time

    # Métriques à comparer
    metrics = {
        "frontier_size": len(result.frontier),
        "iterations": result.iterations,
        "early_stopped": result.early_stopped,
        "elapsed_ms": elapsed * 1000,
    }

    logger.info(f"  Frontière: {metrics['frontier_size']} points")
    logger.info(f"  Itérations: {metrics['iterations']}")
    logger.info(f"  Early stop: {'oui' if metrics['early_stopped'] else 'non'}")

    # Vérifier si une baseline existe
    if not baseline_path.exists():
        # Créer une nouvelle baseline
        logger.info("  Pas de baseline existante, création d'une nouvelle")

        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return True, metrics

    # Charger la baseline existante
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    # Comparer les métriques avec une tolérance
    is_ok = True
    for key in ["frontier_size", "iterations"]:
        if key in baseline:
            diff_pct = abs(metrics[key] - baseline[key]) / baseline[key]
            if diff_pct > KPI_PARETO_TOLERANCE:
                logger.warning(
                    f"  Régression détectée sur {key}: "
                    f"{metrics[key]} vs {baseline[key]} (diff: {diff_pct:.1%})"
                )
                is_ok = False

    if is_ok:
        logger.info("  Test Pareto réussi: non-régressif ✓")
    else:
        logger.warning("  Test Pareto échoué: régression détectée ✗")

    return is_ok, metrics


def check_kpi_gates(
    speedup_results: List[Dict[str, Any]],
    cache_results: Dict[str, float],
    determinism_results: Dict[str, bool],
    pareto_result: Tuple[bool, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Vérifie les KPI gates et génère les badges.

    Args:
        speedup_results: Résultats des benchmarks CPU vs GPU
        cache_results: Résultats des tests de cache hit rate
        determinism_results: Résultats des tests de déterminisme
        pareto_result: Résultat du test de Pareto non-régression

    Returns:
        Dict[str, str]: Badges KPI ("OK" ou "KO")
    """
    badges = {}

    # KPI Speedup GPU
    speedup_ok = True
    max_size_results = [
        r for r in speedup_results if r["N"] == max(r["N"] for r in speedup_results)
    ]

    for indicator in set(r["indicator"] for r in max_size_results):
        # Trouver les résultats CPU et GPU pour cet indicateur
        cpu_result = next(
            (
                r
                for r in max_size_results
                if r["indicator"] == indicator and r["device"] == "cpu"
            ),
            None,
        )
        gpu_result = next(
            (
                r
                for r in max_size_results
                if r["indicator"] == indicator and r["device"] == "gpu"
            ),
            None,
        )

        if cpu_result and gpu_result:
            speedup = cpu_result["mean_ms"] / gpu_result["mean_ms"]
            if speedup < KPI_SPEEDUP_THRESHOLD:
                speedup_ok = False
                logger.warning(
                    f"KPI Speedup non atteint pour {indicator}: "
                    f"{speedup:.2f}x < {KPI_SPEEDUP_THRESHOLD}x"
                )

    badges["KPI_SPEEDUP_GPU"] = "OK" if speedup_ok else "KO"

    # KPI Cache Hit Rate
    cache_ok = all(rate >= KPI_CACHE_HIT_THRESHOLD for rate in cache_results.values())
    badges["KPI_CACHE_HIT"] = "OK" if cache_ok else "KO"

    # KPI Determinism
    determinism_ok = all(determinism_results.values())
    badges["KPI_DETERMINISM"] = "OK" if determinism_ok else "KO"

    # KPI Pareto
    pareto_ok = pareto_result[0]
    badges["KPI_PARETO"] = "OK" if pareto_ok else "KO"

    return badges


def generate_markdown_report(
    speedup_results: List[Dict[str, Any]],
    cache_results: Dict[str, float],
    determinism_results: Dict[str, bool],
    pareto_result: Tuple[bool, Dict[str, Any]],
    badges: Dict[str, str],
    timestamp: str,
) -> str:
    """
    Génère un rapport Markdown complet des benchmarks.

    Args:
        speedup_results: Résultats des benchmarks CPU vs GPU
        cache_results: Résultats des tests de cache hit rate
        determinism_results: Résultats des tests de déterminisme
        pareto_result: Résultat du test de Pareto non-régression
        badges: Badges KPI
        timestamp: Timestamp du rapport

    Returns:
        str: Contenu Markdown du rapport
    """
    env = env_snapshot()

    # Formater le contenu Markdown
    md = [
        f"# ThreadX Benchmark Report - {timestamp}",
        "",
        "## Résumé exécutable (badges)",
        "",
        "| KPI | Status |",
        "|-----|--------|",
        f"| Speedup GPU (≥{KPI_SPEEDUP_THRESHOLD}×) | **{badges['KPI_SPEEDUP_GPU']}** |",
        f"| Cache Hit Rate (≥{KPI_CACHE_HIT_THRESHOLD*100:.0f}%) | **{badges['KPI_CACHE_HIT']}** |",
        f"| Déterminisme (hash égal) | **{badges['KPI_DETERMINISM']}** |",
        f"| Pareto (non-régressif ±{KPI_PARETO_TOLERANCE*100:.0f}%) | **{badges['KPI_PARETO']}** |",
        "",
        "## Méthodologie",
        "",
        "### Configuration de mesure",
        "",
        "* **CPU**: Temps mesuré avec `time.perf_counter_ns()`",
        "* **GPU**: Temps des kernels mesuré avec `cupy.cuda.Event()`",
        "* **Warmup**: Premier run ignoré pour GPU",
        "* **Répétitions**: 5 mesures par point",
        "* **Seed**: 191159 pour reproductibilité",
        "* **Données**: Générées synthétiquement avec seed fixe",
        "",
        "## Résultats CPU vs GPU",
        "",
    ]

    # Tableau synthèse pour chaque taille
    for size in sorted(set(r["N"] for r in speedup_results)):
        size_results = [r for r in speedup_results if r["N"] == size]

        md.extend(
            [
                f"### Taille N = {size:,}",
                "",
                "| Indicateur | CPU (ms) | GPU (ms) | Speedup | GPU Kernel Ratio |",
                "|------------|----------|----------|---------|------------------|",
            ]
        )

        for indicator in sorted(set(r["indicator"] for r in size_results)):
            cpu_result = next(
                (
                    r
                    for r in size_results
                    if r["indicator"] == indicator and r["device"] == "cpu"
                ),
                None,
            )
            gpu_result = next(
                (
                    r
                    for r in size_results
                    if r["indicator"] == indicator and r["device"] == "gpu"
                ),
                None,
            )

            if cpu_result and gpu_result:
                speedup = cpu_result["mean_ms"] / gpu_result["mean_ms"]
                kernel_ratio = gpu_result["gpu_kernel_ratio"]

                md.append(
                    f"| {indicator} | {cpu_result['mean_ms']:.2f} ± {cpu_result['std_ms']:.2f} | "
                    f"{gpu_result['mean_ms']:.2f} ± {gpu_result['std_ms']:.2f} | "
                    f"**{speedup:.2f}×** | {kernel_ratio:.2%} |"
                )

        md.append("")

    # Cache Hit Rate
    md.extend(
        [
            "## Cache Hit Rate",
            "",
            "| Indicateur | Hit Rate | Status |",
            "|------------|----------|--------|",
        ]
    )

    for indicator, rate in cache_results.items():
        status = "✓" if rate >= KPI_CACHE_HIT_THRESHOLD else "✗"
        md.append(f"| {indicator} | {rate:.2%} | {status} |")

    md.extend(
        [
            "",
            "## Déterminisme",
            "",
            "| Indicateur | Déterministe | Hash identique |",
            "|------------|--------------|--------------|",
        ]
    )

    for indicator, is_deterministic in determinism_results.items():
        status = "✓" if is_deterministic else "✗"
        md.append(f"| {indicator} | {status} | {is_deterministic} |")

    # Pareto
    pareto_metrics = pareto_result[1]
    md.extend(
        [
            "",
            "## Pareto Early-Stop",
            "",
            "| Métrique | Valeur | Baseline | Diff | Status |",
            "|----------|--------|----------|------|--------|",
            f"| Taille frontière | {pareto_metrics['frontier_size']} | {pareto_metrics.get('baseline_frontier_size', 'N/A')} | - | - |",
            f"| Itérations | {pareto_metrics['iterations']} | {pareto_metrics.get('baseline_iterations', 'N/A')} | - | - |",
            f"| Early stopped | {pareto_metrics['early_stopped']} | - | - | - |",
            f"| Temps (ms) | {pareto_metrics['elapsed_ms']:.2f} | - | - | - |",
        ]
    )

    # Diagnostics en cas de KO
    if "KO" in badges.values():
        md.extend(["", "## Diagnostics", "", "### Problèmes détectés", ""])

        if badges["KPI_SPEEDUP_GPU"] == "KO":
            md.extend(
                [
                    "#### KPI_SPEEDUP_GPU: KO",
                    "",
                    "Causes probables:",
                    "1. **Taille insuffisante** - Le speedup GPU est plus important sur des grandes tailles",
                    "2. **Overhead de transfert** - Transfers CPU↔GPU dominants par rapport au calcul",
                    "3. **Indicateur non vectorisé** - Implémentation non optimisée pour GPU",
                    "",
                    "Actions recommandées:",
                    "1. Augmenter la taille minimale des données pour utilisation GPU (`n_min_gpu`)",
                    "2. Réduire les transfers en augmentant la charge de calcul par transfer",
                    "3. Optimiser l'implémentation GPU avec micro-batching",
                    "",
                ]
            )

        if badges["KPI_CACHE_HIT"] == "KO":
            md.extend(
                [
                    "#### KPI_CACHE_HIT: KO",
                    "",
                    "Causes probables:",
                    "1. **Cache invalidé trop fréquemment** - TTL trop court",
                    "2. **Clés de cache instables** - Génération non déterministe des clés",
                    "3. **Cache désactivé** - Problème de configuration",
                    "",
                    "Actions recommandées:",
                    "1. Augmenter le TTL du cache (actuellement 3600s)",
                    "2. Vérifier la génération des clés de cache dans `bank.py`",
                    "3. S'assurer que le cache est activé dans la configuration",
                    "",
                ]
            )

        if badges["KPI_DETERMINISM"] == "KO":
            md.extend(
                [
                    "#### KPI_DETERMINISM: KO",
                    "",
                    "Causes probables:",
                    "1. **Seed non propagé** - Certains générateurs aléatoires non initialisés",
                    "2. **Opérations non déterministes** - Algorithmes avec composants aléatoires",
                    "3. **Problème de synchronisation GPU** - Ordre d'exécution variable",
                    "",
                    "Actions recommandées:",
                    "1. Vérifier l'utilisation correcte de `set_global_seed()` partout",
                    "2. Remplacer les opérations non déterministes",
                    "3. Utiliser `utils/determinism.py` pour garantir l'ordre d'exécution",
                    "",
                ]
            )

        if badges["KPI_PARETO"] == "KO":
            md.extend(
                [
                    "#### KPI_PARETO: KO",
                    "",
                    "Causes probables:",
                    "1. **Changement d'algorithme** - Modification non intentionnelle",
                    "2. **Changement de paramètres** - Configuration modifiée",
                    "3. **Bug introduit** - Régression dans l'implémentation",
                    "",
                    "Actions recommandées:",
                    "1. Vérifier les changements récents dans `optimization/pareto.py`",
                    "2. Restaurer les paramètres de la baseline",
                    "3. Exécuter tests unitaires pour isoler le problème",
                    "",
                ]
            )

    # Appendix
    md.extend(
        [
            "",
            "## Appendix - Configuration",
            "",
            "### Environnement d'exécution",
            "",
            f"* **Date**: {env['date']}",
            f"* **Python**: {env['python_version']}",
            f"* **NumPy**: {env['numpy_version']}",
            f"* **CuPy**: {env['cupy_version']}",
            f"* **OS**: {env['os']}",
            f"* **CPU**: {env['cpu']}",
            f"* **GPU**: {env['gpu']}",
            f"* **Devices**: {', '.join(env['devices']) if 'devices' in env else 'N/A'}",
            "",
            "### Configuration",
            "",
            f"* **CSV**: `benchmarks/results/bench_cpu_gpu_{timestamp}.csv`",
            f"* **Rapport**: `benchmarks/reports/REPORT_{timestamp}.md`",
            f"* **KPI Speedup Threshold**: {KPI_SPEEDUP_THRESHOLD}×",
            f"* **KPI Cache Hit Threshold**: {KPI_CACHE_HIT_THRESHOLD*100:.0f}%",
            f"* **KPI Pareto Tolerance**: ±{KPI_PARETO_TOLERANCE*100:.0f}%",
        ]
    )

    return "\n".join(md)


def main():
    """Fonction principale du benchmark."""
    # Parser des arguments
    parser = argparse.ArgumentParser(description="ThreadX CPU vs GPU Benchmark")
    parser.add_argument(
        "--indicators",
        type=str,
        default="bollinger,atr",
        help="Indicateurs à tester (séparés par virgules)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="10000,100000,1000000",
        help="Tailles de données à tester (séparées par virgules)",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Nombre de répétitions par test"
    )
    parser.add_argument(
        "--seed", type=int, default=191159, help="Seed pour reproductibilité"
    )
    parser.add_argument(
        "--export",
        type=str,
        choices=["csv", "md", "all"],
        default="all",
        help="Format d'export des résultats",
    )

    args = parser.parse_args()

    # Traiter les arguments
    indicators = args.indicators.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]

    # Créer les répertoires si nécessaire
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    # Timestamp pour les noms de fichiers
    timestamp = now_tag()

    # Définir les chemins de sortie
    csv_path = RESULTS_DIR / f"bench_cpu_gpu_{timestamp}.csv"
    md_path = REPORTS_DIR / f"REPORT_{timestamp}.md"

    # Afficher la configuration
    logger.info(f"ThreadX Benchmark - Phase C (Timestamp: {timestamp})")
    logger.info(f"Indicateurs: {indicators}")
    logger.info(f"Tailles: {sizes}")
    logger.info(f"Répétitions: {args.repeats}")
    logger.info(f"Seed: {args.seed}")

    # Définir le seed global
    set_global_seed(args.seed)

    # 1. Benchmark CPU vs GPU
    speedup_results = benchmark_indicators(
        indicators=indicators, sizes=sizes, repeats=args.repeats, seed=args.seed
    )

    # 2. Test cache hit rate
    cache_results = benchmark_cache_hit_rate(
        indicators=indicators, size=50000, n_params=200
    )

    # 3. Test déterminisme
    determinism_results = test_determinism(indicators=indicators, size=100000, runs=3)

    # 4. Test Pareto non-régression
    pareto_result = test_pareto_non_regression()

    # 5. Vérifier les KPI gates
    badges = check_kpi_gates(
        speedup_results=speedup_results,
        cache_results=cache_results,
        determinism_results=determinism_results,
        pareto_result=pareto_result,
    )

    # 6. Exporter les résultats
    if args.export in ["csv", "all"]:
        write_csv(csv_path, speedup_results)
        logger.info(f"Résultats CSV exportés: {csv_path}")

    if args.export in ["md", "all"]:
        md_content = generate_markdown_report(
            speedup_results=speedup_results,
            cache_results=cache_results,
            determinism_results=determinism_results,
            pareto_result=pareto_result,
            badges=badges,
            timestamp=timestamp,
        )
        write_md(md_path, md_content)
        logger.info(f"Rapport Markdown exporté: {md_path}")

    # Afficher le résumé des badges
    logger.info("Résumé des KPI gates:")
    for badge, status in badges.items():
        logger.info(f"  {badge}: {status}")

    # Code de retour non-zéro si un KPI est KO
    if "KO" in badges.values():
        logger.warning("Au moins un KPI gate a échoué!")
        sys.exit(1)
    else:
        logger.info("Tous les KPI gates sont OK!")
        sys.exit(0)


if __name__ == "__main__":
    main()