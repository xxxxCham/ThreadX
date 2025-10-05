"""
ThreadX KPI Gates Test - Validation des seuils de performance
===========================================================

Tests automatisés pour valider les seuils KPI minimaux:
- Speedup GPU ≥ 3× vs CPU
- Cache hit-rate ≥ 80%
- Déterminisme sur plusieurs exécutions
- Pareto non-régressif (±5% de tolérance)

Ces tests sont exécutés pendant le CI/CD et doivent être verts
pour toute PR avant d'être mergée.

Author: ThreadX Framework
Version: Phase C - Benchmarks & KPI Gates
"""

import os
import time
import json
import pytest
import numpy as np
from pathlib import Path

# Add project root to path for imports
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from threadx.utils.determinism import set_global_seed, hash_df
from threadx.utils.xp import CUPY_AVAILABLE, get_xp
from threadx.indicators import get_gpu_accelerated_bank
from threadx.optimization.pareto import optimize_pareto, ParetoResult
from benchmarks.utils import (
    gpu_timer,
    hash_series,
    stable_hash,
    KPI_SPEEDUP_THRESHOLD,
    KPI_CACHE_HIT_THRESHOLD,
    KPI_PARETO_TOLERANCE,
)


@pytest.fixture
def synthetic_data():
    """Génère des données synthétiques pour les tests."""
    set_global_seed(191159)
    np.random.seed(191159)

    # Simuler une série de prix réaliste
    size = 100_000  # Taille réduite pour les tests rapides
    price = 100.0
    prices = np.zeros(size)

    for i in range(size):
        # Random walk avec drift
        change_pct = np.random.normal(0, 0.01)
        price *= 1 + change_pct
        prices[i] = price

    return prices


def test_speedup_gpu_ge_3x(synthetic_data):
    """
    Test que le speedup GPU est ≥ 3× par rapport au CPU.
    Skip automatiquement si GPU non disponible.
    """
    if not CUPY_AVAILABLE:
        pytest.skip("Pas de GPU disponible pour ce test")

    bank = get_gpu_accelerated_bank()

    # Test sur au moins 2 indicateurs
    indicators = [
        ("bollinger_bands", {"period": 20, "std_dev": 2.0}),
        ("atr", {"period": 14}),
    ]

    for func_name, params in indicators:
        # Obtenir la fonction
        func = getattr(bank, func_name)

        # Mesurer temps CPU
        start_time = time.perf_counter()
        _ = func(synthetic_data, use_gpu=False, **params)
        cpu_time = time.perf_counter() - start_time

        # Mesurer temps GPU
        start_time = time.perf_counter()
        _ = func(synthetic_data, use_gpu=True, **params)
        gpu_time = time.perf_counter() - start_time

        # Calculer speedup
        speedup = cpu_time / gpu_time

        # Vérifier que le speedup est suffisant
        assert (
            speedup >= KPI_SPEEDUP_THRESHOLD
        ), f"Speedup insuffisant pour {func_name}: {speedup:.2f}× < {KPI_SPEEDUP_THRESHOLD}×"


def test_cache_hit_rate_ge_80pct():
    """
    Test que le taux de cache hit est ≥ 80%.
    """
    bank = get_gpu_accelerated_bank()

    # Génération de données test
    np.random.seed(191159)
    size = 50_000  # Taille réduite pour les tests
    data = np.cumsum(np.random.normal(0, 1, size)) + 100

    # Générer multiples paramètres pour le test
    params_list = [
        {"period": period, "std_dev": std}
        for period in range(10, 21)
        for std in [1.5, 2.0, 2.5]
    ]

    # Reset compteurs cache
    bank._reset_cache_stats()

    # Exécuter les calculs
    _ = bank.batch_ensure("bollinger", params_list, data)

    # Obtenir stats cache
    stats = bank._get_cache_stats()
    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0

    assert (
        hit_rate >= KPI_CACHE_HIT_THRESHOLD
    ), f"Cache hit rate insuffisant: {hit_rate:.2%} < {KPI_CACHE_HIT_THRESHOLD:.2%}"


def test_determinism_hash_equal():
    """
    Test que les résultats sont déterministes avec un seed fixe.
    """
    bank = get_gpu_accelerated_bank()

    # Paramètres indicateur
    indicator = "bollinger"
    params = {"period": 20, "std_dev": 2.0}

    # Exécuter 3 fois avec seed fixe
    hashes = []

    for run in range(3):
        # Reset seed global
        set_global_seed(191159)

        # Générer données
        np.random.seed(191159)
        size = 10_000  # Taille réduite pour test rapide
        data = np.cumsum(np.random.normal(0, 1, size)) + 100

        # Calculer l'indicateur
        result = bank.ensure(indicator, params, data)

        # Hash les résultats
        if isinstance(result, tuple):
            run_hash = [hash_series(arr) for arr in result]
        else:
            run_hash = hash_series(result)

        hashes.append(stable_hash(run_hash))

    # Vérifier que tous les hashes sont identiques
    first_hash = hashes[0]
    for i, h in enumerate(hashes[1:], 1):
        assert h == first_hash, f"Run {i} hash ({h}) différent du run 0 ({first_hash})"


def test_pareto_non_regressif():
    """
    Test que l'algorithme Pareto est non-régressif vs baseline.
    """
    # Chemin baseline
    baseline_file = "pareto_baseline.json"
    baseline_path = ROOT_DIR / "benchmarks" / "baselines" / baseline_file

    # Générer données de test
    np.random.seed(191159)
    n_points = 100  # Réduit pour test rapide

    # Créer des points dans l'espace 2D
    points = []
    for i in range(n_points):
        if i < 15:  # Points sur la frontière
            x = np.random.uniform(0.1, 1.0)
            y = 1.0 / x + np.random.normal(0, 0.01)
        else:  # Points dominés
            x = np.random.uniform(0.1, 1.0)
            y = 1.0 / x + np.random.uniform(0.5, 2.0)

        points.append({"id": i, "metrics": {"x": x, "y": y}})

    # Paramètres d'optimisation
    params = {
        "metrics": ["x", "y"],
        "objectives": ["min", "min"],
        "early_stop": True,
        "early_stop_generations": 3,
        "early_stop_threshold": 0.01,
    }

    # Exécuter l'algorithme
    result = optimize_pareto(points, **params)

    # Métriques à comparer
    metrics = {
        "frontier_size": len(result.frontier),
        "iterations": result.iterations,
        "early_stopped": result.early_stopped,
    }

    # Si baseline n'existe pas, la créer
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(metrics, f, indent=2)
        pytest.skip("Baseline créée, skip ce test")

    # Charger baseline existante
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    # Comparer avec tolérance
    for key in ["frontier_size", "iterations"]:
        if key in baseline:
            current = metrics[key]
            reference = baseline[key]
            diff_pct = abs(current - reference) / reference if reference > 0 else 0

            assert diff_pct <= KPI_PARETO_TOLERANCE, (
                f"Régression détectée sur {key}: "
                f"{current} vs {reference} (diff: {diff_pct:.1%} > {KPI_PARETO_TOLERANCE:.1%})"
            )
