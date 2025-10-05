"""
Démonstration du système de décision dynamique CPU/GPU et auto-profilage multi-GPU.

Ce script montre comment:
1. Le système de décision dynamique CPU/GPU fonctionne
2. L'auto-balance des GPUs est effectuée
3. Le déterminisme est préservé avec la fusion déterministe

Usage:
    python -m threadx.demo_gpu_auto
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configuration manuelle du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from threadx.utils.log import get_logger
from threadx.utils.gpu import get_default_manager
from threadx.utils.gpu.device_manager import CUPY_AVAILABLE, list_devices
from threadx.indicators import get_gpu_accelerated_bank
from threadx.utils.gpu.profile_persistence import (
    get_gpu_thresholds,
    get_multigpu_ratios,
    is_profile_valid,
)

# Configuration du logger pour afficher plus d'informations
logger = get_logger(__name__)


def demo_dynamic_decision():
    """Démontre le système de décision dynamique CPU/GPU."""
    print("\n===== DÉMONSTRATION: Décision Dynamique CPU/GPU =====")

    # Vérifier disponibilité GPU
    if not CUPY_AVAILABLE:
        print("GPU non disponible, impossible de démontrer le système de décision.")
        return False

    # Obtenir banque d'indicateurs avec accélération
    bank = get_gpu_accelerated_bank()

    # Générer données synthétiques
    np.random.seed(42)  # Pour reproductibilité
    sample_sizes = [5000, 50000, 500000]

    for size in sample_sizes:
        prices = np.cumsum(np.random.normal(0, 1, size)) + 100

        # Paramètres de l'indicateur
        params = {"period": 20, "std_dev": 2.0}

        # Mesurer temps CPU
        start_cpu = time.time()
        cpu_result = bank.bollinger_bands(prices, use_gpu=False, **params)
        cpu_time = time.time() - start_cpu

        # Mesurer temps GPU
        start_gpu = time.time()
        gpu_result = bank.bollinger_bands(prices, use_gpu=True, **params)
        gpu_time = time.time() - start_gpu

        # Mesurer temps décision automatique
        start_dyn = time.time()
        dyn_result = bank.bollinger_bands(prices, use_dynamic=True, **params)
        dyn_time = time.time() - start_dyn

        # Déterminer la meilleure option
        best = "CPU" if cpu_time <= gpu_time else "GPU"
        speedup = max(cpu_time, gpu_time) / min(cpu_time, gpu_time)

        # La décision a-t-elle été optimale?
        if best == "CPU" and dyn_time <= 1.1 * cpu_time:
            decision_quality = "✓ Optimale"
        elif best == "GPU" and dyn_time <= 1.1 * gpu_time:
            decision_quality = "✓ Optimale"
        else:
            decision_quality = "✗ Non-optimale"

        print(f"\nTaille échantillon: {size}")
        print(f"Temps CPU: {cpu_time:.4f}s")
        print(f"Temps GPU: {gpu_time:.4f}s")
        print(f"Temps auto-décision: {dyn_time:.4f}s")
        print(f"Meilleure option: {best} (speedup: {speedup:.2f}x)")
        print(f"Qualité décision: {decision_quality}")

    # Afficher le contenu du profil actuel
    profile = get_gpu_thresholds()

    if "entries" in profile and profile["entries"]:
        print("\nProfil de décision CPU/GPU:")
        for key, entry in profile["entries"].items():
            use_gpu = entry.get("gpu_ms_avg", float("inf")) < entry.get("cpu_ms_avg", 0)
            speedup = entry.get("cpu_ms_avg", 0) / entry.get("gpu_ms_avg", float("inf"))
            decision = "GPU" if use_gpu else "CPU"
            print(f"  {key[:30]}...: {decision}, speedup={speedup:.2f}x")
    else:
        print("\nPas d'entrées dans le profil de décision.")

    return True


def demo_auto_balance():
    """Démontre l'auto-balance des GPUs."""
    print("\n===== DÉMONSTRATION: Auto-Balance Multi-GPU =====")

    # Vérifier disponibilité multi-GPU
    if not CUPY_AVAILABLE:
        print("GPU non disponible, impossible de démontrer l'auto-balance.")
        return False

    devices = list_devices()
    if len(devices) < 2:
        print(f"Auto-balance requiert au moins 2 GPUs. {len(devices)} GPU trouvé.")
        return False

    # Afficher les GPUs disponibles
    print(f"GPUs disponibles: {len(devices)}")
    for i, dev in enumerate(devices):
        print(f"  {i}: {dev.name} (ID: {dev.id})")

    # Créer un manager multi-GPU
    gpu_manager = get_default_manager()

    # Vérifier si un profil existe déjà
    profile = get_multigpu_ratios()
    if is_profile_valid(profile) and "ratios" in profile and profile["ratios"]:
        print("\nProfil d'auto-balance existant trouvé:")
        print(f"  Date: {profile.get('updated_at', 'inconnue')}")
        print(f"  Tag: {profile.get('workload_tag', 'default')}")
        print(f"  Ratios: {profile.get('ratios', {})}")

        # Charger le profil existant
        gpu_manager._try_load_balance_from_profile()

    # Afficher balance actuelle
    print("\nBalance actuelle:")
    for name, ratio in gpu_manager.device_balance.items():
        print(f"  {name}: {ratio:.2%}")

    # Fonction de test
    def test_func(data):
        # Convolution (représentatif d'indicateurs techniques)
        kernel = np.ones(20) / 20.0
        return np.convolve(data, kernel, mode="same")

    # Générer données de test
    size = 500000
    np.random.seed(42)
    test_data = np.random.normal(0, 1, size).astype(np.float32)

    # Test avec balance actuelle
    print("\nTest avec balance actuelle...")
    start_time = time.time()
    _ = gpu_manager.distribute_workload(test_data, test_func, seed=42)
    current_time = time.time() - start_time
    print(f"  Temps d'exécution: {current_time:.4f}s")

    # Auto-profiler pour déterminer balance optimale
    print("\nLancement auto-profiling...")
    optimized_balance = gpu_manager.profile_auto_balance(
        sample_size=size, runs=3, workload_tag="demo_auto_balance"
    )

    # Afficher nouvelle balance
    print("\nNouvelle balance optimisée:")
    for name, ratio in optimized_balance.items():
        print(f"  {name}: {ratio:.2%}")

    # Test avec la nouvelle balance
    print("\nTest avec balance optimisée...")
    start_time = time.time()
    _ = gpu_manager.distribute_workload(test_data, test_func, seed=42)
    optimized_time = time.time() - start_time

    # Calculer amélioration
    improvement = (current_time - optimized_time) / current_time * 100
    print(f"  Temps d'exécution: {optimized_time:.4f}s")
    print(f"  Amélioration: {improvement:.1f}%")

    return True


def demo_deterministic_merge():
    """Démontre la fusion déterministe des résultats."""
    print("\n===== DÉMONSTRATION: Fusion Déterministe =====")

    if not CUPY_AVAILABLE:
        print("GPU non disponible, impossible de démontrer la fusion déterministe.")
        return False

    # Créer un manager multi-GPU
    gpu_manager = get_default_manager()

    if len(gpu_manager._gpu_devices) < 1:
        print("Pas de GPU disponible pour la démonstration.")
        return False

    # Fonction de test qui génère des résultats sensibles à l'ordre
    def sensitive_func(x):
        # Cette fonction est sensible à l'ordre à cause du np.cumsum
        # Les résultats diffèrent légèrement à cause des erreurs d'arrondi
        # en particulier sur différents devices
        return np.cumsum(np.sin(x) * 0.01) + x

    # Générer données de test
    size = 10000
    np.random.seed(42)
    test_data = np.random.normal(0, 1, size).astype(np.float32)

    print("Exécution multiple avec seed fixe, vérification déterminisme...")

    # Exécuter plusieurs fois et vérifier identité des résultats
    results = []
    for i in range(3):
        start_time = time.time()
        result = gpu_manager.distribute_workload(test_data, sensitive_func, seed=42)
        results.append(result)
        print(f"  Run {i+1}: {time.time() - start_time:.4f}s")

    # Vérifier si résultats identiques
    identical = all(np.array_equal(results[0], r) for r in results[1:])

    if identical:
        print("\n✓ Tous les résultats sont identiques - Déterminisme confirmé!")
    else:
        print("\n✗ Résultats non identiques - Problème de déterminisme!")
        # Calculer différence max
        diffs = [np.max(np.abs(results[0] - r)) for r in results[1:]]
        print(f"  Différences max: {diffs}")

    return identical


def main():
    """Fonction principale."""
    print("DÉMONSTRATION DU SYSTÈME AUTO-PROFILING GPU ET MULTI-GPU")
    print("=" * 60)

    # Afficher infos sur l'environnement
    print("\nEnvironnement:")
    if CUPY_AVAILABLE:
        devices = list_devices()
        print(f"  GPUs détectés: {len(devices)}")
        for i, dev in enumerate(devices):
            print(
                f"    {i}: {dev.name} (ID: {dev.id}, mémoire: {dev.total_memory/1024**3:.1f} Go)"
            )
    else:
        print("  Pas de GPU détecté (CuPy non disponible)")

    # Vérifier existence dossier profiles
    profiles_dir = Path("artifacts/profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Exécuter les démonstrations
    success_dynamic = demo_dynamic_decision()
    success_balance = demo_auto_balance()
    success_determinism = demo_deterministic_merge()

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE LA DÉMONSTRATION:")
    print(f"  Décision dynamique CPU/GPU: {'✓' if success_dynamic else '✗'}")
    print(f"  Auto-balance multi-GPU: {'✓' if success_balance else '✗'}")
    print(f"  Fusion déterministe: {'✓' if success_determinism else '✗'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
