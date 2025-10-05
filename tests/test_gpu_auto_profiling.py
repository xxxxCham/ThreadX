"""
Tests pour le système de profilage automatique multi-GPU.

Ce module teste:
- Le profilage automatique de performance des GPUs
- La distribution dynamique de charge
- La persistance des profils
- La fusion déterministe des résultats

Usage:
    python -m pytest -xvs tests/test_gpu_auto_profiling.py
"""

import os
import time
import shutil
import logging
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from threadx.utils.gpu import (
    get_default_manager,
    MultiGPUManager,
    profile_persistence,
)
from threadx.utils.gpu.profile_persistence import (
    update_gpu_threshold_entry,
    get_gpu_thresholds,
    get_multigpu_ratios,
    update_multigpu_ratios,
)
from threadx.utils.gpu.device_manager import CUPY_AVAILABLE, list_devices
from threadx.indicators import get_gpu_accelerated_bank

# Configuration des tests
SKIP_SLOW = os.environ.get("SKIP_SLOW_TESTS", "0") == "1"
SAMPLE_SIZE = 10000  # Taille réduite pour tests rapides


def setup_module():
    """Configuration globale des tests."""
    # Nettoyer les profils de test
    profile_dir = Path("artifacts/profiles")
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder les profils existants
    for p in profile_dir.glob("*.json"):
        shutil.copy2(p, p.with_suffix(".json.bak"))


def teardown_module():
    """Nettoyage après tests."""
    # Restaurer les profils d'origine
    profile_dir = Path("artifacts/profiles")
    for p in profile_dir.glob("*.json.bak"):
        shutil.move(p, p.with_suffix(""))


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="Requiert CuPy/GPU")
def test_profile_auto_balance():
    """Teste la détermination automatique de la balance multi-GPU."""
    gpu_manager = get_default_manager()

    if len(gpu_manager._gpu_devices) <= 1:
        pytest.skip("Requiert au moins 2 GPUs pour tester le multi-GPU")

    # Profiler avec des données réduites pour test
    ratios = gpu_manager.profile_auto_balance(
        sample_size=SAMPLE_SIZE, runs=2, workload_tag="test_auto_profile"
    )

    # Vérifier que les ratios sont cohérents
    assert len(ratios) == len(gpu_manager._gpu_devices)
    assert sum(ratios.values()) == pytest.approx(1.0)

    # Vérifier que profil a été sauvegardé
    profile = get_multigpu_ratios()
    assert profile["workload_tag"] == "test_auto_profile"
    assert "ratios" in profile

    # Vérifier que la balance a été mise à jour
    for dev_name, ratio in ratios.items():
        assert gpu_manager.device_balance[dev_name] == pytest.approx(ratio)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="Requiert CuPy/GPU")
def test_deterministic_workload_distribution():
    """Vérifie que la distribution et fusion sont déterministes."""
    gpu_manager = get_default_manager()

    # Générer données de test
    np.random.seed(42)
    data = np.random.normal(0, 1, SAMPLE_SIZE).astype(np.float32)

    # Fonction de test
    def test_func(x):
        return np.sin(x) + np.cos(x * 0.5)

    # Exécuter 3 fois avec même seed
    results = []
    for i in range(3):
        result = gpu_manager.distribute_workload(data, test_func, seed=42)
        results.append(result)

    # Vérifier que résultats sont identiques (déterministes)
    for i in range(1, len(results)):
        assert np.array_equal(results[0], results[i])


@pytest.mark.skipif(not CUPY_AVAILABLE or SKIP_SLOW, reason="Test lent requérant GPU")
def test_auto_profile_indicators():
    """Teste le profilage automatique avec indicateurs techniques."""
    bank = get_gpu_accelerated_bank()

    # Générer données de test
    np.random.seed(42)
    size = 50000
    close = np.cumsum(np.random.normal(0, 1, size)) + 100

    # Paramètres pour indicateurs
    params = {"period": 20, "std_dev": 2.0}

    # Tester décision dynamique CPU/GPU
    use_gpu = bank._should_use_gpu_dynamic(
        indicator="bollinger", n_rows=len(close), params=params, dtype=np.float32
    )

    # La décision devrait être basée sur profil
    print(f"Décision dynamique: utiliser GPU = {use_gpu}")

    # Calcul avec profil
    upper, middle, lower = bank.bollinger_bands(close, **params)

    assert upper is not None
    assert middle is not None
    assert lower is not None


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="Requiert CuPy/GPU")
def test_profile_persistence():
    """Vérifie la persistance et rechargement des profils."""
    # Écrire un profil de test
    devices_info = [
        {"id": 0, "name": "test_gpu_0", "throughput": 1.0},
        {"id": 1, "name": "test_gpu_1", "throughput": 0.75},
    ]
    ratios = {"0": 0.6, "1": 0.4}

    # Sauvegarder profil
    update_multigpu_ratios(
        devices=devices_info,
        ratios=ratios,
        sample_size=3,
        workload_tag="test_persistence",
    )

    # Relire profil
    profile = get_multigpu_ratios()

    # Vérifier contenu
    assert profile["workload_tag"] == "test_persistence"
    assert len(profile["devices"]) == 2
    assert profile["ratios"] == ratios


if __name__ == "__main__":
    # Exécution directe pour tests manuels
    print("Tests du système de profilage auto multi-GPU")

    setup_module()

    try:
        # Infos sur environnement
        if CUPY_AVAILABLE:
            devices = list_devices()
            print(f"GPUs disponibles: {len(devices)}")
            for d in devices:
                print(f" - {d.name} (ID: {d.id})")
        else:
            print("CuPy non disponible, tests GPU ignorés")

        # Test du profiling si GPUs disponibles
        if CUPY_AVAILABLE and len(list_devices()) > 0:
            test_profile_persistence()

            if len(list_devices()) > 1:
                print("Exécution test auto-balance multi-GPU...")
                test_profile_auto_balance()

            print("Exécution test déterminisme...")
            test_deterministic_workload_distribution()
    finally:
        teardown_module()
