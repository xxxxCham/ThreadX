#!/usr/bin/env python3
"""
ThreadX Benchmark Demo - Phase C
===============================

Script de démonstration pour lancer les benchmarks et générer un rapport.
Utile pour vérifier rapidement que l'environnement est correctement configuré.

Usage:
    python run_benchmark_demo.py

Ce script exécute:
1. Un benchmark des indicateurs sur CPU et GPU
2. Un test du taux de hit du cache
3. Un test de déterminisme
4. Un test de non-régression Pareto
5. Génère un rapport complet
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Importer les modules ThreadX
from threadx.utils.log import configure_logging, get_logger
from threadx.utils.determinism import set_global_seed

# Configuration du logger
logger = get_logger("benchmark_demo")
configure_logging(level="INFO")

# Définir le seed global pour reproductibilité
SEED = 191159
set_global_seed(SEED)


def main():
    """Point d'entrée principal du script."""
    logger.info("ThreadX Benchmark Demo - Phase C")

    # Vérifier si CuPy est disponible
    try:
        import cupy

        gpu_available = True
        logger.info("GPU disponible ✓")
    except ImportError:
        gpu_available = False
        logger.warning("GPU non disponible ✗")

    # Lancer le benchmark principal
    from tools.benchmarks_cpu_gpu import main as benchmark_main

    # Préparation des arguments
    sys.argv = [
        sys.argv[0],  # script name
        "--indicators",
        "bollinger,atr",
        "--sizes",
        "10000,100000",  # tailles réduites pour la démo
        "--repeats",
        "3",
        "--seed",
        str(SEED),
        "--export",
        "all",
    ]

    # Appeler le benchmark principal
    logger.info("Lancement du benchmark...")
    try:
        benchmark_main()
    except SystemExit as e:
        if e.code != 0:
            logger.error("Le benchmark a échoué avec des KPI non respectés.")
        else:
            logger.info("Benchmark terminé avec succès!")
    except Exception as e:
        logger.error(f"Erreur pendant le benchmark: {e}")

    logger.info("Démo terminée!")


if __name__ == "__main__":
    main()
