"""
ThreadX Benchmark Runner - Indicateurs
=====================================

Exécute des benchmarks sur les indicateurs ThreadX:
- Compare CPU vs GPU sur divers indicateurs
- Mesure le taux de hit du cache
- Vérifie le déterminisme des calculs
- Génère rapport CSV et badge Markdown

Exécution:
    python -m threadx.benchmarks.run_indicators [--size 1000000] [--runs 5]
"""
# type: ignore  # Trop d'erreurs de type, analyse désactivée

import argparse
import os
from pathlib import Path

# Ajouter la racine du projet au sys.path pour les imports
import sys

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.benchmarks_cpu_gpu import run_benchmark


def main():
    """Point d'entrée principal pour benchmark des indicateurs."""
    parser = argparse.ArgumentParser(
        description="ThreadX Benchmark Runner - Indicateurs"
    )

    # Options de benchmark
    parser.add_argument(
        "--size", type=int, default=1_000_000, help="Taille des données (défaut: 1M)"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Nombre d'exécutions (défaut: 5)"
    )
    parser.add_argument(
        "--indicators",
        type=str,
        default="bollinger,atr",
        help="Liste d'indicateurs à benchmarker (défaut: bollinger,atr)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "benchmarks" / "results"),
        help="Répertoire de sortie (défaut: benchmarks/results)",
    )
    parser.add_argument(
        "--skip-determinism", action="store_true", help="Skip les tests de déterminisme"
    )
    parser.add_argument(
        "--skip-cache", action="store_true", help="Skip les tests de cache"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbose")

    args = parser.parse_args()

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)

    # Exécuter le benchmark
    indicator_list = [i.strip() for i in args.indicators.split(",")]

    # Exécuter benchmark complet
    run_benchmark(
        indicators=indicator_list,
        size=args.size,
        runs=args.runs,
        output_dir=args.output_dir,
        test_determinism=not args.skip_determinism,
        test_cache=not args.skip_cache,
        verbose=args.verbose,
    )

    print(f"✅ Benchmark terminé. Résultats dans {args.output_dir}")


if __name__ == "__main__":
    main()
