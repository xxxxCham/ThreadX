"""
Test simple des utilitaires de benchmark ThreadX
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Importer les modules requis
from benchmarks.utils import now_tag, env_snapshot
from threadx.utils.log import get_logger, configure_logging

# Configuration du logger
configure_logging(level="INFO")
logger = get_logger("benchmark_test")


# Test fonctions
def main():
    # Test now_tag
    tag = now_tag()
    logger.info(f"Tag horodatage: {tag}")

    # Test env_snapshot
    env = env_snapshot()
    logger.info(f"Environnement système:")
    logger.info(f"OS: {env['os']}")
    logger.info(f"CPU: {env['cpu']}")
    logger.info(f"RAM: {env['ram_gb']} GB")
    logger.info(f"GPU: {env['gpu']}")
    logger.info(f"CUDA: {env['cuda_version']}")


if __name__ == "__main__":
    main()
