"""Configuration pytest pour ThreadX.

Ajoute automatiquement src/ au sys.path pour permettre les imports
depuis le package threadx sans installation éditable.
"""

import sys
from pathlib import Path

# Ajouter src/ au sys.path pour que pytest trouve le package threadx
PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"

# Insérer en position 0 pour avoir priorité sur les autres chemins
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def pytest_configure(config):
    """Configuration pytest exécutée avant tous les tests."""
    # S'assurer que src/ est toujours en premier dans sys.path
    src_str = str(SRC_PATH)
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)
