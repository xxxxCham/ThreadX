"""
ThreadX Data Manager - Launcher
Point d'entrée pour lancer le Data Manager GUI
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire ThreadX au path
threadx_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(threadx_root))

# Importer et lancer l'application
if __name__ == "__main__":
    from apps.data_manager.main_window import main

    main()
