"""
ThreadX Data Manager - Point d'entrée
Lancement de l'interface de gestion des données
"""

import sys
from pathlib import Path

# Ajouter le chemin source pour les imports ThreadX
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from apps.data_manager.main_window import main

if __name__ == "__main__":
    main()
