#!/usr/bin/env python3
"""
ThreadX - Lanceur Interface GUI
==============================

Script de lancement pour l'interface ThreadX Data Manager.
"""

import sys
from pathlib import Path

# Ajouter le chemin du module
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def main():
    """Lance l'interface ThreadX."""
    try:
        from threadx_gui import ThreadXDataManagerGUI

        print("🚀 Lancement ThreadX Data Manager GUI...")
        print("=" * 50)
        print("📊 Interface de gestion des tokens crypto")
        print("⚡ Mise à jour automatique avec indicateurs")
        print("📤 Export multi-formats (CSV, Excel, Parquet)")
        print("📋 Logs temps réel avec coloration")
        print("=" * 50)

        app = ThreadXDataManagerGUI()
        app.run()

    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        print("Vérifiez que threadx_gui.py est dans le même dossier.")
        input("Appuyez sur Entrée pour fermer...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        input("Appuyez sur Entrée pour fermer...")
        sys.exit(1)


if __name__ == "__main__":
    main()
