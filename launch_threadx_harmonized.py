#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanceur ThreadX Interface Harmonisée
Lance l'interface TradXPro avec la structure de données harmonisée
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire ThreadX au path
THREADX_ROOT = Path(__file__).parent
sys.path.insert(0, str(THREADX_ROOT))
sys.path.insert(0, str(THREADX_ROOT / "src"))


def main():
    """Lance l'interface ThreadX harmonisée"""
    try:
        print("🚀 Lancement ThreadX Interface Harmonisée...")
        print(f"📁 Racine ThreadX: {THREADX_ROOT}")

        # Vérifier structure de données
        data_dirs = [
            THREADX_ROOT / "data" / "crypto_data_json",
            THREADX_ROOT / "data" / "crypto_data_parquet",
            THREADX_ROOT / "data" / "indicateurs_tech_data",
            THREADX_ROOT / "data" / "cache",
        ]

        print("\n📊 Vérification structure de données:")
        for data_dir in data_dirs:
            exists = "✅" if data_dir.exists() else "❌"
            print(f"  {exists} {data_dir.name}")

        # Import et lancement interface
        print("\n🖼️ Lancement interface GUI...")
        from apps.threadx_tradxpro_interface import main as run_interface

        run_interface()

    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        print("🔄 Tentative mode CLI...")

        try:
            from apps.threadx_tradxpro_interface import run_cli_mode

            run_cli_mode()
        except Exception as cli_error:
            print(f"❌ Erreur CLI: {cli_error}")
            return 1

    except Exception as e:
        print(f"❌ Erreur lancement: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
