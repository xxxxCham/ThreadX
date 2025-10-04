#!/usr/bin/env python3
"""
Lanceur TechinTerror Interface - ThreadX Phase 8
===============================================

Script de lancement pour l'interface TechinTerror ThreadX.
Démarre l'application avec BTC par défaut et Nord dark theme.

Usage:
    python launch_techinterror.py

Features:
- Interface TechinTerror avec 5 onglets
- Homepage BTC automatique
- Thème sombre Nord
- Threading non-bloquant
- Téléchargements manuels 1m+3h

Author: ThreadX Framework
Version: Phase 8 - TechinTerror
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Lance l'interface TechinTerror ThreadX."""
    print("🚀 Lancement de l'interface TechinTerror ThreadX...")
    print("📊 Homepage BTC - Thème Nord Dark")
    print("⚡ Threading non-bloquant activé")
    print("💾 Téléchargements manuels 1m+3h disponibles")
    print("-" * 50)
    
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Import and run app
        from threadx.ui.app import run_app
        
        print("✅ Interface TechinTerror chargée avec succès")
        print("🎯 Ouverture de l'application...")
        
        # Run the TechinTerror interface
        run_app()
        
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        print("💡 Vérifiez que ThreadX est installé correctement")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de l'interface TechinTerror")
        print("👋 À bientôt !")
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()