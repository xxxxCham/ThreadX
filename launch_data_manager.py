#!/usr/bin/env python3
"""
ThreadX Data Manager - Launcher Principal
Script de lancement unifié et stable
"""

# --- import fallback for direct execution ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# -------------------------------------------

import logging
from pathlib import Path

def main():
    """Point d'entrée principal"""
    try:
        print("🚀 Démarrage ThreadX Data Manager")
        
        # Import et lancement de l'application 
        from apps.data_manager.main_window import ThreadXDataManagerApp
        
        app = ThreadXDataManagerApp()
        print("✅ Interface initialisée")
        
        # Lancement de la boucle principale
        app.root.mainloop()
        
        print("👋 Fermeture ThreadX Data Manager")
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Vérifiez que ThreadX est installé et que PYTHONPATH est configuré")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
