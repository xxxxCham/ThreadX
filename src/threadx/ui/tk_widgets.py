"""
ThreadX Point d'entrée unifié pour l'interface Tkinter
======================================================

Centralisation du point d'entrée pour l'application Tkinter.

Cette interface unifiée remplace les précédents scripts ad-hoc
et utilise l'infrastructure centralisée de ThreadX.

Usage:
------
    python -m threadx.ui.tkinter
    python -m threadx.ui.tkinter --theme=dark

Author: ThreadX Team
Version: Phase A
"""

import sys
import argparse
from pathlib import Path

# Pour assurer que le module threadx est dans le PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.parent


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="ThreadX - Interface Tkinter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--debug", action="store_true", help="Mode debug avec logs détaillés"
    )

    parser.add_argument(
        "--dev", action="store_true", help="Mode développement (options avancées)"
    )

    parser.add_argument(
        "--theme",
        choices=["dark", "light", "auto"],
        default="dark",
        help="Thème de l'interface (défaut: dark)",
    )

    parser.add_argument(
        "--config", type=Path, help="Chemin vers fichier de configuration personnalisé"
    )

    return parser.parse_args()


def main():
    """Point d'entrée principal pour l'application Tkinter."""
    # Import dynamique pour éviter les dépendances circulaires
    try:
        from threadx.ui.app import ThreadXApp

        # Parse les arguments
        args = parse_arguments()

        # Créer et lancer l'application
        app = ThreadXApp(debug=args.debug, theme=args.theme, dev_mode=args.dev)

        # Lancement
        app.run()

        return 0

    except ImportError as e:
        print(f"❌ Erreur: {e}")
        print("\n💡 Vérifiez que tkinter est correctement installé:")
        print("   - Sur Windows: Tkinter est inclus avec Python")
        print("   - Sur Linux: sudo apt-get install python3-tk")
        return 1
    except Exception as e:
        print(f"❌ Erreur lors du lancement de l'application: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
