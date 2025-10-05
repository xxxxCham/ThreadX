"""
Script de lancement centralisé pour ThreadX
==========================================

Ce script sert de point d'entrée unique pour lancer les différentes
interfaces utilisateur de ThreadX:
- Interface Tkinter (desktop)
- Interface Streamlit (web)

Usage:
    python run_threadx.py [tkinter|streamlit] [--options]

Examples:
    python run_threadx.py tkinter --theme=dark
    python run_threadx.py streamlit --port=8505

Author: ThreadX Team
Version: Phase A
"""

import sys
import argparse
from pathlib import Path

# Ajout du chemin src au PYTHONPATH
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="ThreadX - Plateforme de Trading Algorithmique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "interface",
        choices=["tkinter", "streamlit"],
        default="tkinter",
        nargs="?",
        help="Interface utilisateur à lancer",
    )

    # Options Tkinter
    parser.add_argument(
        "--theme",
        choices=["dark", "light", "auto"],
        default="dark",
        help="Thème de l'interface (pour Tkinter)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Mode debug avec logs détaillés"
    )

    # Options Streamlit
    parser.add_argument(
        "--port", type=int, default=8504, help="Port pour serveur Streamlit"
    )

    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_arguments()

    try:
        if args.interface == "tkinter":
            # Lancement interface Tkinter
            from threadx.ui.tkinter import main as tkinter_main

            tkinter_args = ["--theme", args.theme]
            if args.debug:
                tkinter_args.append("--debug")
            sys.argv = ["tkinter"] + tkinter_args
            return tkinter_main()

        elif args.interface == "streamlit":
            # Lancement interface Streamlit
            from threadx.ui.streamlit import main as streamlit_main

            streamlit_args = ["--server.port", str(args.port)]
            if args.debug:
                streamlit_args.append("--logger.level=debug")
            # Les args sont gérés en interne par streamlit_main
            return streamlit_main()

    except ImportError as e:
        print(f"❌ Erreur: {e}")
        print("\n💡 Vérifiez que toutes les dépendances sont installées:")
        print("   pip install -e .")
        return 1
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
