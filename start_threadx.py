"""
ThreadX - Lanceur d'Interface Simplifié
========================================

Script de lancement pratique pour l'interface Dash ThreadX.
Vérifie l'environnement, installe les dépendances manquantes,
et lance l'interface web sur http://127.0.0.1:8050

Usage:
    python start_threadx.py

Auteur: ThreadX Framework
Date: 14 Octobre 2025
"""

import subprocess
import sys
from pathlib import Path


def print_banner():
    """Affiche la bannière ThreadX."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║                     THREADX LAUNCHER                        ║
║                  Crypto Trading Framework                   ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_python_version():
    """Vérifie que Python 3.10+ est installé."""
    print("🔍 Vérification de Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ requis (installé: {version.major}.{version.minor})")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Vérifie et installe les dépendances manquantes."""
    print("\n🔍 Vérification des dépendances...")

    required_packages = [
        "dash",
        "dash-bootstrap-components",
        "pandas",
        "plotly",
        "typer",
        "rich",
    ]

    missing = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} - MANQUANT")

    if missing:
        print(f"\n📦 Installation de {len(missing)} package(s) manquant(s)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + missing,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✅ Dépendances installées avec succès!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Échec de l'installation des dépendances")
            print(f"   Installez manuellement: pip install {' '.join(missing)}")
            return False

    print("✅ Toutes les dépendances sont installées")
    return True


def check_project_structure():
    """Vérifie que la structure du projet est correcte."""
    print("\n🔍 Vérification de la structure du projet...")

    required_paths = [
        Path("apps/dash_app.py"),
        Path("src/threadx/ui/callbacks.py"),
        Path("src/threadx/ui/layout.py"),
        Path("src/threadx/bridge/async_coordinator.py"),
    ]

    all_exist = True
    for path in required_paths:
        if path.exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} - MANQUANT")
            all_exist = False

    if not all_exist:
        print("❌ Structure de projet incomplète")
        return False

    print("✅ Structure du projet valide")
    return True


def launch_dash():
    """Lance l'interface Dash."""
    print("\n🚀 Lancement de l'interface Dash...")
    print("=" * 60)
    print("  URL:   http://127.0.0.1:8050")
    print("  Port:  8050")
    print("  Theme: Bootstrap DARKLY")
    print("=" * 60)
    print("\n💡 Appuyez sur Ctrl+C pour arrêter le serveur\n")

    try:
        # Lance le serveur Dash
        subprocess.run(
            [sys.executable, "apps/dash_app.py"],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du serveur Dash...")
        print("✅ Interface fermée proprement")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors du lancement: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ Fichier apps/dash_app.py introuvable")
        return False

    return True


def show_cli_help():
    """Affiche l'aide CLI."""
    print("\n" + "=" * 60)
    print("💡 AUTRES COMMANDES UTILES")
    print("=" * 60)
    print("\n📋 CLI Interface:")
    print("   python -m threadx.cli --help")
    print("\n🔧 Commandes disponibles:")
    print("   python -m threadx.cli data --help      # Gestion des données")
    print("   python -m threadx.cli indicators --help # Indicateurs")
    print("   python -m threadx.cli backtest --help   # Backtesting")
    print("   python -m threadx.cli optimize --help   # Optimisation")
    print("   python -m threadx.cli version           # Version")
    print()


def main():
    """Point d'entrée principal."""
    print_banner()

    # Vérifications préalables
    if not check_python_version():
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    if not check_project_structure():
        sys.exit(1)

    # Lancement de l'interface
    print("\n✅ Toutes les vérifications passées!")

    try:
        if launch_dash():
            show_cli_help()
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
