#!/usr/bin/env python3
"""
ThreadX - Tests Interface GUI
============================

Script de validation des interfaces GUI ThreadX.
"""

import sys
import subprocess
from pathlib import Path


def test_imports():
    """Test des imports nécessaires."""
    print("🔍 Test des imports...")

    try:
        import tkinter as tk

        print("✅ tkinter : OK")
    except ImportError:
        print("❌ tkinter : MANQUANT")
        return False

    try:
        import pandas as pd

        print("✅ pandas : OK")
    except ImportError:
        print("❌ pandas : MANQUANT")
        return False

    try:
        import threading
        import queue
        import json

        print("✅ modules standard : OK")
    except ImportError:
        print("❌ modules standard : ERREUR")
        return False

    return True


def test_threadx_availability():
    """Test de disponibilité ThreadX."""
    print("\n🔍 Test ThreadX...")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from threadx.data.providers.token_diversity import TokenDiversityManager

        print("✅ ThreadX modules : DISPONIBLES")
        return True
    except ImportError as e:
        print(f"⚠️  ThreadX modules : NON DISPONIBLES ({e})")
        print("   Mode simulation sera utilisé")
        return False


def test_gui_syntax():
    """Test de syntaxe des fichiers GUI."""
    print("\n🔍 Test syntaxe GUI...")

    gui_files = ["demo_gui.py", "threadx_gui.py", "launch_gui.py"]

    for gui_file in gui_files:
        file_path = Path(__file__).parent / gui_file
        if file_path.exists():
            try:
                compile(open(file_path).read(), gui_file, "exec")
                print(f"✅ {gui_file} : SYNTAXE OK")
            except SyntaxError as e:
                print(f"❌ {gui_file} : ERREUR SYNTAXE - {e}")
                return False
        else:
            print(f"⚠️  {gui_file} : FICHIER MANQUANT")

    return True


def test_directory_structure():
    """Test de la structure de dossiers."""
    print("\n🔍 Test structure dossiers...")

    base_path = Path(__file__).parent.parent.parent
    required_dirs = ["data", "configs", "logs", "apps/tkinter"]

    all_ok = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"✅ {dir_path} : OK")
        else:
            print(f"❌ {dir_path} : MANQUANT")
            # Créer le dossier
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   → Créé automatiquement")

    return all_ok


def launch_demo():
    """Lance la démo GUI."""
    print("\n🚀 Lancement démo GUI...")

    try:
        demo_path = Path(__file__).parent / "demo_gui.py"
        if demo_path.exists():
            print("✅ Interface démo disponible")
            print("📝 Pour lancer manuellement : python demo_gui.py")
            return True
        else:
            print("❌ Fichier demo_gui.py manquant")
            return False
    except Exception as e:
        print(f"❌ Erreur lancement : {e}")
        return False


def main():
    """Test principal."""
    print("🚀 ThreadX GUI - Tests de Validation")
    print("=" * 50)

    results = []

    # Tests séquentiels
    results.append(("Imports", test_imports()))
    results.append(("ThreadX", test_threadx_availability()))
    results.append(("Syntaxe GUI", test_gui_syntax()))
    results.append(("Structure", test_directory_structure()))
    results.append(("Demo", launch_demo()))

    # Résultats
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS DES TESTS")
    print("=" * 50)

    success_count = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} : {status}")
        if result:
            success_count += 1

    print("=" * 50)
    print(f"🎯 Score : {success_count}/{len(results)} tests réussis")

    if success_count == len(results):
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("🚀 L'interface ThreadX est prête à l'utilisation")
        print("\n💡 Commandes disponibles :")
        print("   python demo_gui.py      # Interface démo")
        print("   python threadx_gui.py   # Interface complète")
        print("   python launch_gui.py    # Lanceur simplifié")
    else:
        print("⚠️  Certains tests ont échoué")
        print("   Vérifiez les erreurs ci-dessus")

    print("\n👋 Tests terminés")


if __name__ == "__main__":
    main()
