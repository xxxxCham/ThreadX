#!/usr/bin/env python3
"""
ThreadX - Test Rapide Interface GUI
===================================

Test simple des interfaces GUI sans dépendances ThreadX.
"""


def test_basic_gui():
    """Test basique de l'interface."""
    print("🔍 Test interface GUI basique...")

    try:
        import tkinter as tk
        from tkinter import ttk

        # Test création fenêtre
        root = tk.Tk()
        root.title("Test ThreadX")
        root.geometry("400x300")

        # Test widgets
        label = ttk.Label(root, text="✅ ThreadX GUI Test")
        label.pack(pady=20)

        button = ttk.Button(root, text="OK", command=root.quit)
        button.pack(pady=10)

        # Fermeture automatique
        root.after(1000, root.quit)  # Ferme après 1 seconde
        root.mainloop()

        print("✅ Interface GUI : OK")
        return True

    except Exception as e:
        print(f"❌ Interface GUI : {e}")
        return False


def test_demo_import():
    """Test import de la démo."""
    print("🔍 Test import demo...")

    try:
        import sys
        from pathlib import Path

        # Ajouter le chemin local
        sys.path.insert(0, str(Path(__file__).parent))

        # Test import (sans exécution)
        import demo_gui

        print("✅ Import demo_gui : OK")

        # Vérifier la classe principale
        if hasattr(demo_gui, "ThreadXDemoGUI"):
            print("✅ Classe ThreadXDemoGUI : OK")
        else:
            print("❌ Classe ThreadXDemoGUI : MANQUANTE")
            return False

        return True

    except Exception as e:
        print(f"❌ Import demo : {e}")
        return False


def test_file_structure():
    """Test structure des fichiers."""
    print("🔍 Test structure fichiers...")

    from pathlib import Path

    files_to_check = ["demo_gui.py", "threadx_gui.py", "launch_gui.py", "README.md"]

    current_dir = Path(__file__).parent
    all_ok = True

    for filename in files_to_check:
        file_path = current_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"✅ {filename} : OK ({size_kb:.1f} KB)")
        else:
            print(f"❌ {filename} : MANQUANT")
            all_ok = False

    return all_ok


def main():
    """Test principal."""
    print("🚀 ThreadX - Test Rapide Interface GUI")
    print("=" * 45)

    tests = [
        ("GUI Basique", test_basic_gui),
        ("Import Demo", test_demo_import),
        ("Structure", test_file_structure),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))

    # Résultats
    print("\n" + "=" * 45)
    print("📊 RÉSULTATS")
    print("=" * 45)

    success = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:12} : {status}")
        if result:
            success += 1

    print("=" * 45)
    print(f"🎯 Score : {success}/{len(results)}")

    if success == len(results):
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("\n🚀 L'interface est fonctionnelle")
        print("📝 Commande pour démo : python demo_gui.py")
    else:
        print("⚠️  Quelques tests ont échoué")

    print("\n✨ Test terminé")


if __name__ == "__main__":
    main()
