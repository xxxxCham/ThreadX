#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThreadX TechinTerror - Validation automatique
==============================================

Script de validation pour vérifier que l'interface TechinTerror 
se lance correctement avec tous les composants.
"""

import sys
import time
from pathlib import Path

# Ajoute /src au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_imports():
    """Teste que tous les imports principaux fonctionnent."""
    print("🔍 Test des imports...")
    
    try:
        from threadx.ui.app_techintetor import ThreadXApp, THEME
        print("✅ ThreadXApp importé")
    except Exception as e:
        print(f"❌ Erreur ThreadXApp: {e}")
        return False
    
    try:
        from threadx.indicators.bank import IndicatorBank
        print("✅ IndicatorBank importé")
    except Exception as e:
        print(f"⚠️ IndicatorBank indisponible: {e}")
    
    try:
        from threadx.ui.downloads import create_downloads_page
        print("✅ Downloads page importée")
    except Exception as e:
        print(f"⚠️ Downloads page indisponible: {e}")
    
    try:
        from threadx.ui.sweep import create_sweep_page
        print("✅ Sweep page importée")
    except Exception as e:
        print(f"⚠️ Sweep page indisponible: {e}")
    
    return True

def test_theme():
    """Teste que le thème Nord est bien défini."""
    print("\n🎨 Test du thème Nord...")
    
    try:
        from threadx.ui.app_techintetor import THEME
        
        required_colors = ['background', 'panel', 'text', 'info', 'positive', 'danger']
        for color in required_colors:
            if color not in THEME:
                print(f"❌ Couleur manquante: {color}")
                return False
            print(f"✅ {color}: {THEME[color]}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur thème: {e}")
        return False

def test_app_creation():
    """Teste la création de l'application sans mainloop."""
    print("\n🏗️ Test création application...")
    
    try:
        from threadx.ui.app_techintetor import ThreadXApp
        import tkinter as tk
        
        # Teste la création (sans mainloop)
        app = ThreadXApp()
        
        # Vérifie les attributs principaux
        assert hasattr(app, 'nb'), "Notebook manquant"
        assert hasattr(app, 'current_data'), "current_data manquant"  
        assert hasattr(app, 'last_equity'), "last_equity manquant"
        assert hasattr(app, 'executor'), "ThreadPoolExecutor manquant"
        
        # Vérifie que le notebook a des onglets
        tabs = app.nb.tabs()
        print(f"✅ Application créée avec {len(tabs)} onglets")
        
        # Vérifie les noms d'onglets
        expected_tabs = ['🏠 Home', '📁 Data', '🔧 Indicators', '🎯 Optimization', 
                        '🎯 Sweep', '🚀 Backtest', '📊 Performance', '📥 Downloads', '📝 Logs']
        
        for i, expected in enumerate(expected_tabs):
            if i < len(tabs):
                actual = app.nb.tab(tabs[i], "text")
                if expected == actual:
                    print(f"✅ Onglet {i+1}: {actual}")
                else:
                    print(f"⚠️ Onglet {i+1}: attendu '{expected}', trouvé '{actual}'")
            else:
                print(f"❌ Onglet manquant: {expected}")
        
        # Ferme proprement
        app.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Erreur création app: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de validation."""
    print("🚀 ThreadX TechinTerror - Validation automatique")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Thème Nord", test_theme), 
        ("Création App", test_app_creation)
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {name}: RÉUSSI")
            else:
                print(f"\n❌ {name}: ÉCHOUÉ")
        except Exception as e:
            print(f"\n💥 {name}: ERREUR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Résultats: {passed}/{len(tests)} tests réussis")
    
    if passed == len(tests):
        print("🎉 Tous les tests sont passés !")
        print("\n💡 Pour lancer l'interface:")
        print("   python run_tkinter.py")
        return True
    else:
        print("⚠️ Certains tests ont échoué.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)