#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Simple du Module Token Diversity Manager
==============================================

Test simple pour vérifier que le module fonctionne correctement.

Usage:
    python test_module.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import sys
from pathlib import Path

# Ajout du chemin du module
module_path = Path(__file__).parent
sys.path.append(str(module_path))

def test_module_import():
    """Test d'importation du module"""

    print("📦 TEST D'IMPORTATION")
    print("-" * 30)

    try:
        # Test import principal
        from tradxpro_core_manager import TradXProManager
        print("✅ TradXProManager importé")

        # Test info module
        from __init__ import MODULE_INFO
        print("✅ Module complet importé")

        # Affichage info module
        print(f"📋 Version: {MODULE_INFO['version']}")

        return True

    except Exception as e:
        print(f"❌ Erreur importation: {e}")
        return False

def test_manager_init():
    """Test d'initialisation du gestionnaire"""

    print("\n🔧 TEST D'INITIALISATION")
    print("-" * 30)

    try:
        from tradxpro_core_manager import TradXProManager

        # Initialisation
        manager = TradXProManager()
        print("✅ TradXProManager initialisé")

        # Vérification des chemins
        print(f"📁 Racine: {manager.paths.root}")
        print(f"📄 JSON: {manager.paths.json_root}")
        print(f"⚡ Parquet: {manager.paths.parquet_root}")

        # Vérification configuration
        print(f"⚙️ Historique: {manager.history_days} jours")
        print(f"🔗 Workers: {manager.max_workers}")
        print(f"📊 Intervals: {manager.intervals}")

        return True

    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        return False

def test_diversity_features():
    """Test des fonctionnalités de diversité"""

    print("\n🔒 TEST FONCTIONNALITÉS DIVERSITÉ")
    print("-" * 30)

    try:
        from tradxpro_core_manager import TradXProManager

        manager = TradXProManager()

        # Test des méthodes de diversité
        print("🧪 Test analyze_token_diversity...")
        test_tokens = [
            {"symbol": "BTC", "name": "Bitcoin", "score": 100},
            {"symbol": "ETH", "name": "Ethereum", "score": 95},
            {"symbol": "UNI", "name": "Uniswap", "score": 80}
        ]

        diversity_stats = manager.analyze_token_diversity(test_tokens)
        print("✅ analyze_token_diversity fonctionne")

        # Test rapport
        print("🧪 Test print_diversity_report...")
        manager.print_diversity_report(test_tokens)
        print("✅ print_diversity_report fonctionne")

        return True

    except Exception as e:
        print(f"❌ Erreur test diversité: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test complet du module"""

    print("🧪 TEST COMPLET - TOKEN DIVERSITY MANAGER")
    print("=" * 50)

    tests = [
        ("Import", test_module_import),
        ("Initialisation", test_manager_init),
        ("Diversité", test_diversity_features)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}...")
        success = test_func()
        results.append((test_name, success))

    # Résultats finaux
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS DES TESTS")
    print("=" * 50)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n📈 Score: {passed}/{len(tests)} tests réussis")

    if passed == len(tests):
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("✅ Le module Token Diversity Manager est opérationnel")

        print(f"\n💡 UTILISATION:")
        print("from tradxpro_core_manager import TradXProManager")
        print("manager = TradXProManager()")
        print("tokens = manager.get_top_100_tokens()  # Avec diversité garantie !")

    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("🔧 Vérifiez la configuration du module")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()