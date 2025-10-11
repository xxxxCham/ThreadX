#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Lanceur Principal - Token Diversity Manager
==============================================

Lanceur principal pour utiliser facilement le Token Diversity Manager.

Usage:
    python launch.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

def main():
    """Lanceur principal avec menu interactif"""

    print("🚀 TRADXPRO TOKEN DIVERSITY MANAGER")
    print("=" * 50)
    print("Module avec diversité garantie des tokens crypto")
    print()

    while True:
        print("📋 OPTIONS DISPONIBLES:")
        print("1. 🧪 Test simple du module")
        print("2. 🔒 Test de diversité des tokens")
        print("3. 💡 Exemple de démarrage rapide")
        print("4. 📊 Exemple d'intégration complète")
        print("5. ⚙️ Setup et configuration")
        print("6. 📚 Voir la documentation")
        print("0. ❌ Quitter")
        print()

        try:
            choice = input("Votre choix (0-6): ").strip()

            if choice == "1":
                print("\n🧪 Lancement du test simple...")
                from tradxpro_core_manager import TradXProManager

                manager = TradXProManager()
                print("✅ TradXProManager initialisé avec succès !")
                print(f"📁 Racine: {manager.paths.root}")
                print(f"⚙️ Configuration: {manager.history_days} jours, {manager.max_workers} workers")

            elif choice == "2":
                print("\n🔒 Test de diversité des tokens...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "tests/test_diversite_simple.py"])

            elif choice == "3":
                print("\n💡 Lancement de l'exemple rapide...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "examples/quick_start_tradxpro.py"])

            elif choice == "4":
                print("\n📊 Lancement de l'exemple complet...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "examples/exemple_integration_tradxpro.py"])

            elif choice == "5":
                print("\n⚙️ Lancement du setup...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "setup_module.py"])

            elif choice == "6":
                print("\n📚 DOCUMENTATION DISPONIBLE:")
                print("📄 README.md - Guide principal")
                print("📄 docs/README_CORE_MANAGER.md - Documentation complète")
                print("📄 docs/DIVERSITE_GARANTIE.md - Détails sur la diversité")
                print("📄 INDEX_TOKEN_DIVERSITY_MANAGER.md - Index des fichiers")

            elif choice == "0":
                print("👋 Au revoir !")
                break

            else:
                print("❌ Choix invalide, veuillez choisir entre 0 et 6")

        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

        print("\n" + "-" * 50 + "\n")

def demo_rapide():
    """Démonstration rapide des fonctionnalités"""

    print("🎯 DÉMONSTRATION RAPIDE")
    print("=" * 30)

    try:
        # Import et initialisation
        from tradxpro_core_manager import TradXProManager
        manager = TradXProManager()

        print("✅ Module initialisé")

        # Test des fonctionnalités de diversité
        test_tokens = [
            {"symbol": "BTC", "name": "Bitcoin", "score": 100},
            {"symbol": "ETH", "name": "Ethereum", "score": 95},
            {"symbol": "ADA", "name": "Cardano", "score": 85},
            {"symbol": "UNI", "name": "Uniswap", "score": 80},
            {"symbol": "AAVE", "name": "Aave", "score": 75},
            {"symbol": "MATIC", "name": "Polygon", "score": 70},
        ]

        print(f"📊 Test avec {len(test_tokens)} tokens...")
        diversity_stats = manager.analyze_token_diversity(test_tokens)

        print(f"✅ Score de diversité: {diversity_stats['global']['diversity_score']:.1f}%")
        print(f"✅ Tokens catégorisés: {diversity_stats['global']['categorized_tokens']}/{len(test_tokens)}")

        print("\n🎉 Le module Token Diversity Manager fonctionne parfaitement !")

    except Exception as e:
        print(f"❌ Erreur lors de la démo: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_rapide()
    else:
        main()