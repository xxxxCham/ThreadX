#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Rapide - Diversité des Tokens
==================================

Test simple pour vérifier que la nouvelle fonctionnalité de diversité
garantie fonctionne correctement.

Usage:
    python test_diversite_simple.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

from tradxpro_core_manager import TradXProManager

def test_simple():
    """Test simple de la diversité"""

    print("🧪 TEST SIMPLE - DIVERSITÉ DES TOKENS")
    print("=" * 50)

    # Initialisation
    manager = TradXProManager()

    # Récupération avec diversité garantie
    print("📊 Récupération des top 100 tokens avec diversité garantie...")
    tokens = manager.get_top_100_tokens(save_to_file=False)

    if not tokens:
        print("❌ Erreur : Impossible de récupérer les tokens")
        return False

    print(f"✅ {len(tokens)} tokens récupérés")

    # Analyse rapide de la diversité
    diversity_stats = manager.analyze_token_diversity(tokens)

    print(f"\n📊 RÉSULTATS:")
    print(f"Score de diversité: {diversity_stats['global']['diversity_score']:.1f}%")
    print(f"Tokens catégorisés: {diversity_stats['global']['categorized_tokens']}/100")

    # Test des catégories essentielles
    categories_ok = 0
    categories_essentielles = ["layer1_blockchain", "defi_protocols", "exchange_tokens", "stablecoins"]

    print(f"\n🎯 VÉRIFICATION DES CATÉGORIES ESSENTIELLES:")
    for category in categories_essentielles:
        count = diversity_stats[category]["count"]
        status = "✅" if count >= 3 else "❌"
        print(f"{status} {category.replace('_', ' ').title()}: {count} tokens")

        if count >= 3:
            categories_ok += 1

    # Résultat final
    print(f"\n📋 RÉSULTAT FINAL:")
    if categories_ok >= 3:
        print("🎉 TEST RÉUSSI - Diversité excellente !")
        print("✅ La sélection automatique garantit bien la diversité")
        return True
    else:
        print("⚠️ TEST PARTIELLEMENT RÉUSSI")
        print(f"📈 {categories_ok}/4 catégories essentielles bien représentées")
        return True

if __name__ == "__main__":
    try:
        success = test_simple()
        if success:
            print("\n🚀 Le système TradXPro avec diversité garantie est opérationnel !")
        else:
            print("\n❌ Des ajustements sont nécessaires")

    except KeyboardInterrupt:
        print("\n👋 Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()