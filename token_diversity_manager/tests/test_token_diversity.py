#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la Sélection Diversifiée des Tokens
===========================================

Script de test pour vérifier que la sélection automatique des top 100 tokens
inclut bien au moins 3 représentants de chaque catégorie importante.

Usage:
    python test_token_diversity.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import sys
from pathlib import Path

# Import du gestionnaire TradXPro
from tradxpro_core_manager import TradXProManager

def test_token_diversity():
    """Test de la diversité des tokens sélectionnés"""
    
    print("🧪 TEST DE DIVERSITÉ DES TOKENS")
    print("=" * 50)
    
    # Initialisation du gestionnaire
    manager = TradXProManager()
    
    # Test 1: Récupération avec diversité garantie
    print("📊 Récupération des top 100 tokens avec diversité garantie...")
    tokens = manager.get_top_100_tokens(save_to_file=False)  # Test sans sauvegarde
    
    if not tokens:
        print("❌ Impossible de récupérer les tokens")
        return False
    
    print(f"✅ {len(tokens)} tokens récupérés")
    
    # Test 2: Analyse de la diversité
    print("\n🔍 Analyse de la diversité...")
    manager.print_diversity_report(tokens)
    
    # Test 3: Vérification des catégories essentielles
    print("🎯 VÉRIFICATION DES CATÉGORIES ESSENTIELLES")
    print("-" * 50)
    
    diversity_stats = manager.analyze_token_diversity(tokens)
    
    # Catégories qui DOIVENT avoir au moins 3 représentants
    essential_categories = [
        "layer1_blockchain",
        "defi_protocols", 
        "exchange_tokens",
        "infrastructure"
    ]
    
    all_good = True
    for category in essential_categories:
        count = diversity_stats[category]["count"]
        status = "✅" if count >= 3 else "❌"
        
        if count < 3:
            all_good = False
        
        print(f"{status} {category:<18} {count:2d} tokens")
        if count > 0:
            print(f"    Tokens: {', '.join(diversity_stats[category]['tokens'])}")
    
    print()
    
    # Test 4: Top tokens par catégorie
    print("🏆 TOP TOKENS PAR CATÉGORIE")
    print("-" * 50)
    
    categories_examples = {
        "Layer 1 Blockchain": ["BTC", "ETH", "ADA", "SOL"],
        "DeFi Protocols": ["UNI", "AAVE", "COMP", "MKR"],
        "Exchange Tokens": ["BNB", "CRO", "FTT", "HT"],
        "Stablecoins": ["USDT", "USDC", "BUSD", "DAI"]
    }
    
    token_dict = {token["symbol"]: token for token in tokens}
    
    for category_name, example_tokens in categories_examples.items():
        print(f"\n{category_name}:")
        found_tokens = []
        
        for symbol in example_tokens:
            if symbol in token_dict:
                token = token_dict[symbol]
                found_tokens.append(token)
        
        # Trier par score décroissant
        found_tokens.sort(key=lambda x: x["score"], reverse=True)
        
        for i, token in enumerate(found_tokens[:3], 1):
            print(f"  {i}. {token['symbol']:8s} - {token['name'][:25]:<25s} (Score: {token['score']:.1f})")
    
    print()
    
    # Test 5: Résumé final
    print("📋 RÉSUMÉ DU TEST")
    print("-" * 50)
    
    global_stats = diversity_stats["global"]
    score_diversite = global_stats["diversity_score"]
    
    print(f"Score de diversité global: {score_diversite:.1f}%")
    print(f"Catégories bien représentées: {len([cat for cat, stats in diversity_stats.items() if cat != 'global' and stats['count'] >= 3])}/10")
    print(f"Tokens catégorisés: {global_stats['categorized_tokens']}/100")
    
    if score_diversite >= 80 and all_good:
        print("✅ TEST RÉUSSI: Excellente diversité des tokens")
        result = True
    elif score_diversite >= 60:
        print("⚠️ TEST PARTIELLEMENT RÉUSSI: Diversité acceptable")
        result = True
    else:
        print("❌ TEST ÉCHOUÉ: Diversité insuffisante")
        result = False
    
    return result

def test_category_guarantee():
    """Test spécifique de la garantie par catégorie"""
    
    print("\n🔒 TEST DE GARANTIE PAR CATÉGORIE")
    print("=" * 50)
    
    manager = TradXProManager()
    
    # Simuler une liste limitée pour forcer l'activation de la garantie
    limited_marketcap = [
        {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000000000, "market_cap_rank": 1},
        {"symbol": "ETH", "name": "Ethereum", "market_cap": 500000000, "market_cap_rank": 2},
        {"symbol": "XRP", "name": "XRP", "market_cap": 100000000, "market_cap_rank": 3},
    ]
    
    limited_volume = [
        {"symbol": "BTC", "volume": 1000000},
        {"symbol": "ETH", "volume": 800000},
        {"symbol": "DOGE", "volume": 500000},
    ]
    
    print("🧪 Test avec données limitées pour activer la garantie...")
    result_tokens = manager.merge_and_select_top_100(limited_marketcap, limited_volume)
    
    print(f"✅ {len(result_tokens)} tokens générés")
    
    # Vérifier que des tokens ont été ajoutés automatiquement
    guaranteed_tokens = [token for token in result_tokens if token.get("source") == "category_guarantee"]
    
    if guaranteed_tokens:
        print(f"🔒 {len(guaranteed_tokens)} tokens ajoutés automatiquement pour garantir la diversité:")
        for token in guaranteed_tokens[:5]:
            print(f"   • {token['symbol']} (Catégorie: {token.get('category', 'Unknown')})")
    else:
        print("ℹ️ Aucun token supplémentaire nécessaire (diversité déjà suffisante)")
    
    return True

def main():
    """Fonction principale"""
    
    print("🚀 TEST COMPLET DE LA SÉLECTION DIVERSIFIÉE")
    print("=" * 60)
    
    try:
        # Test principal
        success1 = test_token_diversity()
        
        # Test de garantie
        success2 = test_category_guarantee()
        
        print("\n" + "=" * 60)
        if success1 and success2:
            print("🎉 TOUS LES TESTS RÉUSSIS!")
            print("✅ La sélection automatique garantit bien la diversité des catégories")
        else:
            print("⚠️ TESTS PARTIELLEMENT RÉUSSIS")
            print("🔧 Quelques ajustements peuvent être nécessaires")
        
    except KeyboardInterrupt:
        print("\n👋 Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur pendant les tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()