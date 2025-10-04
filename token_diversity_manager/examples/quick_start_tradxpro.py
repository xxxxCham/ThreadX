#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradXPro - Démarrage Rapide
===========================

Script de démarrage rapide pour incorporer facilement toute la logique TradXPro
dans votre programme en 3 étapes simples.

Usage:
    python quick_start_tradxpro.py

Ce script vous montre comment :
1. Récupérer automatiquement les 100 meilleurs tokens crypto
2. Télécharger leurs données historiques
3. Les analyser avec des indicateurs techniques

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import sys
import time
from pathlib import Path

# Assurez-vous que le module TradXPro est dans le chemin
sys.path.append(str(Path(__file__).parent))

from tradxpro_core_manager import TradXProManager

def quick_start_demo():
    """Démonstration en 3 étapes simples"""
    
    print("🚀 TRADXPRO - DÉMARRAGE RAPIDE")
    print("=" * 50)
    print("Incorporez toute la logique TradXPro en 3 étapes simples !")
    print()
    
    # ========================================
    # ÉTAPE 1: Initialisation
    # ========================================
    print("📋 ÉTAPE 1: Initialisation du gestionnaire TradXPro")
    print("-" * 50)
    
    # Créer le gestionnaire - il gère tout automatiquement !
    manager = TradXProManager()
    
    print("✅ Gestionnaire TradXPro initialisé")
    print(f"   📁 Dossier racine: {manager.paths.root}")
    print(f"   💾 Données JSON: {manager.paths.json_root}")
    print(f"   ⚡ Données Parquet: {manager.paths.parquet_root}")
    print()
    
    # ========================================
    # ÉTAPE 2: Récupération des meilleurs tokens
    # ========================================
    print("📊 ÉTAPE 2: Récupération des 100 meilleurs tokens crypto")
    print("-" * 50)
    
    # Le gestionnaire récupère automatiquement les top 100 depuis CoinGecko + Binance
    tokens = manager.get_top_100_tokens(save_to_file=True)
    
    if tokens:
        print(f"✅ {len(tokens)} tokens récupérés et sauvegardés")
        print("🏆 Top 10 tokens par score composite:")
        
        for i, token in enumerate(tokens[:10], 1):
            print(f"   {i:2d}. {token['symbol']:8s} - {token['name'][:30]:<30s} "
                  f"(Score: {token['score']:.1f})")
    else:
        print("❌ Erreur lors de la récupération des tokens")
        return False
    
    print()
    
    # ========================================
    # ÉTAPE 3: Analyse avec indicateurs techniques
    # ========================================
    print("📈 ÉTAPE 3: Analyse avec indicateurs techniques")
    print("-" * 50)
    
    # Sélectionner quelques tokens pour la démo
    demo_symbols = [token["symbol"] + "USDC" for token in tokens[:5]]
    
    print(f"🔍 Analyse de {len(demo_symbols)} tokens avec indicateurs...")
    
    for symbol in demo_symbols:
        print(f"\n📊 Analyse {symbol}:")
        
        # Chargement des données avec indicateurs automatique
        df = manager.get_trading_data(
            symbol=symbol,
            interval="1h",
            indicators=["rsi", "bollinger", "atr", "macd"]
        )
        
        if df is not None and len(df) > 0:
            latest = df.iloc[-1]
            
            print(f"   💰 Prix actuel: ${latest['close']:.4f}")
            print(f"   📈 RSI (14): {latest['rsi']:.1f}")
            print(f"   🔵 Bollinger Upper: ${latest['bb_upper']:.4f}")
            print(f"   🔵 Bollinger Lower: ${latest['bb_lower']:.4f}")
            print(f"   ⚡ ATR (14): {latest['atr']:.6f}")
            print(f"   📊 MACD: {latest['macd']:.6f}")
            
            # Signal simple
            if latest['rsi'] < 30:
                print("   🟢 SIGNAL: RSI Oversold - Potentiel d'achat")
            elif latest['rsi'] > 70:
                print("   🔴 SIGNAL: RSI Overbought - Potentiel de vente")
            else:
                print("   🟡 SIGNAL: RSI Neutre")
                
        else:
            print("   ❌ Données non disponibles")
    
    print()
    print("=" * 50)
    print("✅ DÉMARRAGE RAPIDE TERMINÉ AVEC SUCCÈS!")
    print()
    
    return True

def integration_template():
    """Template pour intégrer TradXPro dans votre code"""
    
    print("💡 TEMPLATE D'INTÉGRATION POUR VOTRE CODE")
    print("=" * 50)
    
    template_code = '''
# ========================================
# INTÉGRATION TRADXPRO DANS VOTRE CODE
# ========================================

from tradxpro_core_manager import TradXProManager

class MonApplication:
    def __init__(self):
        # Initialiser TradXPro
        self.tradx = TradXProManager()
        
    def obtenir_meilleurs_tokens(self, nombre=100):
        """Récupère les N meilleurs tokens"""
        return self.tradx.get_top_100_tokens()[:nombre]
    
    def analyser_token(self, symbol, interval="1h"):
        """Analyse complète d'un token"""
        return self.tradx.get_trading_data(
            symbol=symbol,
            interval=interval,
            indicators=["rsi", "bollinger", "atr", "macd"]
        )
    
    def telecharger_donnees(self, symbols):
        """Télécharge les données pour une liste de tokens"""
        return self.tradx.download_crypto_data(symbols)
    
    def ma_strategie_personnalisee(self):
        """Votre stratégie personnalisée"""
        # 1. Récupérer les meilleurs tokens
        tokens = self.obtenir_meilleurs_tokens(20)
        
        # 2. Analyser chaque token
        for token in tokens:
            symbol = token["symbol"] + "USDC"
            df = self.analyser_token(symbol)
            
            if df is not None:
                # 3. Appliquer votre logique
                latest = df.iloc[-1]
                
                # Exemple de condition d'achat
                if (latest['rsi'] < 35 and 
                    latest['close'] < latest['bb_lower'] and
                    latest['macd'] > latest['macd_signal']):
                    print(f"🟢 SIGNAL ACHAT: {symbol}")
                
                # Vos autres conditions...

# Usage:
app = MonApplication()
app.ma_strategie_personnalisee()
'''
    
    print(template_code)
    
    # Sauvegarder le template
    template_file = Path(__file__).parent / "template_integration_tradxpro.py"
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(template_code)
    
    print(f"💾 Template sauvegardé: {template_file}")

def show_features():
    """Affiche toutes les fonctionnalités disponibles"""
    
    print("🎯 FONCTIONNALITÉS DISPONIBLES")
    print("=" * 50)
    
    features = [
        ("🏆 Top 100 Tokens", "Récupération automatique via CoinGecko + Binance"),
        ("📥 Téléchargement", "Données historiques OHLCV multi-timeframes"),
        ("💾 Stockage Optimisé", "JSON + Parquet avec compression"),
        ("📈 Indicateurs", "RSI, Bollinger, ATR, EMA, MACD et plus"),
        ("⚡ Performance", "Chargement parallèle et cache automatique"),
        ("🔄 Mise à jour", "Actualisation automatique des données"),
        ("📊 Analyse", "Outils d'analyse technique intégrés"),
        ("🛠️ API Simple", "Interface unifiée facile à utiliser"),
        ("📁 Gestion Fichiers", "Organisation automatique des données"),
        ("🚀 Extensible", "Facilement intégrable dans vos projets")
    ]
    
    for feature, description in features:
        print(f"{feature:<20} {description}")
    
    print()
    print("📚 MÉTHODES PRINCIPALES:")
    methods = [
        "manager.get_top_100_tokens()",
        "manager.download_crypto_data(symbols)",
        "manager.get_trading_data(symbol, interval, indicators)",
        "manager.get_multiple_trading_data(pairs)",
        "manager.get_data_statistics()",
        "manager.get_available_data()"
    ]
    
    for method in methods:
        print(f"   • {method}")

def main():
    """Fonction principale"""
    
    print("Choisissez une option:")
    print("1. 🚀 Démarrage rapide (démo complète)")
    print("2. 💡 Template d'intégration")
    print("3. 🎯 Voir toutes les fonctionnalités")
    print("4. ❌ Quitter")
    
    try:
        choice = input("\nVotre choix (1-4): ").strip()
        
        if choice == "1":
            success = quick_start_demo()
            if success:
                print("🎉 Vous pouvez maintenant utiliser TradXPro dans vos projets!")
                
        elif choice == "2":
            integration_template()
            
        elif choice == "3":
            show_features()
            
        elif choice == "4":
            print("👋 Au revoir!")
            
        else:
            print("❌ Choix invalide")
            
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")

if __name__ == "__main__":
    main()