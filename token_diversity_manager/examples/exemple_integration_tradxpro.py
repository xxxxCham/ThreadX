#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple d'Incorporation du TradXPro Core Manager
===============================================

Exemple concret montrant comment incorporer toute la logique TradXPro 
(téléchargements, tokens, indicateurs) dans un autre programme.

Usage:
    python exemple_integration_tradxpro.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Import du gestionnaire TradXPro
from tradxpro_core_manager import TradXProManager

def exemple_basic_usage():
    """Exemple d'usage basique du gestionnaire"""
    print("=== EXEMPLE BASIQUE ===")
    
    # Initialisation du gestionnaire
    manager = TradXProManager()
    
    # 1. Récupérer les top 100 tokens avec diversité garantie
    print("📊 Récupération des top 100 tokens avec diversité garantie...")
    tokens = manager.get_top_100_tokens()
    
    if tokens:
        print(f"✅ {len(tokens)} tokens récupérés")
        print("Top 5 tokens:")
        for i, token in enumerate(tokens[:5], 1):
            print(f"  {i}. {token['symbol']:8s} - {token['name'][:25]:<25s} (Score: {token['score']:.1f})")
        
        # Afficher le rapport de diversité
        print("\n📊 Rapport de diversité:")
        manager.print_diversity_report(tokens)
    
    # 2. Charger des données existantes avec indicateurs
    print("\n📈 Chargement de données avec indicateurs...")
    df = manager.get_trading_data(
        symbol="BTCUSDC", 
        interval="1h", 
        indicators=["rsi", "bollinger", "atr"]
    )
    
    if df is not None:
        print(f"✅ DataFrame chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"Colonnes: {list(df.columns)}")
        print("\nAperçu des dernières valeurs:")
        print(df.tail(3))
    else:
        print("❌ Données non disponibles")

def exemple_trading_strategy():
    """Exemple d'une stratégie de trading simple utilisant TradXPro"""
    print("\n=== EXEMPLE STRATÉGIE TRADING ===")
    
    manager = TradXProManager()
    
    # Symboles à analyser
    symbols = ["BTCUSDC", "ETHUSDC", "ADAUSDC"]
    interval = "1h"
    indicators = ["rsi", "bollinger", "atr", "macd"]
    
    print(f"🔍 Analyse de {len(symbols)} symboles avec indicateurs...")
    
    signals = []
    
    for symbol in symbols:
        print(f"\n📊 Analyse {symbol}...")
        
        # Chargement des données avec indicateurs
        df = manager.get_trading_data(symbol, interval, indicators)
        
        if df is None or len(df) < 50:
            print(f"  ❌ Données insuffisantes pour {symbol}")
            continue
        
        # Calcul des signaux simples
        latest = df.iloc[-1]
        
        # Signal RSI
        rsi_signal = "OVERSOLD" if latest['rsi'] < 30 else ("OVERBOUGHT" if latest['rsi'] > 70 else "NEUTRAL")
        
        # Signal Bollinger
        bb_signal = "BELOW_LOWER" if latest['close'] < latest['bb_lower'] else \
                   ("ABOVE_UPPER" if latest['close'] > latest['bb_upper'] else "MIDDLE")
        
        # Signal MACD
        macd_signal = "BULLISH" if latest['macd'] > latest['macd_signal'] else "BEARISH"
        
        signal = {
            "symbol": symbol,
            "price": latest['close'],
            "rsi": latest['rsi'],
            "rsi_signal": rsi_signal,
            "bb_signal": bb_signal,
            "macd_signal": macd_signal
        }
        
        signals.append(signal)
        
        print(f"  💰 Prix: ${latest['close']:.2f}")
        print(f"  📈 RSI: {latest['rsi']:.1f} ({rsi_signal})")
        print(f"  🔵 Bollinger: {bb_signal}")
        print(f"  📊 MACD: {macd_signal}")
    
    # Résumé des signaux
    print("\n=== RÉSUMÉ DES SIGNAUX ===")
    for signal in signals:
        print(f"{signal['symbol']:8s} | RSI: {signal['rsi']:5.1f} ({signal['rsi_signal']:10s}) | "
              f"BB: {signal['bb_signal']:12s} | MACD: {signal['macd_signal']}")

def exemple_portfolio_analysis():
    """Exemple d'analyse de portfolio utilisant les top tokens"""
    print("\n=== EXEMPLE ANALYSE PORTFOLIO ===")
    
    manager = TradXProManager()
    
    # Récupérer et sélectionner les meilleurs tokens
    tokens = manager.load_saved_tokens()
    if not tokens:
        print("📊 Récupération des tokens...")
        tokens = manager.get_top_100_tokens()
    
    if not tokens:
        print("❌ Impossible de récupérer les tokens")
        return
    
    # Sélectionner le top 10 pour l'analyse
    top_10_symbols = [token["symbol"] + "USDC" for token in tokens[:10]]
    
    print(f"🔍 Analyse du top 10 portfolio...")
    
    # Chargement des données en parallèle
    pairs = [(symbol, "1h") for symbol in top_10_symbols]
    results = manager.get_multiple_trading_data(pairs, indicators=["rsi", "atr"])
    
    portfolio_data = []
    
    for symbol_interval, df in results.items():
        if df is not None and len(df) > 0:
            symbol = symbol_interval.replace("_1h", "")
            latest = df.iloc[-1]
            
            # Calculs de volatilité et momentum
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * 100  # Volatilité en %
            momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100  # Momentum 20 périodes
            
            portfolio_data.append({
                "symbol": symbol,
                "price": latest['close'],
                "rsi": latest['rsi'],
                "atr": latest['atr'],
                "volatility": volatility,
                "momentum_20d": momentum
            })
    
    # Tri par momentum
    portfolio_data.sort(key=lambda x: x['momentum_20d'], reverse=True)
    
    print(f"\n📊 Portfolio Analysis ({len(portfolio_data)} tokens):")
    print("Symbol    | Price     | RSI  | Volatility | Momentum 20d")
    print("-" * 60)
    
    for data in portfolio_data:
        print(f"{data['symbol']:8s} | ${data['price']:8.2f} | {data['rsi']:4.1f} | "
              f"{data['volatility']:8.2f}% | {data['momentum_20d']:8.1f}%")

def exemple_data_management():
    """Exemple de gestion des données"""
    print("\n=== EXEMPLE GESTION DONNÉES ===")
    
    manager = TradXProManager()
    
    # Statistiques des données disponibles
    print("📊 Statistiques des données...")
    stats = manager.get_data_statistics()
    
    print(f"✅ Données disponibles:")
    print(f"  Symboles: {stats['symbols_count']}")
    print(f"  Fichiers total: {stats['total_files']}")
    print(f"  Intervals: {stats['intervals']}")
    print(f"  Taille totale: {stats['total_size_mb']} MB")
    
    # Données disponibles
    available = manager.get_available_data()
    if available:
        print(f"\nTop 5 symboles disponibles:")
        for i, (symbol, intervals) in enumerate(list(available.items())[:5], 1):
            print(f"  {i}. {symbol}: {intervals}")
    
    # Exemple de téléchargement de nouvelles données
    print(f"\n📥 Exemple de téléchargement...")
    print("(Simulation - pas de téléchargement réel)")
    
    # Test avec quelques symboles
    test_symbols = ["BTCUSDC", "ETHUSDC"]
    print(f"Téléchargement simulé pour: {test_symbols}")
    
    # Dans un vrai cas, vous appelleriez:
    # results = manager.download_crypto_data(test_symbols, ["1h", "4h"])

class MonProgrammeAvecTradXPro:
    """
    Exemple d'intégration du TradXProManager dans votre propre classe
    """
    
    def __init__(self, custom_root_path: Optional[str] = None):
        """Initialisation avec TradXPro intégré"""
        self.tradx = TradXProManager(custom_root_path)
        self.watchlist = []
        
        print("🚀 Mon Programme avec TradXPro initialisé")
    
    def setup_watchlist(self, top_n=20):
        """Configure une watchlist basée sur les top tokens"""
        print(f"📋 Configuration watchlist (top {top_n})...")
        
        tokens = self.tradx.load_saved_tokens()
        if not tokens:
            tokens = self.tradx.get_top_100_tokens()
        
        self.watchlist = [token["symbol"] + "USDC" for token in tokens[:top_n]]
        print(f"✅ Watchlist configurée: {len(self.watchlist)} symboles")
        
        return self.watchlist
    
    def analyze_watchlist(self):
        """Analyse tous les symboles de la watchlist"""
        if not self.watchlist:
            self.setup_watchlist()
        
        print(f"🔍 Analyse de la watchlist ({len(self.watchlist)} symboles)...")
        
        # Chargement en parallèle
        pairs = [(symbol, "1h") for symbol in self.watchlist]
        results = self.tradx.get_multiple_trading_data(pairs, indicators=["rsi", "bollinger"])
        
        analysis_results = []
        
        for symbol_interval, df in results.items():
            if df is not None and len(df) > 0:
                symbol = symbol_interval.replace("_1h", "")
                latest = df.iloc[-1]
                
                # Votre logique d'analyse personnalisée ici
                score = self._calculate_custom_score(df, latest)
                
                analysis_results.append({
                    "symbol": symbol,
                    "score": score,
                    "price": latest['close'],
                    "rsi": latest['rsi']
                })
        
        # Tri par score
        analysis_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📊 Top 5 recommandations:")
        for i, result in enumerate(analysis_results[:5], 1):
            print(f"  {i}. {result['symbol']:8s} - Score: {result['score']:.2f} "
                  f"(Prix: ${result['price']:.2f}, RSI: {result['rsi']:.1f})")
        
        return analysis_results
    
    def _calculate_custom_score(self, df, latest_row):
        """Calcule un score personnalisé (exemple)"""
        # Exemple de logique de scoring
        rsi_score = max(0, min(100, 100 - abs(latest_row['rsi'] - 50))) / 100
        
        # Position relative dans les bandes de Bollinger
        bb_position = (latest_row['close'] - latest_row['bb_lower']) / \
                     (latest_row['bb_upper'] - latest_row['bb_lower'])
        bb_score = 1 - abs(bb_position - 0.5) * 2  # Score max au milieu
        
        # Score composite
        return (rsi_score * 0.6) + (bb_score * 0.4)
    
    def run_custom_strategy(self):
        """Lance votre stratégie personnalisée"""
        print("🎯 Lancement de la stratégie personnalisée...")
        
        # Configuration
        self.setup_watchlist(10)
        
        # Analyse
        results = self.analyze_watchlist()
        
        # Vos actions personnalisées ici
        print("✅ Stratégie exécutée avec succès!")
        
        return results

def main():
    """Fonction principale avec tous les exemples"""
    print("🚀 EXEMPLES D'INTÉGRATION TRADXPRO CORE MANAGER")
    print("=" * 60)
    
    try:
        # Exemple 1: Usage basique
        exemple_basic_usage()
        
        # Exemple 2: Stratégie de trading
        exemple_trading_strategy()
        
        # Exemple 3: Analyse de portfolio
        exemple_portfolio_analysis()
        
        # Exemple 4: Gestion des données
        exemple_data_management()
        
        # Exemple 5: Intégration dans une classe personnalisée
        print("\n=== EXEMPLE CLASSE PERSONNALISÉE ===")
        mon_programme = MonProgrammeAvecTradXPro()
        mon_programme.run_custom_strategy()
        
        print("\n" + "=" * 60)
        print("✅ TOUS LES EXEMPLES TERMINÉS AVEC SUCCÈS!")
        print("💡 Vous pouvez maintenant adapter ces exemples à vos besoins")
        
    except KeyboardInterrupt:
        print("\n👋 Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()