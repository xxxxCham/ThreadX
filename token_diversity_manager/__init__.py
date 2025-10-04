#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradXPro Token Diversity Manager
===============================

Module spécialisé pour la gestion des tokens crypto avec diversité garantie.

Ce module fournit un gestionnaire complet qui :
- Récupère automatiquement les top 100 tokens crypto
- Garantit une diversité par catégorie (≥3 tokens par catégorie importante)
- Gère le téléchargement et le stockage des données historiques
- Calcule les indicateurs techniques
- Fournit une API unifiée pour l'intégration

Usage:
    from tradxpro_token_diversity_manager import TradXProManager
    
    manager = TradXProManager()
    tokens = manager.get_top_100_tokens()  # Avec diversité garantie !
    df = manager.get_trading_data("BTCUSDC", "1h", ["rsi", "bollinger"])

Auteur: TradXPro Team
Date: 2 octobre 2025
Version: 1.1 - Diversité Garantie
"""

from .tradxpro_core_manager import TradXProManager, TradXProPaths

__version__ = "1.1.0"
__author__ = "TradXPro Team"
__email__ = "support@tradxpro.com"

__all__ = [
    "TradXProManager",
    "TradXProPaths"
]

# Informations sur le module
MODULE_INFO = {
    "name": "TradXPro Token Diversity Manager",
    "version": __version__,
    "description": "Gestionnaire de tokens crypto avec diversité garantie",
    "features": [
        "🏆 Récupération top 100 tokens (CoinGecko + Binance)",
        "🔒 Diversité garantie (≥3 tokens par catégorie)",
        "📥 Téléchargement données historiques multi-threading",
        "📈 Indicateurs techniques (RSI, Bollinger, ATR, EMA, MACD)",
        "💾 Stockage optimisé (JSON + Parquet)",
        "📊 Analyse et rapport de diversité",
        "⚡ Chargement parallèle et cache automatique",
        "🛠️ API simple et unifiée"
    ],
    "categories": [
        "Layer 1 Blockchain", "DeFi Protocols", "Layer 2 Scaling",
        "Smart Contracts", "Meme Coins", "Exchange Tokens", 
        "Stablecoins", "AI Gaming", "Privacy Coins", "Infrastructure"
    ]
}

def get_module_info():
    """Retourne les informations du module"""
    return MODULE_INFO

def print_module_info():
    """Affiche les informations du module"""
    info = MODULE_INFO
    print(f"📦 {info['name']} v{info['version']}")
    print("=" * 50)
    print(f"📋 {info['description']}")
    print()
    print("🎯 Fonctionnalités:")
    for feature in info['features']:
        print(f"   {feature}")
    print()
    print(f"📊 Catégories couvertes ({len(info['categories'])}):")
    for i, category in enumerate(info['categories'], 1):
        print(f"   {i:2d}. {category}")

# Auto-configuration du logging si importé directement
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)