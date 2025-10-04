#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradXPro Core Manager - Module Unifié
=====================================

Module tout-en-un qui incorpore toute la logique TradXPro :
- Gestion des téléchargements crypto
- Sélection des tokens (top 100 marketcap/volume)
- Chargement et traitement des données OHLCV
- Calcul et cache des indicateurs techniques
- Gestion des fichiers JSON/Parquet
- API simplifiée pour intégration

Utilisation :
    from tradxpro_core_manager import TradXProManager
    
    manager = TradXProManager()
    
    # Récupérer les 100 meilleurs tokens
    top_tokens = manager.get_top_100_tokens()
    
    # Télécharger les données
    manager.download_crypto_data(["BTCUSDC", "ETHUSDC"])
    
    # Charger avec indicateurs
    df = manager.get_trading_data("BTCUSDC", "1h", indicators=["rsi", "bollinger"])

Auteur: TradXPro Team
Date: 2 octobre 2025
Version: 1.0
"""

import os
import sys
import json
import time
import logging
import platform
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Configuration des chemins TradXPro
IS_WINDOWS = platform.system() == "Windows"

class TradXProPaths:
    """Gestionnaire centralisé des chemins TradXPro"""
    
    def __init__(self, root_path: Optional[str] = None):
        if root_path is None:
            self.root = Path(r"D:\TradXPro") if IS_WINDOWS else Path("/home/user/TradXPro")
        else:
            self.root = Path(root_path)
        
        # Dossiers principaux
        self.json_root = self.root / "crypto_data_json"
        self.parquet_root = self.root / "crypto_data_parquet"
        self.indicators_db = self.root / "indicators_db"
        self.scripts_dir = self.root / "scripts" / "mise_a_jour_dataframe"
        
        # Fichiers de configuration
        self.tokens_json = self.scripts_dir / "resultats_choix_des_100tokens.json"
        self.log_file = self.scripts_dir / "unified_data_historique.log"
        
        # Création des dossiers si nécessaire
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crée les dossiers nécessaires s'ils n'existent pas"""
        for path in [self.json_root, self.parquet_root, self.indicators_db, self.scripts_dir]:
            path.mkdir(parents=True, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class TradXProManager:
    """
    Gestionnaire principal TradXPro - API unifiée pour toutes les fonctionnalités
    """
    
    def __init__(self, root_path: Optional[str] = None):
        """
        Initialise le gestionnaire TradXPro
        
        Args:
            root_path: Chemin racine personnalisé (optionnel)
        """
        self.paths = TradXProPaths(root_path)
        self.logger = logger
        
        # Configuration par défaut
        self.history_days = 365
        self.binance_limit = 1000
        self.intervals = ["3m", "5m", "15m", "30m", "1h"]
        self.max_workers = max(4, (os.cpu_count() or 8) // 2)
        
        logger.info(f"TradXPro Manager initialisé - Racine: {self.paths.root}")
    
    # =========================================================
    #  SECTION 1: Gestion des tokens (Top 100)
    # =========================================================
    
    def get_top_100_marketcap_coingecko(self) -> List[Dict]:
        """
        Récupère les 100 cryptos avec la plus grosse capitalisation via CoinGecko
        
        Returns:
            Liste des tokens avec marketcap, nom, symbol, rang
        """
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1
        }
        
        try:
            logger.info("Récupération top 100 marketcap CoinGecko...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            result = []
            for entry in data:
                result.append({
                    "symbol": entry["symbol"].upper(),
                    "name": entry["name"],
                    "market_cap": entry.get("market_cap", 0),
                    "market_cap_rank": entry.get("market_cap_rank", 999),
                    "volume": entry.get("total_volume", 0)
                })
            
            logger.info(f"✅ {len(result)} tokens récupérés via CoinGecko")
            return result
            
        except Exception as e:
            logger.error(f"Erreur CoinGecko API: {e}")
            return []
    
    def get_top_100_volume_binance(self) -> List[Dict]:
        """
        Récupère les 100 cryptos USDC avec le plus gros volume 24h via Binance
        
        Returns:
            Liste des tokens USDC avec volume 24h
        """
        url = "https://api.binance.com/api/v3/ticker/24hr"
        
        try:
            logger.info("Récupération top 100 volume USDC Binance...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Filtrer uniquement les paires USDC
            usdc_pairs = []
            for entry in data:
                if entry["symbol"].endswith("USDC"):
                    base_asset = entry["symbol"].replace("USDC", "")
                    usdc_pairs.append({
                        "symbol": base_asset,
                        "volume": float(entry["quoteVolume"]) if entry["quoteVolume"] else 0,
                        "price_change": float(entry["priceChangePercent"]) if entry["priceChangePercent"] else 0
                    })
            
            # Trier par volume décroissant et prendre les 100 premiers
            usdc_pairs.sort(key=lambda x: x["volume"], reverse=True)
            result = usdc_pairs[:100]
            
            logger.info(f"✅ {len(result)} tokens USDC récupérés via Binance")
            return result
            
        except Exception as e:
            logger.error(f"Erreur Binance API: {e}")
            return []
    
    def _ensure_category_representation(self, tokens: List[Dict]) -> List[Dict]:
        """
        Garantit qu'au moins les 3 meilleures cryptos de chaque catégorie importante sont incluses
        
        Args:
            tokens: Liste des tokens triés par score
            
        Returns:
            Liste ajustée avec représentation garantie par catégorie
        """
        # Définition des catégories importantes avec leurs tokens représentatifs
        essential_categories = {
            "layer1_blockchain": ["BTC", "ETH", "ADA", "SOL", "AVAX", "DOT", "NEAR", "ALGO"],
            "defi_protocols": ["UNI", "AAVE", "COMP", "MKR", "SUSHI", "CRV", "1INCH", "YFI"],
            "layer2_scaling": ["MATIC", "ARB", "OP", "IMX", "LRC", "MINA"],
            "smart_contracts": ["ETH", "ADA", "SOL", "AVAX", "DOT", "ALGO", "NEAR", "ATOM"],
            "meme_coins": ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK"],
            "exchange_tokens": ["BNB", "CRO", "FTT", "HT", "KCS", "OKB"],
            "stablecoins": ["USDT", "USDC", "BUSD", "DAI", "FRAX", "TUSD"],
            "ai_gaming": ["FET", "AGIX", "OCEAN", "AXS", "SAND", "MANA", "ENJ"],
            "privacy_coins": ["XMR", "ZEC", "DASH", "SCRT"],
            "infrastructure": ["LINK", "GRT", "FIL", "AR", "STORJ", "SIA"]
        }
        
        logger.info("🔍 Vérification de la représentation par catégorie...")
        
        # Créer un index des tokens actuels
        current_symbols = {token["symbol"] for token in tokens}
        guaranteed_tokens = []
        
        # Pour chaque catégorie, garantir au moins 3 tokens du top marketcap
        for category, category_tokens in essential_categories.items():
            category_count = 0
            category_found = []
            
            # Vérifier les tokens déjà présents dans cette catégorie
            for token in tokens:
                if token["symbol"] in category_tokens:
                    category_found.append(token)
                    category_count += 1
            
            # Si moins de 3 tokens de cette catégorie, essayer d'en ajouter
            if category_count < 3:
                missing_count = 3 - category_count
                logger.debug(f"Catégorie {category}: {category_count} tokens présents, besoin de {missing_count} supplémentaires")
                
                # Chercher les tokens manquants dans les données originales
                for symbol in category_tokens:
                    if symbol not in current_symbols and missing_count > 0:
                        # Créer un token de base avec score élevé pour garantir l'inclusion
                        guaranteed_token = {
                            "symbol": symbol,
                            "name": symbol,
                            "market_cap": 0,
                            "market_cap_rank": 999,
                            "volume": 0,
                            "price_change": 0,
                            "source": "category_guarantee",
                            "category": category,
                            "score": 150  # Score élevé pour garantir l'inclusion
                        }
                        guaranteed_tokens.append(guaranteed_token)
                        current_symbols.add(symbol)
                        missing_count -= 1
                        logger.debug(f"Token {symbol} ajouté pour garantir la catégorie {category}")
        
        # Fusionner les tokens garantis avec la liste originale
        if guaranteed_tokens:
            combined_tokens = tokens + guaranteed_tokens
            # Retrier par score
            combined_tokens.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"✅ {len(guaranteed_tokens)} tokens ajoutés pour garantir la diversité des catégories")
            return combined_tokens[:100]  # Toujours retourner 100 tokens max
        
        return tokens

    def merge_and_select_top_100(self, marketcap_list: List[Dict], volume_list: List[Dict]) -> List[Dict]:
        """
        Fusionne les listes marketcap et volume pour sélectionner les 100 meilleurs tokens
        avec garantie de représentation par catégorie
        
        Args:
            marketcap_list: Liste des tokens par marketcap
            volume_list: Liste des tokens par volume
            
        Returns:
            Liste fusionnée des 100 meilleurs tokens avec représentation garantie
        """
        logger.info("Fusion des listes marketcap et volume avec garantie de diversité...")
        
        # Index par symbole
        marketcap_dict = {token["symbol"]: token for token in marketcap_list}
        volume_dict = {token["symbol"]: token for token in volume_list}
        
        # Fusion des données
        merged_tokens = {}
        all_symbols = set(marketcap_dict.keys()) | set(volume_dict.keys())
        
        for symbol in all_symbols:
            mc_data = marketcap_dict.get(symbol, {})
            vol_data = volume_dict.get(symbol, {})
            
            merged_tokens[symbol] = {
                "symbol": symbol,
                "name": mc_data.get("name", symbol),
                "market_cap": mc_data.get("market_cap", 0),
                "market_cap_rank": mc_data.get("market_cap_rank", 999),
                "volume": vol_data.get("volume", 0),
                "price_change": vol_data.get("price_change", 0),
                "source": "both" if (symbol in marketcap_dict and symbol in volume_dict) else 
                         ("marketcap" if symbol in marketcap_dict else "volume")
            }
        
        # Scoring composite pour sélectionner les meilleurs
        scored_tokens = []
        for token in merged_tokens.values():
            # Score basé sur marketcap (inversé car rang 1 = meilleur) et volume
            mc_score = max(0, 101 - token["market_cap_rank"]) if token["market_cap_rank"] < 999 else 0
            vol_score = min(100, token["volume"] / 1_000_000)  # Normalisation volume
            
            # Bonus si présent dans les deux listes
            bonus = 20 if token["source"] == "both" else 0
            
            total_score = mc_score + vol_score + bonus
            token["score"] = total_score
            scored_tokens.append(token)
        
        # Trier par score décroissant
        scored_tokens.sort(key=lambda x: x["score"], reverse=True)
        
        # Appliquer la garantie de représentation par catégorie
        diversified_tokens = self._ensure_category_representation(scored_tokens)
        
        # Prendre les 100 premiers après diversification
        top_100 = diversified_tokens[:100]
        
        # Statistiques finales
        avg_score = np.mean([t['score'] for t in top_100])
        category_stats = {}
        for token in top_100:
            source = token.get("source", "unknown")
            category_stats[source] = category_stats.get(source, 0) + 1
        
        logger.info(f"✅ Top 100 tokens sélectionnés avec diversité garantie:")
        logger.info(f"   Score moyen: {avg_score:.1f}")
        logger.info(f"   Répartition: {category_stats}")
        
        return top_100
    
    def analyze_token_diversity(self, tokens: List[Dict]) -> Dict[str, Any]:
        """
        Analyse la diversité des tokens sélectionnés par catégorie
        
        Args:
            tokens: Liste des tokens à analyser
            
        Returns:
            Dictionnaire avec statistiques de diversité
        """
        # Définition des catégories (même que dans _ensure_category_representation)
        categories = {
            "layer1_blockchain": ["BTC", "ETH", "ADA", "SOL", "AVAX", "DOT", "NEAR", "ALGO"],
            "defi_protocols": ["UNI", "AAVE", "COMP", "MKR", "SUSHI", "CRV", "1INCH", "YFI"],
            "layer2_scaling": ["MATIC", "ARB", "OP", "IMX", "LRC", "MINA"],
            "smart_contracts": ["ETH", "ADA", "SOL", "AVAX", "DOT", "ALGO", "NEAR", "ATOM"],
            "meme_coins": ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK"],
            "exchange_tokens": ["BNB", "CRO", "FTT", "HT", "KCS", "OKB"],
            "stablecoins": ["USDT", "USDC", "BUSD", "DAI", "FRAX", "TUSD"],
            "ai_gaming": ["FET", "AGIX", "OCEAN", "AXS", "SAND", "MANA", "ENJ"],
            "privacy_coins": ["XMR", "ZEC", "DASH", "SCRT"],
            "infrastructure": ["LINK", "GRT", "FIL", "AR", "STORJ", "SIA"]
        }
        
        diversity_stats = {}
        token_symbols = {token["symbol"] for token in tokens}
        
        for category, category_tokens in categories.items():
            found_tokens = [symbol for symbol in category_tokens if symbol in token_symbols]
            diversity_stats[category] = {
                "count": len(found_tokens),
                "tokens": found_tokens,
                "coverage": len(found_tokens) / len(category_tokens) * 100
            }
        
        # Statistiques globales
        total_categorized = sum(len(stats["tokens"]) for stats in diversity_stats.values())
        diversity_stats["global"] = {
            "total_tokens": len(tokens),
            "categorized_tokens": total_categorized,
            "uncategorized_tokens": len(tokens) - total_categorized,
            "diversity_score": len([cat for cat, stats in diversity_stats.items() 
                                  if cat != "global" and stats["count"] >= 3]) / len(categories) * 100
        }
        
        return diversity_stats
    
    def print_diversity_report(self, tokens: List[Dict]):
        """
        Affiche un rapport détaillé de la diversité des tokens
        
        Args:
            tokens: Liste des tokens à analyser
        """
        diversity_stats = self.analyze_token_diversity(tokens)
        
        print("\n📊 RAPPORT DE DIVERSITÉ DES TOKENS")
        print("=" * 50)
        
        # Statistiques globales
        global_stats = diversity_stats["global"]
        print(f"Total de tokens: {global_stats['total_tokens']}")
        print(f"Tokens catégorisés: {global_stats['categorized_tokens']}")
        print(f"Score de diversité: {global_stats['diversity_score']:.1f}%")
        print()
        
        # Détail par catégorie
        print("Représentation par catégorie:")
        print("-" * 30)
        
        for category, stats in diversity_stats.items():
            if category == "global":
                continue
                
            status = "✅" if stats["count"] >= 3 else ("⚠️" if stats["count"] >= 1 else "❌")
            category_name = category.replace("_", " ").title()
            
            print(f"{status} {category_name:<18} {stats['count']:2d}/10 ({stats['coverage']:4.1f}%)")
            if stats["tokens"]:
                tokens_str = ", ".join(stats["tokens"][:5])
                if len(stats["tokens"]) > 5:
                    tokens_str += f" (+{len(stats['tokens'])-5} autres)"
                print(f"    {tokens_str}")
        
        print()
    
    def get_top_100_tokens(self, save_to_file: bool = True) -> List[Dict]:
        """
        API principale : récupère et fusionne les top 100 tokens
        
        Args:
            save_to_file: Sauvegarder le résultat dans resultats_choix_des_100tokens.json
            
        Returns:
            Liste des 100 meilleurs tokens
        """
        logger.info("🚀 Récupération des top 100 tokens...")
        
        # Récupération des données depuis les APIs
        marketcap_tokens = self.get_top_100_marketcap_coingecko()
        volume_tokens = self.get_top_100_volume_binance()
        
        if not marketcap_tokens and not volume_tokens:
            logger.error("❌ Impossible de récupérer les données des APIs")
            return []
        
        # Fusion et sélection avec garantie de diversité
        top_100 = self.merge_and_select_top_100(marketcap_tokens, volume_tokens)
        
        # Analyse de la diversité finale
        diversity_stats = self.analyze_token_diversity(top_100)
        logger.info(f"📊 Analyse de diversité:")
        logger.info(f"   Score de diversité: {diversity_stats['global']['diversity_score']:.1f}%")
        logger.info(f"   Tokens catégorisés: {diversity_stats['global']['categorized_tokens']}/100")
        
        # Afficher les catégories bien représentées
        well_represented = [cat for cat, stats in diversity_stats.items() 
                          if cat != "global" and stats["count"] >= 3]
        logger.info(f"   Catégories bien représentées (≥3): {len(well_represented)}/10")
        
        # Sauvegarde optionnelle
        if save_to_file and top_100:
            try:
                with open(self.paths.tokens_json, 'w', encoding='utf-8') as f:
                    json.dump(top_100, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Top 100 sauvegardé: {self.paths.tokens_json}")
            except Exception as e:
                logger.error(f"Erreur sauvegarde: {e}")
        
        return top_100
    
    def load_saved_tokens(self) -> List[Dict]:
        """
        Charge les tokens sauvegardés depuis le fichier JSON
        
        Returns:
            Liste des tokens ou liste vide si erreur
        """
        try:
            if self.paths.tokens_json.exists():
                with open(self.paths.tokens_json, 'r', encoding='utf-8') as f:
                    tokens = json.load(f)
                logger.info(f"✅ {len(tokens)} tokens chargés depuis {self.paths.tokens_json}")
                return tokens
            else:
                logger.warning(f"Fichier tokens non trouvé: {self.paths.tokens_json}")
                return []
        except Exception as e:
            logger.error(f"Erreur chargement tokens: {e}")
            return []
    
    # =========================================================
    #  SECTION 2: Téléchargement des données crypto
    # =========================================================
    
    def _interval_to_ms(self, interval: str) -> int:
        """Convertit un interval en millisecondes"""
        multipliers = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000
        }
        
        if interval[-1] in multipliers:
            return int(interval[:-1]) * multipliers[interval[-1]]
        return 60 * 1000  # Default 1 minute
    
    def _download_single_pair(self, symbol: str, interval: str, 
                             progress_callback: Optional[Callable] = None) -> bool:
        """
        Télécharge les données pour une paire symbol/interval via Binance API
        
        Args:
            symbol: Symbol (ex: BTCUSDC)
            interval: Interval (ex: 1h)
            progress_callback: Callback optionnel pour progression
            
        Returns:
            True si succès, False sinon
        """
        url = "https://api.binance.com/api/v3/klines"
        
        # Calcul période de téléchargement
        end_time = int(time.time() * 1000)
        start_time = end_time - (self.history_days * 24 * 60 * 60 * 1000)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": self.binance_limit
        }
        
        try:
            logger.debug(f"Téléchargement {symbol}_{interval}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning(f"Aucune donnée pour {symbol}_{interval}")
                return False
            
            # Conversion en DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Nettoyage et typage
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp').sort_index()
            
            # Suppression des doublons et NaN
            df = df[~df.index.duplicated(keep='last')]
            df = df.dropna()
            
            if len(df) == 0:
                logger.warning(f"DataFrame vide après nettoyage: {symbol}_{interval}")
                return False
            
            # Sauvegarde JSON
            json_file = self.paths.json_root / f"{symbol}_{interval}.json"
            
            # Conversion pour JSON (timestamp en ms)
            df_json = df.reset_index()
            df_json['timestamp'] = df_json['timestamp'].astype(int) // 1_000_000
            
            with open(json_file, 'w') as f:
                json.dump(df_json.to_dict('records'), f)
            
            # Sauvegarde Parquet (plus efficace)
            parquet_file = self.paths.parquet_root / f"{symbol}_{interval}.parquet"
            df.to_parquet(parquet_file, compression='zstd')
            
            logger.debug(f"✅ {symbol}_{interval}: {len(df)} lignes sauvegardées")
            
            if progress_callback:
                progress_callback(symbol, interval, len(df))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement {symbol}_{interval}: {e}")
            return False
    
    def download_crypto_data(self, symbols: List[str], intervals: Optional[List[str]] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Télécharge les données crypto pour plusieurs symboles/intervals
        
        Args:
            symbols: Liste des symboles (ex: ["BTCUSDC", "ETHUSDC"])
            intervals: Liste des intervals (par défaut: ["3m", "5m", "15m", "30m", "1h"])
            progress_callback: Callback optionnel(symbol, interval, nb_rows)
            
        Returns:
            Dictionnaire avec statistiques de téléchargement
        """
        if intervals is None:
            intervals = self.intervals
        
        logger.info(f"🔄 Téléchargement de {len(symbols)} symboles × {len(intervals)} intervals...")
        
        # Préparation des tâches
        tasks = []
        for symbol in symbols:
            for interval in intervals:
                tasks.append((symbol, interval))
        
        # Téléchargement parallèle
        results = {"success": 0, "errors": 0, "details": []}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._download_single_pair, symbol, interval, progress_callback): (symbol, interval)
                for symbol, interval in tasks
            }
            
            for future in as_completed(future_to_task):
                symbol, interval = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        results["success"] += 1
                        results["details"].append(f"✅ {symbol}_{interval}")
                    else:
                        results["errors"] += 1
                        results["details"].append(f"❌ {symbol}_{interval}")
                except Exception as e:
                    results["errors"] += 1
                    results["details"].append(f"❌ {symbol}_{interval}: {e}")
        
        logger.info(f"✅ Téléchargement terminé: {results['success']} succès, {results['errors']} erreurs")
        return results
    
    def download_top_100_data(self, intervals: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Télécharge les données pour tous les top 100 tokens
        
        Args:
            intervals: Liste des intervals (optionnel)
            
        Returns:
            Statistiques de téléchargement
        """
        # Chargement des tokens
        tokens = self.load_saved_tokens()
        if not tokens:
            logger.info("Aucun token sauvegardé, récupération des top 100...")
            tokens = self.get_top_100_tokens()
        
        if not tokens:
            logger.error("❌ Impossible de récupérer les tokens")
            return {"success": 0, "errors": 1, "details": ["Pas de tokens disponibles"]}
        
        # Conversion en symboles USDC
        symbols = [token["symbol"] + "USDC" for token in tokens]
        
        logger.info(f"🚀 Téléchargement des données pour {len(symbols)} tokens...")
        return self.download_crypto_data(symbols, intervals)
    
    # =========================================================
    #  SECTION 3: Chargement et traitement des données
    # =========================================================
    
    def load_ohlcv_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Charge les données OHLCV avec priorité Parquet → JSON
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            interval: Interval (ex: 1h)
            
        Returns:
            DataFrame OHLCV avec DatetimeIndex UTC ou None
        """
        # Priorité 1: Parquet
        parquet_file = self.paths.parquet_root / f"{symbol}_{interval}.parquet"
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                logger.debug(f"Chargé depuis Parquet: {symbol}_{interval} ({len(df)} lignes)")
                return df
            except Exception as e:
                logger.warning(f"Erreur lecture Parquet {parquet_file}: {e}")
        
        # Priorité 2: JSON
        json_file = self.paths.json_root / f"{symbol}_{interval}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                # Vérification des colonnes
                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Colonnes manquantes dans {json_file}")
                    return None
                
                # Conversion types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp').sort_index()
                
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                # Création automatique du Parquet pour optimiser les futurs accès
                try:
                    df.to_parquet(parquet_file, compression="zstd")
                    logger.debug(f"Parquet créé: {parquet_file}")
                except Exception as e:
                    logger.warning(f"Impossible de créer {parquet_file}: {e}")
                
                logger.debug(f"Chargé depuis JSON: {symbol}_{interval} ({len(df)} lignes)")
                return df
                
            except Exception as e:
                logger.error(f"Erreur lecture JSON {json_file}: {e}")
        
        logger.warning(f"Aucune donnée trouvée pour {symbol}_{interval}")
        return None
    
    # =========================================================
    #  SECTION 4: Calcul des indicateurs techniques
    # =========================================================
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calcule les bandes de Bollinger"""
        close = df['close']
        ma = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()
        
        return {
            'bb_middle': ma,
            'bb_upper': ma + (std_dev * std),
            'bb_lower': ma - (std_dev * std),
            'bb_width': (2 * std_dev * std) / ma
        }
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule l'ATR (Average True Range)"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period, min_periods=period).mean()
    
    def calculate_ema(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calcule l'EMA (Exponential Moving Average)"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calcule le MACD"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def add_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """
        Ajoute plusieurs indicateurs à un DataFrame OHLCV
        
        Args:
            df: DataFrame OHLCV
            indicators: Liste des indicateurs ('rsi', 'bollinger', 'atr', 'ema', 'macd')
            
        Returns:
            DataFrame avec indicateurs ajoutés
        """
        result = df.copy()
        
        for indicator in indicators:
            try:
                if indicator == 'rsi':
                    result['rsi'] = self.calculate_rsi(df)
                elif indicator == 'bollinger':
                    bb_data = self.calculate_bollinger_bands(df)
                    for key, series in bb_data.items():
                        result[key] = series
                elif indicator == 'atr':
                    result['atr'] = self.calculate_atr(df)
                elif indicator == 'ema':
                    result['ema'] = self.calculate_ema(df)
                elif indicator == 'macd':
                    macd_data = self.calculate_macd(df)
                    for key, series in macd_data.items():
                        result[key] = series
                else:
                    logger.warning(f"Indicateur non supporté: {indicator}")
                    
            except Exception as e:
                logger.error(f"Erreur calcul {indicator}: {e}")
        
        # Suppression des lignes avec NaN
        result = result.dropna()
        
        return result
    
    # =========================================================
    #  SECTION 5: API principale unifiée
    # =========================================================
    
    def get_trading_data(self, symbol: str, interval: str, 
                        indicators: Optional[List[str]] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        API principale : charge les données OHLCV + indicateurs
        
        Args:
            symbol: Symbole crypto (ex: BTCUSDC)
            interval: Interval (ex: 1h, 5m)
            indicators: Liste des indicateurs à calculer (optionnel)
            start_date: Date de début (format YYYY-MM-DD, optionnel)
            end_date: Date de fin (format YYYY-MM-DD, optionnel)
            
        Returns:
            DataFrame complet avec OHLCV + indicateurs ou None
            
        Example:
            >>> manager = TradXProManager()
            >>> df = manager.get_trading_data("BTCUSDC", "1h", 
            ...                              indicators=["rsi", "bollinger", "atr"])
            >>> print(f"DataFrame: {len(df)} lignes, {len(df.columns)} colonnes")
        """
        # Chargement des données de base
        df = self.load_ohlcv_data(symbol, interval)
        
        if df is None:
            logger.error(f"Impossible de charger {symbol}_{interval}")
            return None
        
        # Filtrage temporel si demandé
        if start_date or end_date:
            df = df.loc[start_date:end_date]
            logger.info(f"Filtrage temporel: {len(df)} lignes après filtrage")
        
        # Ajout des indicateurs si demandés
        if indicators:
            df = self.add_indicators(df, indicators)
            logger.info(f"Indicateurs ajoutés: {indicators}")
        
        logger.info(f"✅ {symbol}_{interval}: {len(df)} lignes, {len(df.columns)} colonnes")
        return df
    
    def get_multiple_trading_data(self, pairs: List[Tuple[str, str]], 
                                 indicators: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Charge les données pour plusieurs paires en parallèle
        
        Args:
            pairs: Liste de tuples (symbol, interval)
            indicators: Liste des indicateurs à calculer
            
        Returns:
            Dictionnaire {symbol_interval: DataFrame}
        """
        logger.info(f"Chargement de {len(pairs)} paires en parallèle...")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pair = {
                executor.submit(self.get_trading_data, symbol, interval, indicators): (symbol, interval)
                for symbol, interval in pairs
            }
            
            for future in as_completed(future_to_pair):
                symbol, interval = future_to_pair[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[f"{symbol}_{interval}"] = df
                        logger.debug(f"✅ {symbol}_{interval} chargé")
                    else:
                        logger.warning(f"❌ Échec chargement {symbol}_{interval}")
                except Exception as e:
                    logger.error(f"Erreur {symbol}_{interval}: {e}")
        
        logger.info(f"✅ {len(results)}/{len(pairs)} paires chargées avec succès")
        return results
    
    # =========================================================
    #  SECTION 6: Utilitaires et statistiques
    # =========================================================
    
    def get_available_data(self) -> Dict[str, List[str]]:
        """
        Scanne les données disponibles sur disque
        
        Returns:
            Dictionnaire {symbol: [list_of_intervals]}
        """
        available = {}
        
        # Scan des fichiers Parquet
        for file_path in self.paths.parquet_root.glob("*.parquet"):
            filename = file_path.stem
            if "_" in filename:
                symbol, interval = filename.rsplit("_", 1)
                if symbol not in available:
                    available[symbol] = []
                available[symbol].append(interval)
        
        # Compléter avec les fichiers JSON s'ils ne sont pas en Parquet
        for file_path in self.paths.json_root.glob("*.json"):
            filename = file_path.stem
            if "_" in filename and not filename.startswith("resultats"):
                symbol, interval = filename.rsplit("_", 1)
                if symbol not in available:
                    available[symbol] = []
                if interval not in available[symbol]:
                    available[symbol].append(interval)
        
        # Tri des intervals
        for symbol in available:
            available[symbol] = sorted(available[symbol])
        
        return available
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les données disponibles
        
        Returns:
            Dictionnaire avec statistiques
        """
        available_data = self.get_available_data()
        
        total_files = sum(len(intervals) for intervals in available_data.values())
        
        # Taille des dossiers
        json_size = sum(f.stat().st_size for f in self.paths.json_root.glob("*.json")) / 1024 / 1024
        parquet_size = sum(f.stat().st_size for f in self.paths.parquet_root.glob("*.parquet")) / 1024 / 1024
        
        stats = {
            "symbols_count": len(available_data),
            "total_files": total_files,
            "intervals": list(set(interval for intervals in available_data.values() for interval in intervals)),
            "json_size_mb": round(json_size, 1),
            "parquet_size_mb": round(parquet_size, 1),
            "total_size_mb": round(json_size + parquet_size, 1),
            "top_symbols": sorted(available_data.keys())[:10] if available_data else [],
            "sample_data": dict(list(available_data.items())[:5]) if available_data else {}
        }
        
        return stats
    
    def cleanup_old_files(self, days_old: int = 7) -> Dict[str, int]:
        """
        Nettoie les fichiers anciens
        
        Args:
            days_old: Supprimer les fichiers plus anciens que X jours
            
        Returns:
            Statistiques de nettoyage
        """
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        stats = {"json_removed": 0, "parquet_removed": 0}
        
        # Nettoyage JSON
        for file_path in self.paths.json_root.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                stats["json_removed"] += 1
        
        # Nettoyage Parquet
        for file_path in self.paths.parquet_root.glob("*.parquet"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                stats["parquet_removed"] += 1
        
        logger.info(f"Nettoyage terminé: {stats['json_removed']} JSON, {stats['parquet_removed']} Parquet supprimés")
        return stats

# =========================================================
#  SECTION 7: Interface en ligne de commande
# =========================================================

def main():
    """Interface en ligne de commande pour tester le gestionnaire"""
    print("🚀 TradXPro Core Manager - Test Interface")
    print("=" * 50)
    
    manager = TradXProManager()
    
    while True:
        print("\nOptions disponibles:")
        print("1. 📊 Récupérer top 100 tokens")
        print("2. 📥 Télécharger données crypto")
        print("3. 📈 Charger données avec indicateurs")
        print("4. 📋 Statistiques des données")
        print("5. 🧹 Nettoyer anciens fichiers")
        print("0. ❌ Quitter")
        
        choice = input("\nVotre choix: ").strip()
        
        try:
            if choice == "1":
                print("\n🔄 Récupération des top 100 tokens...")
                tokens = manager.get_top_100_tokens()
                print(f"✅ {len(tokens)} tokens récupérés")
                for i, token in enumerate(tokens[:10], 1):
                    print(f"  {i:2d}. {token['symbol']:10s} - {token['name'][:30]:<30s} (Score: {token['score']:.1f})")
                
            elif choice == "2":
                symbols = input("Symboles (séparés par des virgules, ex: BTCUSDC,ETHUSDC): ").strip()
                if symbols:
                    symbol_list = [s.strip().upper() for s in symbols.split(",")]
                    print(f"\n🔄 Téléchargement de {len(symbol_list)} symboles...")
                    results = manager.download_crypto_data(symbol_list)
                    print(f"✅ Résultats: {results['success']} succès, {results['errors']} erreurs")
                
            elif choice == "3":
                symbol = input("Symbole (ex: BTCUSDC): ").strip().upper()
                interval = input("Interval (ex: 1h): ").strip()
                indicators_str = input("Indicateurs (ex: rsi,bollinger,atr): ").strip()
                
                indicators = [i.strip() for i in indicators_str.split(",") if i.strip()] if indicators_str else None
                
                print(f"\n🔄 Chargement {symbol}_{interval} avec indicateurs {indicators}...")
                df = manager.get_trading_data(symbol, interval, indicators)
                
                if df is not None:
                    print(f"✅ DataFrame chargé: {len(df)} lignes, {len(df.columns)} colonnes")
                    print(f"Colonnes: {list(df.columns)}")
                    print(f"Période: {df.index[0]} à {df.index[-1]}")
                    print("\nAperçu des dernières valeurs:")
                    print(df.tail(3))
                else:
                    print("❌ Impossible de charger les données")
                
            elif choice == "4":
                print("\n📊 Calcul des statistiques...")
                stats = manager.get_data_statistics()
                print(f"✅ Statistiques des données:")
                print(f"  Symboles: {stats['symbols_count']}")
                print(f"  Fichiers total: {stats['total_files']}")
                print(f"  Intervals disponibles: {stats['intervals']}")
                print(f"  Taille JSON: {stats['json_size_mb']} MB")
                print(f"  Taille Parquet: {stats['parquet_size_mb']} MB")
                print(f"  Taille totale: {stats['total_size_mb']} MB")
                
            elif choice == "5":
                days = input("Supprimer fichiers plus anciens que X jours (défaut: 7): ").strip()
                days = int(days) if days.isdigit() else 7
                print(f"\n🧹 Nettoyage des fichiers > {days} jours...")
                stats = manager.cleanup_old_files(days)
                print(f"✅ {stats['json_removed'] + stats['parquet_removed']} fichiers supprimés")
                
            elif choice == "0":
                print("👋 Au revoir!")
                break
                
            else:
                print("❌ Choix invalide")
                
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            logger.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()