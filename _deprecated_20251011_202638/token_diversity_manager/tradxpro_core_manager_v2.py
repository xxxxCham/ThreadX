#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradXPro Core Manager v2.0 - Refactorisé Sans Redondances
==========================================================

Module unifié REFACTORISÉ qui délègue au code de référence
unified_data_historique_with_indicators.py pour éliminer redondances.

CHANGEMENTS v2.0:
- ✅ Téléchargement: Délègue à fetch_klines (référence)
- ✅ Indicateurs: Utilise threadx.indicators.numpy
- ✅ Top 100 tokens: Délègue à merge_and_update_tokens
- ✅ Chemins: Utilise constantes globales de référence
- ✅ Conservation: Fonctionnalités uniques (_ensure_category_representation)

VALEURS AJOUTÉES CONSERVÉES:
- Garantie de diversité par catégorie (unique)
- Analyse statistique de diversité
- API simplifiée get_trading_data()

Utilisation:
    from tradxpro_core_manager import TradXProManager

    manager = TradXProManager()

    # Récupérer les 100 meilleurs tokens avec diversité garantie
    top_tokens = manager.get_top_100_tokens()

    # Télécharger avec retry automatique
    manager.download_crypto_data(["BTCUSDC", "ETHUSDC"])

    # Charger avec indicateurs
    df = manager.get_trading_data("BTCUSDC", "1h", indicators=["rsi", "macd"])

Auteur: TradXPro Team (Refactorisé ThreadX Core)
Date: 11 octobre 2025
Version: 2.0 (Sans Redondances)
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# =========================================================
#  Imports depuis ThreadX consolidé (DÉLÉGATION)
# =========================================================

# Ajouter racine ThreadX au path pour imports
THREADX_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(THREADX_ROOT))

try:
    # Nouveaux modules consolidés ThreadX
    from src.threadx.data.tokens import TokenManager, get_top100_tokens
    from src.threadx.data.loader import BinanceDataLoader
    from src.threadx.indicators.indicators_np import (
        rsi_np,
        boll_np,
        macd_np,
        atr_np,
        ema_np,
        vwap_np,
        obv_np,
    )

    # Legacy support - importer chemins depuis unified_data (temporaire)
    from unified_data_historique_with_indicators import (
        JSON_ROOT,
        PARQUET_ROOT,
        INDICATORS_DB_ROOT,
        OUTPUT_DIR,
        JSON_PATH,
        parquet_path,
        json_path_symbol,
        indicator_path,
        HISTORY_DAYS,
        INTERVALS,
    )

except ImportError as e:
    raise ImportError(
        f"Impossible d'importer modules ThreadX: {e}\n"
        "Vérifiez que src/threadx/data/tokens.py et loader.py existent."
    ) from e

except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer depuis le code de référence: {e}")
    print(
        "Assurez-vous que unified_data_historique_with_indicators.py existe à la racine."
    )
    sys.exit(1)

# Retirer le chemin temporaire
sys.path.pop(0)

# =========================================================
#  Configuration Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================
#  TradXProManager v2.0 - Délégation Complète
# =========================================================


class TradXProManager:
    """
    Gestionnaire principal TradXPro v2.0 - SANS REDONDANCES

    Délègue toute la logique métier au code de référence et conserve
    UNIQUEMENT les fonctionnalités à valeur ajoutée unique.
    """

    def __init__(self):
        """Initialise le gestionnaire (chemins déjà gérés par référence)."""
        self.logger = logger
        self.logger.info("TradXPro Manager v2.0 initialisé (mode délégation)")

    # =========================================================
    #  SECTION 1: Gestion des tokens (VALEUR AJOUTÉE)
    # =========================================================

    def get_top_100_tokens(self, save_to_file: bool = True) -> List[Dict]:
        """
        Récupère les 100 meilleurs tokens avec GARANTIE DE DIVERSITÉ.

        Utilise le nouveau TokenManager consolidé,
        puis applique la logique unique de diversité garantie.

        Args:
            save_to_file: Sauvegarder dans JSON_PATH

        Returns:
            Liste des 100 tokens avec diversité garantie par catégorie
        """
        self.logger.info("🚀 Récupération Top 100 tokens avec diversité garantie")

        try:
            # Utiliser TokenManager consolidé
            token_mgr = TokenManager(cache_path=Path(JSON_PATH))

            # Récupération données
            marketcap_list = token_mgr.get_top100_marketcap()
            volume_list = token_mgr.get_top100_volume()

            # Fusion
            base_tokens = token_mgr.merge_and_rank_tokens(
                marketcap_list, volume_list, save_cache=save_to_file
            )

            # VALEUR AJOUTÉE: Garantie de diversité
            diversified_tokens = self._ensure_category_representation(base_tokens)

            # Limiter à 100
            top_100 = diversified_tokens[:100]

            # Statistiques
            avg_score = np.mean([t.get("score", 0) for t in top_100])
            self.logger.info(f"✅ Top 100 sélectionnés - Score moyen: {avg_score:.1f}")

            return top_100

        except Exception as e:
            self.logger.error(f"❌ Erreur get_top_100_tokens: {e}")
            return []

    def _ensure_category_representation(self, tokens: List[Dict]) -> List[Dict]:
        """
        FONCTIONNALITÉ UNIQUE - Garantit représentation par catégorie.

        S'assure qu'au moins 3 tokens de chaque catégorie importante
        sont inclus dans la sélection finale.

        Args:
            tokens: Liste des tokens triés par score

        Returns:
            Liste ajustée avec représentation garantie
        """
        # Catégories essentielles avec tokens représentatifs
        essential_categories = {
            "layer1_blockchain": [
                "BTC",
                "ETH",
                "ADA",
                "SOL",
                "AVAX",
                "DOT",
                "NEAR",
                "ALGO",
            ],
            "defi_protocols": [
                "UNI",
                "AAVE",
                "COMP",
                "MKR",
                "SUSHI",
                "CRV",
                "1INCH",
                "YFI",
            ],
            "layer2_scaling": ["MATIC", "ARB", "OP", "IMX", "LRC", "MINA"],
            "smart_contracts": [
                "ETH",
                "ADA",
                "SOL",
                "AVAX",
                "DOT",
                "ALGO",
                "NEAR",
                "ATOM",
            ],
            "meme_coins": ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK"],
            "exchange_tokens": ["BNB", "CRO", "FTT", "HT", "KCS", "OKB"],
            "stablecoins": ["USDT", "USDC", "BUSD", "DAI", "FRAX", "TUSD"],
            "ai_gaming": ["FET", "AGIX", "OCEAN", "AXS", "SAND", "MANA", "ENJ"],
            "privacy_coins": ["XMR", "ZEC", "DASH", "SCRT"],
            "infrastructure": ["LINK", "GRT", "FIL", "AR", "STORJ", "SIA"],
        }

        self.logger.info("🔍 Vérification diversité par catégorie...")

        # Index des tokens actuels
        current_symbols = {token["symbol"].upper() for token in tokens}
        guaranteed_tokens = []

        # Pour chaque catégorie, garantir au moins 3 tokens
        for category, category_tokens in essential_categories.items():
            # Trouver tokens de cette catégorie déjà présents
            present = [t for t in tokens if t["symbol"].upper() in category_tokens]

            if len(present) < 3:
                # Manque de représentation, ajouter les meilleurs tokens de la catégorie
                missing_count = 3 - len(present)

                # Chercher dans marketcap original
                marketcap_candidates = get_top100_marketcap_coingecko()
                category_candidates = [
                    t
                    for t in marketcap_candidates
                    if t["symbol"].upper() in category_tokens
                    and t["symbol"].upper() not in current_symbols
                ]

                # Ajouter les meilleurs manquants
                to_add = category_candidates[:missing_count]
                guaranteed_tokens.extend(to_add)

                self.logger.info(
                    f"  📌 {category}: Ajout de {len(to_add)} tokens "
                    f"({[t['symbol'] for t in to_add]})"
                )

        # Fusionner tokens garantis avec liste originale
        if guaranteed_tokens:
            # Marquer les tokens garantis
            for token in guaranteed_tokens:
                token["guaranteed"] = True
                token["score"] = token.get("score", 0) + 1000  # Boost score

            # Fusionner sans doublons
            all_tokens = tokens + guaranteed_tokens
            seen = set()
            deduplicated = []
            for token in all_tokens:
                sym = token["symbol"].upper()
                if sym not in seen:
                    deduplicated.append(token)
                    seen.add(sym)

            # Re-trier par score
            deduplicated.sort(key=lambda x: x.get("score", 0), reverse=True)

            self.logger.info(
                f"✅ Diversité garantie: {len(guaranteed_tokens)} tokens ajoutés"
            )

            return deduplicated

        return tokens

    def analyze_token_diversity(self, tokens: List[Dict]) -> Dict[str, Any]:
        """
        FONCTIONNALITÉ UNIQUE - Analyse la diversité des tokens.

        Retourne statistiques détaillées par catégorie.
        """
        # Définition des catégories (même que _ensure_category_representation)
        categories_def = {
            "layer1_blockchain": [
                "BTC",
                "ETH",
                "ADA",
                "SOL",
                "AVAX",
                "DOT",
                "NEAR",
                "ALGO",
            ],
            "defi_protocols": [
                "UNI",
                "AAVE",
                "COMP",
                "MKR",
                "SUSHI",
                "CRV",
                "1INCH",
                "YFI",
            ],
            "layer2_scaling": ["MATIC", "ARB", "OP", "IMX", "LRC", "MINA"],
            "meme_coins": ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK"],
            "exchange_tokens": ["BNB", "CRO", "FTT", "HT", "KCS", "OKB"],
            "stablecoins": ["USDT", "USDC", "BUSD", "DAI", "FRAX", "TUSD"],
            "ai_gaming": ["FET", "AGIX", "OCEAN", "AXS", "SAND", "MANA", "ENJ"],
        }

        stats = {"global": {}, "categories": {}}
        token_symbols = {t["symbol"].upper() for t in tokens}
        categorized = 0

        for category, cat_tokens in categories_def.items():
            count = len([s for s in cat_tokens if s in token_symbols])
            stats["categories"][category] = count
            categorized += count

        stats["global"]["total"] = len(tokens)
        stats["global"]["categorized_tokens"] = categorized
        stats["global"]["diversity_score"] = (
            (categorized / len(tokens)) * 100 if tokens else 0
        )

        return stats

    def print_diversity_report(self, tokens: List[Dict]):
        """FONCTIONNALITÉ UNIQUE - Affiche rapport de diversité formaté."""
        stats = self.analyze_token_diversity(tokens)

        print("\n" + "=" * 60)
        print("📊 RAPPORT DE DIVERSITÉ DES TOKENS")
        print("=" * 60)
        print(f"Total tokens: {stats['global']['total']}")
        print(f"Tokens catégorisés: {stats['global']['categorized_tokens']}")
        print(f"Score de diversité: {stats['global']['diversity_score']:.1f}%")
        print("\n📂 Répartition par catégorie:")

        for category, count in stats["categories"].items():
            status = "✅" if count >= 3 else "⚠️" if count > 0 else "❌"
            print(f"  {status} {category.replace('_', ' ').title()}: {count} tokens")

        print("=" * 60)

    def load_saved_tokens(self) -> List[Dict]:
        """Charge les tokens sauvegardés (DÉLÉGATION au fichier JSON_PATH)."""
        try:
            if not os.path.exists(JSON_PATH):
                self.logger.warning(f"Fichier tokens non trouvé: {JSON_PATH}")
                return []

            with open(JSON_PATH, "r", encoding="utf-8") as f:
                tokens = json.load(f)

            self.logger.info(f"✅ {len(tokens)} tokens chargés depuis {JSON_PATH}")
            return tokens

        except Exception as e:
            self.logger.error(f"❌ Erreur chargement tokens: {e}")
            return []

    # =========================================================
    #  SECTION 2: Téléchargement (DÉLÉGATION COMPLÈTE)
    # =========================================================

    def download_crypto_data(
        self,
        symbols: List[str],
        intervals: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Télécharge données OHLCV (DÉLÈGUE à fetch_klines).

        Args:
            symbols: Liste symboles (ex: ["BTCUSDC", "ETHUSDC"])
            intervals: Timeframes (défaut: ["3m", "5m", "15m", "30m", "1h"])
            progress_callback: Callback optionnel(progress_pct, done, total)

        Returns:
            Dict avec statistiques du téléchargement
        """
        intervals = intervals or INTERVALS

        now = int(datetime.now().timestamp() * 1000)
        start_ms = now - (HISTORY_DAYS * 86_400_000)

        total_tasks = len(symbols) * len(intervals)
        done = 0
        success = 0

        self.logger.info(
            f"📥 Téléchargement: {len(symbols)} symboles × {len(intervals)} TFs"
        )

        for symbol in symbols:
            for interval in intervals:
                try:
                    # DÉLÉGATION au code de référence
                    candles = fetch_klines(symbol, interval, start_ms, now)

                    if candles:
                        # Sauvegarde JSON
                        json_file = json_path_symbol(symbol, interval)
                        os.makedirs(os.path.dirname(json_file), exist_ok=True)

                        with open(json_file, "w", encoding="utf-8") as f:
                            json.dump(candles, f, indent=2)

                        success += 1
                        self.logger.info(
                            f"✅ {symbol}@{interval}: {len(candles)} bougies"
                        )

                except Exception as e:
                    self.logger.error(f"❌ Erreur {symbol}@{interval}: {e}")

                done += 1
                if progress_callback:
                    progress_callback((done / total_tasks) * 100, done, total_tasks)

        result = {
            "total_tasks": total_tasks,
            "success": success,
            "failed": total_tasks - success,
            "success_rate": (success / total_tasks) * 100 if total_tasks > 0 else 0,
        }

        self.logger.info(
            f"✅ Téléchargement terminé: {success}/{total_tasks} "
            f"({result['success_rate']:.1f}%)"
        )

        return result

    def download_top_100_data(
        self, intervals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Télécharge données des top 100 tokens (avec diversité)."""
        top_tokens = self.get_top_100_tokens(save_to_file=False)
        symbols = [f"{t['symbol']}USDC" for t in top_tokens]

        return self.download_crypto_data(symbols, intervals)

    # =========================================================
    #  SECTION 3: Chargement données (DÉLÉGATION)
    # =========================================================

    def load_ohlcv_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Charge données OHLCV depuis Parquet ou JSON (DÉLÈGUE).

        Args:
            symbol: Symbole (ex: "BTCUSDC")
            interval: Timeframe (ex: "1h")

        Returns:
            DataFrame OHLCV ou None si non trouvé
        """
        # Priorité 1: Parquet (plus rapide)
        pq_file = parquet_path(symbol, interval)
        if os.path.exists(pq_file):
            try:
                df = pd.read_parquet(pq_file)
                self.logger.debug(f"✅ Chargé depuis Parquet: {pq_file}")
                return df
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur lecture Parquet: {e}")

        # Priorité 2: JSON (avec conversion)
        json_file = json_path_symbol(symbol, interval)
        if os.path.exists(json_file):
            try:
                # DÉLÉGATION à _json_to_df du code de référence
                df = _json_to_df(json_file)

                if df is not None:
                    # Optionnel: sauvegarder en Parquet pour prochaine fois
                    json_candles_to_parquet(json_file, os.path.dirname(pq_file))
                    self.logger.debug(f"✅ Chargé depuis JSON: {json_file}")
                    return df

            except Exception as e:
                self.logger.error(f"❌ Erreur conversion JSON: {e}")

        self.logger.warning(f"❌ Données non trouvées: {symbol}@{interval}")
        return None

    # =========================================================
    #  SECTION 4: Indicateurs (DÉLÉGATION threadx.indicators.numpy)
    # =========================================================

    def add_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """
        Ajoute indicateurs à un DataFrame (DÉLÈGUE aux fonctions NumPy).

        Args:
            df: DataFrame OHLCV
            indicators: Liste noms (ex: ["rsi", "macd", "bb", "atr"])

        Returns:
            DataFrame enrichi avec indicateurs
        """
        df = df.copy()

        for indicator in indicators:
            try:
                if indicator == "rsi":
                    df["rsi"] = rsi_np(df["close"].values, 14)

                elif indicator in ["bollinger", "bb"]:
                    lower, middle, upper, z = boll_np(df["close"].values, 20, 2.0)
                    df["bb_lower"] = lower
                    df["bb_middle"] = middle
                    df["bb_upper"] = upper
                    df["bb_zscore"] = z

                elif indicator == "macd":
                    macd, signal, hist = macd_np(df["close"].values, 12, 26, 9)
                    df["macd"] = macd
                    df["macd_signal"] = signal
                    df["macd_hist"] = hist

                elif indicator == "atr":
                    df["atr"] = atr_np(
                        df["high"].values, df["low"].values, df["close"].values, 14
                    )

                elif indicator in ["ema20", "ema_20"]:
                    df["ema_20"] = ema_np(df["close"].values, 20)

                elif indicator in ["ema50", "ema_50"]:
                    df["ema_50"] = ema_np(df["close"].values, 50)

                elif indicator == "vwap":
                    df["vwap"] = vwap_np(
                        df["close"].values,
                        df["high"].values,
                        df["low"].values,
                        df["volume"].values,
                        96,
                    )

                elif indicator == "obv":
                    df["obv"] = obv_np(df["close"].values, df["volume"].values)

                else:
                    self.logger.warning(f"⚠️ Indicateur inconnu: {indicator}")

            except Exception as e:
                self.logger.error(f"❌ Erreur calcul {indicator}: {e}")

        return df

    # =========================================================
    #  SECTION 5: API unifiée (FAÇADE)
    # =========================================================

    def get_trading_data(
        self,
        symbol: str,
        interval: str,
        indicators: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        API FAÇADE - Charge OHLCV + indicateurs.

        Args:
            symbol: Symbole (ex: "BTCUSDC")
            interval: Timeframe (ex: "1h")
            indicators: Liste indicateurs optionnelle
            start_date: Date début filtrage (format ISO)
            end_date: Date fin filtrage (format ISO)

        Returns:
            DataFrame enrichi ou None
        """
        # Chargement OHLCV (délégué)
        df = self.load_ohlcv_data(symbol, interval)

        if df is None:
            return None

        # Filtrage dates si spécifié
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]

        # Ajout indicateurs si demandé (délégué)
        if indicators:
            df = self.add_indicators(df, indicators)

        return df

    def get_multiple_trading_data(
        self, pairs: List[Tuple[str, str]], indicators: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Charge plusieurs paires en parallèle.

        Args:
            pairs: Liste de tuples (symbol, interval)
            indicators: Indicateurs à calculer

        Returns:
            Dict {f"{symbol}@{interval}": DataFrame}
        """
        results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.get_trading_data, symbol, interval, indicators): (
                    symbol,
                    interval,
                )
                for symbol, interval in pairs
            }

            for future in as_completed(futures):
                symbol, interval = futures[future]
                key = f"{symbol}@{interval}"

                try:
                    df = future.result()
                    if df is not None:
                        results[key] = df
                except Exception as e:
                    self.logger.error(f"❌ Erreur {key}: {e}")

        return results


# =========================================================
#  Interface CLI simplifiée
# =========================================================


def main():
    """Interface en ligne de commande pour tester le gestionnaire v2."""
    print("🚀 TradXPro Core Manager v2.0 - Sans Redondances")
    print("=" * 60)

    manager = TradXProManager()

    # Menu interactif simple
    while True:
        print("\n📋 Menu:")
        print("1. Récupérer Top 100 tokens (avec diversité)")
        print("2. Télécharger données crypto")
        print("3. Charger données + indicateurs")
        print("4. Analyser diversité tokens")
        print("5. Quitter")

        choice = input("\nChoix: ").strip()

        if choice == "1":
            tokens = manager.get_top_100_tokens()
            print(f"\n✅ {len(tokens)} tokens récupérés")
            manager.print_diversity_report(tokens)

        elif choice == "2":
            symbols_input = input("Symboles (ex: BTCUSDC,ETHUSDC): ").strip()
            symbols = [s.strip() for s in symbols_input.split(",")]

            result = manager.download_crypto_data(symbols)
            print(f"\n✅ Téléchargement: {result['success']}/{result['total_tasks']}")

        elif choice == "3":
            symbol = input("Symbole (ex: BTCUSDC): ").strip()
            interval = input("Timeframe (ex: 1h): ").strip()
            indicators = input("Indicateurs (ex: rsi,macd,bb): ").strip().split(",")

            df = manager.get_trading_data(symbol, interval, indicators)

            if df is not None:
                print(f"\n✅ Données chargées: {len(df)} lignes")
                print(f"Colonnes: {', '.join(df.columns)}")
                print(f"\nAperçu:\n{df.tail()}")
            else:
                print("\n❌ Données non trouvées")

        elif choice == "4":
            tokens = manager.load_saved_tokens()
            if tokens:
                manager.print_diversity_report(tokens)
            else:
                print("❌ Aucun token sauvegardé")

        elif choice == "5":
            print("\n👋 Au revoir!")
            break

        else:
            print("❌ Choix invalide")


if __name__ == "__main__":
    main()
