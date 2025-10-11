"""
Script de synchronisation automatique des données ThreadX
Télécharge et vérifie les données depuis le 1er janvier jusqu'à hier
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# Ajouter le chemin src au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from threadx.data.ingest import IngestionManager
from threadx.data.registry import scan_symbols, dataset_exists
from threadx.config import get_settings

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Ajouter le chemin src au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from threadx.data.providers.binance import BinanceProvider
from threadx.data.storage.parquet_manager import ParquetManager
from threadx.data.preprocessing.data_cleaner import DataCleaner
from threadx.config.settings import load_settings
from threadx.utils.log import get_logger

logger = get_logger(__name__)


class AutoDataSync:
    """Gestionnaire de synchronisation automatique des données"""

    def __init__(self):
        self.settings = load_settings()
        self.provider = BinanceProvider()
        self.storage = ParquetManager()
        self.cleaner = DataCleaner()

        # Période de synchronisation : 1er janvier de l'année en cours jusqu'à hier
        self.start_date = datetime(datetime.now().year, 1, 1)
        self.end_date = datetime.now() - timedelta(days=1)
        self.end_date = self.end_date.replace(hour=23, minute=59, second=59)

        logger.info(f"📅 Période de sync: {self.start_date} → {self.end_date}")

    def get_symbols_to_sync(self) -> List[str]:
        """Récupère la liste des symboles à synchroniser depuis la config"""
        # Symboles par défaut si non spécifié dans la config
        default_symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "AVAX/USDT",
            "DOT/USDT",
        ]

        # Essayer de récupérer depuis la config
        try:
            symbols = self.settings.get("data", {}).get("symbols", default_symbols)
        except:
            symbols = default_symbols

        logger.info(f"🎯 Symboles à synchroniser: {', '.join(symbols)}")
        return symbols

    def get_timeframes_to_sync(self) -> List[str]:
        """Récupère la liste des timeframes à synchroniser"""
        default_timeframes = ["1h", "4h", "1d"]

        try:
            timeframes = self.settings.get("data", {}).get(
                "timeframes", default_timeframes
            )
        except:
            timeframes = default_timeframes

        logger.info(f"⏱️  Timeframes à synchroniser: {', '.join(timeframes)}")
        return timeframes

    def check_existing_data(self, symbol: str, timeframe: str) -> Dict:
        """Vérifie les données existantes pour un symbole/timeframe"""
        symbol_normalized = symbol.replace("/", "")

        available = self.storage.list_available_data(symbol_normalized, timeframe)

        if not available:
            return {
                "has_data": False,
                "last_date": None,
                "gap_days": (self.end_date - self.start_date).days,
            }

        # Charger la dernière date disponible
        try:
            last_df = self.storage.load_ohlcv(
                symbol_normalized,
                timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
            )

            if last_df.empty:
                return {
                    "has_data": False,
                    "last_date": None,
                    "gap_days": (self.end_date - self.start_date).days,
                }

            last_date = last_df.index[-1]
            gap_days = (self.end_date - last_date).days

            return {
                "has_data": True,
                "last_date": last_date,
                "gap_days": gap_days,
                "total_bars": len(last_df),
            }
        except Exception as e:
            logger.warning(f"⚠️ Erreur vérification {symbol}/{timeframe}: {e}")
            return {
                "has_data": False,
                "last_date": None,
                "gap_days": (self.end_date - self.start_date).days,
            }

    def sync_symbol_timeframe(self, symbol: str, timeframe: str) -> bool:
        """Synchronise un symbole/timeframe spécifique"""
        symbol_normalized = symbol.replace("/", "")

        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Synchronisation: {symbol} / {timeframe}")
        logger.info(f"{'='*60}")

        # Vérifier les données existantes
        check = self.check_existing_data(symbol, timeframe)

        if check["has_data"]:
            logger.info(f"📊 Données existantes: {check['total_bars']:,} barres")
            logger.info(f"📅 Dernière date: {check['last_date']}")
            logger.info(f"⏳ Écart: {check['gap_days']} jours")

            if check["gap_days"] <= 1:
                logger.info(f"✅ Données à jour, aucun téléchargement nécessaire")
                return True

            # Télécharger uniquement les données manquantes
            download_start = check["last_date"] + timedelta(days=1)
        else:
            logger.info(f"📥 Aucune donnée existante, téléchargement complet")
            download_start = self.start_date

        try:
            # Télécharger les données
            logger.info(
                f"⬇️  Téléchargement depuis {download_start} jusqu'à {self.end_date}"
            )

            df = self.provider.fetch_historical_range(
                symbol=symbol,
                timeframe=timeframe,
                start_date=download_start,
                end_date=self.end_date,
            )

            if df.empty:
                logger.warning(f"⚠️ Aucune donnée reçue pour {symbol}/{timeframe}")
                return False

            logger.info(f"✅ Téléchargé: {len(df):,} barres")

            # Nettoyer les données
            df_clean = self.cleaner.clean_ohlcv(df)
            logger.info(f"🧹 Nettoyé: {len(df_clean):,} barres valides")

            # Sauvegarder par mois
            months = df_clean.index.to_period("M").unique()

            for month in months:
                month_data = df_clean[df_clean.index.to_period("M") == month]
                month_date = month.to_timestamp()

                self.storage.save_ohlcv(
                    month_data, symbol_normalized, timeframe, month_date
                )

                logger.info(f"💾 Sauvegardé {month}: {len(month_data):,} barres")

            logger.info(f"✅ Synchronisation terminée pour {symbol}/{timeframe}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur synchronisation {symbol}/{timeframe}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def run_full_sync(self) -> Dict:
        """Exécute la synchronisation complète"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 DÉMARRAGE SYNCHRONISATION AUTOMATIQUE ThreadX")
        logger.info("=" * 80)

        symbols = self.get_symbols_to_sync()
        timeframes = self.get_timeframes_to_sync()

        total = len(symbols) * len(timeframes)
        success = 0
        failed = 0

        results = {"total": total, "success": 0, "failed": 0, "details": []}

        for i, symbol in enumerate(symbols, 1):
            for j, timeframe in enumerate(timeframes, 1):
                progress = ((i - 1) * len(timeframes) + j) / total * 100
                logger.info(
                    f"\n📊 Progression: {progress:.1f}% ({(i-1)*len(timeframes)+j}/{total})"
                )

                result = self.sync_symbol_timeframe(symbol, timeframe)

                if result:
                    success += 1
                    results["details"].append(
                        {"symbol": symbol, "timeframe": timeframe, "status": "success"}
                    )
                else:
                    failed += 1
                    results["details"].append(
                        {"symbol": symbol, "timeframe": timeframe, "status": "failed"}
                    )

        results["success"] = success
        results["failed"] = failed

        # Résumé final
        logger.info("\n" + "=" * 80)
        logger.info("📊 RÉSUMÉ DE LA SYNCHRONISATION")
        logger.info("=" * 80)
        logger.info(f"✅ Réussies: {success}/{total}")
        logger.info(f"❌ Échouées: {failed}/{total}")
        logger.info(f"📅 Période: {self.start_date.date()} → {self.end_date.date()}")
        logger.info("=" * 80)

        return results

    def verify_data_integrity(self) -> Dict:
        """Vérifie l'intégrité de toutes les données synchronisées"""
        logger.info("\n" + "=" * 80)
        logger.info("🔍 VÉRIFICATION DE L'INTÉGRITÉ DES DONNÉES")
        logger.info("=" * 80)

        symbols = self.get_symbols_to_sync()
        timeframes = self.get_timeframes_to_sync()

        report = {"total_checked": 0, "valid": 0, "issues": [], "summary": {}}

        for symbol in symbols:
            symbol_normalized = symbol.replace("/", "")

            for timeframe in timeframes:
                report["total_checked"] += 1

                try:
                    df = self.storage.load_ohlcv(
                        symbol_normalized, timeframe, self.start_date, self.end_date
                    )

                    if df.empty:
                        report["issues"].append(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "issue": "Aucune donnée",
                            }
                        )
                        continue

                    # Vérifier la continuité temporelle
                    expected_bars = self._calculate_expected_bars(timeframe)
                    actual_bars = len(df)
                    completeness = (
                        (actual_bars / expected_bars) * 100 if expected_bars > 0 else 0
                    )

                    # Vérifier la qualité
                    nulls = df.isnull().sum().sum()
                    duplicates = df.index.duplicated().sum()

                    if completeness >= 95 and nulls == 0 and duplicates == 0:
                        report["valid"] += 1
                    else:
                        report["issues"].append(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "completeness": f"{completeness:.1f}%",
                                "nulls": nulls,
                                "duplicates": duplicates,
                            }
                        )

                    logger.info(
                        f"✓ {symbol}/{timeframe}: {actual_bars:,} barres ({completeness:.1f}% complet)"
                    )

                except Exception as e:
                    report["issues"].append(
                        {"symbol": symbol, "timeframe": timeframe, "issue": str(e)}
                    )

        report["summary"] = {
            "total": report["total_checked"],
            "valid": report["valid"],
            "with_issues": len(report["issues"]),
            "success_rate": (
                (report["valid"] / report["total_checked"] * 100)
                if report["total_checked"] > 0
                else 0
            ),
        }

        logger.info("\n📊 Résumé de vérification:")
        logger.info(f"✅ Valides: {report['valid']}/{report['total_checked']}")
        logger.info(f"⚠️  Avec problèmes: {len(report['issues'])}")
        logger.info(f"📈 Taux de réussite: {report['summary']['success_rate']:.1f}%")

        return report

    def _calculate_expected_bars(self, timeframe: str) -> int:
        """Calcule le nombre de barres attendues pour un timeframe"""
        total_days = (self.end_date - self.start_date).days

        bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}

        return total_days * bars_per_day.get(timeframe, 0)


def main():
    """Point d'entrée principal"""
    try:
        syncer = AutoDataSync()

        # Exécuter la synchronisation
        sync_results = syncer.run_full_sync()

        # Vérifier l'intégrité
        verification_results = syncer.verify_data_integrity()

        # Rapport final
        logger.info("\n" + "=" * 80)
        logger.info("🎉 SYNCHRONISATION TERMINÉE")
        logger.info("=" * 80)
        logger.info(
            f"📥 Données synchronisées: {sync_results['success']}/{sync_results['total']}"
        )
        logger.info(
            f"✅ Données valides: {verification_results['valid']}/{verification_results['total_checked']}"
        )
        logger.info(
            f"📊 Taux de réussite global: {verification_results['summary']['success_rate']:.1f}%"
        )
        logger.info("=" * 80)

        return sync_results, verification_results

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
