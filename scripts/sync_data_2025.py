"""
Script de synchronisation automatique des données ThreadX
Télécharge depuis le 1er janvier 2025 jusqu'à hier
Utilise IngestionManager pour gérer les données
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Ajouter src au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from threadx.data.ingest import IngestionManager
from threadx.data.registry import scan_symbols
from threadx.config import get_settings

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Synchronisation automatique des données"""

    logger.info("=" * 80)
    logger.info("🚀 SYNCHRONISATION AUTOMATIQUE ThreadX")
    logger.info("=" * 80)

    # Configuration
    settings = get_settings()
    manager = IngestionManager(settings)

    # Période : 1er janvier 2025 → hier
    start_date = datetime(2025, 1, 1)
    end_date = datetime.now() - timedelta(days=1)
    end_date = end_date.replace(hour=23, minute=59, second=59)

    logger.info(f"📅 Période: {start_date.date()} → {end_date.date()}")

    # Symboles à synchroniser (USDC au lieu de USDT)
    default_symbols = [
        "BTCUSDC",
        "ETHUSDC",
        "BNBUSDC",
        "SOLUSDC",
        "XRPUSDC",
        "ADAUSDC",
        "AVAXUSDC",
        "DOTUSDC",
    ]

    logger.info(f"🎯 Symboles: {', '.join(default_symbols)}")

    # Timeframes
    timeframes = ["1h", "4h", "1d"]
    logger.info(f"⏱️  Timeframes: {', '.join(timeframes)}")

    # Statistiques
    total = len(default_symbols) * len(timeframes)
    success = 0
    failed = 0

    # Synchroniser chaque symbole/timeframe
    for i, symbol in enumerate(default_symbols, 1):
        for j, timeframe in enumerate(timeframes, 1):
            progress = ((i - 1) * len(timeframes) + j) / total * 100
            current = (i - 1) * len(timeframes) + j

            logger.info("")
            logger.info("=" * 60)
            logger.info(f"📊 Progression: {progress:.1f}% " f"({current}/{total})")
            logger.info(f"🔄 Sync: {symbol} / {timeframe}")
            logger.info("=" * 60)

            try:
                # Télécharger 1m truth
                logger.info(f"⬇️  Téléchargement 1m pour {symbol}")
                df_1m = manager.download_ohlcv_1m(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    force=False,  # Utilise cache local si disponible
                )

                if df_1m.empty:
                    logger.warning(f"⚠️  Aucune donnée 1m pour {symbol}")
                    failed += 1
                    continue

                logger.info(f"✅ Téléchargé: {len(df_1m):,} barres 1m")

                # Resample vers le timeframe cible
                if timeframe != "1m":
                    logger.info(f"🔄 Resample 1m → {timeframe}")
                    df_tf = manager.resample_from_1m(
                        symbol=symbol,
                        target_tf=timeframe,
                        start=start_date,
                        end=end_date,
                    )

                    if df_tf.empty:
                        logger.warning(f"⚠️  Échec resample vers {timeframe}")
                        failed += 1
                        continue

                    logger.info(f"✅ Resampled: {len(df_tf):,} barres {timeframe}")

                success += 1
                logger.info(f"✅ Synchronisation OK: {symbol}/{timeframe}")

            except Exception as e:
                logger.error(f"❌ Erreur {symbol}/{timeframe}: {e}")
                failed += 1

    # Résumé final
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RÉSUMÉ")
    logger.info("=" * 80)
    logger.info(f"✅ Réussies: {success}/{total}")
    logger.info(f"❌ Échouées: {failed}/{total}")
    logger.info(f"📅 Période: {start_date.date()} → {end_date.date()}")

    # Afficher les stats de l'ingestion
    stats = manager.session_stats
    logger.info("")
    logger.info("📈 Statistiques de session:")
    logger.info(f"  - Symboles traités: {stats['symbols_processed']}")
    logger.info(f"  - Fichiers téléchargés: {stats['files_downloaded']}")
    logger.info(f"  - Fichiers resamplés: {stats['files_resampled']}")
    logger.info(f"  - Gaps comblés: {stats['gaps_filled']}")
    logger.info(f"  - Avertissements: {stats['verification_warnings']}")
    logger.info("=" * 80)
    logger.info("🎉 SYNCHRONISATION TERMINÉE")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interruption utilisateur")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
