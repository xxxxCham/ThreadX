"""
Script de synchronisation INTELLIGENTE des données ThreadX
Utilise les données existantes (3m/5m/15m/1h) au lieu de tout re-télécharger
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd

# Ajouter src au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from threadx.data.ingest import IngestionManager
from threadx.config import get_settings

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DownloadStats:
    """Compte les bougies téléchargées et calcule le taux par minute (fenêtre glissante 60s)."""

    def __init__(self):
        from collections import deque
        import threading

        self.total = 0
        self._records = deque()  # tuples (timestamp, count)
        self._lock = threading.Lock()

    def add(self, count: int):
        """Ajouter `count` bougies téléchargées."""
        import time

        now = time.time()
        with self._lock:
            self.total += int(count)
            self._records.append((now, int(count)))
            # Nettoyer les enregistrements vieux de plus de 120s pour sécurité
            cutoff = now - 120
            while self._records and self._records[0][0] < cutoff:
                self._records.popleft()

    def rate_per_minute(self) -> int:
        """Retourne le total de bougies ajoutées dans la dernière minute."""
        import time

        now = time.time()
        cutoff = now - 60
        s = 0
        with self._lock:
            for ts, c in self._records:
                if ts >= cutoff:
                    s += c
        return int(s)


class SmartSyncManager:
    """Gestionnaire intelligent de synchronisation"""

    def __init__(self, settings):
        self.settings = settings
        self.manager = IngestionManager(settings)
        self.data_root = Path(settings.DATA_ROOT)
        # Statistiques de téléchargement
        self.stats = DownloadStats()

        # Timeframes disponibles (du plus au moins granulaire)
        self.available_timeframes = ["1h", "15m", "5m", "3m", "1m"]
        self.target_timeframes = ["1h", "4h", "1d"]

    def find_existing_data(self, symbol: str) -> Optional[str]:
        """Trouve le meilleur timeframe existant pour ce symbole"""

        logger.info(f"🔍 Recherche données existantes pour {symbol}...")

        # Chercher dans l'ordre de priorité
        for tf in self.available_timeframes:
            # Vérifier dans processed/
            processed_path = self.data_root / "processed" / symbol / f"{tf}.parquet"

            if processed_path.exists():
                try:
                    df = pd.read_parquet(processed_path)
                    if not df.empty and len(df) > 100:
                        logger.info(
                            f"✅ Trouvé: {tf} ({len(df):,} barres) "
                            f"dans {processed_path.parent.name}"
                        )
                        return tf
                except Exception as e:
                    logger.warning(f"⚠️  Erreur lecture {tf}: {e}")
                    continue

            # Vérifier dans raw/
            raw_path = self.data_root / "raw" / tf / f"{symbol}.parquet"
            if raw_path.exists():
                try:
                    df = pd.read_parquet(raw_path)
                    if not df.empty and len(df) > 100:
                        logger.info(
                            f"✅ Trouvé: {tf} ({len(df):,} barres) " f"dans raw/{tf}"
                        )
                        return tf
                except Exception as e:
                    logger.warning(f"⚠️  Erreur lecture {tf}: {e}")
                    continue

        logger.warning(f"⚠️  Aucune donnée existante pour {symbol}")
        return None

    def check_data_completeness(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> Dict:
        """Vérifie si les données sont complètes pour la période"""

        try:
            # Charger les données
            df = self.load_data(symbol, timeframe)

            if df.empty:
                return {"complete": False, "coverage": 0.0, "bars": 0, "gaps": []}

            # Calculer la couverture
            total_days = (end_date - start_date).days
            data_start = df.index.min()
            data_end = df.index.max()

            # Convertir les dates en timezone-aware si nécessaire
            if data_start.tzinfo is not None:
                # Les données ont une timezone, convertir start/end en UTC
                import pytz

                if start_date.tzinfo is None:
                    start_date = pytz.UTC.localize(start_date)
                if end_date.tzinfo is None:
                    end_date = pytz.UTC.localize(end_date)
            else:
                # Les données n'ont pas de timezone, retirer la timezone de start/end
                if start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)

            # Vérifier si la plage est couverte
            coverage_start = data_start <= start_date
            coverage_end = data_end >= end_date - timedelta(days=2)

            expected = self.calculate_expected_bars(timeframe, start_date, end_date)
            coverage_pct = len(df) / expected if total_days > 0 and expected > 0 else 0
            missing = max(0, expected - len(df)) if expected > 0 else 0

            is_complete = coverage_start and coverage_end and coverage_pct >= 0.90

            return {
                "complete": is_complete,
                "coverage": coverage_pct,
                "bars": len(df),
                "expected": expected,
                "missing": missing,
                "data_start": data_start,
                "data_end": data_end,
                "gaps": [],  # À implémenter si besoin
            }

        except Exception as e:
            logger.error(f"❌ Erreur vérification complétude: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {"complete": False, "coverage": 0.0, "bars": 0}

    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Charge les données depuis processed/ ou raw/"""

        # Essayer processed/ d'abord
        processed_path = self.data_root / "processed" / symbol / f"{timeframe}.parquet"
        if processed_path.exists():
            return pd.read_parquet(processed_path)

        # Essayer raw/
        raw_path = self.data_root / "raw" / timeframe / f"{symbol}.parquet"
        if raw_path.exists():
            return pd.read_parquet(raw_path)

        return pd.DataFrame()

    def calculate_expected_bars(
        self, timeframe: str, start: datetime, end: datetime
    ) -> int:
        """Calcule le nombre de barres attendu"""

        total_hours = (end - start).total_seconds() / 3600

        bars_per_hour = {
            "1m": 60,
            "3m": 20,
            "5m": 12,
            "15m": 4,
            "1h": 1,
            "4h": 0.25,
            "1d": 1 / 24,
        }

        return int(total_hours * bars_per_hour.get(timeframe, 0))

    def find_missing_ranges(
        self, df: pd.DataFrame, timeframe: str, start: datetime, end: datetime
    ):
        """Retourne une liste de tuples (start, end) des plages manquantes pour le timeframe donné.

        Se base sur un DateRange attendu et compare à l'index présent.
        """
        try:
            if df is None or df.empty:
                return [(start, end)]

            freq_map = {
                "1m": "1min",
                "3m": "3min",
                "5m": "5min",
                "15m": "15min",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
            }
            freq = freq_map.get(timeframe, "1min")

            # Normaliser timezone comme dans check_data_completeness
            data_tz = df.index.tz
            try:
                import pytz

                if data_tz is not None:
                    if start.tzinfo is None:
                        start = pytz.UTC.localize(start)
                    if end.tzinfo is None:
                        end = pytz.UTC.localize(end)
                else:
                    if start.tzinfo is not None:
                        start = start.replace(tzinfo=None)
                    if end.tzinfo is not None:
                        end = end.replace(tzinfo=None)
            except Exception:
                # Si pytz indisponible, continuer sans timezone
                pass

            expected = pd.date_range(start=start, end=end, freq=freq)
            present = pd.Index(df.index)
            missing = expected.difference(present)
            if missing.empty:
                return []

            # Grouper en plages contiguës
            ranges = []
            cur_start = missing[0]
            prev = missing[0]
            td = pd.Timedelta(freq)
            for ts in missing[1:]:
                if (ts - prev) > td:
                    ranges.append((cur_start.to_pydatetime(), prev.to_pydatetime()))
                    cur_start = ts
                prev = ts

            ranges.append((cur_start.to_pydatetime(), prev.to_pydatetime()))
            return ranges
        except Exception as e:
            logger.warning(f"⚠️ Erreur détermination plages manquantes: {e}")
            return []

    def sync_symbol_smart(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> bool:
        """Synchronisation intelligente d'un symbole"""

        logger.info("=" * 60)
        logger.info(f"🚀 Synchronisation intelligente: {symbol}")
        logger.info("=" * 60)

        # 1. Trouver les données existantes
        best_tf = self.find_existing_data(symbol)

        if best_tf:
            # 2. Vérifier la complétude
            check = self.check_data_completeness(symbol, best_tf, start_date, end_date)

            logger.info(
                f"📊 Complétude {best_tf}: "
                f"{check['coverage']*100:.1f}% "
                f"({check['bars']:,} barres) "
                f"— manquantes: {check.get('missing', 0):,}"
            )

            if check["complete"]:
                logger.info(
                    f"✅ Données {best_tf} suffisantes, "
                    f"génération des timeframes supérieurs"
                )

                # Si des bougies manquent encore, télécharger les segments manquants (cas 1m principalement)
                if check.get("missing", 0) > 0:
                    logger.info(
                        f"⬇️  {check['missing']:,} bougies manquantes détectées: téléchargement des segments manquants..."
                    )

                    # Charger les données existantes
                    df_exist = self.load_data(symbol, best_tf)
                    ranges = self.find_missing_ranges(
                        df_exist, best_tf, start_date, end_date
                    )

                    if ranges:
                        for rstart, rend in ranges:
                            logger.info(f"⬇️  Segment: {rstart} → {rend}")
                            try:
                                seg = self.manager.download_ohlcv_1m(
                                    symbol=symbol, start=rstart, end=rend, force=True
                                )
                                if seg is not None and not seg.empty:
                                    self.stats.add(len(seg))
                                    logger.info(
                                        f"✅ Segment téléchargé: {len(seg):,} bougies"
                                    )
                                else:
                                    logger.warning(
                                        "⚠️  Aucun segment téléchargé pour cette plage"
                                    )
                            except Exception as e:
                                logger.error(f"❌ Erreur téléchargement segment: {e}")

                        # Après avoir comblé les segments, rejouer la génération
                        logger.info(
                            "🔁 Re-vérification après téléchargement des segments"
                        )
                        check2 = self.check_data_completeness(
                            symbol, best_tf, start_date, end_date
                        )
                        if not check2.get("complete", False):
                            logger.warning(
                                "⚠️  Après téléchargement, données toujours incomplètes — fallback sur téléchargement complet"
                            )
                            return self.download_and_resample(
                                symbol, start_date, end_date
                            )

                    # Charger à nouveau et générer
                    return self.generate_from_existing(
                        symbol, best_tf, start_date, end_date
                    )
                # 3. Générer les timeframes manquants depuis les données existantes
                return self.generate_from_existing(
                    symbol, best_tf, start_date, end_date
                )
            else:
                logger.warning(
                    f"⚠️  Données {best_tf} incomplètes "
                    f"({check['coverage']*100:.1f}%), "
                    f"fallback sur téléchargement"
                )

        # 4. Fallback: télécharger depuis 1m
        logger.info("📥 Téléchargement 1m nécessaire")
        return self.download_and_resample(symbol, start_date, end_date)

    def generate_from_existing(
        self, symbol: str, source_tf: str, start_date: datetime, end_date: datetime
    ) -> bool:
        """Génère les timeframes cibles depuis un timeframe existant"""

        success_count = 0

        # Charger les données source (1m dans la plupart des cas)
        logger.info(f"📂 Chargement données {source_tf} pour {symbol}")
        df_source = self.load_data(symbol, source_tf)

        if df_source.empty:
            logger.error(f"❌ Impossible de charger les données {source_tf}")
            return False

        logger.info(f"✅ Chargé: {len(df_source):,} barres {source_tf}")

        for target_tf in self.target_timeframes:
            # Ne pas régénérer le même timeframe
            if source_tf == target_tf:
                logger.info(f"⏭️  {target_tf} déjà disponible")
                success_count += 1
                continue

            # Ne générer que les timeframes supérieurs
            tf_hierarchy = {
                "1m": 0,
                "3m": 1,
                "5m": 2,
                "15m": 3,
                "1h": 4,
                "4h": 5,
                "1d": 6,
            }

            if tf_hierarchy.get(source_tf, 0) > tf_hierarchy.get(target_tf, 0):
                logger.warning(
                    f"⚠️  Impossible de générer {target_tf} "
                    f"depuis {source_tf} (timeframe plus petit)"
                )
                continue

            try:
                logger.info(f"🔄 Génération {target_tf} depuis {source_tf}")

                # Resample depuis le DataFrame source
                # Si source_tf == "1m", utiliser resample_from_1m
                if source_tf == "1m":
                    df_target = self.manager.resample_from_1m(
                        df_1m=df_source, timeframe=target_tf
                    )
                else:
                    # Pour les autres timeframes, utiliser la fonction générique
                    from threadx.data.resample import resample_ohlcv

                    df_target = resample_ohlcv(
                        df=df_source, source_tf=source_tf, target_tf=target_tf
                    )

                if not df_target.empty:
                    logger.info(f"✅ Généré: {len(df_target):,} barres {target_tf}")
                    # Ces barres sont générées localement (pas de téléchargement)
                    success_count += 1
                else:
                    logger.warning(f"⚠️  Échec génération {target_tf}")

            except Exception as e:
                logger.error(f"❌ Erreur génération {target_tf}: {e}")
                import traceback

                logger.error(traceback.format_exc())

        return success_count >= len(self.target_timeframes) - 1

    def download_and_resample(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> bool:
        """Télécharge 1m et génère tous les timeframes"""

        try:
            # Télécharger 1m
            logger.info(f"⬇️  Téléchargement 1m pour {symbol}")
            df_1m = self.manager.download_ohlcv_1m(
                symbol=symbol, start=start_date, end=end_date, force=False
            )

            if df_1m.empty:
                logger.error(f"❌ Aucune donnée 1m reçue pour {symbol}")
                return False

            logger.info(f"✅ Téléchargé: {len(df_1m):,} barres 1m")

            # Mettre à jour les statistiques de téléchargement
            try:
                self.stats.add(len(df_1m))
                logger.info(
                    f"📈 Statistiques: {self.stats.rate_per_minute():,} bougies/min — total téléchargées: {self.stats.total:,}"
                )
            except Exception:
                # Ne doit pas interrompre le flux principal
                logger.debug(
                    "⚠️  Impossible de mettre à jour les stats de téléchargement"
                )

            # Générer tous les timeframes cibles
            success_count = 0
            for target_tf in self.target_timeframes:
                if target_tf == "1m":
                    success_count += 1
                    continue

                try:
                    logger.info(f"🔄 Resample 1m → {target_tf}")
                    # Resample depuis le DataFrame 1m téléchargé
                    df_tf = self.manager.resample_from_1m(
                        df_1m=df_1m, timeframe=target_tf
                    )

                    if not df_tf.empty:
                        logger.info(f"✅ Resampled: {len(df_tf):,} barres {target_tf}")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️  Échec resample {target_tf}")

                except Exception as e:
                    logger.error(f"❌ Erreur resample {target_tf}: {e}")

            return success_count >= len(self.target_timeframes) - 1

        except Exception as e:
            logger.error(f"❌ Erreur téléchargement {symbol}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False


def main(test_mode: bool = True):
    """Point d'entrée principal"""

    logger.info("=" * 80)
    logger.info("🚀 SYNCHRONISATION INTELLIGENTE ThreadX")
    logger.info("Utilise les données existantes avant de télécharger")
    logger.info("=" * 80)

    # Configuration
    settings = get_settings()
    smart_sync = SmartSyncManager(settings)

    # Période
    start_date = datetime(2025, 1, 1)
    end_date = datetime.now() - timedelta(days=1)
    end_date = end_date.replace(hour=23, minute=59, second=59)

    logger.info(f"📅 Période: {start_date.date()} → {end_date.date()}")

    # Symboles
    all_symbols = [
        "BTCUSDC",
        "ETHUSDC",
        "BNBUSDC",
        "SOLUSDC",
        "XRPUSDC",
        "ADAUSDC",
        "AVAXUSDC",
        "DOTUSDC",
    ]

    # Mode test: un seul symbole
    if test_mode:
        symbols = ["BTCUSDC"]
        logger.info("🧪 MODE TEST: BTCUSDC uniquement")
    else:
        symbols = all_symbols
        logger.info(f"🎯 {len(symbols)} symboles à synchroniser")

    logger.info(f"⏱️  Timeframes cibles: {', '.join(smart_sync.target_timeframes)}")
    logger.info("")

    # Synchroniser
    success = 0
    failed = 0

    for i, symbol in enumerate(symbols, 1):
        logger.info("")
        logger.info(f"📊 Symbole {i}/{len(symbols)}: {symbol}")

        try:
            if smart_sync.sync_symbol_smart(symbol, start_date, end_date):
                success += 1
                logger.info(f"✅ {symbol} synchronisé avec succès")
            else:
                failed += 1
                logger.error(f"❌ Échec synchronisation {symbol}")
        except Exception as e:
            failed += 1
            logger.error(f"❌ Erreur {symbol}: {e}")
            import traceback

            logger.error(traceback.format_exc())

    # Résumé
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RÉSUMÉ SYNCHRONISATION INTELLIGENTE")
    logger.info("=" * 80)
    logger.info(f"✅ Réussies: {success}/{len(symbols)}")
    logger.info(f"❌ Échouées: {failed}/{len(symbols)}")
    logger.info(f"📅 Période: {start_date.date()} → {end_date.date()}")

    # Afficher statistiques de téléchargement
    try:
        logger.info("")
        logger.info("📈 Statistiques de téléchargement globales")
        logger.info(f"🔢 Total bougies téléchargées: {smart_sync.stats.total:,}")
        logger.info(
            f"⚡ Taux actuel (60s): {smart_sync.stats.rate_per_minute():,} bougies/min"
        )
    except Exception:
        logger.debug("⚠️  Impossible d'afficher les statistiques de téléchargement")

    if test_mode:
        logger.info("")
        logger.info("🧪 Test terminé avec BTCUSDC")
        logger.info(
            "💡 Pour lancer tous les symboles: python scripts/sync_data_smart.py --full"
        )

    logger.info("=" * 80)
    logger.info("🎉 SYNCHRONISATION TERMINÉE")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Synchroniser tous les symboles (sinon test avec BTCUSDC)",
    )

    args = parser.parse_args()

    try:
        main(test_mode=not args.full)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interruption utilisateur")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
