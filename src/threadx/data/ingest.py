"""
ThreadX Data Ingestion - Phase 2 Extension
API principale pour ingestion avec système "1m truth".

Architecture:
- Source canonique = données 1m uniquement
- Tous les timeframes dérivés (3m, 5m, 15m, 1h, 3h) via resample depuis 1m
- Sanity checks 1h/3h optionnels (non bloquants, pour validation uniquement)
- Banque locale prioritaire (lecture existing + complétion gaps seulement)
- Idempotence totale (re-run safe)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np

from ..config import get_settings, Settings

# Logger simple si utils.logging_utils pas disponible
try:
    from ..utils.logging_utils import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

from .io import read_frame, write_frame, normalize_ohlcv, DataNotFoundError
from .resample import resample_from_1m, TimeframeError
from .registry import dataset_exists, scan_symbols, file_checksum
from .legacy_adapter import LegacyAdapter, IngestionError, APIError


class IngestionManager:
    """
    Gestionnaire principal d'ingestion ThreadX avec système "1m truth".

    Principe:
    1. Banque locale prioritaire (scan datasets existants)
    2. Téléchargement 1m seulement pour gaps manquants
    3. Resample vers tous timeframes depuis 1m truth
    4. Sanity checks 1h/3h optionnels (validation, non source)
    5. Idempotence garantie
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.adapter = LegacyAdapter(settings)

        # Chemins relatifs depuis TOML
        self.raw_1m_path = Path(self.settings.DATA_ROOT) / "raw" / "1m"
        self.processed_path = Path(self.settings.DATA_ROOT) / "processed"

        # Configuration tolerances (depuis TOML ou defaults)
        self.verification_config = {
            "atol_price": getattr(self.settings, "VERIFY_ATOL_PRICE", 1e-8),
            "rtol_price": getattr(self.settings, "VERIFY_RTOL_PRICE", 1e-8),
            "atol_vol": getattr(self.settings, "VERIFY_ATOL_VOL", 1e-8),
            "enabled_slow_tfs": getattr(self.settings, "VERIFY_SLOW_TFS", ["1h", "3h"]),
        }

        # Thread safety pour UI non-bloquante
        self._lock = threading.RLock()

        # Stats session
        self.session_stats = {
            "symbols_processed": 0,
            "files_downloaded": 0,
            "files_resampled": 0,
            "gaps_filled": 0,
            "verification_warnings": 0,
        }

    def download_ohlcv_1m(
        self, symbol: str, start: datetime, end: datetime, *, force: bool = False
    ) -> pd.DataFrame:
        """
        Télécharge OHLCV 1m (source canonique) avec gestion banque locale prioritaire.

        Args:
            symbol: Symbole trading (ex. "BTCUSDT")
            start: Date début (UTC)
            end: Date fin (UTC)
            force: Forcer téléchargement même si données locales existent

        Returns:
            DataFrame 1m normalisé (index UTC, colonnes OHLCV float64)

        Raises:
            IngestionError: En cas d'échec téléchargement/normalisation
        """
        with self._lock:
            logger.info(f"Processing {symbol} 1m: {start.date()} → {end.date()}")

            parquet_path = self.raw_1m_path / f"{symbol}.parquet"

            # 1. Vérification banque locale (prioritaire)
            existing_df = None
            if parquet_path.exists() and not force:
                try:
                    existing_df = read_frame(parquet_path)
                    logger.info(
                        f"Local data found: {len(existing_df)} rows, {existing_df.index.min()} → {existing_df.index.max()}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to read local data: {e}")

            # 2. Détermination plages manquantes
            download_ranges = self._calculate_missing_ranges(existing_df, start, end)

            if not download_ranges and existing_df is not None:
                logger.info("All requested data available locally, no download needed")
                # Extraction de la plage demandée
                mask = (existing_df.index >= start) & (existing_df.index <= end)
                return existing_df[mask].copy()

            # 3. Téléchargement segments manquants seulement
            new_segments = []
            for dl_start, dl_end in download_ranges:
                logger.info(
                    f"Downloading missing segment: {dl_start.date()} → {dl_end.date()}"
                )
                try:
                    raw_klines = self.adapter.fetch_klines_1m(symbol, dl_start, dl_end)
                    if raw_klines:
                        segment_df = self.adapter.json_to_dataframe(raw_klines)
                        new_segments.append(segment_df)
                        self.session_stats["files_downloaded"] += 1
                except APIError as e:
                    logger.error(f"Download failed for {symbol}: {e}")
                    if existing_df is None:
                        raise IngestionError(f"No local data and download failed: {e}")

            # 4. Fusion avec données existantes (idempotent)
            final_df = self._merge_dataframes_idempotent(existing_df, new_segments)

            # 5. Sauvegarde mise à jour
            if new_segments:  # Seulement si nouvelles données
                self.raw_1m_path.mkdir(parents=True, exist_ok=True)
                # Autoriser l'écrasement du fichier existant lors de la mise à jour
                write_frame(final_df, parquet_path, overwrite=True)
                logger.info(f"Saved updated {symbol} 1m data: {len(final_df)} rows")

            # 6. Extraction plage demandée
            # Filtrer selon date range (attention timezone)
            try:

                def to_utc_timestamp(x):
                    ts = pd.to_datetime(x)
                    # pd.Timestamp has .tz attribute in newer pandas
                    if getattr(ts, "tz", None) is None:
                        ts = ts.tz_localize("UTC")
                    else:
                        ts = ts.tz_convert("UTC")
                    return ts

                start_dt = to_utc_timestamp(start)
                end_dt = to_utc_timestamp(end)
            except Exception:
                # Fallback simple conversion
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)

            mask = (final_df.index >= start_dt) & (final_df.index <= end_dt)
        result = final_df[mask].copy()

        logger.info(f"Returning {len(result)} rows for requested period")
        return result

    def resample_from_1m(self, df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample depuis 1m truth vers timeframe cible (wrapper ThreadX).

        Args:
            df_1m: DataFrame 1m source
            timeframe: Timeframe cible (3m, 5m, 15m, 1h, 3h, ...)

        Returns:
            DataFrame resamplé avec agrégations canoniques

        Raises:
            TimeframeError: Si timeframe invalide
        """
        if df_1m.empty:
            logger.warning("Empty 1m DataFrame for resampling")
            return pd.DataFrame()

        try:
            # Utilisation du module resample existant Phase 2
            result = resample_from_1m(df_1m, timeframe)
            logger.debug(f"Resampled {len(df_1m)} → {len(result)} rows ({timeframe})")
            return result
        except Exception as e:
            raise TimeframeError(f"Resample failed for {timeframe}: {e}")

    def verify_resample_consistency(
        self,
        df_1m: pd.DataFrame,
        df_slow: pd.DataFrame,
        timeframe: str,
        *,
        atol_price: Optional[float] = None,
        rtol_price: Optional[float] = None,
        atol_vol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Sanity check: compare resample 1m vs téléchargement direct slow TF.

        Args:
            df_1m: DataFrame 1m truth
            df_slow: DataFrame téléchargé direct (1h/3h)
            timeframe: Timeframe slow testé
            atol_price/rtol_price/atol_vol: Tolérances (défaut depuis config)

        Returns:
            Rapport {"ok": bool, "anomalies": list, "stats": dict}
        """
        # Tolérances depuis config si non spécifiées
        atol_price = atol_price or self.verification_config["atol_price"]
        rtol_price = rtol_price or self.verification_config["rtol_price"]
        atol_vol = atol_vol or self.verification_config["atol_vol"]

        if df_1m.empty or df_slow.empty:
            return {"ok": False, "anomalies": ["Empty input DataFrames"], "stats": {}}

        try:
            # Resample 1m → timeframe pour comparaison
            df_resampled = self.resample_from_1m(df_1m, timeframe)

            if df_resampled.empty:
                return {
                    "ok": False,
                    "anomalies": ["Empty resampled DataFrame"],
                    "stats": {},
                }

            # Intersection timestamps (alignement)
            common_idx = df_resampled.index.intersection(df_slow.index)

            if len(common_idx) == 0:
                return {
                    "ok": False,
                    "anomalies": [
                        "No common timestamps between resampled and slow data"
                    ],
                    "stats": {
                        "resampled_count": len(df_resampled),
                        "slow_count": len(df_slow),
                    },
                }

            df_r_common = df_resampled.loc[common_idx]
            df_s_common = df_slow.loc[common_idx]

            anomalies = []

            # Vérification colonnes OHLCV
            for col in ["open", "high", "low", "close"]:
                if col in df_r_common.columns and col in df_s_common.columns:
                    # Pour open/close: égalité exacte attendue (première/dernière valeur)
                    if col in ["open", "close"]:
                        diff = np.abs(df_r_common[col] - df_s_common[col])
                        max_diff = diff.max()
                        if max_diff > atol_price:
                            anomalies.append(
                                f"{col}: max diff {max_diff:.2e} > {atol_price:.2e}"
                            )

                    # Pour high/low: tolérance relative (agrégation numérique)
                    else:
                        if not np.allclose(
                            df_r_common[col],
                            df_s_common[col],
                            atol=atol_price,
                            rtol=rtol_price,
                        ):
                            diff = np.abs(df_r_common[col] - df_s_common[col])
                            max_diff = diff.max()
                            anomalies.append(f"{col}: max diff {max_diff:.2e} > tol")

            # Vérification volume (agrégation sum)
            if "volume" in df_r_common.columns and "volume" in df_s_common.columns:
                if not np.allclose(
                    df_r_common["volume"], df_s_common["volume"], atol=atol_vol
                ):
                    diff = np.abs(df_r_common["volume"] - df_s_common["volume"])
                    max_diff = diff.max()
                    anomalies.append(
                        f"volume: max diff {max_diff:.2e} > {atol_vol:.2e}"
                    )

            # Stats rapport
            stats = {
                "common_timestamps": len(common_idx),
                "resampled_total": len(df_resampled),
                "slow_total": len(df_slow),
                "coverage_ratio": (
                    len(common_idx) / len(df_slow) if len(df_slow) > 0 else 0.0
                ),
            }

            report = {"ok": len(anomalies) == 0, "anomalies": anomalies, "stats": stats}

            if anomalies:
                self.session_stats["verification_warnings"] += len(anomalies)
                logger.warning(
                    f"Consistency check {timeframe}: {len(anomalies)} anomalies"
                )
                for anomaly in anomalies:
                    logger.warning(f"  - {anomaly}")
            else:
                logger.info(
                    f"Consistency check {timeframe}: OK ({len(common_idx)} points)"
                )

            return report

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"ok": False, "anomalies": [f"Verification error: {e}"], "stats": {}}

    def detect_and_fill_gaps_1m(
        self, df_1m: pd.DataFrame, *, max_gap_ratio: float = 0.05
    ) -> pd.DataFrame:
        """
        Détecte et comble les gaps 1m conservativement (wrapper adapter).

        Args:
            df_1m: DataFrame 1m avec potentiels gaps
            max_gap_ratio: Ratio max gaps à combler (défaut 5%)

        Returns:
            DataFrame avec gaps < 5% comblés, gaps > 5% laissés + WARNING
        """
        if df_1m.empty:
            return df_1m

        gaps_before = len(self.adapter.detect_gaps_1m(df_1m))
        result = self.adapter.fill_gaps_conservative(df_1m, max_gap_ratio)
        gaps_after = len(self.adapter.detect_gaps_1m(result))

        filled_count = gaps_before - gaps_after
        if filled_count > 0:
            self.session_stats["gaps_filled"] += filled_count
            logger.info(f"Filled {filled_count} small gaps (< {max_gap_ratio:.1%})")

        return result

    def update_assets_batch(
        self,
        symbols: List[str],
        timeframes: List[str],
        start: datetime,
        end: datetime,
        *,
        force: bool = False,
        enable_verification: bool = True,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Mise à jour batch avec système "1m truth" + resample + sanity checks.

        Args:
            symbols: Liste symboles à traiter
            timeframes: Liste timeframes à générer (3m, 5m, 15m, 1h, 3h, ...)
            start/end: Période (UTC)
            force: Forcer re-téléchargement
            enable_verification: Activer sanity checks 1h/3h
            max_workers: Parallélisme

        Returns:
            Rapport complet {"summary": {...}, "details": [...], "errors": [...]}
        """
        logger.info(f"Batch update: {len(symbols)} symbols × {len(timeframes)} TFs")

        with self._lock:
            # Reset stats session
            self.session_stats = {k: 0 for k in self.session_stats}

        results = {"summary": {}, "details": [], "errors": []}

        # Traitement en parallèle par symbole (1m truth + resample)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for symbol in symbols:
                future = executor.submit(
                    self._process_symbol_complete,
                    symbol,
                    timeframes,
                    start,
                    end,
                    force,
                    enable_verification,
                )
                futures[future] = symbol

            # Collecte résultats
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    symbol_result = future.result()
                    results["details"].append(symbol_result)

                    if symbol_result["success"]:
                        self.session_stats["symbols_processed"] += 1
                    else:
                        results["errors"].extend(symbol_result.get("errors", []))

                except Exception as e:
                    error_msg = f"Symbol {symbol} failed: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)

        # Résumé final
        results["summary"] = {
            "symbols_requested": len(symbols),
            "symbols_processed": self.session_stats["symbols_processed"],
            "timeframes_requested": len(timeframes),
            "files_downloaded": self.session_stats["files_downloaded"],
            "files_resampled": self.session_stats["files_resampled"],
            "gaps_filled": self.session_stats["gaps_filled"],
            "verification_warnings": self.session_stats["verification_warnings"],
            "total_errors": len(results["errors"]),
        }

        logger.info(f"Batch complete: {results['summary']}")
        return results

    def _calculate_missing_ranges(
        self,
        existing_df: Optional[pd.DataFrame],
        requested_start: datetime,
        requested_end: datetime,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Calcule les plages manquantes à télécharger (banque locale prioritaire).

        Returns:
            Liste des plages [(start, end), ...] à télécharger
        """
        if existing_df is None or existing_df.empty:
            return [(requested_start, requested_end)]

        # Vérification couverture existante
        data_start = existing_df.index.min().to_pydatetime()
        data_end = existing_df.index.max().to_pydatetime()

        # Normaliser requested_start/end pour éviter comparaison tz-aware vs tz-naive
        try:
            import pytz

            if getattr(data_start, "tzinfo", None) is not None:
                # data timestamps are tz-aware -> ensure requested_* are UTC-aware
                if getattr(requested_start, "tzinfo", None) is None:
                    requested_start = pytz.UTC.localize(requested_start)
                if getattr(requested_end, "tzinfo", None) is None:
                    requested_end = pytz.UTC.localize(requested_end)
            else:
                # data timestamps are naive -> strip tz from requested_* if present
                if getattr(requested_start, "tzinfo", None) is not None:
                    requested_start = requested_start.replace(tzinfo=None)
                if getattr(requested_end, "tzinfo", None) is not None:
                    requested_end = requested_end.replace(tzinfo=None)
        except Exception:
            # Si pytz non disponible, poursuivre sans conversion (risque minime)
            pass

        missing_ranges = []

        # Plage avant (si demandée)
        if requested_start < data_start:
            missing_ranges.append((requested_start, min(data_start, requested_end)))

        # Plage après (si demandée)
        if requested_end > data_end:
            missing_ranges.append((max(data_end, requested_start), requested_end))

        # TODO: Détection gaps internes (version future)
        # Pour l'instant: continuité supposée entre data_start et data_end

        return missing_ranges

    def _merge_dataframes_idempotent(
        self, existing: Optional[pd.DataFrame], new_segments: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Fusion idempotente: existing + new_segments sans doublons.

        Returns:
            DataFrame fusionné, index UTC trié unique
        """
        all_dfs = []

        if existing is not None and not existing.empty:
            all_dfs.append(existing)

        all_dfs.extend([seg for seg in new_segments if not seg.empty])

        if not all_dfs:
            return pd.DataFrame()

        # Concat + déduplication + tri
        merged = pd.concat(all_dfs, ignore_index=False)
        merged = merged[~merged.index.duplicated(keep="first")]
        merged = merged.sort_index()

        return merged

    def _process_symbol_complete(
        self,
        symbol: str,
        timeframes: List[str],
        start: datetime,
        end: datetime,
        force: bool,
        enable_verification: bool,
    ) -> Dict[str, Any]:
        """
        Traitement complet d'un symbole: 1m truth → resample → verification.

        Returns:
            Rapport {"symbol": str, "success": bool, "timeframes": [...], "errors": [...]}
        """
        report = {"symbol": symbol, "success": False, "timeframes": [], "errors": []}

        try:
            # 1. Download/update 1m truth
            df_1m = self.download_ohlcv_1m(symbol, start, end, force=force)

            if df_1m.empty:
                report["errors"].append(f"No 1m data available for {symbol}")
                return report

            # 2. Fill gaps conservatively
            df_1m_filled = self.detect_and_fill_gaps_1m(df_1m)

            # 3. Resample to all requested timeframes
            for tf in timeframes:
                try:
                    df_output = None
                    if tf == "1m":
                        # 1m = source directe
                        df_output = df_1m_filled
                        output_path = self.processed_path / tf / f"{symbol}.parquet"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        write_frame(df_1m_filled, output_path)
                    else:
                        # Resample depuis 1m truth
                        df_resampled = self.resample_from_1m(df_1m_filled, tf)
                        df_output = df_resampled

                        if not df_resampled.empty:
                            output_path = self.processed_path / tf / f"{symbol}.parquet"
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            write_frame(df_resampled, output_path)
                            self.session_stats["files_resampled"] += 1

                    report["timeframes"].append(
                        {
                            "tf": tf,
                            "rows": len(df_output) if df_output is not None else 0,
                            "status": "ok",
                        }
                    )

                except Exception as e:
                    error_msg = f"Resample {tf} failed: {e}"
                    report["errors"].append(error_msg)
                    report["timeframes"].append(
                        {"tf": tf, "rows": 0, "status": "error"}
                    )

            # 4. Optional verification avec téléchargement slow TFs
            if enable_verification:
                slow_tfs = [
                    tf
                    for tf in self.verification_config["enabled_slow_tfs"]
                    if tf in timeframes
                ]

                for slow_tf in slow_tfs:
                    try:
                        # Téléchargement direct slow TF pour comparaison
                        raw_slow = self.adapter.fetch_klines_1m(
                            symbol, start, end, interval=slow_tf
                        )
                        df_slow = self.adapter.json_to_dataframe(raw_slow)

                        if not df_slow.empty:
                            verify_report = self.verify_resample_consistency(
                                df_1m_filled, df_slow, slow_tf
                            )

                            if not verify_report["ok"]:
                                report["errors"].extend(
                                    [
                                        f"Verification {slow_tf}: {a}"
                                        for a in verify_report["anomalies"]
                                    ]
                                )

                    except Exception as e:
                        logger.warning(f"Verification {slow_tf} skipped: {e}")

            report["success"] = (
                len([tf for tf in report["timeframes"] if tf["status"] == "ok"]) > 0
            )

        except Exception as e:
            report["errors"].append(f"Symbol processing failed: {e}")

        return report


# API publique simplifiée pour UI/autres modules
def download_ohlcv_1m(
    symbol: str, start: datetime, end: datetime, *, force: bool = False
) -> pd.DataFrame:
    """API publique: téléchargement 1m truth."""
    manager = IngestionManager()
    return manager.download_ohlcv_1m(symbol, start, end, force=force)


def resample_from_1m_api(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """API publique: resample depuis 1m."""
    manager = IngestionManager()
    return manager.resample_from_1m(df_1m, timeframe)


def update_assets_batch(
    symbols: List[str], timeframes: List[str], start: datetime, end: datetime, **kwargs
) -> Dict[str, Any]:
    """API publique: mise à jour batch."""
    manager = IngestionManager()
    return manager.update_assets_batch(symbols, timeframes, start, end, **kwargs)
