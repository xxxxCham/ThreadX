"""
ThreadX Legacy Adapter - Phase 2 Data Extension
Adaptation sélective du code legacy unified_data_historique_with_indicators.py
pour téléchargement/normalisation OHLCV avec architecture ThreadX.

Principes d'adaptation:
- Suppression variables d'environnement → Settings/TOML
- Chemins absolus → chemins relatifs
- UI/CLI → ThreadX Phase 8 UI
- Conservation logique métier (retry, normalisation, gaps)
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

import pandas as pd
import numpy as np
import requests

from ..config import get_settings, Settings
from .io import normalize_ohlcv, write_frame
from .registry import file_checksum

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


class IngestionError(Exception):
    """Erreur lors de l'ingestion de données."""

    pass


class APIError(Exception):
    """Erreur API Binance."""

    pass


class LegacyAdapter:
    """
    Adapter pour réutiliser les fonctions legacy de téléchargement/normalisation.

    Remplace les dépendances env vars par Settings TOML,
    paths absolus par chemins relatifs, logging par logger ThreadX.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.binance_endpoint = "https://api.binance.com/api/v3/klines"
        self.binance_limit = 1000
        self.max_retries = 3
        self.backoff_factor = 2.0
        self.request_timeout = 10

        # Chemins relatifs depuis TOML
        self.raw_json_path = Path(self.settings.DATA_ROOT) / "raw" / "json"
        self.processed_path = Path(self.settings.DATA_ROOT) / "processed"

        # Création répertoires
        self.raw_json_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def fetch_klines_1m(
        self, symbol: str, start: datetime, end: datetime, interval: str = "1m"
    ) -> List[Dict[str, Any]]:
        """
        Télécharge les klines depuis l'API Binance (adaptation legacy fetch_klines).

        Args:
            symbol: Symbole trading (ex. "BTCUSDT")
            start: Date début (UTC)
            end: Date fin (UTC)
            interval: Interval (fixé à "1m" pour 1m truth)

        Returns:
            Liste raw des klines JSON

        Raises:
            APIError: En cas d'échec API après retries
        """
        all_candles: List[Dict[str, Any]] = []
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        current_ms = start_ms

        logger.info(
            f"Téléchargement {symbol} {interval}: {start.date()} → {end.date()}"
        )

        while current_ms < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_ms,
                "endTime": end_ms,
                "limit": self.binance_limit,
            }

            data = None
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")
                    response = requests.get(
                        self.binance_endpoint,
                        params=params,
                        timeout=self.request_timeout,
                    )

                    if response.status_code == 429:
                        raise requests.exceptions.RequestException("Rate limited")

                    response.raise_for_status()
                    data = response.json()

                    if isinstance(data, dict) and "code" in data:
                        raise APIError(f"Binance API error: {data}")

                    break

                except Exception as e:
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries}: {e}")
                    if attempt == self.max_retries - 1:
                        raise APIError(f"Failed after {self.max_retries} retries: {e}")

                    delay = self.backoff_factor**attempt
                    time.sleep(delay)

            if not data:
                logger.warning("No data received, stopping download")
                break

            all_candles.extend(data)
            logger.debug(f"Downloaded {len(data)} candles, total: {len(all_candles)}")

            # Progression pour éviter boucle infinie
            if data:
                last_time = data[-1][0]  # timestamp de la dernière bougie
                current_ms = last_time + 1
            else:
                break

        logger.info(f"Download complete: {len(all_candles)} candles pour {symbol}")
        return all_candles

    def json_to_dataframe(
        self, raw_klines: List[Dict[str, Any]], symbol: str = None
    ) -> pd.DataFrame:
        """
        Convertit les klines JSON bruts en DataFrame normalisé (adaptation _json_to_df).

        Args:
            raw_klines: Données brutes JSON de l'API Binance
            symbol: Symbole trading (ex. "BTCUSDT") - ajouté comme colonne

        Returns:
            DataFrame normalisé avec index DatetimeIndex UTC, colonnes OHLCV + symbol en float64

        Raises:
            IngestionError: Si conversion échoue
        """
        if not raw_klines:
            logger.warning("Empty klines data")
            return pd.DataFrame()

        try:
            # Colonnes standard Binance API
            columns = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ]

            df = pd.DataFrame(raw_klines, columns=columns)

            # Fix timestamps (adaptation _fix_timestamp_conversion)
            df = self._fix_timestamp_conversion(df, "open_time")

            # Conversion types OHLCV
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Index DatetimeIndex UTC trié unique
            df = df.set_index("open_time").sort_index()
            df = df[~df.index.duplicated(keep="first")]  # Suppression doublons

            # Sélection colonnes canoniques
            result = df[["open", "high", "low", "close", "volume"]].copy()

            # Validation schéma OHLCV (sans symbol pour l'instant)
            result = normalize_ohlcv(result)

            # Ajout de la colonne symbol APRÈS normalisation
            if symbol:
                result["symbol"] = symbol

            logger.info(
                f"DataFrame created: {len(result)} rows, {result.index.min()} → {result.index.max()}"
            )
            return result

        except Exception as e:
            raise IngestionError(f"Failed to convert JSON to DataFrame: {e}")

    def _fix_timestamp_conversion(
        self, df: pd.DataFrame, timestamp_col: str = "open_time"
    ) -> pd.DataFrame:
        """
        Normalise les timestamps avec gestion ms/s et forçage UTC (adaptation legacy).

        Args:
            df: DataFrame avec colonne timestamp
            timestamp_col: Nom de la colonne timestamp

        Returns:
            DataFrame avec timestamps normalisés en UTC
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Column {timestamp_col} not found")
            return df

        try:
            # Copie pour éviter SettingWithCopyWarning
            df_copy = df.copy()

            # Détection automatique ms vs secondes
            sample_val = df_copy[timestamp_col].iloc[0] if len(df_copy) > 0 else 0

            if sample_val > 1e12:  # Millisecondes (> ~2001)
                df_copy[timestamp_col] = pd.to_datetime(
                    df_copy[timestamp_col], unit="ms", utc=True
                )
            else:  # Secondes
                df_copy[timestamp_col] = pd.to_datetime(
                    df_copy[timestamp_col], unit="s", utc=True
                )

            # Forçage UTC si pas déjà défini
            if df_copy[timestamp_col].dt.tz is None:
                df_copy[timestamp_col] = df_copy[timestamp_col].dt.tz_localize("UTC")
            elif df_copy[timestamp_col].dt.tz != pd.Timestamp.now(tz="UTC").tz:
                df_copy[timestamp_col] = df_copy[timestamp_col].dt.tz_convert("UTC")

            logger.debug(f"Timestamps normalized to UTC: {timestamp_col}")
            return df_copy

        except Exception as e:
            logger.error(f"Timestamp conversion failed: {e}")
            return df

    def detect_gaps_1m(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Détecte les trous dans une série 1m (adaptation detect_missing).

        Args:
            df: DataFrame avec index DatetimeIndex 1m

        Returns:
            Liste des gaps [(start, end), ...]
        """
        if len(df) < 2:
            return []

        # Génération série continue 1m
        full_range = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq="1min", tz="UTC"
        )

        # Identification des timestamps manquants
        missing = full_range.difference(pd.DatetimeIndex(df.index))

        if len(missing) == 0:
            return []

        # Regroupement des trous consécutifs
        gaps = []
        if len(missing) > 0:
            # Détection des plages de timestamps consécutifs manquants
            missing_sorted = missing.sort_values()
            gap_start = missing_sorted[0]
            gap_end = missing_sorted[0]

            for ts in missing_sorted[1:]:
                if (ts - gap_end).total_seconds() <= 60:  # Consécutif (1min)
                    gap_end = ts
                else:
                    gaps.append((gap_start, gap_end))
                    gap_start = ts
                    gap_end = ts

            # Dernier gap
            gaps.append((gap_start, gap_end))

        logger.info(f"Detected {len(gaps)} gaps in 1m data")
        return gaps

    def fill_gaps_conservative(
        self, df: pd.DataFrame, max_gap_ratio: float = 0.05
    ) -> pd.DataFrame:
        """
        Remplit les trous < 5% avec forward fill, sinon WARNING (adaptation legacy).

        Args:
            df: DataFrame 1m avec trous
            max_gap_ratio: Ratio max de trous à remplir (défaut 5%)

        Returns:
            DataFrame avec trous comblés conservativement
        """
        gaps = self.detect_gaps_1m(df)

        if not gaps:
            return df

        total_minutes = (df.index.max() - df.index.min()).total_seconds() / 60

        df_filled = df.copy()

        for gap_start, gap_end in gaps:
            gap_minutes = (gap_end - gap_start).total_seconds() / 60 + 1
            gap_ratio = gap_minutes / total_minutes

            if gap_ratio <= max_gap_ratio:
                logger.debug(
                    f"Filling small gap {gap_start} → {gap_end} ({gap_ratio:.2%})"
                )
                # Forward fill contrôlé
                gap_range = pd.date_range(gap_start, gap_end, freq="1min", tz="UTC")
                for ts in gap_range:
                    if ts not in df_filled.index:
                        # Forward fill de la dernière valeur disponible
                        last_valid_idx = df_filled.index[df_filled.index < ts]
                        if len(last_valid_idx) > 0:
                            last_valid = df_filled.loc[last_valid_idx[-1]]
                            # Ajout nouvelle ligne avec forward fill
                            new_row = pd.DataFrame(
                                [last_valid.values],
                                columns=df_filled.columns,
                                index=[ts],
                            )
                            df_filled = pd.concat([df_filled, new_row])
            else:
                logger.warning(
                    f"Gap too large to fill: {gap_start} → {gap_end} "
                    f"({gap_ratio:.2%} > {max_gap_ratio:.2%})"
                )

        return df_filled.sort_index()
