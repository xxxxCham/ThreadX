"""
ThreadX Data Manager - Scanner de données locales
Découverte et catalogage des données d'indicateurs existantes
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
import logging

from ..models import (
    DataCatalog,
    SymbolData,
    TimeframeData,
    IndicatorFile,
    DataQuality,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


class LocalDataScanner:
    """Scanner pour découvrir les données d'indicateurs locales"""

    def __init__(self):
        self.supported_extensions = {".parquet", ".csv", ".feather"}
        self.known_indicators = {
            "atr",
            "bollinger",
            "ema",
            "sma",
            "rsi",
            "macd",
            "stochastic",
            "williams",
            "adx",
            "cci",
            "mfi",
        }

    def scan_indicators_db(self, paths: List[str]) -> DataCatalog:
        """
        Scan les répertoires indicators_db pour découvrir la structure

        Args:
            paths: Liste des chemins à scanner (e.g., ["g:\\indicators_db", "i:\\indicators_db"])

        Returns:
            DataCatalog avec toutes les données découvertes
        """
        logger.info(f"Début du scan de {len(paths)} chemins...")

        catalog = DataCatalog(
            root_paths=[Path(p) for p in paths],
            symbols={},
            unique_symbols=set(),
            unique_timeframes=set(),
            unique_indicators=set(),
        )

        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"Chemin inexistant: {path}")
                continue

            logger.info(f"Scan de {path}...")
            self._scan_path(path, catalog)

        # Calcul des statistiques finales
        self._calculate_catalog_stats(catalog)

        logger.info(
            f"Scan terminé: {catalog.total_files} fichiers, "
            f"{len(catalog.unique_symbols)} symboles, "
            f"{catalog.size_mb:.1f} MB"
        )

        return catalog

    def _scan_path(self, root_path: Path, catalog: DataCatalog) -> None:
        """Scan récursif d'un chemin racine"""
        try:
            for item in root_path.iterdir():
                if item.is_dir():
                    # Structure attendue: indicators_db/SYMBOL/TIMEFRAME/
                    if self._looks_like_symbol_dir(item):
                        self._scan_symbol_directory(item, catalog)
                    else:
                        # Continuer la recherche récursive
                        self._scan_path(item, catalog)

        except PermissionError:
            logger.warning(f"Permission refusée: {root_path}")
        except Exception as e:
            logger.error(f"Erreur lors du scan de {root_path}: {e}")

    def _looks_like_symbol_dir(self, path: Path) -> bool:
        """Détermine si un répertoire ressemble à un répertoire de symbole"""
        # Heuristiques:
        # - Nom court et majuscules (e.g., ETHUSDC, BTCUSDT)
        # - Contient des sous-répertoires de timeframes
        name = path.name.upper()

        if not (3 <= len(name) <= 12):  # Longueur raisonnable
            return False

        if not name.replace("/", "").replace("_", "").isalnum():
            return False

        # Vérifier s'il y a des sous-répertoires qui ressemblent à des timeframes
        try:
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            timeframe_patterns = {
                "1m",
                "3m",
                "5m",
                "15m",
                "30m",
                "1h",
                "2h",
                "4h",
                "6h",
                "8h",
                "12h",
                "1d",
                "1w",
            }
            return any(subdir in timeframe_patterns for subdir in subdirs)
        except:
            return False

    def _scan_symbol_directory(self, symbol_path: Path, catalog: DataCatalog) -> None:
        """Scan d'un répertoire de symbole"""
        symbol = symbol_path.name.upper()
        logger.debug(f"Scan symbole: {symbol}")

        if symbol not in catalog.symbols:
            catalog.symbols[symbol] = SymbolData(symbol=symbol, timeframes={})
            catalog.unique_symbols.add(symbol)

        symbol_data = catalog.symbols[symbol]

        # Scanner chaque timeframe
        for tf_path in symbol_path.iterdir():
            if tf_path.is_dir():
                timeframe = tf_path.name.lower()
                logger.debug(f"Scan timeframe: {symbol}/{timeframe}")

                self._scan_timeframe_directory(tf_path, symbol_data, timeframe, catalog)
                catalog.unique_timeframes.add(timeframe)

    def _scan_timeframe_directory(
        self,
        tf_path: Path,
        symbol_data: SymbolData,
        timeframe: str,
        catalog: DataCatalog,
    ) -> None:
        """Scan d'un répertoire de timeframe"""
        if timeframe not in symbol_data.timeframes:
            symbol_data.timeframes[timeframe] = TimeframeData(
                timeframe=timeframe, indicators={}
            )

        tf_data = symbol_data.timeframes[timeframe]

        # Scanner tous les fichiers d'indicateurs
        for file_path in tf_path.iterdir():
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            ):
                indicator_file = self._analyze_indicator_file(
                    file_path, symbol_data.symbol, timeframe
                )
                if indicator_file:
                    # Ajouter au catalogue
                    indicator = indicator_file.indicator
                    if indicator not in tf_data.indicators:
                        tf_data.indicators[indicator] = []
                    tf_data.indicators[indicator].append(indicator_file)

                    # Statistiques globales
                    catalog.unique_indicators.add(indicator)
                    catalog.total_files += 1
                    catalog.total_size_bytes += indicator_file.size_bytes

    def _analyze_indicator_file(
        self, file_path: Path, symbol: str, timeframe: str
    ) -> Optional[IndicatorFile]:
        """Analyse un fichier d'indicateur individuel"""
        try:
            # Analyse du nom de fichier pour extraire indicateur et paramètres
            indicator, parameters = self._parse_filename(file_path.name)
            if not indicator:
                logger.debug(f"Impossible de parser: {file_path.name}")
                return None

            # Métadonnées du fichier
            stat = file_path.stat()
            size_bytes = stat.st_size
            modified_time = datetime.fromtimestamp(stat.st_mtime)

            # Analyse rapide du contenu (optionnelle pour performance)
            row_count, date_range, columns = self._quick_content_analysis(file_path)

            return IndicatorFile(
                path=file_path,
                symbol=symbol,
                timeframe=timeframe,
                indicator=indicator,
                parameters=parameters,
                size_bytes=size_bytes,
                modified_time=modified_time,
                row_count=row_count,
                date_range=date_range,
                columns=columns,
                quality=DataQuality.PENDING,  # Sera déterminé par la validation
            )

        except Exception as e:
            logger.warning(f"Erreur analyse {file_path}: {e}")
            return None

    def _parse_filename(self, filename: str) -> tuple[Optional[str], Dict[str, any]]:
        """
        Parse un nom de fichier pour extraire indicateur et paramètres

        Exemples:
        - atr_p14.parquet -> ("atr", {"period": 14})
        - bollinger_p20_s2.0.parquet -> ("bollinger", {"period": 20, "sigma": 2.0})
        - ema_p50.csv -> ("ema", {"period": 50})
        """
        # Enlever l'extension
        name_no_ext = filename.lower()
        for ext in self.supported_extensions:
            if name_no_ext.endswith(ext):
                name_no_ext = name_no_ext[: -len(ext)]
                break

        parts = name_no_ext.split("_")
        if not parts:
            return None, {}

        # Premier élément = indicateur
        indicator = parts[0]
        if indicator not in self.known_indicators:
            # Essayer de détecter automatiquement
            logger.debug(f"Indicateur inconnu: {indicator}")

        # Parser les paramètres
        parameters = {}
        for part in parts[1:]:
            param_name, param_value = self._parse_parameter(part)
            if param_name:
                parameters[param_name] = param_value

        return indicator, parameters

    def _parse_parameter(self, param_str: str) -> tuple[Optional[str], any]:
        """
        Parse un paramètre individuel

        Exemples:
        - p14 -> ("period", 14)
        - s2.0 -> ("sigma", 2.0)
        - len21 -> ("length", 21)
        """
        if not param_str:
            return None, None

        # Mapping des préfixes connus
        prefix_mapping = {
            "p": "period",
            "s": "sigma",
            "len": "length",
            "fast": "fast_period",
            "slow": "slow_period",
            "signal": "signal_period",
        }

        # Chercher un préfixe connu
        for prefix, full_name in prefix_mapping.items():
            if param_str.startswith(prefix):
                value_str = param_str[len(prefix) :]
                try:
                    # Essayer int puis float
                    if "." in value_str:
                        return full_name, float(value_str)
                    else:
                        return full_name, int(value_str)
                except ValueError:
                    return full_name, value_str  # Garder comme string

        # Fallback: essayer de parser comme nombre
        try:
            if "." in param_str:
                return "value", float(param_str)
            else:
                return "value", int(param_str)
        except ValueError:
            return "param", param_str

    def _quick_content_analysis(
        self, file_path: Path
    ) -> tuple[Optional[int], Optional[tuple], Optional[List[str]]]:
        """Analyse rapide du contenu d'un fichier (optionnelle)"""
        try:
            if file_path.suffix.lower() == ".parquet":
                # Pour parquet, on peut obtenir les métadonnées sans charger tout
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(file_path)

                # Nombre de lignes
                row_count = parquet_file.metadata.num_rows

                # Colonnes
                columns = [col.name for col in parquet_file.schema]

                # Pour la plage de dates, il faudrait lire les données
                # On skip pour la performance
                date_range = None

                return row_count, date_range, columns

        except Exception as e:
            logger.debug(f"Impossible d'analyser le contenu de {file_path}: {e}")

        return None, None, None

    def _calculate_catalog_stats(self, catalog: DataCatalog) -> None:
        """Calcule les statistiques finales du catalogue"""
        catalog.total_files = 0
        catalog.total_size_bytes = 0

        for symbol_data in catalog.symbols.values():
            symbol_files = 0
            symbol_size = 0

            for tf_data in symbol_data.timeframes.values():
                tf_files = 0
                tf_size = 0

                for files_list in tf_data.indicators.values():
                    for file in files_list:
                        tf_files += 1
                        tf_size += file.size_bytes

                tf_data.file_count = tf_files
                tf_data.size_bytes = tf_size
                symbol_files += tf_files
                symbol_size += tf_size

            symbol_data.total_files = symbol_files
            symbol_data.total_size_bytes = symbol_size
            catalog.total_files += symbol_files
            catalog.total_size_bytes += symbol_size


def create_demo_catalog() -> DataCatalog:
    """Crée un catalogue de démonstration pour les tests"""
    catalog = DataCatalog(
        root_paths=[Path("g:/indicators_db")],
        symbols={
            "ETHUSDC": SymbolData(
                symbol="ETHUSDC",
                timeframes={
                    "5m": TimeframeData(
                        timeframe="5m",
                        indicators={
                            "atr": [
                                IndicatorFile(
                                    path=Path(
                                        "g:/indicators_db/ETHUSDC/5m/atr_p14.parquet"
                                    ),
                                    symbol="ETHUSDC",
                                    timeframe="5m",
                                    indicator="atr",
                                    parameters={"period": 14},
                                    size_bytes=1024 * 1024,  # 1MB
                                    row_count=10000,
                                )
                            ],
                            "bollinger": [
                                IndicatorFile(
                                    path=Path(
                                        "g:/indicators_db/ETHUSDC/5m/bollinger_p20_s2.0.parquet"
                                    ),
                                    symbol="ETHUSDC",
                                    timeframe="5m",
                                    indicator="bollinger",
                                    parameters={"period": 20, "sigma": 2.0},
                                    size_bytes=2 * 1024 * 1024,  # 2MB
                                    row_count=10000,
                                )
                            ],
                        },
                    )
                },
            )
        },
        unique_symbols={"ETHUSDC"},
        unique_timeframes={"5m"},
        unique_indicators={"atr", "bollinger"},
    )

    # Calculer les stats
    scanner = LocalDataScanner()
    scanner._calculate_catalog_stats(catalog)

    return catalog
