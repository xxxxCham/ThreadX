# -*- coding: utf-8 -*-
"""
ThreadX Unified Data Historique with Indicators v1.0
Interface exactement identique au système TradXPro original
Adaptée pour ThreadX avec toutes les fonctionnalités originales
"""

from __future__ import annotations

import os
import re
import json
import time
import queue
import logging
import threading
import platform
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple, Callable
from functools import lru_cache

import pandas as pd
import requests

# Import numpy pour calculs indicateurs
try:
    import numpy as np
except ImportError:
    np = None

# Progress bar
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_):
        return it


# =========================================================
#  Tkinter: import **optionnel** (GUI si dispo, sinon CLI)
# =========================================================
TK_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
    from tkinter import ttk, messagebox, filedialog

    TK_AVAILABLE = True
except Exception:
    tk = None
    ScrolledText = None
    ttk = None

# =========================================================
#  Configuration ThreadX (adaptée de TradXPro)
# =========================================================
IS_WINDOWS = platform.system() == "Windows"

# Chemins ThreadX (structure harmonisée avec l'existant)
THREADX_ROOT = Path(__file__).parent.parent
JSON_ROOT = THREADX_ROOT / "data" / "crypto_data_json"
PARQUET_ROOT = THREADX_ROOT / "data" / "crypto_data_parquet"
INDICATORS_ROOT = THREADX_ROOT / "data" / "indicateurs_tech_data"
INDICATORS_ATR = INDICATORS_ROOT / "atr"
INDICATORS_BOLLINGER = INDICATORS_ROOT / "bollinger"
INDICATORS_REGISTRY = INDICATORS_ROOT / "registry"
CACHE_ROOT = THREADX_ROOT / "data" / "cache"
OUTPUT_DIR = THREADX_ROOT / "data" / "best_token_DataFrame"

# Chemins de fichiers
DEFAULT_JSON_PATH = THREADX_ROOT / "data" / "resultats_choix_des_100tokens.json"
DEFAULT_LOG_FILE = THREADX_ROOT / "logs" / "unified_data_historique.log"

JSON_PATH = DEFAULT_JSON_PATH
LOG_FILE = DEFAULT_LOG_FILE

# Compatibilité ancienne variable DATA_FOLDER
DATA_FOLDER = JSON_ROOT

# Pattern pour les noms de fichiers harmonisés
FILENAME_PATTERN = re.compile(r"([A-Z0-9]+)_([a-z0-9]+)_12months\.json")

# Création des répertoires ThreadX harmonisés
for _p in (
    JSON_ROOT,
    PARQUET_ROOT,
    INDICATORS_ROOT,
    INDICATORS_ATR,
    INDICATORS_BOLLINGER,
    INDICATORS_REGISTRY,
    CACHE_ROOT,
    OUTPUT_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)

JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
#  Logging + canal UI/CLI (identique TradXPro)
# =========================================================
logger = logging.getLogger("UnifiedDataHistorique")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_queue: queue.Queue[logging.LogRecord] = queue.Queue()


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue[logging.LogRecord]):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(record)


queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

# Console handler pour le mode CLI
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =========================================================
#  Paramètres opérationnels (identiques TradXPro)
# =========================================================
HISTORY_DAYS = 365
BINANCE_LIMIT = 1000
# Pattern pour fichiers existants avec format TOKEN_TIMEFRAME_12months.json
FILENAME_PATTERN = re.compile(r"^([A-Za-z0-9]+)_([0-9mhdws]+)_12months\.json$")
# Intervalles disponibles (ajout 1s pour temps réel)
INTERVALS = ["1s", "3m", "5m", "15m", "30m", "1h"]
FORCE_UPDATE = False
# Flag global pour arrêt propre
STOP_REQUESTED = False
MAX_WORKERS = max(4, (os.cpu_count() or 8) // 2)

# =========================================================
#  Utilitaires (identiques TradXPro)
# =========================================================


def interval_to_ms(interval: str) -> int:
    """Convertit un intervalle en millisecondes"""
    amount = int(re.sub(r"\D", "", interval))
    unit = re.sub(r"\d", "", interval)
    if unit == "s":
        return amount * 1_000  # secondes
    if unit == "m":
        return amount * 60_000  # minutes
    if unit == "h":
        return amount * 3_600_000  # heures
    return amount * 60_000


def detect_existing_filename_format(token: str, interval: str) -> str:
    """Détecte le format de fichier existant et retourne le nom harmonisé"""
    # Format existant : TOKEN_TIMEFRAME_12months.json
    existing_format = f"{token}_{interval}_12months.json"
    existing_path = JSON_ROOT / existing_format

    # Format nouveau erroné : TOKENUSDC_TIMEFRAME.json
    new_format = f"{token}USDC_{interval}.json"
    new_path = JSON_ROOT / new_format

    # Si le format existant existe, l'utiliser
    if existing_path.exists():
        return existing_format

    # Si le nouveau format existe, le renommer vers l'existant
    if new_path.exists():
        try:
            new_path.rename(existing_path)
            logger.info(f"Renommé {new_format} → {existing_format}")
        except Exception as e:
            logger.error(f"Erreur renommage {new_format}: {e}")

    # Toujours retourner le format standardisé
    return existing_format


def cleanup_duplicate_files():
    """Nettoie les fichiers dupliqués avec mauvais nommage"""
    logger.info("=== NETTOYAGE FICHIERS DUPLIQUES ===")
    cleaned = 0

    # Chercher les fichiers avec format TOKENUSDC_TIMEFRAME.json
    for file_path in JSON_ROOT.glob("*USDC_*.json"):
        if "_12months" not in file_path.name:
            # Extraire token et interval
            match = re.match(
                r"^([A-Za-z0-9]+)USDC_([0-9mhdws]+)\.json$", file_path.name
            )
            if match:
                token, interval = match.groups()
                target_name = f"{token}_{interval}_12months.json"
                target_path = JSON_ROOT / target_name

                try:
                    if target_path.exists():
                        # Si le format correct existe, supprimer le doublon
                        file_path.unlink()
                        logger.info(f"Supprimé doublon: {file_path.name}")
                    else:
                        # Sinon, renommer vers le format correct
                        file_path.rename(target_path)
                        logger.info(f"Renommé: {file_path.name} → {target_name}")
                    cleaned += 1
                except Exception as e:
                    logger.error(f"Erreur nettoyage {file_path.name}: {e}")

    logger.info(f"=== NETTOYAGE TERMINE: {cleaned} fichiers traités ===")


# =========================================================
#  Sélection des tokens (sources publiques - identique TradXPro)
# =========================================================


def get_usdc_base_assets() -> List[str]:
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        resp = requests.get(url, timeout=10).json()
        return [
            s["baseAsset"].upper()
            for s in resp.get("symbols", [])
            if s["symbol"].endswith("USDC")
        ]
    except Exception as e:
        logger.error(f"Erreur get_usdc_base_assets: {e}")
        return []


def get_top100_marketcap_coingecko() -> List[Dict]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
    }
    try:
        data = requests.get(url, params=params, timeout=10).json()
        return [
            {
                "symbol": entry["symbol"].upper(),
                "name": entry["name"],
                "market_cap": entry["market_cap"],
                "market_cap_rank": entry["market_cap_rank"],
            }
            for entry in data
        ]
    except Exception as e:
        logger.error(f"Erreur get_top100_marketcap_coingecko: {e}")
        return []


def get_top100_volume_usdc() -> List[Dict]:
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        data = requests.get(url, timeout=10).json()
        usdc_volumes = [
            {
                "baseAsset": entry["symbol"].replace("USDC", ""),
                "volume": float(entry["quoteVolume"]),
            }
            for entry in data
            if entry["symbol"].endswith("USDC")
        ]
        usdc_volumes.sort(key=lambda x: x["volume"], reverse=True)
        return usdc_volumes[:100]
    except Exception as e:
        logger.error(f"Erreur get_top100_volume_usdc: {e}")
        return []


def merge_and_update_tokens(
    market_cap_list: List[Dict], volume_list: List[Dict]
) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for mc in market_cap_list:
        sym = mc["symbol"].upper()
        merged[sym] = dict(mc, volume=0)
    for v in volume_list:
        sym = v["baseAsset"].upper()
        merged.setdefault(
            sym,
            {
                "symbol": sym,
                "name": sym,
                "market_cap": None,
                "market_cap_rank": None,
                "volume": 0,
            },
        )
        merged[sym]["volume"] = v["volume"]

    old_dict: Dict[str, Dict] = {}
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                old = json.load(f)
            old_dict = {
                i["symbol"].upper(): i
                for i in old
                if isinstance(i, dict) and "symbol" in i
            }
        except Exception:
            pass

    old_dict.update(merged)
    final = sorted(
        old_dict.values(),
        key=lambda x: ((x.get("market_cap_rank") or 999), -x.get("volume", 0)),
    )
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4)
    logger.info(f"Mis à jour {len(final)} tokens dans {JSON_PATH}")
    return final


# =========================================================
#  Téléchargement OHLCV (JSON in-place - identique TradXPro)
# =========================================================


def fetch_klines(
    symbol: str, interval: str, start_ms: int, end_ms: int, retries: int = 3
) -> List[Dict]:
    """Télécharge les klines depuis Binance - Version TradXPro exacte"""
    all_candles: List[Dict] = []
    cur = start_ms

    while cur < end_ms:
        # Vérifier arrêt demandé
        if STOP_REQUESTED:
            logger.warning("Téléchargement interrompu dans fetch_klines")
            break

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": BINANCE_LIMIT,
        }

        for attempt in range(retries):
            try:
                response = requests.get(
                    "https://api.binance.com/api/v3/klines", params=params, timeout=10
                )
                data = response.json()

                if isinstance(data, dict) and "code" in data:
                    logger.error(f"API error pour {symbol}: {data}")
                    return all_candles

                break
            except Exception as e:
                logger.warning(
                    f"fetch_klines retry {attempt+1}/{retries} pour {symbol}: {e}"
                )
                if attempt < retries - 1:
                    time.sleep(2**attempt)  # Backoff exponentiel
        else:
            logger.error(f"Echec téléchargement {symbol} après {retries} tentatives")
            return all_candles

        if not data:
            break

        # Conversion format TradXPro
        batch_candles = [
            {
                "timestamp": c[0],
                "open": str(c[1]),
                "high": str(c[2]),
                "low": str(c[3]),
                "close": str(c[4]),
                "volume": str(c[5]),
                "extra": {
                    "close_time": c[6],
                    "quote_asset_volume": str(c[7]),
                    "trades_count": c[8],
                    "taker_buy_base_volume": str(c[9]),
                    "taker_buy_quote_volume": str(c[10]),
                },
            }
            for c in data
        ]

        all_candles.extend(batch_candles)
        cur = data[-1][0] + 1
        time.sleep(0.1)  # Rate limiting

    return all_candles


def detect_missing(
    candles: List[Dict], interval: str, start_ms: int, end_ms: int
) -> List[Tuple[int, int]]:
    """Détecte les trous dans les données - Version TradXPro exacte"""
    if not candles:
        return [(start_ms, end_ms)]

    interval_ms = interval_to_ms(interval)
    gaps = []

    # Vérifier le début
    first_ts = candles[0]["timestamp"]
    if first_ts > start_ms:
        gaps.append((start_ms, first_ts - interval_ms))

    # Vérifier les trous entre les bougies
    for i in range(len(candles) - 1):
        current_ts = candles[i]["timestamp"]
        next_ts = candles[i + 1]["timestamp"]
        expected_next = current_ts + interval_ms

        if next_ts > expected_next:
            gaps.append((expected_next, next_ts - interval_ms))

    # Vérifier la fin
    last_ts = candles[-1]["timestamp"]
    if last_ts < end_ms:
        gaps.append((last_ts + interval_ms, end_ms))

    return gaps


def verify_and_complete(
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> None:
    """Vérifie et complète les données manquantes - Version TradXPro exacte"""
    logger.info("=== DÉBUT VÉRIFICATION ET COMPLÉTION ===")

    json_files = list(JSON_ROOT.glob("*_12months.json"))
    total_files = len(json_files)
    processed = 0

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - HISTORY_DAYS * 86_400_000

    for json_file in json_files:
        if STOP_REQUESTED:
            break

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                candles = json.load(f)

            if not candles:
                continue

            # Extraire interval du nom de fichier
            match = FILENAME_PATTERN.match(json_file.name)
            if not match:
                continue

            symbol, interval = match.groups()
            symbol_usdc = f"{symbol}USDC"

            # Détecter les trous
            gaps = detect_missing(candles, interval, start_ms, now_ms)

            if gaps:
                logger.info(
                    f"Complétion {len(gaps)} trous pour {symbol_usdc}_{interval}"
                )

                # Compléter chaque trou
                for gap_start, gap_end in gaps:
                    if STOP_REQUESTED:
                        break

                    gap_data = fetch_klines(symbol_usdc, interval, gap_start, gap_end)
                    if gap_data:
                        candles.extend(gap_data)

                # Trier et sauvegarder
                candles.sort(key=lambda x: x["timestamp"])
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(candles, f, indent=2)

        except Exception as e:
            logger.error(f"Erreur vérification {json_file.name}: {e}")

        processed += 1
        if progress_callback:
            progress_callback(processed / total_files * 100, processed, total_files)

    logger.info(f"=== VÉRIFICATION TERMINÉE: {processed} fichiers traités ===\n")


def download_ohlcv(
    tokens: List[str],
    usdc_symbols: set,
    force_update: bool = False,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> None:
    """Télécharge les données OHLCV - Version TradXPro exacte"""
    global STOP_REQUESTED
    STOP_REQUESTED = False  # Reset flag

    logger.info("=== DÉBUT TÉLÉCHARGEMENT OHLCV ===")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - HISTORY_DAYS * 86_400_000
    total_operations = len(tokens) * len(INTERVALS)
    processed = 0

    # Nettoyage préalable des doublons
    cleanup_duplicate_files()

    for token in tokens:
        if STOP_REQUESTED:
            logger.warning("Téléchargement interrompu par utilisateur")
            break

        symbol = f"{token}USDC"
        if symbol not in usdc_symbols:
            logger.warning(f"{symbol} non disponible sur Binance")
            processed += len(INTERVALS)  # Skip tous les intervalles
            continue

        for interval in INTERVALS:
            if STOP_REQUESTED:
                break

            # Nom de fichier harmonisé
            fname = f"{token}_{interval}_12months.json"
            fpath = JSON_ROOT / fname

            try:
                # Vérifier si fichier existe et est récent
                if fpath.exists() and not force_update:
                    age_hours = (time.time() - fpath.stat().st_mtime) / 3600
                    if age_hours < 1:  # Moins d'1h
                        logger.info(f"{fname} récent ({age_hours:.1f}h), skip")
                        processed += 1
                        continue

                # Charger données existantes ou créer nouveau
                existing_data = []
                if fpath.exists():
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)

                        # Vérifier intégrité des données
                        if existing_data:
                            timestamps = [c["timestamp"] for c in existing_data]
                            if len(set(timestamps)) != len(timestamps):
                                logger.warning(f"{fname} doublons, nettoyage")
                                # Supprimer doublons, garder ordre
                                seen = set()
                                existing_data = [
                                    c
                                    for c in existing_data
                                    if (
                                        c["timestamp"] not in seen
                                        and not seen.add(c["timestamp"])
                                    )
                                ]
                                existing_data.sort(key=lambda x: x["timestamp"])

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"{fname} corrompu: {e}, re-dl complet")
                        existing_data = []

                # Déterminer plage de téléchargement
                if existing_data:
                    last_timestamp = max(c["timestamp"] for c in existing_data)
                    download_start = last_timestamp + interval_to_ms(interval)
                else:
                    download_start = start_ms

                # Télécharger nouvelles données si nécessaire
                if download_start < now_ms:
                    logger.info(f"Téléchargement {symbol}_{interval}...")
                    new_data = fetch_klines(symbol, interval, download_start, now_ms)

                    if new_data:
                        # Fusionner avec données existantes
                        all_data = existing_data + new_data
                        # Supprimer doublons et trier
                        seen = set()
                        all_data = [
                            c
                            for c in all_data
                            if c["timestamp"] not in seen
                            and not seen.add(c["timestamp"])
                        ]
                        all_data.sort(key=lambda x: x["timestamp"])

                        # Sauvegarder
                        fpath.parent.mkdir(parents=True, exist_ok=True)
                        with open(fpath, "w", encoding="utf-8") as f:
                            json.dump(all_data, f, indent=2)

                        logger.info(f"Sauvé {len(all_data)} bougies pour {fname}")
                    else:
                        logger.warning(
                            f"Aucune nouvelle donnée pour {symbol}_{interval}"
                        )
                else:
                    logger.info(f"{fname} déjà à jour")

            except Exception as e:
                logger.error(f"Erreur traitement {symbol}_{interval}: {e}")

            processed += 1
            if progress_callback:
                progress_callback(
                    processed / total_operations * 100, processed, total_operations
                )

            # Vérifier arrêt
            if STOP_REQUESTED:
                break

        if STOP_REQUESTED:
            break

    logger.info(f"=== TÉLÉCHARGEMENT TERMINÉ: {processed} opérations ===")


# =========================================================
#  Conversion JSON→Parquet (identique TradXPro)
# =========================================================


def _fix_timestamp_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige l'index d'un DataFrame pour garantir DatetimeIndex UTC."""
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    # Chercher colonne timestamp
    timestamp_cols = [
        col
        for col in df.columns
        if col.lower() in ["timestamp", "datetime", "time", "open_time"]
    ]

    if timestamp_cols:
        ts_col = timestamp_cols[0]
        if df[ts_col].dtype == "int64":
            df.index = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        else:
            df.index = pd.to_datetime(df[ts_col], utc=True)
        df = df.drop(columns=[ts_col])
        logger.debug(f"Index corrigé: {ts_col} → DatetimeIndex UTC")
    else:
        logger.warning("Aucune colonne timestamp trouvée pour correction d'index")

    return df


def _json_to_df(json_path: Path) -> Optional[pd.DataFrame]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not data:
        return None
    df = pd.DataFrame(data)

    # Correction critique des timestamps
    df = _fix_timestamp_conversion(df)

    # Colonnes OHLCV obligatoires
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Colonne {col} manquante dans {json_path}")

    # Nettoyage final
    df = df[numeric_cols].dropna()
    df = df.astype("float64")

    # Validation index UTC
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError(f"Index non-UTC après correction dans {json_path}")

    logger.debug(f"JSON→DataFrame: {len(df)} lignes, {df.index[0]} → {df.index[-1]}")
    return df


def json_candles_to_parquet(
    json_path: Path, out_dir: Optional[Path] = None, compression: str = "snappy"
) -> Optional[Path]:
    root = json_path.stem
    out_path = (out_dir or json_path.parent) / f"{root}.parquet"

    try:
        json_mtime = json_path.stat().st_mtime
        if out_path.exists():
            pq_mtime = out_path.stat().st_mtime
            if pq_mtime >= json_mtime:
                return out_path
    except Exception:
        pass

    df = _json_to_df(json_path)
    if df is None or df.empty:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=True, compression=compression)
    logger.info(f"Parquet écrit: {out_path}")
    return out_path


def convert_all_candles(
    data_dir: Path,
    out_dir: Optional[Path] = None,
    workers: int = MAX_WORKERS,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> int:
    files = [f for f in data_dir.glob("*.json") if FILENAME_PATTERN.match(f.name)]
    total = len(files)
    written = 0

    start_time = time.time()
    logger.info(f"Conversion JSON→Parquet: {total} fichiers")

    def _one(p: Path) -> int:
        try:
            outp = json_candles_to_parquet(p, out_dir)
            return 1 if outp else 0
        except Exception as e:
            logger.error(f"Parquet conversion failed for {p}: {e}")
            return 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, p) for p in files]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                written += fut.result()
            except Exception as e:
                logger.error(f"convert_all_candles error: {e}")
            if progress_callback:
                progress_callback(i / total * 100, i, total)

    elapsed = time.time() - start_time
    logger.info(f"Parquet (bougies) écrits/validés: {written} en {elapsed:.2f}s")
    return written


# =========================================================
#  Gestion harmonisée des indicateurs techniques
# =========================================================


def setup_indicators_structure():
    """Initialise la structure harmonisée des indicateurs techniques"""
    # Vérifier et créer les dossiers d'indicateurs
    for indicator_dir in [INDICATORS_ATR, INDICATORS_BOLLINGER, INDICATORS_REGISTRY]:
        indicator_dir.mkdir(parents=True, exist_ok=True)

    # Créer le fichier registry s'il n'existe pas
    registry_file = INDICATORS_REGISTRY / "indicators_registry.json"
    if not registry_file.exists():
        default_registry = {
            "indicators": {
                "ATR": {
                    "name": "Average True Range",
                    "path": "atr",
                    "enabled": True,
                    "default_period": 14,
                },
                "BOLLINGER": {
                    "name": "Bollinger Bands",
                    "path": "bollinger",
                    "enabled": True,
                    "default_period": 20,
                    "default_std": 2,
                },
            },
            "last_update": datetime.now().isoformat(),
            "version": "1.0",
        }
        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(default_registry, f, indent=2)
        logger.info(f"Registry indicateurs créé: {registry_file}")


def load_indicators_registry():
    """Charge le registry des indicateurs"""
    registry_file = INDICATORS_REGISTRY / "indicators_registry.json"
    try:
        with open(registry_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur chargement registry: {e}")
        return {}


def calculate_indicators_for_symbol(
    symbol: str, interval: str, with_indicators: bool = True
) -> Optional[Path]:
    """Calcule les indicateurs pour un symbole donné"""
    if not with_indicators:
        return None

    try:
        # Charger données OHLCV
        json_file = JSON_ROOT / f"{symbol}_{interval}.json"
        if not json_file.exists():
            logger.warning(f"Données OHLCV manquantes: {json_file}")
            return None

        df = _json_to_df(json_file)
        if df is None or df.empty:
            return None

        # Calculer ATR
        atr_data = calculate_atr(df)
        if atr_data is not None:
            atr_file = INDICATORS_ATR / f"{symbol}_{interval}_atr.parquet"
            atr_data.to_parquet(atr_file, engine="pyarrow", index=True)
            logger.debug(f"ATR calculé: {atr_file}")

        # Calculer Bollinger Bands
        bb_data = calculate_bollinger_bands(df)
        if bb_data is not None:
            bb_file = INDICATORS_BOLLINGER / f"{symbol}_{interval}_bollinger.parquet"
            bb_data.to_parquet(bb_file, engine="pyarrow", index=True)
            logger.debug(f"Bollinger calculé: {bb_file}")

        return json_file

    except Exception as e:
        logger.error(f"Erreur calcul indicateurs {symbol}_{interval}: {e}")
        return None


def calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[pd.DataFrame]:
    """Calcule l'Average True Range"""
    try:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # True Range calculation
        hl = high - low
        hc = np.abs(high - np.roll(close, 1))
        lc = np.abs(low - np.roll(close, 1))

        tr = np.maximum(hl, np.maximum(hc, lc))
        tr[0] = hl[0]  # Premier élément

        # ATR avec moyenne mobile
        atr = pd.Series(tr, index=df.index).rolling(window=period).mean()

        result_df = pd.DataFrame({"atr": atr, "tr": tr}, index=df.index)

        return result_df.dropna()

    except Exception as e:
        logger.error(f"Erreur calcul ATR: {e}")
        return None


def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> Optional[pd.DataFrame]:
    """Calcule les Bollinger Bands"""
    try:
        close = df["close"]

        # Moyenne mobile
        sma = close.rolling(window=period).mean()

        # Écart-type mobile
        std = close.rolling(window=period).std()

        # Bandes
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        result_df = pd.DataFrame(
            {
                "bb_upper": upper_band,
                "bb_middle": sma,
                "bb_lower": lower_band,
                "bb_width": upper_band - lower_band,
                "bb_percent": (close - lower_band) / (upper_band - lower_band),
            },
            index=df.index,
        )

        return result_df.dropna()

    except Exception as e:
        logger.error(f"Erreur calcul Bollinger: {e}")
        return None


def verify_and_clean_cache():
    """Vérifie et nettoie le cache de données"""
    try:
        cache_files = list(CACHE_ROOT.glob("**/*"))
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

        logger.info(
            f"Cache: {len(cache_files)} fichiers, {total_size / 1024 / 1024:.1f} MB"
        )

        # Nettoyer les fichiers temporaires anciens (>7 jours)
        cutoff_time = time.time() - (7 * 24 * 3600)
        cleaned = 0

        for cache_file in cache_files:
            if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                    cleaned += 1
                except Exception:
                    pass

        if cleaned > 0:
            logger.info(f"Cache nettoyé: {cleaned} fichiers supprimés")

    except Exception as e:
        logger.error(f"Erreur nettoyage cache: {e}")


# =========================================================
#  Interface GUI (EXACTEMENT identique à TradXPro)
# =========================================================


class UnifiedDataHistoriqueGUI:
    """Interface graphique identique à TradXPro original"""

    def __init__(self):
        if not TK_AVAILABLE:
            raise ImportError("Tkinter non disponible - mode GUI impossible")

        self.root = tk.Tk()
        self.root.title("ThreadX - Unified Data Historique with Indicators v1.0")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # Variables d'état
        self.is_running = False
        self.current_operation = None
        self.log_text = None
        self.progress_var = None
        self.status_var = None

        # Setup
        self.setup_styles()
        self.setup_ui()
        self.setup_logging_display()

    def setup_styles(self):
        """Configuration des styles visuels"""
        style = ttk.Style()

        # Utiliser le thème moderne si disponible
        available_themes = style.theme_names()
        if "vista" in available_themes:
            style.theme_use("vista")
        elif "clam" in available_themes:
            style.theme_use("clam")

        # Styles personnalisés
        style.configure("Title.TLabel", font=("Arial", 12, "bold"))
        style.configure("Header.TLabel", font=("Arial", 10, "bold"))
        style.configure("Success.TButton", foreground="green")
        style.configure("Warning.TButton", foreground="orange")
        style.configure("Error.TButton", foreground="red")

    def setup_ui(self):
        """Configuration de l'interface utilisateur - IDENTIQUE TradXPro"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Titre
        title_label = ttk.Label(
            main_frame,
            text="ThreadX - Unified Data Historique with Indicators",
            style="Title.TLabel",
        )
        title_label.pack(pady=(0, 20))

        # Frame des boutons d'action - EXACTEMENT comme TradXPro
        self.create_action_buttons(main_frame)

        # Frame de progression
        self.create_progress_section(main_frame)

        # Zone de logs - IDENTIQUE TradXPro
        self.create_logs_section(main_frame)

        # Barre de statut
        self.create_status_bar()

    def create_action_buttons(self, parent):
        """Création des boutons d'action - EXACTEMENT identiques TradXPro"""
        buttons_frame = ttk.LabelFrame(parent, text="Actions", padding=15)
        buttons_frame.pack(fill=tk.X, pady=(0, 20))

        # Ligne 1: Gestion des tokens
        row1_frame = ttk.Frame(buttons_frame)
        row1_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            row1_frame,
            text="🔄 Refresh 100 meilleures monnaies",
            command=self.refresh_top100_tokens,
            style="Success.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            row1_frame, text="📋 Voir liste tokens", command=self.view_tokens_list
        ).pack(side=tk.LEFT, padx=10)

        # Ligne 2: Téléchargement
        row2_frame = ttk.Frame(buttons_frame)
        row2_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            row2_frame,
            text="📥 Télécharger OHLCV (sans indicateurs)",
            command=self.download_ohlcv_only,
            style="Header.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            row2_frame,
            text="📊 Télécharger OHLCV + Indicateurs",
            command=self.download_ohlcv_with_indicators,
            style="Warning.TButton",
        ).pack(side=tk.LEFT, padx=10)

        # Ligne 3: Conversion et utilitaires
        row3_frame = ttk.Frame(buttons_frame)
        row3_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            row3_frame,
            text="🔄 Convertir JSON → Parquet",
            command=self.convert_json_to_parquet,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            row3_frame,
            text="✅ Vérifier & Compléter données",
            command=self.verify_and_complete_data,
        ).pack(side=tk.LEFT, padx=10)

        # Ligne 4: Contrôles
        row4_frame = ttk.Frame(buttons_frame)
        row4_frame.pack(fill=tk.X)

        ttk.Button(
            row4_frame,
            text="⏹️ Arrêter opération",
            command=self.stop_operation,
            style="Error.TButton",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(row4_frame, text="🧹 Nettoyer logs", command=self.clear_logs).pack(
            side=tk.LEFT, padx=10
        )

        ttk.Button(
            row4_frame, text="📁 Ouvrir dossier données", command=self.open_data_folder
        ).pack(side=tk.LEFT, padx=10)

    def create_progress_section(self, parent):
        """Section de progression - IDENTIQUE TradXPro"""
        progress_frame = ttk.LabelFrame(parent, text="Progression", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 20))

        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, length=400, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        # Label de statut
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack()

    def create_logs_section(self, parent):
        """Section des logs - EXACTEMENT identique TradXPro"""
        logs_frame = ttk.LabelFrame(parent, text="Journal d'activité", padding=10)
        logs_frame.pack(fill=tk.BOTH, expand=True)

        # Zone de texte avec scrollbar - IDENTIQUE TradXPro
        self.log_text = ScrolledText(logs_frame, height=20, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Message initial
        self.log_text.insert(
            tk.END, "=== ThreadX - Unified Data Historique with Indicators v1.0 ===\n"
        )
        self.log_text.insert(
            tk.END, "Interface prête. Sélectionnez une action pour commencer.\n\n"
        )

    def create_status_bar(self):
        """Barre de statut - identique TradXPro"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Labels d'information harmonisée
        info_text = f"JSON: {JSON_ROOT.name} | Parquet: {PARQUET_ROOT.name}"
        info_label = ttk.Label(status_frame, text=info_text)
        info_label.pack(side=tk.LEFT, padx=5)

        # Info indicateurs
        indicators_text = (
            f"Indicateurs: {INDICATORS_ROOT.name} | Cache: {CACHE_ROOT.name}"
        )
        indicators_label = ttk.Label(
            status_frame, text=indicators_text, font=("Arial", 8)
        )
        indicators_label.pack(side=tk.LEFT, padx=10)

        # Séparateur
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=5
        )

        # Heure
        self.time_label = ttk.Label(status_frame, text="")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        self.update_time()

    def update_time(self):
        """Met à jour l'heure affichée"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def setup_logging_display(self):
        """Configuration de l'affichage des logs - IDENTIQUE TradXPro"""

        def check_log_queue():
            try:
                while True:
                    record = log_queue.get_nowait()
                    if self.log_text:
                        formatted = queue_handler.format(record)
                        self.log_text.insert(tk.END, formatted + "\n")
                        self.log_text.see(tk.END)
            except queue.Empty:
                pass

            # Programmer la prochaine vérification
            self.root.after(100, check_log_queue)

        self.root.after(100, check_log_queue)

    # =========================================================
    #  Actions des boutons - EXACTEMENT identiques TradXPro
    # =========================================================

    def refresh_top100_tokens(self):
        """Refresh des 100 meilleures monnaies - IDENTIQUE TradXPro"""
        if self.is_running:
            messagebox.showwarning("Attention", "Une opération est déjà en cours!")
            return

        def worker():
            try:
                self.is_running = True
                self.current_operation = "refresh_tokens"
                self.update_status("🔄 Récupération des 100 meilleures monnaies...")
                self.update_progress(10)

                logger.info("=== DEBUT REFRESH 100 MEILLEURES MONNAIES ===")

                # Récupération market cap CoinGecko
                self.update_status("📊 Récupération données CoinGecko...")
                market_cap_list = get_top100_marketcap_coingecko()
                self.update_progress(40)
                logger.info(f"Récupéré {len(market_cap_list)} tokens CoinGecko")

                # Récupération volumes Binance
                self.update_status("💰 Récupération volumes Binance...")
                volume_list = get_top100_volume_usdc()
                self.update_progress(70)
                logger.info(f"Récupéré {len(volume_list)} tokens Binance")

                # Fusion et sauvegarde
                self.update_status("💾 Fusion et sauvegarde...")
                final_tokens = merge_and_update_tokens(market_cap_list, volume_list)
                self.update_progress(100)

                logger.info(
                    f"=== REFRESH TERMINE: {len(final_tokens)} tokens sauvés ==="
                )
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Succès",
                        f"Refresh terminé!\n{len(final_tokens)} tokens mis à jour dans:\n{JSON_PATH}",
                    ),
                )

            except Exception as e:
                logger.error(f"Erreur refresh tokens: {e}")
                self.root.after(
                    0, lambda: messagebox.showerror("Erreur", f"Erreur refresh: {e}")
                )
            finally:
                self.is_running = False
                self.current_operation = None
                self.update_status("Prêt")
                self.update_progress(0)

        threading.Thread(target=worker, daemon=True).start()

    def download_ohlcv_only(self):
        """Téléchargement OHLCV seulement - IDENTIQUE TradXPro"""
        if self.is_running:
            messagebox.showwarning("Attention", "Une opération est déjà en cours!")
            return

        if not JSON_PATH.exists():
            messagebox.showerror(
                "Erreur",
                "Fichier des tokens non trouvé!\nEffectuez d'abord un refresh des tokens.",
            )
            return

        def worker():
            try:
                self.is_running = True
                self.current_operation = "download_ohlcv"

                logger.info("=== DEBUT TELECHARGEMENT OHLCV ===")

                # Charger liste des tokens
                with open(JSON_PATH, "r", encoding="utf-8") as f:
                    tokens_data = json.load(f)

                tokens = [
                    t["symbol"]
                    for t in tokens_data
                    if isinstance(t, dict) and "symbol" in t
                ]
                logger.info(f"Chargé {len(tokens)} tokens à télécharger")

                # Récupérer symboles USDC disponibles
                self.update_status("🔍 Vérification symboles Binance...")
                usdc_symbols = set(get_usdc_base_assets())
                usdc_symbols = {f"{s}USDC" for s in usdc_symbols}
                logger.info(f"Trouvé {len(usdc_symbols)} symboles USDC sur Binance")

                # Télécharger
                def progress_callback(pct, current, total):
                    self.update_progress(pct)
                    self.update_status(
                        f"📥 Téléchargement: {current}/{total} tokens ({pct:.1f}%)"
                    )

                download_ohlcv(tokens, usdc_symbols, FORCE_UPDATE, progress_callback)

                logger.info("=== TELECHARGEMENT OHLCV TERMINE ===")
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Succès", "Téléchargement OHLCV terminé!"
                    ),
                )

            except Exception as e:
                logger.error(f"Erreur téléchargement OHLCV: {e}")
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Erreur", f"Erreur téléchargement: {e}"
                    ),
                )
            finally:
                self.is_running = False
                self.current_operation = None
                self.update_status("Prêt")
                self.update_progress(0)

        threading.Thread(target=worker, daemon=True).start()

    def download_ohlcv_with_indicators(self):
        """Téléchargement OHLCV + Indicateurs - IMPLÉMENTATION COMPLÈTE"""
        if self.is_running:
            messagebox.showwarning("Attention", "Une opération est déjà en cours!")
            return

        if not JSON_PATH.exists():
            messagebox.showerror(
                "Erreur",
                "Fichier des tokens non trouvé!\nEffectuez d'abord un refresh des tokens.",
            )
            return

        def worker():
            try:
                self.is_running = True
                self.current_operation = "download_ohlcv_indicators"

                logger.info("=== DEBUT TELECHARGEMENT OHLCV + INDICATEURS ===")

                # Initialiser structure des indicateurs
                setup_indicators_structure()

                # Charger liste des tokens
                with open(JSON_PATH, "r", encoding="utf-8") as f:
                    tokens_data = json.load(f)

                tokens = [
                    t["symbol"]
                    for t in tokens_data
                    if isinstance(t, dict) and "symbol" in t
                ]
                logger.info(f"Chargé {len(tokens)} tokens à traiter")

                # Récupérer symboles USDC disponibles
                self.update_status("🔍 Vérification symboles Binance...")
                usdc_symbols = set(get_usdc_base_assets())
                usdc_symbols = {f"{s}USDC" for s in usdc_symbols}

                # Phase 1: Téléchargement OHLCV
                self.update_status("📥 Phase 1: Téléchargement OHLCV...")

                def ohlcv_progress(pct, current, total):
                    self.update_progress(pct * 0.6)  # 60% pour OHLCV
                    self.update_status(
                        f"📥 OHLCV: {current}/{total} tokens ({pct:.1f}%)"
                    )

                download_ohlcv(tokens, usdc_symbols, FORCE_UPDATE, ohlcv_progress)

                # Phase 2: Calcul des indicateurs
                self.update_status("📊 Phase 2: Calcul indicateurs techniques...")
                processed_indicators = 0
                total_to_process = len(tokens) * len(INTERVALS)

                for i, token in enumerate(tokens):
                    symbol = f"{token}USDC"
                    if symbol not in usdc_symbols:
                        continue

                    for interval in INTERVALS:
                        try:
                            calculate_indicators_for_symbol(symbol, interval, True)
                            processed_indicators += 1
                        except Exception as e:
                            logger.error(f"Erreur indicateurs {symbol}_{interval}: {e}")

                    # Mise à jour progression
                    progress_pct = 60 + (processed_indicators / total_to_process * 40)
                    self.update_progress(progress_pct)
                    self.update_status(
                        f"📊 Indicateurs: {processed_indicators}/{total_to_process} "
                        f"({progress_pct:.1f}%)"
                    )

                logger.info(
                    f"=== TELECHARGEMENT + INDICATEURS TERMINE: "
                    f"{processed_indicators} indicateurs calculés ==="
                )
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Succès",
                        f"Téléchargement + Indicateurs terminé!\n"
                        f"{processed_indicators} indicateurs calculés",
                    ),
                )

            except Exception as e:
                logger.error(f"Erreur téléchargement + indicateurs: {e}")
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Erreur", f"Erreur téléchargement + indicateurs: {e}"
                    ),
                )
            finally:
                self.is_running = False
                self.current_operation = None
                self.update_status("Prêt")
                self.update_progress(0)

        threading.Thread(target=worker, daemon=True).start()

    def convert_json_to_parquet(self):
        """Conversion JSON vers Parquet - IDENTIQUE TradXPro"""
        if self.is_running:
            messagebox.showwarning("Attention", "Une opération est déjà en cours!")
            return

        def worker():
            try:
                self.is_running = True
                self.current_operation = "convert_parquet"

                logger.info("=== DEBUT CONVERSION JSON → PARQUET ===")

                def progress_callback(pct, current, total):
                    self.update_progress(pct)
                    self.update_status(
                        f"🔄 Conversion: {current}/{total} fichiers ({pct:.1f}%)"
                    )

                written = convert_all_candles(
                    JSON_ROOT, PARQUET_ROOT, MAX_WORKERS, progress_callback
                )

                logger.info(f"=== CONVERSION TERMINEE: {written} fichiers Parquet ===")
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Succès",
                        f"Conversion terminée!\n{written} fichiers Parquet créés dans:\n{PARQUET_ROOT}",
                    ),
                )

            except Exception as e:
                logger.error(f"Erreur conversion Parquet: {e}")
                self.root.after(
                    0, lambda: messagebox.showerror("Erreur", f"Erreur conversion: {e}")
                )
            finally:
                self.is_running = False
                self.current_operation = None
                self.update_status("Prêt")
                self.update_progress(0)

        threading.Thread(target=worker, daemon=True).start()

    def verify_and_complete_data(self):
        """Vérification et complétion des données - IMPLÉMENTATION COMPLÈTE"""
        if self.is_running:
            messagebox.showwarning("Attention", "Une opération est déjà en cours!")
            return

        def worker():
            try:
                self.is_running = True
                self.current_operation = "verify_data"

                logger.info("=== DEBUT VERIFICATION ET COMPLETION DONNEES ===")

                # Phase 1: Vérification structure
                self.update_status("🔍 Vérification structure des dossiers...")
                self.update_progress(10)
                setup_indicators_structure()

                # Phase 2: Vérification cache
                self.update_status("🧹 Nettoyage du cache...")
                self.update_progress(30)
                verify_and_clean_cache()

                # Phase 3: Vérification fichiers JSON
                self.update_status("📋 Vérification fichiers JSON...")
                self.update_progress(50)
                json_files = list(JSON_ROOT.glob("*.json"))
                valid_json = 0
                corrupted_json = []

                for json_file in json_files:
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data and isinstance(data, list):
                            valid_json += 1
                        else:
                            corrupted_json.append(json_file.name)
                    except Exception:
                        corrupted_json.append(json_file.name)

                # Phase 4: Vérification fichiers Parquet
                self.update_status("📊 Vérification fichiers Parquet...")
                self.update_progress(70)
                parquet_files = list(PARQUET_ROOT.glob("*.parquet"))
                valid_parquet = len(parquet_files)

                # Phase 5: Vérification indicateurs
                self.update_status("📈 Vérification indicateurs...")
                self.update_progress(90)
                atr_files = list(INDICATORS_ATR.glob("*.parquet"))
                bb_files = list(INDICATORS_BOLLINGER.glob("*.parquet"))

                # Rapport final
                report = {
                    "json_files": {
                        "total": len(json_files),
                        "valid": valid_json,
                        "corrupted": corrupted_json,
                    },
                    "parquet_files": {"total": valid_parquet},
                    "indicators": {"atr": len(atr_files), "bollinger": len(bb_files)},
                    "timestamp": datetime.now().isoformat(),
                }

                # Sauvegarder rapport
                report_file = THREADX_ROOT / "data" / "verification_report.json"
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)

                self.update_progress(100)
                logger.info(
                    f"=== VERIFICATION TERMINEE: {valid_json} JSON, "
                    f"{valid_parquet} Parquet, {len(atr_files)} ATR, "
                    f"{len(bb_files)} Bollinger ==="
                )

                message = (
                    f"Vérification terminée!\n\n"
                    f"📋 JSON: {valid_json}/{len(json_files)} valides\n"
                    f"📊 Parquet: {valid_parquet} fichiers\n"
                    f"📈 ATR: {len(atr_files)} indicateurs\n"
                    f"📈 Bollinger: {len(bb_files)} indicateurs\n\n"
                    f"Rapport sauvé: verification_report.json"
                )

                if corrupted_json:
                    message += f"\n\n⚠️ Fichiers corrompus: {len(corrupted_json)}"

                self.root.after(0, lambda: messagebox.showinfo("Rapport", message))

            except Exception as e:
                logger.error(f"Erreur vérification: {e}")
                self.root.after(
                    0,
                    lambda: messagebox.showerror("Erreur", f"Erreur vérification: {e}"),
                )
            finally:
                self.is_running = False
                self.current_operation = None
                self.update_status("Prêt")
                self.update_progress(0)

        threading.Thread(target=worker, daemon=True).start()

    def view_tokens_list(self):
        """Affichage de la liste des tokens - IDENTIQUE TradXPro"""
        if not JSON_PATH.exists():
            messagebox.showerror(
                "Erreur",
                "Fichier des tokens non trouvé!\nEffectuez d'abord un refresh des tokens.",
            )
            return

        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                tokens_data = json.load(f)

            # Créer une nouvelle fenêtre
            tokens_window = tk.Toplevel(self.root)
            tokens_window.title("Liste des 100 meilleures monnaies")
            tokens_window.geometry("800x600")

            # Treeview pour afficher les tokens
            columns = ("Rank", "Symbol", "Name", "Market Cap", "Volume 24h")
            tree = ttk.Treeview(tokens_window, columns=columns, show="headings")

            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=150)

            # Scrollbar
            scrollbar = ttk.Scrollbar(
                tokens_window, orient=tk.VERTICAL, command=tree.yview
            )
            tree.configure(yscrollcommand=scrollbar.set)

            # Remplir les données
            for i, token in enumerate(tokens_data[:100], 1):
                if isinstance(token, dict):
                    rank = token.get("market_cap_rank", i)
                    symbol = token.get("symbol", "N/A")
                    name = token.get("name", "N/A")
                    market_cap = token.get("market_cap", 0)
                    volume = token.get("volume", 0)

                    # Formatage
                    mc_str = f"${market_cap:,.0f}" if market_cap else "N/A"
                    vol_str = f"${volume:,.0f}" if volume else "N/A"

                    tree.insert(
                        "", tk.END, values=(rank, symbol, name, mc_str, vol_str)
                    )

            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lecture tokens: {e}")

    def stop_operation(self):
        """Arrêt de l'opération en cours - CORRIGÉ avec flag global"""
        global STOP_REQUESTED

        if not self.is_running:
            messagebox.showinfo("Info", "Aucune opération en cours.")
            return

        result = messagebox.askyesno(
            "Confirmation", "Voulez-vous vraiment arrêter l'opération en cours?"
        )
        if result:
            # Déclencher l'arrêt propre
            STOP_REQUESTED = True
            self.update_status("⏹️ Arrêt en cours...")
            logger.warning("Arrêt demandé par l'utilisateur")

    def clear_logs(self):
        """Nettoyage des logs - IDENTIQUE TradXPro"""
        if self.log_text:
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "=== Logs nettoyés ===\n")

    def open_data_folder(self):
        """Ouverture du dossier de données - IDENTIQUE TradXPro"""
        try:
            if IS_WINDOWS:
                os.startfile(str(JSON_ROOT))
            else:
                os.system(f"xdg-open {JSON_ROOT}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir le dossier: {e}")

    def update_status(self, message: str):
        """Mise à jour du statut"""
        if self.status_var:
            self.status_var.set(message)

    def update_progress(self, value: float):
        """Mise à jour de la barre de progression"""
        if self.progress_var:
            self.progress_var.set(value)

    def run(self):
        """Lancement de l'interface"""
        logger.info("🚀 Démarrage de l'interface ThreadX Unified Data Historique")
        self.root.mainloop()


# =========================================================
#  Mode CLI (identique TradXPro)
# =========================================================


def run_cli_mode():
    """Mode CLI quand GUI non disponible - IDENTIQUE TradXPro"""
    print("=" * 80)
    print("ThreadX - Unified Data Historique with Indicators v1.0 - MODE CLI")
    print("=" * 80)

    while True:
        print("\nActions disponibles:")
        print("1. 🔄 Refresh 100 meilleures monnaies")
        print("2. 📥 Télécharger OHLCV")
        print("3. 🔄 Convertir JSON → Parquet")
        print("4. 📋 Voir liste tokens")
        print("5. 🚪 Quitter")

        try:
            choice = input("\nSélectionnez une action (1-5): ").strip()

            if choice == "1":
                print("\n🔄 Refresh des 100 meilleures monnaies...")
                market_cap_list = get_top100_marketcap_coingecko()
                volume_list = get_top100_volume_usdc()
                final_tokens = merge_and_update_tokens(market_cap_list, volume_list)
                print(f"✅ Refresh terminé: {len(final_tokens)} tokens mis à jour")

            elif choice == "2":
                print("\n📥 Téléchargement OHLCV...")
                if not JSON_PATH.exists():
                    print("❌ Fichier tokens non trouvé! Effectuez d'abord un refresh.")
                    continue

                with open(JSON_PATH, "r", encoding="utf-8") as f:
                    tokens_data = json.load(f)
                tokens = [
                    t["symbol"]
                    for t in tokens_data
                    if isinstance(t, dict) and "symbol" in t
                ]
                usdc_symbols = set(get_usdc_base_assets())
                usdc_symbols = {f"{s}USDC" for s in usdc_symbols}

                download_ohlcv(tokens, usdc_symbols, FORCE_UPDATE)
                print("✅ Téléchargement OHLCV terminé")

            elif choice == "3":
                print("\n🔄 Conversion JSON → Parquet...")
                written = convert_all_candles(JSON_ROOT, PARQUET_ROOT, MAX_WORKERS)
                print(f"✅ Conversion terminée: {written} fichiers Parquet")

            elif choice == "4":
                print("\n📋 Liste des tokens:")
                if JSON_PATH.exists():
                    with open(JSON_PATH, "r", encoding="utf-8") as f:
                        tokens_data = json.load(f)
                    for i, token in enumerate(
                        tokens_data[:20], 1
                    ):  # Afficher 20 premiers
                        if isinstance(token, dict):
                            print(
                                f"{i:2d}. {token.get('symbol', 'N/A'):10s} - {token.get('name', 'N/A')}"
                            )
                    if len(tokens_data) > 20:
                        print(f"... et {len(tokens_data) - 20} autres")
                else:
                    print("❌ Fichier tokens non trouvé")

            elif choice == "5":
                print("👋 Au revoir!")
                break

            else:
                print("❌ Choix invalide. Sélectionnez 1-5.")

        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


# =========================================================
#  Point d'entrée principal (identique TradXPro)
# =========================================================


def main():
    """Point d'entrée principal - IDENTIQUE TradXPro"""
    print("🚀 ThreadX - Unified Data Historique with Indicators v1.0")
    print(f"Platform: {platform.system()}")
    print(f"GUI Available: {'✅' if TK_AVAILABLE else '❌'}")

    # Parser d'arguments (comme TradXPro)
    parser = argparse.ArgumentParser(description="ThreadX Unified Data Historique")
    parser.add_argument("--cli", action="store_true", help="Forcer le mode CLI")
    parser.add_argument("--force", action="store_true", help="Forcer la mise à jour")
    args = parser.parse_args()

    global FORCE_UPDATE
    FORCE_UPDATE = args.force

    if args.cli or not TK_AVAILABLE:
        print("🖥️ Mode CLI")
        run_cli_mode()
    else:
        try:
            print("🖼️ Mode GUI")
            app = UnifiedDataHistoriqueGUI()
            app.run()
        except Exception as e:
            print(f"❌ Erreur GUI: {e}")
            print("🔄 Basculement vers mode CLI...")
            run_cli_mode()


if __name__ == "__main__":
    main()
