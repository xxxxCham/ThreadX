# -*- coding: utf-8 -*-
"""
UnifiedDataHistorique_with_Indicators v2.4
- FIX CRITIQUE: Timestamps 1970 → conversion ms + UTC forcé
- FIX CRITIQUE: Formatage prix adaptatif (évite 0.00 sur micro-caps)
- FIX CRITIQUE: Calcul périodes sûres réalistes (évite max_safe=0)
- FIX CRITIQUE: Anti-doublons logging + gestion symboles indisponibles
- PERF: Cache LRU + optimisations I/O
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

import numpy as np
import pandas as pd
import requests

# Import du nouveau système de performance centralisé
try:
    from perf_manager import PerfLogger

    PERF_AVAILABLE = True
except ImportError:
    PERF_AVAILABLE = False

    class PerfLogger:
        @staticmethod
        def log_run(**kwargs):
            pass


try:  # dotenv facultatif
    from dotenv import load_dotenv
except Exception:  # pragma: no cover

    def load_dotenv(*_, **__):
        return False


try:
    from tqdm import tqdm
except Exception:  # fallback simple si tqdm indisponible

    def tqdm(it, **_):
        return it


# =========================================================
#  Tkinter: import **optionnel** (GUI si dispo, sinon CLI)
# =========================================================
TK_AVAILABLE = False
try:  # noqa: SIM105
    import tkinter as tk  # type: ignore
    from tkinter.scrolledtext import ScrolledText  # type: ignore
    from tkinter import ttk  # type: ignore

    TK_AVAILABLE = True
except Exception:
    tk = None  # type: ignore
    ScrolledText = None  # type: ignore
    ttk = None  # type: ignore

# =========================================================
#  Chargement .env (prioritaire sur defaults ci-dessous)
# =========================================================
load_dotenv()

# =========================================================
#  Defaults alignés sur D:\\TradXPro (Windows)
# =========================================================
IS_WINDOWS = platform.system() == "Windows"

# =========================================================
#  Chemins ThreadX (structure normalisée)
# =========================================================
JSON_ROOT = os.getenv("THREADX_JSON_ROOT", r"D:\ThreadX\data\crypto_data_json")
PARQUET_ROOT = os.getenv("THREADX_PARQUET_ROOT", r"D:\ThreadX\data\crypto_data_parquet")
INDICATORS_DB_ROOT = os.getenv("THREADX_IND_DB", r"D:\ThreadX\data\indicators")
OUTPUT_DIR = os.getenv("THREADX_OUTPUT_DIR", r"D:\ThreadX\data\exports")

# Chemins historiques pour compatibilité
DEFAULT_JSON_PATH = (
    r"D:\ThreadX\data\crypto_data_json\resultats_choix_des_100tokens.json"
    if IS_WINDOWS
    else "/home/user/resultats_choix_des_100tokens.json"
)
DEFAULT_LOG_FILE = (
    r"D:\ThreadX\logs\unified_data_historique.log"
    if IS_WINDOWS
    else "/home/user/unified_data_historique.log"
)

# Variables complémentaires
JSON_PATH = os.path.normpath(os.getenv("JSON_PATH", DEFAULT_JSON_PATH))
LOG_FILE = os.path.normpath(os.getenv("LOG_FILE", DEFAULT_LOG_FILE))

# Compatibilité ancienne variable DATA_FOLDER
DATA_FOLDER = JSON_ROOT  # Rétrocompatibilité

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")

# Création des répertoires TradXPro
for _p in (JSON_ROOT, PARQUET_ROOT, INDICATORS_DB_ROOT, OUTPUT_DIR):
    os.makedirs(_p, exist_ok=True)

os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# =========================================================
#  Fonctions utilitaires de chemins
# =========================================================


def _fix_dataframe_index(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige l'index d'un DataFrame pour garantir DatetimeIndex UTC."""
    if isinstance(df.index, pd.DatetimeIndex):
        # Déjà un DatetimeIndex, vérifier timezone
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
        # Conversion selon le format
        if df[ts_col].dtype == "int64":  # Millisecondes
            df.index = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        else:  # String ISO
            df.index = pd.to_datetime(df[ts_col], utc=True)

        # Supprimer la colonne timestamp devenue index
        df = df.drop(columns=[ts_col])
        logger.debug(f"Index corrigé: {ts_col} → DatetimeIndex UTC")
    else:
        logger.warning("Aucune colonne timestamp trouvée pour correction d'index")

    return df


def parquet_path(symbol: str, tf: str) -> str:
    """Chemin Parquet pour un symbole/timeframe."""
    return os.path.join(PARQUET_ROOT, f"{symbol.upper()}_{tf.lower()}.parquet")


def json_path_symbol(symbol: str, tf: str) -> str:
    """Chemin JSON pour un symbole/timeframe."""
    return os.path.join(JSON_ROOT, f"{symbol.upper()}_{tf.lower()}.json")


def indicator_path(symbol: str, tf: str, name: str, key: str) -> str:
    """Chemin indicateur selon conventions ThreadX.

    Args:
        symbol: Symbole crypto (ex: 'BTCUSDC')
        tf: Timeframe (ex: '1h', '3m', '5m', '15m', '30m')
        name: Nom indicateur (ex: 'bollinger', 'rsi', 'atr')
        key: Clé paramètres (ex: 'period14', 'period20_std2.0')

    Returns:
        Chemin complet du fichier Parquet de l'indicateur
        Format: D:/ThreadX/data/indicators/{SYMBOL}/{tf}/{name}_{key}.parquet
        Exemple: D:/ThreadX/data/indicators/ZK/1h/bollinger_period20_std2.0.parquet
    """
    # Extraire le symbole de base (sans USDC)
    base_symbol = symbol.upper().replace("USDC", "")
    return os.path.join(
        INDICATORS_DB_ROOT, base_symbol, tf.lower(), f"{name}_{key}.parquet"
    )


# =========================================================
#  Logging + canal UI/CLI
# =========================================================
logger = logging.getLogger("UnifiedDataHistorique")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_queue: "queue.Queue[logging.LogRecord]" = queue.Queue()


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: "queue.Queue[logging.LogRecord]"):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        self.log_queue.put(record)


queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

# Console handler pour le mode CLI
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =========================================================
#  Paramètres opérationnels
# =========================================================
HISTORY_DAYS = 365
BINANCE_LIMIT = 1000
FILENAME_PATTERN = re.compile(r"^([A-Za-z0-9]+USDC)_([0-9mhdw]+)\.json$")
INTERVALS = ["3m", "5m", "15m", "30m", "1h"]
FORCE_UPDATE = False
MAX_WORKERS = max(4, (os.cpu_count() or 8) // 2)

# =========================================================
#  Utilitaires
# =========================================================


def interval_to_ms(interval: str) -> int:
    amount = int(re.sub(r"\D", "", interval))
    unit = re.sub(r"\d", "", interval)
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    return amount * 60_000


# =========================================================
#  Sélection des tokens (sources publiques)
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
    except Exception as e:  # pragma: no cover (IO)
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
    except Exception as e:  # pragma: no cover (IO)
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
    except Exception as e:  # pragma: no cover (IO)
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
    if os.path.exists(JSON_PATH):
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
#  Téléchargement OHLCV (JSON in-place)
# =========================================================


def fetch_klines(
    symbol: str, interval: str, start_ms: int, end_ms: int, retries: int = 3
) -> List[Dict]:
    all_candles: List[Dict] = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": BINANCE_LIMIT,
        }
        for attempt in range(retries):
            try:
                data = requests.get(
                    "https://api.binance.com/api/v3/klines", params=params, timeout=10
                ).json()
                if isinstance(data, dict) and "code" in data:
                    raise ValueError(f"API error: {data}")
                break
            except Exception as e:  # pragma: no cover (IO)
                logger.warning(f"fetch_klines retry {attempt+1}/{retries}: {e}")
                time.sleep(5)
        else:
            return all_candles
        if not data:
            break
        all_candles.extend(
            [
                {
                    "timestamp": c[0],
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "volume": c[5],
                    "extra": {
                        "close_time": c[6],
                        "quote_asset_volume": c[7],
                        "trades_count": c[8],
                        "taker_buy_base_volume": c[9],
                        "taker_buy_quote_volume": c[10],
                    },
                }
                for c in data
            ]
        )
        cur = data[-1][0] + 1
        time.sleep(0.2)
    return all_candles


def download_ohlcv(
    tokens: List[str],
    usdc_symbols: set,
    force_update: bool = False,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> None:
    now = int(time.time() * 1000)
    start = now - HISTORY_DAYS * 86_400_000
    total_tokens = len(tokens)
    treated = 0
    for token in tqdm(tokens, desc="Téléchargement Tokens"):
        symbol = f"{token}USDC"
        if symbol not in usdc_symbols:
            continue
        for interval in INTERVALS:
            fname = f"{symbol}_{interval}.json"
            fpath = os.path.join(DATA_FOLDER, fname)
            if os.path.exists(fpath):
                recent = (time.time() - os.path.getmtime(fpath)) < 86_400
                if recent and not force_update:
                    logger.info(f"{fname} récent (<1j), skip.")
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    ts = [c["timestamp"] for c in old]
                    if sorted(ts) != ts or len(set(ts)) != len(ts):
                        logger.warning(f"{fname} corrompu → re-download complet.")
                        data = fetch_klines(symbol, interval, start, now)
                    else:
                        last_ts = max(ts) if ts else start
                        upd_start = last_ts + interval_to_ms(interval)
                        if upd_start >= now:
                            logger.info(f"{fname} à jour, skip.")
                            continue
                        data = fetch_klines(symbol, interval, upd_start, now)
                        data = (old + data) if data else old
                        data.sort(key=lambda x: x["timestamp"])
                except json.JSONDecodeError:
                    logger.error(f"{fname} JSON invalide → re-download complet.")
                    data = fetch_klines(symbol, interval, start, now)
            else:
                data = fetch_klines(symbol, interval, start, now)

            if data:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Enregistré/Mis à jour {len(data)} bougies pour {fname}")
        treated += 1
        if progress_callback:
            progress_callback(treated / total_tokens * 100, treated, total_tokens)


# =========================================================
#  Vérification & complétion (JSON in-place)
# =========================================================


def detect_missing(
    candles: List[Dict], interval: str, start_ms: int, end_ms: int
) -> List[Tuple[int, int]]:
    if not candles:
        return [(start_ms, end_ms)]
    candles.sort(key=lambda x: x["timestamp"])
    step = interval_to_ms(interval)
    missing: List[Tuple[int, int]] = []
    first = candles[0]["timestamp"]
    if first > start_ms:
        missing.append((start_ms, first - 1))
    for i in range(len(candles) - 1):
        a, b = candles[i]["timestamp"], candles[i + 1]["timestamp"]
        if b - a > step:
            missing.append((a + step, b - 1))
    last = candles[-1]["timestamp"]
    if last < end_ms:
        missing.append((last + step, end_ms))
    return [g for g in missing if g[0] < g[1]]


def verify_and_complete(
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> None:
    now = datetime.utcnow()
    start_ms = int((now - timedelta(days=HISTORY_DAYS)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    files = [f for f in os.listdir(DATA_FOLDER) if FILENAME_PATTERN.match(f)]
    total = len(files)
    done = 0
    for file in tqdm(files, desc="Vérification Fichiers"):
        m = FILENAME_PATTERN.match(file)
        symbol, interval = m.groups()  # type: ignore[union-attr]
        fpath = os.path.join(DATA_FOLDER, file)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts = [c["timestamp"] for c in data]
            if sorted(ts) != ts or len(set(ts)) != len(ts):
                logger.warning(f"{file} corrompu → re-complétion totale.")
                data = []
        except json.JSONDecodeError:
            logger.error(f"{file} invalide → re-complétion totale.")
            data = []
        gaps = detect_missing(data, interval, start_ms, end_ms)
        if gaps:
            add: List[Dict] = []
            for a, b in gaps:
                add.extend(fetch_klines(symbol, interval, a, b))
            if add:
                merged = list({c["timestamp"]: c for c in (data + add)}.values())
                merged.sort(key=lambda x: x["timestamp"])  # type: ignore[index]
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2)
                logger.info(f"Complété {file} (+{len(add)} bougies)")
        done += 1
        if progress_callback:
            progress_callback(done / total * 100, done, total)


# =========================================================
#  Conversion JSON→Parquet (bougies historiques)
# =========================================================


def _json_to_df(json_path: str) -> Optional[pd.DataFrame]:
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

    # Colonnes OHLCV obligatoires avec conversion numérique sécurisée
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Colonne {col} manquante dans {path}")

    # Nettoyage final
    df = df[numeric_cols].dropna()
    df = df.astype("float64")

    # Validation index UTC
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError(f"Index non-UTC après correction dans {path}")

    logger.debug(f"JSON→DataFrame: {len(df)} lignes, {df.index[0]} → {df.index[-1]}")
    return df


def json_candles_to_parquet(
    json_path: str, out_dir: Optional[str] = None, compression: str = "snappy"
) -> Optional[str]:
    root, _ = os.path.splitext(os.path.basename(json_path))
    out_path = os.path.join(out_dir or os.path.dirname(json_path), root + ".parquet")
    try:
        json_mtime = os.path.getmtime(json_path)
        if os.path.exists(out_path):
            pq_mtime = os.path.getmtime(out_path)
            if pq_mtime >= json_mtime:  # déjà à jour
                return out_path
    except Exception:
        pass

    df = _json_to_df(json_path)
    if df is None or df.empty:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Clean and sanitize the DataFrame before saving
    df_clean = _sanitize_ohlcv_df(df)

    # Write the cleaned DataFrame to Parquet
    df_clean.to_parquet(out_path, engine="pyarrow", index=True, compression="zstd")
    logger.info(f"Parquet écrit: {out_path}")
    return out_path


def convert_all_candles(
    data_dir: str,
    out_dir: Optional[str] = None,
    workers: int = MAX_WORKERS,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> int:
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if FILENAME_PATTERN.match(f)
    ]
    total = len(files)
    written = 0

    start_time = time.time()
    logger.info(f"Conversion JSON→Parquet: {total} fichiers")

    def _one(p: str) -> int:
        try:
            outp = json_candles_to_parquet(p, out_dir)
            return 1 if outp else 0
        except Exception as e:  # pragma: no cover (IO)
            logger.error(f"Parquet conversion failed for {p}: {e}")
            return 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, p) for p in files]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                written += fut.result()
            except Exception as e:  # pragma: no cover
                logger.error(f"convert_all_candles error: {e}")
            if progress_callback:
                progress_callback(i / total * 100, i, total)

    elapsed = time.time() - start_time
    logger.info(f"Parquet (bougies) écrits/validés: {written} en {elapsed:.2f}s")

    # Logging performance
    if PERF_AVAILABLE:
        try:
            PerfLogger.log_run(
                elapsed_sec=elapsed,
                n_tasks=total,
                n_input_rows=total,
                n_results_rows=written,
                backend="json_to_parquet",
                symbol="multiple",
                start="conversion",
                end="completed",
            )
        except Exception as e:
            logger.warning(f"Erreur logging performance: {e}")

    return written


# =========================================================
#  Indicateurs (numpy/pandas) + écriture Parquet
# =========================================================


def _ewm(x: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    if span <= 1:
        out[:] = x
        return out
    a = 2.0 / (span + 1.0)
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def ema_np(arr: np.ndarray, span: int) -> np.ndarray:
    return _ewm(arr, span)


def atr_np(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    prev = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return _ewm(tr, period)


def boll_np(close: np.ndarray, period: int = 20, std: float = 2.0):
    ma = _ewm(close, period)
    var = _ewm((close - ma) ** 2, period)
    sd = np.sqrt(np.maximum(var, 1e-12))
    upper = ma + std * sd
    lower = ma - std * sd
    z = (close - ma) / sd
    return lower, ma, upper, z


def rsi_np(close: np.ndarray, period: int = 14) -> np.ndarray:
    if close.size == 0:
        return np.array([], dtype=np.float64)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ewm(gain, period)
    avg_loss = _ewm(loss, period)
    rs = np.divide(avg_gain, np.maximum(avg_loss, 1e-12))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def macd_np(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema_np(close, fast)
    ema_slow = ema_np(close, slow)
    macd = ema_fast - ema_slow
    sig = ema_np(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def vwap_np(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    window: int = 96,
):
    typical = (high + low + close) / 3.0
    vol_ema = ema_np(volume, window)
    pv_ema = ema_np(typical * volume, window)
    vwap = np.divide(pv_ema, np.maximum(vol_ema, 1e-12))
    return vwap


def obv_np(close: np.ndarray, volume: np.ndarray):
    obv = np.zeros_like(close, dtype=np.float64)
    for i in range(1, close.size):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv


# =========================================================
#  Système de cache d'indicateurs TradXPro
# =========================================================

# =========================================================
#  Chargement de la liste des 100 tokens TradXPro
# =========================================================


def load_100_tokens_list() -> List[str]:
    """Charge la liste des 100 tokens depuis le fichier JSON TradXPro.

    Returns:
        Liste des symboles avec USDC (ex: ['BTCUSDC', 'ETHUSDC', ...])
    """
    try:
        if not os.path.exists(JSON_PATH):
            logger.warning(f"Fichier 100 tokens non trouvé: {JSON_PATH}")
            return []

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            tokens_data = json.load(f)

        # Extraction des symboles et ajout de USDC
        symbols = []
        for token in tokens_data:
            if isinstance(token, dict) and "symbol" in token:
                symbol = token["symbol"]
                # Ajouter USDC si pas déjà présent
                if not symbol.endswith("USDC"):
                    symbol += "USDC"
                symbols.append(symbol)

        logger.info(f"Chargé {len(symbols)} tokens depuis {JSON_PATH}")
        return sorted(symbols)

    except Exception as e:
        logger.error(f"Erreur chargement 100 tokens: {e}")
        return []


def get_all_available_symbols() -> List[str]:
    """Récupère tous les symboles disponibles (100 tokens + symboles manuels).

    Returns:
        Liste complète des symboles disponibles
    """
    # Symboles de base (manuels)
    base_symbols = [
        "BTCUSDC",
        "ETHUSDC",
        "ADAUSDC",
        "SOLUSDC",
        "DOTUSDC",
        "AVAXUSDC",
        "MATICUSDC",
        "LINKUSDC",
        "UNIUSDC",
        "ATOMUSDC",
    ]

    # Charger les 100 tokens
    tokens_100 = load_100_tokens_list()

    # Fusion et suppression des doublons
    all_symbols = list(set(base_symbols + tokens_100))
    return sorted(all_symbols)


# =========================================================
#  API simplifiée fonctionnelle
# =========================================================


def vortex_df(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> pd.DataFrame:
    n = len(closes)
    if n == 0:
        return pd.DataFrame({"vi_plus": [], "vi_minus": []})
    h = highs.astype(np.float64)
    l = lows.astype(np.float64)
    c = closes.astype(np.float64)
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    prev_c = np.roll(c, 1)
    prev_h[0] = h[0]
    prev_l[0] = l[0]
    prev_c[0] = c[0]
    vm_plus = np.abs(h - prev_l)
    vm_minus = np.abs(l - prev_h)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    vm_p_sum = (
        pd.Series(vm_plus, dtype="float64")
        .rolling(window=period, min_periods=period)
        .sum()
    )
    vm_m_sum = (
        pd.Series(vm_minus, dtype="float64")
        .rolling(window=period, min_periods=period)
        .sum()
    )
    tr_sum = (
        pd.Series(tr, dtype="float64").rolling(window=period, min_periods=period).sum()
    )
    vi_plus = (vm_p_sum / tr_sum).to_numpy()
    vi_minus = (vm_m_sum / tr_sum).to_numpy()
    return pd.DataFrame({"vi_plus": vi_plus, "vi_minus": vi_minus})
