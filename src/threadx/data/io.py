"""
ThreadX Data I/O Module - Phase 2
I/O unifiée JSON/Parquet avec validation Pandera stricte.

Remplace et modernise:
- TradXPro/core/data_io.py (chemins absolus)
- TradXPro/core/io_candles.py (env vars TXP_DF_ROOT)

Nouveautés:
- Validation schéma Pandera obligatoire
- Chemins relatifs depuis threadx.config uniquement
- Exceptions dédiées avec contexte
- Support JSON ISO 8601 + Parquet optimisé
"""

import logging
import json
from pathlib import Path
from typing import Union, Optional, Literal
import warnings

import pandas as pd
import pyarrow as parquet

# Pandera import avec fallback gracieux
try:
    import pandera as pa  # type: ignore
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    # Mock pour éviter erreurs
    class pa:  # type: ignore
        class errors:
            class SchemaError(Exception): pass
        @staticmethod
        def DataFrameSchema(*args, **kwargs):
            return None
        @staticmethod
        def Column(*args, **kwargs):
            return None
        class Check:
            @staticmethod
            def gt(*args, **kwargs):
                return None
            @staticmethod
            def ge(*args, **kwargs):
                return None

# Import config ThreadX (Phase 1)
try:
    from threadx.config import load_settings
    _settings = load_settings()
except ImportError:
    # Fallback pour tests isolés
    from pathlib import Path as _Path
    class _MockSettings:
        DATA_ROOT = _Path("./data")
    _settings = _MockSettings()

logger = logging.getLogger(__name__)

__all__ = [
    "OHLCV_SCHEMA",
    "DataNotFoundError", 
    "FileValidationError",
    "SchemaMismatchError",
    "read_frame",
    "write_frame", 
    "normalize_ohlcv"
]

# ============================================================================
# EXCEPTIONS DÉDIÉES
# ============================================================================

class DataNotFoundError(FileNotFoundError):
    """Données OHLCV introuvables au chemin spécifié."""
    
    def __init__(self, path: Union[Path, str], message: Optional[str] = None):
        self.path = Path(path)
        default_msg = f"Données OHLCV introuvables: {self.path}"
        super().__init__(message or default_msg)


class FileValidationError(ValueError):
    """Erreur de validation du fichier OHLCV."""
    
    def __init__(self, path: Union[Path, str], details: str):
        self.path = Path(path)
        self.details = details
        super().__init__(f"Validation échouée pour {self.path}: {details}")


class SchemaMismatchError(ValueError):
    """Schéma OHLCV non conforme."""
    
    def __init__(self, details: str, expected_schema: Optional[str] = None):
        self.details = details
        self.expected_schema = expected_schema
        message = f"Schéma OHLCV invalide: {details}"
        if expected_schema:
            message += f" (attendu: {expected_schema})"
        super().__init__(message)


# ============================================================================
# SCHÉMA PANDERA OHLCV STRICT
# ============================================================================

if PANDERA_AVAILABLE:
    OHLCV_SCHEMA = pa.DataFrameSchema({
        "open": pa.Column(
            float,
            checks=[
                pa.Check.gt(0, error="Prix open doit être > 0"),
            ],
        ),
        "high": pa.Column(
            float,
            checks=[
                pa.Check.gt(0, error="Prix high doit être > 0"),
            ],
        ),
        "low": pa.Column(
            float,
            checks=[
                pa.Check.gt(0, error="Prix low doit être > 0"),
            ],
        ),
        "close": pa.Column(
            float,
            checks=[
                pa.Check.gt(0, error="Prix close doit être > 0"),
            ],
        ),
        "volume": pa.Column(
            float,
            checks=[
                pa.Check.ge(0, error="Volume doit être >= 0"),
            ],
        )
    }, strict=True, coerce=True)
else:
    # Schéma mock si Pandera indisponible
    OHLCV_SCHEMA = None
    logger.warning("Pandera non disponible - validation schéma désactivée")


def _validate_ohlcv_constraints(df: pd.DataFrame) -> None:
    """Validations OHLCV spécifiques non couvertes par Pandera."""
    
    # Vérification cohérence OHLC
    invalid_ohlc = (
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"]) |
        (df["high"] < df["low"])
    )
    
    if invalid_ohlc.any():
        n_invalid = invalid_ohlc.sum()
        sample_idx = df.index[invalid_ohlc].tolist()[:3]
        raise SchemaMismatchError(
            f"{n_invalid} lignes avec OHLC incohérent (ex: {sample_idx})"
        )
    
    # Vérification ordre chronologique strict
    if not df.index.is_monotonic_increasing:
        raise SchemaMismatchError("Index datetime doit être strictement croissant")
    
    # Vérification unicité timestamps
    if df.index.duplicated().any():
        n_dupes = df.index.duplicated().sum()
        raise SchemaMismatchError(f"{n_dupes} timestamps dupliqués détectés")


# ============================================================================
# NORMALISATION OHLCV
# ============================================================================

def normalize_ohlcv(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """
    Normalise un DataFrame vers le schéma OHLCV ThreadX.
    
    Inspiré de TradXPro io_candles._normalize_columns mais plus strict:
    - Force index DatetimeIndex UTC
    - Trie par timestamp croissant
    - Supprime doublons
    - Convertit dtypes vers schéma
    - Valide cohérence OHLC
    
    Args:
        df: DataFrame brut à normaliser
        tz: Timezone cible (défaut UTC)
        
    Returns:
        DataFrame conforme OHLCV_SCHEMA
        
    Raises:
        SchemaMismatchError: Si normalisation impossible
    """
    if df is None or df.empty:
        raise SchemaMismatchError("DataFrame vide ou null")
    
    df_norm = df.copy()
    
    # === NORMALISATION COLONNES ===
    # Colonnes en minuscules (comme TradXPro)
    df_norm.columns = df_norm.columns.str.lower()
    
    # Alias fréquents (repris de TradXPro)
    alias_map = {
        "o": "open", "h": "high", "l": "low", "c": "close",
        "vol": "volume", "v": "volume",
        "base_asset_volume": "volume"
    }
    
    df_norm = df_norm.rename(columns=alias_map)
    
    # Vérification colonnes obligatoires
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df_norm.columns)
    if missing:
        raise SchemaMismatchError(f"Colonnes OHLCV manquantes: {sorted(missing)}")
    
    # Sélection colonnes OHLCV seulement (ordre canonique garanti)
    canonical_order = ["open", "high", "low", "close", "volume"]
    df_norm = df_norm[canonical_order]
    
    # === NORMALISATION INDEX ===
    if not isinstance(df_norm.index, pd.DatetimeIndex):
        # Cas JSON: colonne 'timestamp' (logique TradXPro)
        if "timestamp" in df.columns:
            ts_col = df["timestamp"]
            
            # Auto-détection unité (ms vs s) comme TradXPro
            if pd.api.types.is_numeric_dtype(ts_col):
                sample_val = ts_col.dropna().iloc[0] if not ts_col.dropna().empty else 0
                if sample_val > 10_000_000_000:  # > 10^10 → millisecondes
                    ts_index = pd.to_datetime(ts_col, unit="ms", utc=True)
                else:
                    ts_index = pd.to_datetime(ts_col, unit="s", utc=True)
            else:
                ts_index = pd.to_datetime(ts_col, utc=True)
                
            df_norm.index = pd.DatetimeIndex(ts_index)  # type: ignore
        else:
            # Conversion index existant
            df_norm.index = pd.to_datetime(df_norm.index, utc=True)
    
    # Force timezone UTC
    if df_norm.index.tz is None:
        df_norm.index = df_norm.index.tz_localize(tz)  # type: ignore
    else:
        df_norm.index = df_norm.index.tz_convert(tz)  # type: ignore
    
    # === NETTOYAGE ===
    # Tri chronologique
    df_norm = df_norm.sort_index()
    
    # Suppression doublons (garde le dernier)
    df_norm = df_norm[~df_norm.index.duplicated(keep="last")]
    
    # Suppression NaN
    initial_len = len(df_norm)
    df_norm = df_norm.dropna()
    dropped = initial_len - len(df_norm)
    if dropped > 0:
        logger.warning(f"Suppression de {dropped} lignes avec NaN")
    
    # === CONVERSION TYPES ===
    for col in ["open", "high", "low", "close", "volume"]:
        df_norm[col] = pd.to_numeric(df_norm[col], errors="coerce").astype("float64")
    
    # === VALIDATION FINALE ===
    _validate_ohlcv_constraints(df_norm)
    
    logger.debug(f"Normalisation OHLCV: {initial_len} → {len(df_norm)} lignes")
    return df_norm


# ============================================================================
# I/O UNIFIÉ JSON/PARQUET
# ============================================================================

def _detect_format(path: Union[Path, str]) -> str:
    """Détecte le format depuis l'extension."""
    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    
    if ext == ".parquet":
        return "parquet"
    elif ext == ".json":
        return "json"
    else:
        raise FileValidationError(path, f"Format non supporté: {ext} (attendu .parquet ou .json)")


def read_frame(
    path: Union[Path, str], 
    fmt: Optional[str] = None, 
    validate: bool = True
) -> pd.DataFrame:
    """
    Lecture unifiée JSON/Parquet → DataFrame OHLCV validé.
    
    Remplace TradXPro read_candles_* avec validation stricte.
    
    Args:
        path: Chemin vers fichier données
        fmt: Format explicite ("json"|"parquet") ou auto-détection
        validate: Si True, valide contre OHLCV_SCHEMA
        
    Returns:
        DataFrame OHLCV normalisé et validé
        
    Raises:
        DataNotFoundError: Fichier introuvable
        FileValidationError: Erreur lecture fichier
        SchemaMismatchError: Validation schéma échouée
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise DataNotFoundError(path, f"Fichier introuvable: {path_obj}")
    
    if not path_obj.is_file():
        raise FileValidationError(path, "Chemin ne pointe pas vers un fichier")
    
    if path_obj.stat().st_size == 0:
        raise FileValidationError(path, "Fichier vide")
    
    # Détection format
    fmt = fmt or _detect_format(path_obj)
    
    try:
        # Lecture selon format
        if fmt == "parquet":
            logger.debug(f"Lecture Parquet: {path_obj}")
            df = pd.read_parquet(path_obj, engine="pyarrow")
            
        elif fmt == "json":
            logger.debug(f"Lecture JSON: {path_obj}")
            # JSON records orienté (comme TradXPro)
            df = pd.read_json(path_obj, orient="records", convert_dates=False)
            
        else:
            raise FileValidationError(path, f"Format non supporté: {fmt}")
            
    except Exception as e:
        raise FileValidationError(path, f"Erreur lecture {fmt.upper()}: {e}")
    
    # Normalisation OHLCV
    try:
        df_normalized = normalize_ohlcv(df)
    except Exception as e:
        raise SchemaMismatchError(f"Normalisation échouée: {e}")
    
    # Validation optionnelle 
    if validate and PANDERA_AVAILABLE and OHLCV_SCHEMA:
        try:
            OHLCV_SCHEMA.validate(df_normalized, lazy=False)
            logger.debug(f"Validation OHLCV réussie: {len(df_normalized)} lignes")
        except pa.errors.SchemaError as e:
            raise SchemaMismatchError(f"Schema Pandera: {e}")
    elif validate and not PANDERA_AVAILABLE:
        logger.warning("Validation demandée mais Pandera indisponible")
    
    return df_normalized


def write_frame(
    df: pd.DataFrame,
    path: Union[Path, str], 
    fmt: Optional[str] = None,
    overwrite: bool = False
) -> None:
    """
    Écriture unifiée DataFrame → JSON/Parquet.
    
    Args:
        df: DataFrame OHLCV à sauvegarder
        path: Chemin destination
        fmt: Format explicite ou auto-détection
        overwrite: Si False, refuse d'écraser fichier existant
        
    Raises:
        FileValidationError: Erreur écriture
        SchemaMismatchError: DataFrame non conforme
    """
    path_obj = Path(path)
    
    # Vérification écrasement
    if path_obj.exists() and not overwrite:
        raise FileValidationError(path, "Fichier existe (utiliser overwrite=True)")
    
    # Création dossiers parents
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Validation DataFrame avant écriture
    if PANDERA_AVAILABLE and OHLCV_SCHEMA:
        try:
            OHLCV_SCHEMA.validate(df, lazy=False)
        except pa.errors.SchemaError as e:
            raise SchemaMismatchError(f"DataFrame invalide pour sauvegarde: {e}")
    
    # Détection format
    fmt = fmt or _detect_format(path_obj)
    
    try:
        if fmt == "parquet":
            logger.debug(f"Écriture Parquet: {path_obj}")
            df.to_parquet(path_obj, engine="pyarrow", compression="snappy")
            
        elif fmt == "json":
            logger.debug(f"Écriture JSON: {path_obj}")
            # JSON records avec timestamps ISO 8601
            df_json = df.copy()
            df_json.index = df_json.index.strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # type: ignore
            df_json.to_json(path_obj, orient="records", date_format="iso", indent=2)
            
        else:
            raise FileValidationError(path, f"Format écriture non supporté: {fmt}")
            
        logger.info(f"Sauvegarde réussie: {path_obj} ({len(df)} lignes, {fmt.upper()})")
        
    except Exception as e:
        # Nettoyage fichier partiellement écrit
        if path_obj.exists():
            path_obj.unlink()
        raise FileValidationError(path, f"Erreur écriture {fmt.upper()}: {e}")
