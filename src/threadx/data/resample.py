"""
ThreadX Data Resampling Module - Phase 2
Resampling canonique 1m → X avec gestion gaps et parallélisation.

Remplace et améliore:
- TradXPro logique resampling éparse
- Gestion gaps intelligente (seuil 5%)
- Batch processing avec multiprocessing

Nouveautés:
- Validation strict entrée/sortie OHLCV
- GPU hooks (CuPy) préparés mais CPU-first
- Forward-fill intelligent avec seuils
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
import numpy as np

# Import GPU optionnel
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Config ThreadX
try:
    from threadx.config import load_settings
    _settings = load_settings()
except ImportError:
    _settings = None

# Import validation OHLCV
try:
    from .io import OHLCV_SCHEMA, SchemaMismatchError, normalize_ohlcv, PANDERA_AVAILABLE  # type: ignore
except ImportError:
    # Fallback pour tests isolés
    OHLCV_SCHEMA = None
    PANDERA_AVAILABLE = False
    class SchemaMismatchError(Exception): pass
    def normalize_ohlcv(df, **kwargs): return df

logger = logging.getLogger(__name__)

__all__ = [
    "resample_from_1m",
    "resample_batch", 
    "TimeframeError",
    "GapFillingError"
]

# ============================================================================
# EXCEPTIONS RESAMPLING
# ============================================================================

class TimeframeError(ValueError):
    """Erreur de timeframe invalide pour resampling."""
    pass


class GapFillingError(ValueError):
    """Erreur lors du gap filling."""
    pass


# ============================================================================
# TIMEFRAMES CANONIQUES
# ============================================================================

# Mapping timeframes → règles pandas resample
TIMEFRAME_RULES = {
    # Minutes
    "1m": "1min", "2m": "2min", "3m": "3min", "5m": "5min",
    "10m": "10min", "15m": "15min", "30m": "30min",
    
    # Heures  
    "1h": "1h", "2h": "2h", "3h": "3h", "4h": "4h", 
    "6h": "6h", "8h": "8h", "12h": "12h",
    
    # Jours
    "1d": "1D", "3d": "3D", "7d": "7D",
    
    # Semaines/Mois
    "1w": "1W", "1M": "1MS"
}


def _validate_timeframe(timeframe: str) -> str:
    """Valide et normalise le timeframe."""
    tf_clean = timeframe.lower().strip()
    
    if tf_clean not in TIMEFRAME_RULES:
        valid_tfs = sorted(TIMEFRAME_RULES.keys())
        raise TimeframeError(f"Timeframe '{timeframe}' non supporté. Valides: {valid_tfs}")
    
    return tf_clean


def _detect_gaps(df_1m: pd.DataFrame, expected_freq: str = "1min") -> tuple[float, int]:
    """
    Détecte les gaps dans une série 1m.
    
    Returns:
        (gap_ratio, n_missing): proportion gaps et nombre timestamps manquants
    """
    if df_1m.empty:
        return 0.0, 0
    
    # Période théorique complète
    start_time = df_1m.index.min()
    end_time = df_1m.index.max()
    expected_range = pd.date_range(start_time, end_time, freq=expected_freq)
    
    n_expected = len(expected_range)
    n_actual = len(df_1m)
    n_missing = max(0, n_expected - n_actual)
    
    gap_ratio = n_missing / n_expected if n_expected > 0 else 0.0
    
    return gap_ratio, n_missing


def _smart_ffill(df: pd.DataFrame, max_gap_ratio: float) -> pd.DataFrame:
    """
    Forward-fill intelligent avec limite de gap.
    
    Args:
        df: DataFrame avec gaps potentiels
        max_gap_ratio: Seuil max de gaps à tolérer pour ffill
        
    Returns:
        DataFrame avec ffill conditionnel
    """
    if df.empty:
        return df
    
    # Détection gaps actuels
    gap_ratio, n_missing = _detect_gaps(df)
    
    if gap_ratio == 0:
        logger.debug("Aucun gap détecté - pas de ffill nécessaire")
        return df
    
    if gap_ratio <= max_gap_ratio:
        # Gaps acceptables → ffill
        logger.debug(f"Gaps {gap_ratio:.2%} <= seuil {max_gap_ratio:.2%} → forward-fill appliqué")
        
        # Reindex sur range complet puis ffill
        full_range = pd.date_range(df.index.min(), df.index.max(), freq="1min")
        df_reindexed = df.reindex(full_range)
        df_filled = df_reindexed.ffill()
        
        return df_filled.dropna()  # Supprime les NaN en début si présents
    else:
        # Gaps trop importants → warning et pas de ffill longue distance
        logger.warning(
            f"Gaps importants détectés: {gap_ratio:.2%} > seuil {max_gap_ratio:.2%} "
            f"({n_missing} timestamps manquants) - forward-fill limité"
        )
        
        # Ffill conservateur: max 5 minutes consécutives
        df_conservative = df.copy()
        df_conservative = df_conservative.ffill(limit=5)
        
        return df_conservative


# ============================================================================
# RESAMPLING CANONIQUE
# ============================================================================

def resample_from_1m(
    df_1m: pd.DataFrame,
    timeframe: str,
    gap_ffill_threshold: float = 0.05,
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Resampling canonique 1m → timeframe avec gestion gaps.
    
    Remplace la logique TradXPro avec validation stricte et gap filling.
    
    Args:
        df_1m: DataFrame OHLCV en 1min (doit être validé)
        timeframe: Timeframe cible (ex: "15m", "1h", "4h")
        gap_ffill_threshold: Seuil gaps pour forward-fill (défaut 5%)
        use_gpu: Si True, utilise CuPy si disponible (TODO Phase 3+)
        
    Returns:
        DataFrame OHLCV resampleé et validé
        
    Raises:
        TimeframeError: Timeframe invalide
        SchemaMismatchError: Données entrée/sortie non conformes
        GapFillingError: Problème gap filling
    """
    if df_1m is None or df_1m.empty:
        raise SchemaMismatchError("DataFrame 1m vide ou null")
    
    # Validation timeframe
    tf_clean = _validate_timeframe(timeframe)
    resample_rule = TIMEFRAME_RULES[tf_clean]
    
    logger.debug(f"Resampling 1m → {tf_clean} (règle: {resample_rule})")
    
    # TODO: GPU acceleration hook (Phase 3+)
    if use_gpu and GPU_AVAILABLE:
        logger.warning("GPU resampling non implémenté - utilisation CPU")
    
    # Validation entrée (schéma OHLCV basique)
    required_cols = {"open", "high", "low", "close", "volume"}
    missing_cols = required_cols - set(df_1m.columns)
    if missing_cols:
        raise SchemaMismatchError(f"Colonnes OHLCV manquantes: {sorted(missing_cols)}")
    
    # Vérification index datetime
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        raise SchemaMismatchError("Index doit être DatetimeIndex")
    
    # Gap filling intelligent
    try:
        df_filled = _smart_ffill(df_1m, gap_ffill_threshold)
        logger.debug(f"Gap filling: {len(df_1m)} → {len(df_filled)} lignes")
    except Exception as e:
        raise GapFillingError(f"Échec gap filling: {e}")
    
    # Resampling canonique OHLCV
    try:
        resampler = df_filled.resample(resample_rule, label="left", closed="left")
        
        df_resampled = pd.DataFrame({
            "open": resampler["open"].first(),
            "high": resampler["high"].max(), 
            "low": resampler["low"].min(),
            "close": resampler["close"].last(),
            "volume": resampler["volume"].sum()
        })
        
        # Suppression périodes sans données
        df_resampled = df_resampled.dropna()
        
        if df_resampled.empty:
            raise SchemaMismatchError("Resampling a produit un DataFrame vide")
        
        logger.debug(f"Resampling terminé: {len(df_filled)} → {len(df_resampled)} barres")
        
    except Exception as e:
        raise SchemaMismatchError(f"Échec resampling: {e}")
    
    # Validation sortie (optionnelle si Pandera disponible)
    if PANDERA_AVAILABLE and OHLCV_SCHEMA:
        try:
            OHLCV_SCHEMA.validate(df_resampled, lazy=False)
        except Exception as e:
            logger.warning(f"Validation Pandera échouée: {e}")
    
    return df_resampled


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def _resample_single_symbol(
    symbol_data: tuple[str, pd.DataFrame],
    timeframe: str,
    gap_ffill_threshold: float = 0.05
) -> tuple[str, pd.DataFrame]:
    """
    Worker function pour resampling d'un symbole (multiprocessing safe).
    
    Args:
        symbol_data: (symbol, df_1m)
        timeframe: Timeframe cible 
        gap_ffill_threshold: Seuil gaps
        
    Returns:
        (symbol, df_resampled)
    """
    symbol, df_1m = symbol_data
    
    try:
        df_resampled = resample_from_1m(
            df_1m, 
            timeframe, 
            gap_ffill_threshold=gap_ffill_threshold,
            use_gpu=False  # Pas de GPU en multiprocessing
        )
        
        logger.debug(f"Resampling {symbol}: {len(df_1m)} → {len(df_resampled)} barres")
        return symbol, df_resampled
        
    except Exception as e:
        logger.error(f"Échec resampling {symbol}: {e}")
        # Retourne DataFrame vide plutôt que crash
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty_df.index = pd.DatetimeIndex([], tz="UTC")
        return symbol, empty_df


def resample_batch(
    frames_by_symbol: Dict[str, pd.DataFrame],
    timeframe: str,
    batch_threshold: int = 10,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Resampling batch avec parallélisation conditionnelle.
    
    Args:
        frames_by_symbol: Dict {symbol: df_1m} à resampler
        timeframe: Timeframe cible pour tous les symboles
        batch_threshold: Seuil pour activer parallélisation
        parallel: Si True, parallélise si > batch_threshold
        max_workers: Nombre workers ProcessPool (défaut: CPU count)
        
    Returns:
        Dict {symbol: df_resampled} dans le même ordre
        
    Raises:
        TimeframeError: Timeframe invalide
    """
    if not frames_by_symbol:
        logger.warning("frames_by_symbol vide - retour dict vide")
        return {}
    
    # Validation timeframe une seule fois
    tf_clean = _validate_timeframe(timeframe)
    n_symbols = len(frames_by_symbol)
    
    logger.info(f"Resampling batch: {n_symbols} symboles → {tf_clean}")
    
    # Choix stratégie: parallèle vs séquentiel
    if parallel and n_symbols > batch_threshold:
        logger.debug(f"Mode parallèle activé: {n_symbols} > {batch_threshold}")
        
        # Préparation données pour ProcessPool
        symbol_data_list = list(frames_by_symbol.items())
        
        # Worker partiellement appliqué
        worker_func = partial(
            _resample_single_symbol,
            timeframe=tf_clean,
            gap_ffill_threshold=0.05
        )
        
        # Exécution parallèle
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(worker_func, symbol_data_list))
            
            logger.debug(f"Resampling parallèle terminé: {len(results)} résultats")
            
        except Exception as e:
            logger.error(f"Échec ProcessPool: {e} - fallback séquentiel")
            # Fallback séquentiel
            results = [worker_func(item) for item in symbol_data_list]
    
    else:
        logger.debug(f"Mode séquentiel: {n_symbols} <= {batch_threshold}")
        
        # Traitement séquentiel
        results = []
        for symbol, df_1m in frames_by_symbol.items():
            try:
                df_resampled = resample_from_1m(df_1m, tf_clean)
                results.append((symbol, df_resampled))
            except Exception as e:
                logger.error(f"Échec resampling séquentiel {symbol}: {e}")
                empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                empty_df.index = pd.DatetimeIndex([], tz="UTC")
                results.append((symbol, empty_df))
    
    # Reconstruction dict avec ordre préservé
    resampled_dict = {symbol: df_resampled for symbol, df_resampled in results}
    
    # Validation résultats
    success_count = sum(1 for df in resampled_dict.values() if not df.empty)
    logger.info(f"Resampling batch terminé: {success_count}/{n_symbols} symboles traités")
    
    return resampled_dict
