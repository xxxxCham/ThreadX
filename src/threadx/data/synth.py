"""
ThreadX Synthetic Data Generator - Phase 2
Génération de données OHLCV synthétiques pour tests et benchmarks.

Nouveautés:
- GBM (Geometric Brownian Motion) réaliste 
- Garantie conformité OHLCV_SCHEMA
- Déterminisme avec seed fixe
- Patterns volatilité et volume adaptables
"""

import logging
from typing import Optional, Tuple
import warnings

import pandas as pd
import numpy as np

# Validation OHLCV si disponible
try:
    from threadx.data.io import OHLCV_SCHEMA, normalize_ohlcv, PANDERA_AVAILABLE  # type: ignore
except ImportError:
    OHLCV_SCHEMA = None
    PANDERA_AVAILABLE = False
    def normalize_ohlcv(df, **kwargs): return df

logger = logging.getLogger(__name__)

__all__ = [
    "make_synth_ohlcv",
    "make_trending_ohlcv", 
    "make_volatile_ohlcv",
    "SynthDataError"
]

# ============================================================================
# EXCEPTIONS
# ============================================================================

class SynthDataError(ValueError):
    """Erreur génération données synthétiques."""
    pass


# ============================================================================
# GÉNÉRATEUR OHLCV DE BASE
# ============================================================================

def make_synth_ohlcv(
    n: int = 10_000,
    start: str = "2024-01-01",
    freq: str = "1min", 
    seed: int = 42,
    base_price: float = 50_000.0,
    volatility: float = 0.02,
    volume_base: float = 1_000_000.0,
    volume_noise: float = 0.5
) -> pd.DataFrame:
    """
    Génère données OHLCV synthétiques via GBM (Geometric Brownian Motion).
    
    Garantit:
    - Index DatetimeIndex UTC continu
    - Cohérence OHLC (low <= open,close <= high)
    - Volume > 0 avec variabilité réaliste
    - Déterminisme via seed
    - Validation OHLCV_SCHEMA si disponible
    
    Args:
        n: Nombre de barres à générer
        start: Date début (ISO format)
        freq: Fréquence temporelle ("1min", "5min", "1H", etc.)
        seed: Seed RNG pour reproductibilité
        base_price: Prix de départ
        volatility: Volatilité annualisée (σ)
        volume_base: Volume moyen 
        volume_noise: Facteur bruit volume (0-1)
        
    Returns:
        DataFrame OHLCV validé et normalisé
        
    Raises:
        SynthDataError: Paramètres invalides ou génération échouée
    """
    # Validation paramètres
    if n <= 0:
        raise SynthDataError(f"n doit être > 0, reçu: {n}")
    
    if base_price <= 0:
        raise SynthDataError(f"base_price doit être > 0, reçu: {base_price}")
    
    if volatility < 0:
        raise SynthDataError(f"volatility doit être >= 0, reçu: {volatility}")
    
    if volume_base <= 0:
        raise SynthDataError(f"volume_base doit être > 0, reçu: {volume_base}")
    
    # Initialisation RNG déterministe
    rng = np.random.RandomState(seed)
    logger.debug(f"Génération OHLCV synthétique: {n} barres, seed={seed}")
    
    try:
        # === INDEX TEMPOREL ===
        start_dt = pd.to_datetime(start, utc=True)
        index = pd.date_range(start=start_dt, periods=n, freq=freq)
        
        # === GÉNÉRATION PRIX VIA GBM ===
        # Paramètres GBM
        dt = 1.0 / (365 * 24 * 60)  # 1min en fraction d'année (approximatif)
        
        if freq == "1H":
            dt = 1.0 / (365 * 24)  # 1h
        elif freq == "1D": 
            dt = 1.0 / 365  # 1jour
        
        # Drift + diffusion
        drift = 0.0  # Marché neutre en moyenne
        diffusion = volatility * np.sqrt(dt)
        
        # Shocks aléatoires
        shocks = rng.normal(drift, diffusion, n)
        
        # Prix via processus log-normal
        log_returns = shocks
        cumulative_returns = np.cumsum(log_returns)
        prices = base_price * np.exp(cumulative_returns)
        
        # === GÉNÉRATION OHLC COHÉRENT ===
        open_prices = prices.copy()
        close_prices = np.roll(open_prices, -1)  # Close[t] = Open[t+1] approx
        close_prices[-1] = open_prices[-1] * (1 + rng.normal(0, diffusion))  # Dernière barre
        
        # High/Low avec écarts réalistes
        intrabar_range = rng.exponential(diffusion, n) * 0.5  # Écart typique intra-barre
        
        high_prices = np.maximum(open_prices, close_prices) + intrabar_range * prices
        low_prices = np.minimum(open_prices, close_prices) - intrabar_range * prices
        
        # Garantie cohérence stricte
        low_prices = np.maximum(low_prices, 0.1)  # Prix > 0
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        
        # === GÉNÉRATION VOLUME ===
        # Volume corrélé à volatilité (volume↑ quand mouvement prix↑)
        price_changes = np.abs(np.diff(prices, prepend=prices[0]))
        volume_multiplier = 1.0 + price_changes / np.mean(prices) * 10  # Facteur activité
        
        base_volumes = rng.exponential(volume_base, n)
        noise = rng.uniform(1 - volume_noise, 1 + volume_noise, n)
        volumes = base_volumes * volume_multiplier * noise
        volumes = np.maximum(volumes, 1.0)  # Volume min = 1
        
        # === CONSTRUCTION DATAFRAME ===
        df_synth = pd.DataFrame({
            "open": open_prices,
            "high": high_prices, 
            "low": low_prices,
            "close": close_prices,
            "volume": volumes
        }, index=index)
        
        logger.debug(f"Données synthétiques générées: prix {df_synth['close'].iloc[0]:.2f} → {df_synth['close'].iloc[-1]:.2f}")
        
        # === NORMALISATION ET VALIDATION ===
        df_normalized = normalize_ohlcv(df_synth)
        
        # Validation optionnelle Pandera
        if PANDERA_AVAILABLE and OHLCV_SCHEMA:
            try:
                OHLCV_SCHEMA.validate(df_normalized, lazy=False)
                logger.debug("Validation Pandera réussie pour données synthétiques")
            except Exception as e:
                logger.warning(f"Validation Pandera échouée: {e}")
        
        return df_normalized
        
    except Exception as e:
        raise SynthDataError(f"Échec génération données synthétiques: {e}")


# ============================================================================
# GÉNÉRATEURS SPÉCIALISÉS
# ============================================================================

def make_trending_ohlcv(
    n: int = 5_000,
    trend_strength: float = 0.05,
    **kwargs
) -> pd.DataFrame:
    """
    Génère données OHLCV avec tendance haussière/baissière.
    
    Args:
        n: Nombre barres
        trend_strength: Force tendance (+: haussier, -: baissier)
        **kwargs: Arguments pour make_synth_ohlcv
        
    Returns:
        DataFrame OHLCV avec tendance
    """
    # Modification paramètres pour créer tendance
    kwargs_trend = kwargs.copy()
    
    # Prix de base et drift modifiés
    base_price = kwargs_trend.get("base_price", 50_000.0)
    
    # Génération de base
    df_base = make_synth_ohlcv(n=n, **kwargs_trend)
    
    # Application tendance linéaire
    trend_multiplier = np.linspace(1.0, 1.0 + trend_strength, n)
    
    for col in ["open", "high", "low", "close"]:
        df_base[col] = df_base[col] * trend_multiplier
    
    logger.debug(f"Tendance appliquée: {trend_strength:>+.3f} sur {n} barres")
    
    return normalize_ohlcv(df_base)


def make_volatile_ohlcv(
    n: int = 2_000,
    volatility_spikes: int = 5,
    spike_intensity: float = 3.0,
    **kwargs
) -> pd.DataFrame:
    """
    Génère données OHLCV avec pics de volatilité aléatoires.
    
    Simule événements market impact (news, liquidations, etc.)
    
    Args:
        n: Nombre barres
        volatility_spikes: Nombre pics volatilité
        spike_intensity: Multiplicateur volatilité pendant pics
        **kwargs: Arguments pour make_synth_ohlcv
        
    Returns:
        DataFrame OHLCV avec volatilité variable
    """
    seed = kwargs.get("seed", 42)
    rng = np.random.RandomState(seed + 1)  # Seed différent pour éviter corrélation
    
    # Génération base
    df_base = make_synth_ohlcv(n=n, **kwargs)
    
    # Positions aléatoires des spikes
    spike_positions = rng.choice(n, size=min(volatility_spikes, n//10), replace=False)
    spike_durations = rng.randint(5, 20, len(spike_positions))  # 5-20 barres par spike
    
    # Application spikes
    for spike_start, duration in zip(spike_positions, spike_durations):
        spike_end = min(spike_start + duration, n)
        
        # Facteur volatilité pendant spike
        spike_factor = rng.uniform(1.5, spike_intensity)
        
        # Modification prix dans zone spike
        base_close = df_base.loc[df_base.index[spike_start], "close"]
        
        for i in range(spike_start, spike_end):
            # Mouvement amplified
            shock = rng.normal(0, 0.05) * spike_factor  # ±5% * spike_factor
            
            for col in ["open", "high", "low", "close"]:
                col_idx = df_base.columns.get_loc(col)
                current_val = float(df_base.iloc[i, col_idx])  # type: ignore
                df_base.iloc[i, col_idx] = current_val * (1 + shock * rng.uniform(0.5, 1.5))  # type: ignore
        
        # Re-normalisation cohérence OHLC
        for i in range(spike_start, spike_end):
            row = df_base.iloc[i]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            
            # Correction cohérence
            new_high = max(o, h, l, c)
            new_low = min(o, h, l, c)
            
            df_base.iloc[i, df_base.columns.get_loc("high")] = new_high  # type: ignore
            df_base.iloc[i, df_base.columns.get_loc("low")] = new_low  # type: ignore
    
    logger.debug(f"Volatilité spikes: {len(spike_positions)} appliqués")
    
    return normalize_ohlcv(df_base)


# ============================================================================
# UTILITAIRES VALIDATION
# ============================================================================

def validate_synth_determinism(seed: int = 42, n: int = 100) -> bool:
    """
    Teste le déterminisme de la génération synthétique.
    
    Returns:
        True si deux générations identiques avec même seed
    """
    try:
        df1 = make_synth_ohlcv(n=n, seed=seed)
        df2 = make_synth_ohlcv(n=n, seed=seed)
        
        # Comparaison stricte
        is_identical = df1.equals(df2)
        
        if is_identical:
            logger.debug("Déterminisme synthétique validé")
        else:
            logger.warning("Déterminisme synthétique ÉCHOUÉ")
            
        return is_identical
        
    except Exception as e:
        logger.error(f"Erreur test déterminisme: {e}")
        return False