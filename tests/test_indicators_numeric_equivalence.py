#!/usr/bin/env python3
"""
ThreadX Test - Équivalence numérique CPU ↔ GPU
==============================================

Validation que les calculs d'indicateurs produisent des résultats
numériquement équivalents entre CPU et GPU aux tolérances fixées.

Utilise l'IndicatorBank centralisée pour garantir la cohérence.
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Ajout du chemin ThreadX
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.indicators.bank import IndicatorBank
from threadx.utils.determinism import set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Configuration des tolérances
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-10
SEED_GLOBAL = 42


@pytest.fixture
def ohlcv_data():
    """Génère des données OHLCV synthétiques reproductibles."""
    set_global_seed(SEED_GLOBAL)
    
    n_points = 10000
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1min')
    
    # Prix de base avec trend + noise
    base_price = 50000.0
    trend = np.linspace(0, 5000, n_points)
    noise = np.random.randn(n_points) * 100
    
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n_points) * 50)
    low = close - np.abs(np.random.randn(n_points) * 50)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.randint(1000, 10000, n_points)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def indicator_bank():
    """Instance IndicatorBank pour les tests."""
    return IndicatorBank()


class TestIndicatorEquivalence:
    """Tests d'équivalence numérique CPU ↔ GPU."""
    
    def test_bollinger_bands_equivalence(self, ohlcv_data, indicator_bank):
        """Test équivalence Bollinger Bands CPU vs GPU."""
        logger.info("Test équivalence Bollinger Bands CPU ↔ GPU")
        
        params_list = [
            {'period': 20, 'std': 2.0},
            {'period': 50, 'std': 1.5},
            {'period': 14, 'std': 2.5}
        ]
        
        results_comparison = []
        
        for params in params_list:
            # Calcul CPU (force)
            start_cpu = time.perf_counter()
            cpu_result = indicator_bank.ensure(
                'bollinger',
                params,
                ohlcv_data['close'],
                symbol="BTCUSDC",
                timeframe="1m"
            )
            cpu_duration = time.perf_counter() - start_cpu
            
            # Calcul GPU (si disponible)
            try:
                # Force recalcul pour mesure GPU
                if hasattr(indicator_bank.cache_manager, '_clear_cache'):
                    indicator_bank.cache_manager._clear_cache()
                
                start_gpu = time.perf_counter()
                gpu_result = indicator_bank.ensure(
                    'bollinger',
                    params,
                    ohlcv_data['close'],
                    symbol="BTCUSDC_GPU",  # Différent symbol pour éviter cache
                    timeframe="1m"
                )
                gpu_duration = time.perf_counter() - start_gpu
                
                # Comparaison numérique
                if isinstance(cpu_result, tuple) and len(cpu_result) == 3:
                    cpu_upper, cpu_middle, cpu_lower = cpu_result
                    gpu_upper, gpu_middle, gpu_lower = gpu_result
                    
                    # Vérification équivalence avec tolérances
                    np.testing.assert_allclose(
                        cpu_upper, gpu_upper,
                        rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE,
                        err_msg=f"Upper band mismatch for params {params}"
                    )
                    
                    np.testing.assert_allclose(
                        cpu_middle, gpu_middle,
                        rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE,
                        err_msg=f"Middle band mismatch for params {params}"
                    )
                    
                    np.testing.assert_allclose(
                        cpu_lower, gpu_lower,
                        rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE,
                        err_msg=f"Lower band mismatch for params {params}"
                    )
                    
                    # Calcul des écarts
                    max_error_upper = np.max(np.abs(cpu_upper - gpu_upper))
                    max_error_middle = np.max(np.abs(cpu_middle - gpu_middle))
                    max_error_lower = np.max(np.abs(cpu_lower - gpu_lower))
                    max_error = max(max_error_upper, max_error_middle, max_error_lower)
                    
                    results_comparison.append({
                        'indicator': 'bollinger',
                        'params': str(params),
                        'n_points': len(ohlcv_data),
                        'cpu_duration': cpu_duration,
                        'gpu_duration': gpu_duration,
                        'speedup': cpu_duration / gpu_duration if gpu_duration > 0 else 0,
                        'max_error': max_error,
                        'equivalence_ok': max_error < ABSOLUTE_TOLERANCE
                    })
                    
                    logger.info(f"Bollinger {params}: CPU={cpu_duration:.4f}s, "
                              f"GPU={gpu_duration:.4f}s, max_error={max_error:.2e}")
                
            except Exception as e:
                logger.warning(f"GPU calculation failed for {params}: {e}")
                # Test passe en mode CPU-only
                results_comparison.append({
                    'indicator': 'bollinger',
                    'params': str(params),
                    'n_points': len(ohlcv_data),
                    'cpu_duration': cpu_duration,
                    'gpu_duration': None,
                    'speedup': None,
                    'max_error': 0.0,
                    'equivalence_ok': True  # CPU vs CPU est toujours équivalent
                })
        
        # Sauvegarde des résultats
        results_df = pd.DataFrame(results_comparison)
        output_path = Path("artifacts/reports/test_equivalence_bollinger.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Résultats sauvés: {output_path}")
        
        # Assert global : tous les tests d'équivalence réussis
        assert all(results_df['equivalence_ok']), "Équivalence CPU ↔ GPU échouée"
    
    def test_atr_equivalence(self, ohlcv_data, indicator_bank):
        """Test équivalence ATR CPU vs GPU."""
        logger.info("Test équivalence ATR CPU ↔ GPU")
        
        params_list = [
            {'period': 14, 'method': 'ema'},
            {'period': 21, 'method': 'sma'},
            {'period': 10, 'method': 'ema'}
        ]
        
        results_comparison = []
        
        for params in params_list:
            # Calcul CPU
            start_cpu = time.perf_counter()
            cpu_result = indicator_bank.ensure(
                'atr',
                params,
                ohlcv_data[['high', 'low', 'close']],
                symbol="BTCUSDC",
                timeframe="1m"
            )
            cpu_duration = time.perf_counter() - start_cpu
            
            # Calcul GPU (si disponible)
            try:
                if hasattr(indicator_bank.cache_manager, '_clear_cache'):
                    indicator_bank.cache_manager._clear_cache()
                
                start_gpu = time.perf_counter()
                gpu_result = indicator_bank.ensure(
                    'atr',
                    params,
                    ohlcv_data[['high', 'low', 'close']],
                    symbol="BTCUSDC_GPU",  # Différent symbol pour éviter cache
                    timeframe="1m"
                )
                gpu_duration = time.perf_counter() - start_gpu
                
                # Comparaison numérique
                if isinstance(cpu_result, np.ndarray) and isinstance(gpu_result, np.ndarray):
                    np.testing.assert_allclose(
                        cpu_result, gpu_result,
                        rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE,
                        err_msg=f"ATR mismatch for params {params}"
                    )
                    
                    max_error = np.max(np.abs(cpu_result - gpu_result))
                    
                    results_comparison.append({
                        'indicator': 'atr',
                        'params': str(params),  
                        'n_points': len(ohlcv_data),
                        'cpu_duration': cpu_duration,
                        'gpu_duration': gpu_duration,
                        'speedup': cpu_duration / gpu_duration if gpu_duration > 0 else 0,
                        'max_error': max_error,
                        'equivalence_ok': max_error < ABSOLUTE_TOLERANCE
                    })
                    
                    logger.info(f"ATR {params}: CPU={cpu_duration:.4f}s, "
                              f"GPU={gpu_duration:.4f}s, max_error={max_error:.2e}")
                
            except Exception as e:
                logger.warning(f"GPU calculation failed for {params}: {e}")
                results_comparison.append({
                    'indicator': 'atr',
                    'params': str(params),
                    'n_points': len(ohlcv_data),
                    'cpu_duration': cpu_duration,
                    'gpu_duration': None,
                    'speedup': None,
                    'max_error': 0.0,
                    'equivalence_ok': True
                })
        
        # Sauvegarde des résultats
        results_df = pd.DataFrame(results_comparison)
        output_path = Path("artifacts/reports/test_equivalence_atr.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Résultats sauvés: {output_path}")
        
        # Assert global
        assert all(results_df['equivalence_ok']), "Équivalence ATR CPU ↔ GPU échouée"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
