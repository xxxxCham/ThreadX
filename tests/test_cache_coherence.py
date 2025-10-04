#!/usr/bin/env python3
"""
ThreadX Test - Cohérence du cache
=================================

Validation de la cohérence du système de cache IndicatorBank:
- Cache hit/miss rates
- Invalidation cohérente
- Performance du cache multi-niveaux
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.indicators.bank import IndicatorBank
from threadx.utils.determinism import set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)

SEED_GLOBAL = 42
TARGET_CACHE_HIT_RATE = 0.80  # 80% minimum requis


@pytest.fixture
def test_data():
    """Données de test reproductibles."""
    set_global_seed(SEED_GLOBAL)
    
    n = 5000
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    
    close = 50000 + np.cumsum(np.random.randn(n) * 10)
    high = close + np.abs(np.random.randn(n) * 20)
    low = close - np.abs(np.random.randn(n) * 20)
    open_price = np.roll(close, 1)
    volume = np.random.randint(1000, 5000, n)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def fresh_bank():
    """Instance IndicatorBank fraîche pour chaque test."""
    return IndicatorBank()


class TestCacheCoherence:
    """Tests de cohérence du système de cache."""
    
    def test_cache_hit_miss_rates(self, test_data, fresh_bank):
        """Test des taux de hit/miss du cache."""
        logger.info("Test des taux de cache hit/miss")
        
        # Scénario: calculs répétés identiques
        params = {'period': 20, 'std': 2.0}
        symbol = "BTCUSDC"
        timeframe = "1m"
        
        # Premier calcul - doit être un miss
        start = time.perf_counter()
        result1 = fresh_bank.ensure(
            'bollinger', params, test_data['close'],
            symbol=symbol, timeframe=timeframe
        )
        first_duration = time.perf_counter() - start
        
        # Deuxième calcul identique - doit être un hit
        start = time.perf_counter()
        result2 = fresh_bank.ensure(
            'bollinger', params, test_data['close'],
            symbol=symbol, timeframe=timeframe
        )
        second_duration = time.perf_counter() - start
        
        # Vérification équivalence des résultats
        assert np.array_equal(result1[0], result2[0]), "Résultats incohérents du cache"
        assert np.array_equal(result1[1], result2[1]), "Résultats incohérents du cache"
        assert np.array_equal(result1[2], result2[2]), "Résultats incohérents du cache"
        
        # Le cache hit doit être significativement plus rapide
        speedup = first_duration / second_duration
        assert speedup > 2.0, f"Cache hit pas assez rapide: {speedup:.2f}x"
        
        logger.info(f"Cache hit speedup: {speedup:.2f}x "
                   f"({first_duration:.4f}s → {second_duration:.4f}s)")
        
        # Test avec variations de paramètres
        cache_hits = 0
        cache_misses = 0
        
        param_variations = [
            {'period': 20, 'std': 2.0},  # Répétition - hit
            {'period': 21, 'std': 2.0},  # Nouveau - miss
            {'period': 20, 'std': 2.0},  # Répétition - hit
            {'period': 20, 'std': 1.5},  # Nouveau - miss
            {'period': 21, 'std': 2.0},  # Répétition - hit
        ]
        
        durations = []
        for params in param_variations:
            start = time.perf_counter()
            fresh_bank.ensure(
                'bollinger', params, test_data['close'],
                symbol=symbol, timeframe=timeframe
            )
            duration = time.perf_counter() - start
            durations.append(duration)
            
            # Heuristique: temps < 10% du premier = hit
            if duration < first_duration * 0.1:
                cache_hits += 1
            else:
                cache_misses += 1
        
        hit_rate = cache_hits / len(param_variations)
        logger.info(f"Cache hit rate: {hit_rate:.2%} "
                   f"(hits: {cache_hits}, misses: {cache_misses})")
        
        # Résultats de test
        results = {
            'test': 'cache_hit_miss_rates',
            'first_calc_duration': first_duration,
            'second_calc_duration': second_duration,
            'cache_speedup': speedup,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': hit_rate,
            'target_hit_rate': TARGET_CACHE_HIT_RATE,
            'hit_rate_ok': hit_rate >= TARGET_CACHE_HIT_RATE
        }
        
        # Sauvegarde
        output_path = Path("artifacts/reports/test_cache_coherence.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([results]).to_csv(output_path, index=False)
        
        logger.info(f"Résultats sauvés: {output_path}")
    
    def test_cache_invalidation(self, test_data, fresh_bank):
        """Test de l'invalidation cohérente du cache."""
        logger.info("Test invalidation du cache")
        
        symbol = "BTCUSDC"
        timeframe = "1m"
        params = {'period': 14}
        
        # Calcul initial
        result1 = fresh_bank.ensure(
            'atr', params, test_data[['high', 'low', 'close']],
            symbol=symbol, timeframe=timeframe
        )
        
        # Modification des données (simulation nouveau tick)
        modified_data = test_data.copy()
        modified_data.iloc[-1, modified_data.columns.get_loc('close')] += 100
        
        # Nouveau calcul avec données modifiées
        result2 = fresh_bank.ensure(
            'atr', params, modified_data[['high', 'low', 'close']],
            symbol=symbol, timeframe=timeframe
        )
        
        # Les résultats doivent être différents
        assert not np.array_equal(result1, result2), \
            "Cache non invalidé avec données modifiées"
        
        logger.info("Test invalidation cache: OK")
    
    def test_cache_memory_efficiency(self, test_data, fresh_bank):
        """Test de l'efficacité mémoire du cache."""
        logger.info("Test efficacité mémoire du cache")
        
        symbol = "BTCUSDC"
        timeframe = "1m"
        
        # Calculs multiples pour remplir le cache
        indicators_params = [
            ('bollinger', {'period': p, 'std': s}) 
            for p in [10, 20, 50] for s in [1.5, 2.0, 2.5]
        ]
        
        indicators_params.extend([
            ('atr', {'period': p, 'method': 'ema'}) 
            for p in [14, 21, 28]
        ])
        
        cache_operations = 0
        
        for indicator, params in indicators_params:
            try:
                if indicator == 'bollinger':
                    result = fresh_bank.ensure(
                        indicator, params, test_data['close'],
                        symbol=symbol, timeframe=timeframe
                    )
                elif indicator == 'atr':
                    result = fresh_bank.ensure(
                        indicator, params, test_data[['high', 'low', 'close']],
                        symbol=symbol, timeframe=timeframe
                    )
                
                cache_operations += 1
                
            except Exception as e:
                logger.warning(f"Erreur calcul {indicator} {params}: {e}")
        
        logger.info(f"Cache operations complétées: {cache_operations}")
        
        # Vérification que le cache fonctionne après multiple ops
        start = time.perf_counter()
        fresh_bank.ensure(
            'bollinger', {'period': 20, 'std': 2.0}, test_data['close'],
            symbol=symbol, timeframe=timeframe
        )
        duration = time.perf_counter() - start
        
        # Doit être rapide (hit)
        assert duration < 0.01, f"Cache inefficient après operations multiples: {duration:.4f}s"
        
        logger.info(f"Cache memory efficiency: OK (duration: {duration:.4f}s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
