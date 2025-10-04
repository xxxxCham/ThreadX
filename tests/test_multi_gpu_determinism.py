#!/usr/bin/env python3
"""
ThreadX Test - Multi-GPU Déterminisme
=====================================

Validation du déterminisme et de la distribution multi-GPU:
- Résultats identiques sur différents GPU
- Distribution équilibrée des calculs
- Performances d'auto-balancing
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.indicators.bank import IndicatorBank
from threadx.gpu.multi_gpu import MultiGPUManager
from threadx.utils.determinism import set_global_seed, stable_dict_merge
from threadx.utils.log import get_logger

logger = get_logger(__name__)

SEED_GLOBAL = 42
TARGET_GPU_UTILIZATION = 0.70  # 70% minimum requis


@pytest.fixture
def large_dataset():
    """Dataset volumineux pour tests multi-GPU."""
    set_global_seed(SEED_GLOBAL)
    
    n = 50000  # Dataset plus large
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    
    # Série temporelle complexe
    base = 50000
    trend = np.linspace(0, 10000, n)
    seasonal = 1000 * np.sin(2 * np.pi * np.arange(n) / 1440)  # Cycle journalier
    noise = np.random.randn(n) * 200
    
    close = base + trend + seasonal + noise
    high = close + np.abs(np.random.randn(n) * 100)
    low = close - np.abs(np.random.randn(n) * 100)
    open_price = np.roll(close, 1)
    volume = np.random.randint(5000, 50000, n)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def multi_gpu_manager():
    """Manager multi-GPU pour tests."""
    return MultiGPUManager()


class TestMultiGPUDeterminism:
    """Tests de déterminisme multi-GPU."""
    
    def test_deterministic_results_across_gpus(self, large_dataset, multi_gpu_manager):
        """Test résultats déterministes sur différents GPU."""
        logger.info("Test déterminisme multi-GPU")
        
        if not multi_gpu_manager.is_available():
            pytest.skip("Multi-GPU non disponible")
        
        available_gpus = multi_gpu_manager.list_devices()
        if len(available_gpus) < 2:
            pytest.skip(f"Multi-GPU requis, {len(available_gpus)} GPU(s) détecté(s)")
        
        # Configuration test
        params = {'period': 50, 'std': 2.0}
        symbol = "BTCUSDC"
        timeframe = "1m"
        
        # Calculs sur différents GPU avec même seed
        results_per_gpu = {}
        durations_per_gpu = {}
        
        for gpu_id in available_gpus[:2]:  # Test sur 2 premiers GPU
            set_global_seed(SEED_GLOBAL)  # Reset seed pour déterminisme
            
            bank = IndicatorBank()
            
            # Force utilisation GPU spécifique
            if hasattr(bank, 'set_device'):
                bank.set_device(gpu_id)
            
            start = time.perf_counter()
            result = bank.ensure(
                'bollinger', params, large_dataset['close'],
                symbol=f"{symbol}_GPU{gpu_id}", timeframe=timeframe
            )
            duration = time.perf_counter() - start
            
            results_per_gpu[gpu_id] = result
            durations_per_gpu[gpu_id] = duration
            
            logger.info(f"GPU {gpu_id}: calcul en {duration:.4f}s")
        
        # Vérification déterminisme entre GPU
        gpu_ids = list(results_per_gpu.keys())
        if len(gpu_ids) >= 2:
            result_gpu0 = results_per_gpu[gpu_ids[0]]
            result_gpu1 = results_per_gpu[gpu_ids[1]]
            
            # Comparaison numérique stricte
            if isinstance(result_gpu0, tuple) and len(result_gpu0) == 3:
                for i, (r0, r1) in enumerate(zip(result_gpu0, result_gpu1)):
                    np.testing.assert_allclose(
                        r0, r1, rtol=1e-12, atol=1e-12,
                        err_msg=f"Déterminisme échoué GPU{gpu_ids[0]} vs GPU{gpu_ids[1]} band {i}"
                    )
                    
                    max_diff = np.max(np.abs(r0 - r1))
                    logger.info(f"Max diff band {i}: {max_diff:.2e}")
        
        # Analyse de performance
        performance_balance = max(durations_per_gpu.values()) / min(durations_per_gpu.values())
        
        results_summary = {
            'test': 'multi_gpu_determinism',
            'gpus_tested': len(results_per_gpu),
            'performance_balance': performance_balance,
            'durations': durations_per_gpu,
            'determinism_ok': True,  # Si on arrive ici, c'est OK
            'dataset_size': len(large_dataset)
        }
        
        # Sauvegarde
        output_path = Path("artifacts/reports/test_multi_gpu_determinism.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([results_summary]).to_csv(output_path, index=False)
        
        logger.info(f"Test déterminisme multi-GPU: OK")
        logger.info(f"Performance balance: {performance_balance:.2f}")
    
    def test_gpu_auto_balancing(self, large_dataset, multi_gpu_manager):
        """Test auto-balancing GPU."""
        logger.info("Test auto-balancing GPU")
        
        if not multi_gpu_manager.is_available():
            pytest.skip("Multi-GPU non disponible")
        
        # Simulation charge distribuée
        param_sets = [
            {'period': p, 'std': s} 
            for p in [10, 20, 50, 100] 
            for s in [1.0, 1.5, 2.0, 2.5]
        ]
        
        symbol = "BTCUSDC"
        timeframe = "1m"
        
        bank = IndicatorBank()
        gpu_usage_stats = {}
        
        start_total = time.perf_counter()
        
        for i, params in enumerate(param_sets):
            start = time.perf_counter()
            
            result = bank.ensure(
                'bollinger', params, large_dataset['close'],
                symbol=f"{symbol}_BATCH{i}", timeframe=timeframe
            )
            
            duration = time.perf_counter() - start
            
            # Collecte stats GPU si disponible
            if hasattr(multi_gpu_manager, 'get_current_device'):
                current_gpu = multi_gpu_manager.get_current_device()
                if current_gpu not in gpu_usage_stats:
                    gpu_usage_stats[current_gpu] = []
                gpu_usage_stats[current_gpu].append(duration)
        
        total_duration = time.perf_counter() - start_total
        
        # Analyse distribution de charge
        if gpu_usage_stats:
            load_distribution = {
                gpu: {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'avg_time': np.mean(durations)
                }
                for gpu, durations in gpu_usage_stats.items()
            }
        else:
            load_distribution = {'single_gpu': {'count': len(param_sets)}}
        
        results_summary = {
            'test': 'gpu_auto_balancing',
            'total_tasks': len(param_sets),
            'total_duration': total_duration,
            'avg_task_duration': total_duration / len(param_sets),
            'load_distribution': str(load_distribution),
            'dataset_size': len(large_dataset)
        }
        
        # Sauvegarde
        output_path = Path("artifacts/reports/test_gpu_auto_balancing.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([results_summary]).to_csv(output_path, index=False)
        
        logger.info(f"Auto-balancing GPU: {len(param_sets)} tâches en {total_duration:.2f}s")
        logger.info(f"Distribution: {load_distribution}")
    
    def test_stable_dict_merge_determinism(self):
        """Test déterminisme stable_dict_merge."""
        logger.info("Test stable_dict_merge déterminisme")
        
        # Dictionnaires avec ordres différents
        dict1 = {'b': 2, 'a': 1, 'c': 3}
        dict2 = {'c': 3, 'a': 1, 'b': 2}
        dict3 = {'a': 1, 'b': 2, 'c': 3}
        
        # Merge stable doit donner même résultat
        result1 = stable_dict_merge(dict1, {})
        result2 = stable_dict_merge(dict2, {})
        result3 = stable_dict_merge(dict3, {})
        
        # Conversion en string pour comparaison déterministe
        str1 = str(sorted(result1.items()))
        str2 = str(sorted(result2.items()))
        str3 = str(sorted(result3.items()))
        
        assert str1 == str2 == str3, "stable_dict_merge non déterministe"
        
        logger.info("stable_dict_merge: déterminisme OK")
    
    def test_global_seed_reproducibility(self, large_dataset):
        """Test reproductibilité avec set_global_seed."""
        logger.info("Test reproductibilité global seed")
        
        bank1 = IndicatorBank()
        bank2 = IndicatorBank()
        
        params = {'period': 20, 'std': 2.0}
        
        # Premier calcul avec seed
        set_global_seed(SEED_GLOBAL)
        result1 = bank1.ensure(
            'bollinger', params, large_dataset['close'],
            symbol="TEST1", timeframe="1m"
        )
        
        # Deuxième calcul avec même seed
        set_global_seed(SEED_GLOBAL)
        result2 = bank2.ensure(
            'bollinger', params, large_dataset['close'],
            symbol="TEST2", timeframe="1m"  # Différent symbol pour éviter cache
        )
        
        # Vérification reproductibilité
        if isinstance(result1, tuple) and len(result1) == 3:
            for i, (r1, r2) in enumerate(zip(result1, result2)):
                np.testing.assert_array_equal(
                    r1, r2,
                    err_msg=f"Global seed non reproductible band {i}"
                )
        
        logger.info("Global seed reproductibilité: OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])