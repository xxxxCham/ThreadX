#!/usr/bin/env python3
"""
Tests unitaires pour ThreadX ATR (Average True Range)
====================================================

Tests complets du module atr.py incluant:
- Calculs CPU vs GPU pour ATR
- Validation mathématique EMA vs SMA
- Performance et benchmarks
- Gestion d'erreurs et edge cases
- Batch processing et multi-GPU
"""

import unittest
import numpy as np
import pandas as pd
import time
from unittest.mock import patch, MagicMock

# Import du module à tester
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from threadx.indicators.atr import (
    ATR,
    ATRSettings,
    compute_atr,
    compute_atr_batch,
    validate_atr_results,
    benchmark_atr_performance,
    ATRGPUManager
)


class TestATRSettings(unittest.TestCase):
    """Tests de la configuration ATRSettings"""
    
    def test_default_settings(self):
        """Test des valeurs par défaut"""
        settings = ATRSettings()
        
        self.assertEqual(settings.period, 14)
        self.assertEqual(settings.method, 'ema')
        self.assertTrue(settings.use_gpu)
        self.assertEqual(settings.gpu_batch_size, 1000)
        self.assertTrue(settings.cpu_fallback)
        self.assertEqual(settings.gpu_split_ratio, (0.75, 0.25))
    
    def test_custom_settings(self):
        """Test configuration personnalisée"""
        settings = ATRSettings(
            period=21,
            method='sma',
            use_gpu=False,
            gpu_batch_size=500
        )
        
        self.assertEqual(settings.period, 21)
        self.assertEqual(settings.method, 'sma')
        self.assertFalse(settings.use_gpu)
        self.assertEqual(settings.gpu_batch_size, 500)
    
    def test_validation_period(self):
        """Test validation période"""
        with self.assertRaises(ValueError):
            ATRSettings(period=0)
        
        # Period = 1 doit être accepté pour ATR
        settings = ATRSettings(period=1)
        self.assertEqual(settings.period, 1)
    
    def test_validation_method(self):
        """Test validation méthode"""
        with self.assertRaises(ValueError):
            ATRSettings(method='invalid')  # type: ignore[arg-type]  # Test de validation
        
        # Méthodes valides
        settings_ema = ATRSettings(method='ema')
        settings_sma = ATRSettings(method='sma')
        self.assertEqual(settings_ema.method, 'ema')
        self.assertEqual(settings_sma.method, 'sma')
    
    def test_validation_gpu_split(self):
        """Test validation répartition GPU"""
        with self.assertRaises(ValueError):
            ATRSettings(gpu_split_ratio=(0.9, 0.9))  # Sum > 1.0
        
        with self.assertRaises(ValueError):
            ATRSettings(gpu_split_ratio=(0.05, 0.02))  # Sum < 0.1


class TestATRGPUManager(unittest.TestCase):
    """Tests du gestionnaire GPU ATR"""
    
    def setUp(self):
        """Setup pour chaque test"""
        self.settings = ATRSettings()
    
    def test_atr_gpu_manager_init_no_gpu(self):
        """Test initialisation sans GPU"""
        with patch('threadx.indicators.atr.HAS_CUPY', False):
            with patch('threadx.indicators.atr.GPU_AVAILABLE', False):
                manager = ATRGPUManager(self.settings)
                self.assertEqual(len(manager.available_gpus), 0)
                self.assertEqual(len(manager.gpu_capabilities), 0)
    
    def test_split_workload_consistency(self):
        """Test cohérence répartition workload"""
        manager = ATRGPUManager(self.settings)
        manager.available_gpus = [0, 1]  # Mock dual GPU
        
        total_size = 1000
        splits = manager.split_workload(total_size)
        
        if len(splits) >= 2:
            # Vérification que la somme des splits = total
            total_processed = sum(end - start for _, start, end in splits)
            self.assertEqual(total_processed, total_size)
            
            # Vérification pas de chevauchement
            splits_sorted = sorted(splits, key=lambda x: x[1])  # Sort by start
            for i in range(len(splits_sorted) - 1):
                current_end = splits_sorted[i][2]
                next_start = splits_sorted[i+1][1]
                self.assertEqual(current_end, next_start)


class TestATR(unittest.TestCase):
    """Tests de la classe ATR"""
    
    def setUp(self):
        """Setup données test OHLC pour chaque test"""
        np.random.seed(42)
        self.n = 1000
        
        # Génération données OHLC cohérentes
        base_price = 100
        returns = np.random.randn(self.n) * 0.02  # Rendements 2%
        close = base_price * np.cumprod(1 + returns)
        
        # High/Low autour du close avec spread réaliste
        spread = np.random.uniform(0.5, 3.0, self.n)  # Spread 0.5-3%
        self.high = close * (1 + spread/200)
        self.low = close * (1 - spread/200)
        self.close = close
        
        # Conversion en séries pandas
        self.high_series = pd.Series(self.high)
        self.low_series = pd.Series(self.low)
        self.close_series = pd.Series(self.close)
        
        # Settings CPU pour éviter problèmes GPU en tests
        self.settings_cpu = ATRSettings(use_gpu=False)
        self.atr_cpu = ATR(self.settings_cpu)
    
    def test_atr_init(self):
        """Test initialisation ATR"""
        atr = ATR()
        self.assertIsInstance(atr.settings, ATRSettings)
        self.assertIsInstance(atr.gpu_manager, ATRGPUManager)
        self.assertIsInstance(atr._cache, dict)
    
    def test_compute_basic_ema(self):
        """Test calcul basique ATR avec EMA"""
        atr_values = self.atr_cpu.compute(
            self.high, self.low, self.close, 
            period=14, method='ema'
        )
        
        # Vérifications de base
        self.assertEqual(len(atr_values), self.n)
        
        # ATR doit être >= 0
        valid_idx = ~np.isnan(atr_values)
        self.assertTrue(np.all(atr_values[valid_idx] >= 0))
        
        # Premier point doit être valide (ATR calculable dès le 2ème point)
        self.assertFalse(np.isnan(atr_values[1]))
    
    def test_compute_basic_sma(self):
        """Test calcul basique ATR avec SMA"""
        atr_values = self.atr_cpu.compute(
            self.high, self.low, self.close,
            period=14, method='sma'
        )
        
        # Vérifications de base
        self.assertEqual(len(atr_values), self.n)
        self.assertTrue(np.all(atr_values[~np.isnan(atr_values)] >= 0))
        
        # Avec SMA, on a period-1 NaN au début
        nan_count = np.sum(np.isnan(atr_values))
        self.assertEqual(nan_count, 13)  # period-1 = 14-1
    
    def test_compute_different_periods(self):
        """Test calcul avec différentes périodes"""
        periods = [7, 14, 21, 30]
        
        for period in periods:
            atr_ema = self.atr_cpu.compute(
                self.high, self.low, self.close,
                period=period, method='ema'
            )
            atr_sma = self.atr_cpu.compute(
                self.high, self.low, self.close,
                period=period, method='sma'
            )
            
            # Validation basique
            self.assertEqual(len(atr_ema), self.n)
            self.assertEqual(len(atr_sma), self.n)
            
            # EMA plus réactif que SMA (généralement)
            valid_idx = ~(np.isnan(atr_ema) | np.isnan(atr_sma))
            if np.any(valid_idx):
                # Les deux doivent être positifs
                self.assertTrue(np.all(atr_ema[valid_idx] >= 0))
                self.assertTrue(np.all(atr_sma[valid_idx] >= 0))
    
    def test_compute_pandas_series(self):
        """Test calcul avec pandas Series"""
        atr_values = self.atr_cpu.compute(
            self.high_series, self.low_series, self.close_series,
            period=14, method='ema'
        )
        
        # Même résultat qu'avec numpy arrays
        atr_numpy = self.atr_cpu.compute(
            self.high, self.low, self.close,
            period=14, method='ema'
        )
        
        np.testing.assert_array_almost_equal(atr_values, atr_numpy, decimal=10)
    
    def test_compute_mismatched_lengths(self):
        """Test avec longueurs différentes"""
        high_short = self.high[:500]  # Plus court
        
        with self.assertRaises(ValueError):
            self.atr_cpu.compute(high_short, self.low, self.close)
    
    def test_compute_insufficient_data(self):
        """Test avec données insuffisantes"""
        # Seulement 1 point (< period+1 requis)
        short_high = np.array([105])
        short_low = np.array([95])
        short_close = np.array([100])
        
        with self.assertRaises(ValueError):
            self.atr_cpu.compute(short_high, short_low, short_close, period=14)
    
    def test_true_range_calculation(self):
        """Test calcul True Range manuel vs implémentation"""
        # Données test simples
        high = np.array([105, 107, 103, 108])
        low = np.array([95, 97, 93, 98])
        close = np.array([100, 102, 98, 105])
        
        # Calcul TR manuel
        # TR[i] = max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
        manual_tr = []
        for i in range(len(high)):
            if i == 0:
                # Premier point: prev_close = close[0]
                prev_close = close[0]
            else:
                prev_close = close[i-1]
            
            hl_diff = high[i] - low[i]
            hc_diff = abs(high[i] - prev_close)
            lc_diff = abs(low[i] - prev_close)
            
            tr = max(hl_diff, hc_diff, lc_diff)
            manual_tr.append(tr)
        
        # Calcul avec notre implémentation
        computed_tr = self.atr_cpu._true_range_cpu(high, low, close)
        
        # Comparaison
        np.testing.assert_array_almost_equal(computed_tr, manual_tr, decimal=10)
    
    def test_compute_batch_simple(self):
        """Test calcul batch simple"""
        params_list = [
            {'period': 14, 'method': 'ema'},
            {'period': 21, 'method': 'sma'},
            {'period': 7, 'method': 'ema'}
        ]
        
        results = self.atr_cpu.compute_batch(
            self.high, self.low, self.close, params_list
        )
        
        # Vérification des clés
        expected_keys = ['14_ema', '21_sma', '7_ema']
        self.assertEqual(set(results.keys()), set(expected_keys))
        
        # Vérification des résultats
        for key, result in results.items():
            self.assertIsNotNone(result)
            self.assertEqual(len(result), self.n)
            
            # Validation
            valid = validate_atr_results(result)
            self.assertTrue(valid)
    
    def test_compute_batch_with_errors(self):
        """Test batch avec paramètres invalides"""
        params_list = [
            {'period': 14, 'method': 'ema'},    # Valide
            {'period': 0, 'method': 'ema'},     # Invalide: period = 0
            {'period': 14, 'method': 'invalid'} # Invalide: méthode inconnue
        ]
        
        results = self.atr_cpu.compute_batch(
            self.high, self.low, self.close, params_list
        )
        
        # Premier résultat doit être valide
        self.assertIsNotNone(results.get('14_ema'))
        
        # Autres peuvent être None (erreurs)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)  # Toutes les clés présentes
    
    def test_mathematical_accuracy_ema(self):
        """Test précision mathématique EMA vs calcul manuel"""
        period = 14
        
        # Calcul avec notre implémentation
        atr_values = self.atr_cpu.compute(
            self.high, self.low, self.close, 
            period=period, method='ema'
        )
        
        # Calcul manuel
        manual_tr = self.atr_cpu._true_range_cpu(self.high, self.low, self.close)
        manual_atr = pd.Series(manual_tr).ewm(span=period, adjust=False).mean().values
        
        # Comparaison
        np.testing.assert_array_almost_equal(np.array(atr_values), np.array(manual_atr), decimal=10)
    
    def test_mathematical_accuracy_sma(self):
        """Test précision mathématique SMA vs calcul manuel"""
        period = 14
        
        # Calcul avec notre implémentation
        atr_values = self.atr_cpu.compute(
            self.high, self.low, self.close,
            period=period, method='sma'
        )
        
        # Calcul manuel
        manual_tr = self.atr_cpu._true_range_cpu(self.high, self.low, self.close)
        manual_atr = pd.Series(manual_tr).rolling(window=period, min_periods=period).mean().values
        
        # Comparaison
        np.testing.assert_array_almost_equal(atr_values, np.asarray(manual_atr), decimal=10)


class TestATRPublicAPI(unittest.TestCase):
    """Tests des fonctions publiques de l'API"""
    
    def setUp(self):
        """Setup données test OHLC"""
        np.random.seed(42)
        n = 500
        
        base_price = 100
        returns = np.random.randn(n) * 0.015
        close = base_price * np.cumprod(1 + returns)
        
        spread = np.random.uniform(0.3, 2.0, n)
        self.high = close * (1 + spread/200)
        self.low = close * (1 - spread/200)
        self.close = close
    
    def test_compute_atr_basic(self):
        """Test API simple compute_atr"""
        atr_values = compute_atr(
            self.high, self.low, self.close,
            period=14, method='ema', use_gpu=False
        )
        
        self.assertEqual(len(atr_values), len(self.close))
        
        # Validation
        valid = validate_atr_results(atr_values)
        self.assertTrue(valid)
    
    def test_compute_atr_batch_basic(self):
        """Test API batch compute_atr_batch"""
        params_list = [
            {'period': 14, 'method': 'ema'},
            {'period': 21, 'method': 'sma'}
        ]
        
        results = compute_atr_batch(
            self.high, self.low, self.close,
            params_list, use_gpu=False
        )
        
        self.assertEqual(len(results), 2)
        self.assertIn('14_ema', results)
        self.assertIn('21_sma', results)
        
        for key, result in results.items():
            if result is not None:
                valid = validate_atr_results(result)
                self.assertTrue(valid)


class TestATRValidation(unittest.TestCase):
    """Tests de validation des résultats ATR"""
    
    def test_validate_atr_results_valid(self):
        """Test validation avec résultats valides"""
        atr_values = np.array([1.5, 2.3, 1.8, 2.1, 1.9])
        
        valid = validate_atr_results(atr_values)
        self.assertTrue(valid)
    
    def test_validate_atr_results_negative(self):
        """Test validation avec valeurs négatives (invalide)"""
        atr_values = np.array([1.5, -0.3, 1.8, 2.1, 1.9])  # Une valeur négative
        
        valid = validate_atr_results(atr_values)
        self.assertFalse(valid)
    
    def test_validate_atr_results_infinite(self):
        """Test validation avec valeurs infinies (invalide)"""
        atr_values = np.array([1.5, np.inf, 1.8, 2.1, 1.9])
        
        valid = validate_atr_results(atr_values)
        self.assertFalse(valid)
    
    def test_validate_atr_results_with_nan(self):
        """Test validation avec valeurs NaN (valide)"""
        atr_values = np.array([np.nan, np.nan, 1.8, 2.1, 1.9])
        
        # NaN au début acceptable (données insuffisantes)
        valid = validate_atr_results(atr_values)
        self.assertTrue(valid)
    
    def test_validate_atr_results_all_nan(self):
        """Test validation avec que des NaN"""
        atr_values = np.full(10, np.nan)
        
        # Valide si toutes NaN (données complètement insuffisantes)
        valid = validate_atr_results(atr_values)
        self.assertTrue(valid)


class TestATRPerformance(unittest.TestCase):
    """Tests de performance ATR"""
    
    def test_benchmark_atr_performance(self):
        """Test du benchmark de performance"""
        bench_results = benchmark_atr_performance(
            data_sizes=[100, 500],
            n_runs=2
        )
        
        # Vérifications structure résultats
        self.assertIn('cpu_times', bench_results)
        self.assertIn('gpu_times', bench_results)
        self.assertIn('speedups', bench_results)
        self.assertIn('gpu_available', bench_results)
        
        # CPU times doivent être positifs
        self.assertGreater(bench_results['cpu_times'][100], 0)
        self.assertGreater(bench_results['cpu_times'][500], 0)
    
    def test_performance_ema_vs_sma(self):
        """Test performance EMA vs SMA"""
        np.random.seed(42)
        n = 2000
        
        # Données OHLC
        base_price = 100
        returns = np.random.randn(n) * 0.02
        close = base_price * np.cumprod(1 + returns)
        spread = np.random.uniform(0.5, 2.0, n)
        high = close * (1 + spread/200)
        low = close * (1 - spread/200)
        
        # Test EMA
        start_time = time.time()
        atr_ema = compute_atr(high, low, close, period=14, method='ema', use_gpu=False)
        ema_time = time.time() - start_time
        
        # Test SMA
        start_time = time.time()
        atr_sma = compute_atr(high, low, close, period=14, method='sma', use_gpu=False)
        sma_time = time.time() - start_time
        
        # Validation que les deux donnent des résultats cohérents
        self.assertTrue(validate_atr_results(atr_ema))
        self.assertTrue(validate_atr_results(atr_sma))
        
        # Log performance (pas d'assertion car dépend de la machine)
        print(f"\nPerformance ATR (n={n}):")
        print(f"  EMA: {ema_time:.4f}s")
        print(f"  SMA: {sma_time:.4f}s")
        print(f"  Ratio SMA/EMA: {sma_time/ema_time:.2f}x")


class TestATREdgeCases(unittest.TestCase):
    """Tests des cas limites ATR"""
    
    def setUp(self):
        """Setup"""
        self.settings_cpu = ATRSettings(use_gpu=False)
        self.atr = ATR(self.settings_cpu)
    
    def test_zero_volatility(self):
        """Test avec volatilité nulle"""
        # Prix constants
        n = 100
        price = 100.0
        high = np.full(n, price)
        low = np.full(n, price)
        close = np.full(n, price)
        
        atr_values = self.atr.compute(high, low, close, period=14, method='ema')
        
        # ATR doit être proche de 0 avec volatilité nulle
        valid_idx = ~np.isnan(atr_values)
        if np.any(valid_idx):
            self.assertTrue(np.all(atr_values[valid_idx] < 1e-10))
    
    def test_extreme_gaps(self):
        """Test avec gaps extrêmes"""
        # Création de gaps importants
        high = np.array([105, 200, 95, 150])    # Gap up puis gap down
        low = np.array([95, 190, 85, 140])
        close = np.array([100, 195, 90, 145])
        
        atr_values = self.atr.compute(high, low, close, period=2, method='ema')
        
        # ATR doit capturer la volatilité des gaps
        valid_idx = ~np.isnan(atr_values)
        if np.any(valid_idx):
            # Avec des gaps importants, ATR doit être élevé
            max_atr = np.max(atr_values[valid_idx])
            self.assertGreater(max_atr, 50)  # Doit refléter les gaps
    
    def test_minimal_period(self):
        """Test avec période minimale (1)"""
        high = np.array([105, 107, 103, 108, 106])
        low = np.array([95, 97, 93, 98, 96])
        close = np.array([100, 102, 98, 105, 103])
        
        # Period = 1 pour ATR
        atr_values = self.atr.compute(high, low, close, period=1, method='ema')
        
        # Doit fonctionner sans erreur
        self.assertEqual(len(atr_values), len(close))
        valid = validate_atr_results(atr_values)
        self.assertTrue(valid)
        
        # Avec period=1, ATR[i] ≈ TR[i] après stabilisation EMA
        # Premier point toujours valide
        self.assertFalse(np.isnan(atr_values[0]))
    
    def test_very_large_period(self):
        """Test avec période très grande"""
        np.random.seed(42)
        n = 100
        
        # Données OHLC
        base_price = 100
        returns = np.random.randn(n) * 0.01
        close = base_price * np.cumprod(1 + returns)
        spread = np.random.uniform(0.2, 1.0, n)
        high = close * (1 + spread/200)
        low = close * (1 - spread/200)
        
        # Period proche de la taille des données
        large_period = n - 5  # 95
        
        atr_values = self.atr.compute(high, low, close, period=large_period, method='sma')
        
        # Avec SMA et grande période, beaucoup de NaN au début
        nan_count = np.sum(np.isnan(atr_values))
        self.assertEqual(nan_count, large_period - 1)
        
        # Mais quelques valeurs valides à la fin
        valid_count = np.sum(~np.isnan(atr_values))
        self.assertGreater(valid_count, 0)
    
    def test_high_low_inversion(self):
        """Test avec high < low (données invalides)"""
        # Données incohérentes
        high = np.array([95, 97, 93])    # Plus bas que low
        low = np.array([105, 107, 103])  # Plus haut que high
        close = np.array([100, 102, 98])
        
        # Doit fonctionner mathématiquement (max prendra les bonnes valeurs)
        atr_values = self.atr.compute(high, low, close, period=2, method='ema')
        
        # Validation que ça ne crash pas
        self.assertEqual(len(atr_values), len(close))
        valid = validate_atr_results(atr_values)
        self.assertTrue(valid)  # ATR reste >= 0 même avec données incohérentes


if __name__ == '__main__':
    # Configuration des tests
    unittest.TestCase.maxDiff = None
    
    # Suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajout des classes de tests
    test_classes = [
        TestATRSettings,
        TestATRGPUManager,
        TestATR,
        TestATRPublicAPI,
        TestATRValidation,
        TestATRPerformance,
        TestATREdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Exécution
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("🎯 ThreadX ATR - Tests unitaires")
    print("=" * 60)
    
    result = runner.run(test_suite)
    
    # Résumé
    print(f"\n{'='*60}")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ ÉCHECS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 ERREURS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Taux de succès: {success_rate:.1f}%")
    
    exit(0 if result.wasSuccessful() else 1)