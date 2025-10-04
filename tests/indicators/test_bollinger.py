#!/usr/bin/env python3
"""
Tests unitaires pour ThreadX Bollinger Bands
============================================

Tests complets du module bollinger.py incluant:
- Calculs CPU vs GPU
- Validation mathématique des résultats
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

from threadx.indicators.bollinger import (
    BollingerBands,
    BollingerSettings,
    compute_bollinger_bands,
    compute_bollinger_batch,
    validate_bollinger_results,
    benchmark_bollinger_performance,
    GPUManager
)


class TestBollingerSettings(unittest.TestCase):
    """Tests de la configuration BollingerSettings"""
    
    def test_default_settings(self):
        """Test des valeurs par défaut"""
        settings = BollingerSettings()
        
        self.assertEqual(settings.period, 20)
        self.assertEqual(settings.std, 2.0)
        self.assertTrue(settings.use_gpu)
        self.assertEqual(settings.gpu_batch_size, 1000)
        self.assertTrue(settings.cpu_fallback)
        self.assertEqual(settings.gpu_split_ratio, (0.75, 0.25))
    
    def test_custom_settings(self):
        """Test configuration personnalisée"""
        settings = BollingerSettings(
            period=50,
            std=1.5,
            use_gpu=False,
            gpu_batch_size=500
        )
        
        self.assertEqual(settings.period, 50)
        self.assertEqual(settings.std, 1.5)
        self.assertFalse(settings.use_gpu)
        self.assertEqual(settings.gpu_batch_size, 500)
    
    def test_validation_period(self):
        """Test validation période"""
        with self.assertRaises(ValueError):
            BollingerSettings(period=1)  # < 2
        
        with self.assertRaises(ValueError):
            BollingerSettings(period=0)
    
    def test_validation_std(self):
        """Test validation écart-type"""
        with self.assertRaises(ValueError):
            BollingerSettings(std=0)
        
        with self.assertRaises(ValueError):
            BollingerSettings(std=-1.0)
    
    def test_validation_gpu_split(self):
        """Test validation répartition GPU"""
        with self.assertRaises(ValueError):
            BollingerSettings(gpu_split_ratio=(0.9, 0.9))  # Sum > 1.0
        
        with self.assertRaises(ValueError):
            BollingerSettings(gpu_split_ratio=(0.05, 0.02))  # Sum < 0.1


class TestGPUManager(unittest.TestCase):
    """Tests du gestionnaire GPU"""
    
    def setUp(self):
        """Setup pour chaque test"""
        self.settings = BollingerSettings()
    
    def test_gpu_manager_init_no_gpu(self):
        """Test initialisation sans GPU"""
        with patch('threadx.indicators.bollinger.HAS_CUPY', False):
            with patch('threadx.indicators.bollinger.GPU_AVAILABLE', False):
                manager = GPUManager(self.settings)
                self.assertEqual(len(manager.available_gpus), 0)
                self.assertEqual(len(manager.gpu_capabilities), 0)
    
    @patch('threadx.indicators.bollinger.HAS_CUPY', True)
    @patch('threadx.indicators.bollinger.GPU_AVAILABLE', True)
    @patch('threadx.indicators.bollinger.N_GPUS', 2)
    def test_gpu_manager_init_with_gpu(self):
        """Test initialisation avec GPU (mock)"""
        # Mock CuPy runtime
        mock_props = {
            'name': b'RTX 5090',
            'major': 8,
            'minor': 9,
            'multiProcessorCount': 128
        }
        
        with patch('threadx.indicators.bollinger.cp') as mock_cp:
            mock_cp.cuda.runtime.getDeviceProperties.return_value = mock_props
            mock_cp.cuda.runtime.memGetInfo.return_value = (25 * 1024**3, 32 * 1024**3)  # 25GB free / 32GB total
            
            manager = GPUManager(self.settings)
            
            # Ne peut pas tester réellement sans GPU, mais on vérifie la structure
            self.assertIsInstance(manager.available_gpus, list)
            self.assertIsInstance(manager.gpu_capabilities, dict)
    
    def test_split_workload_no_gpu(self):
        """Test répartition sans GPU"""
        with patch('threadx.indicators.bollinger.HAS_CUPY', False):
            manager = GPUManager(self.settings)
            splits = manager.split_workload(1000)
            self.assertEqual(len(splits), 0)
    
    def test_split_workload_single_gpu(self):
        """Test répartition GPU unique"""
        manager = GPUManager(self.settings)
        manager.available_gpus = [0]  # Mock single GPU
        
        splits = manager.split_workload(1000)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0], (0, 0, 1000))
    
    def test_split_workload_multi_gpu(self):
        """Test répartition multi-GPU"""
        manager = GPUManager(self.settings)
        manager.available_gpus = [0, 1]  # Mock dual GPU
        
        splits = manager.split_workload(1000)
        self.assertEqual(len(splits), 2)
        
        # Vérification répartition 75/25
        gpu1_size = splits[0][2] - splits[0][1]  # end - start
        gpu2_size = splits[1][2] - splits[1][1]
        
        self.assertEqual(gpu1_size, 750)  # 75% de 1000
        self.assertEqual(gpu2_size, 250)  # 25% de 1000


class TestBollingerBands(unittest.TestCase):
    """Tests de la classe BollingerBands"""
    
    def setUp(self):
        """Setup données test pour chaque test"""
        np.random.seed(42)
        self.n = 1000
        self.close = np.random.randn(self.n) * 10 + 100
        self.close_series = pd.Series(self.close)
        
        # Settings CPU pour éviter problèmes GPU en tests
        self.settings_cpu = BollingerSettings(use_gpu=False)
        self.bb_cpu = BollingerBands(self.settings_cpu)
    
    def test_bollinger_init(self):
        """Test initialisation BollingerBands"""
        bb = BollingerBands()
        self.assertIsInstance(bb.settings, BollingerSettings)
        self.assertIsInstance(bb.gpu_manager, GPUManager)
        self.assertIsInstance(bb._cache, dict)
    
    def test_compute_basic_numpy(self):
        """Test calcul basique avec numpy array"""
        upper, middle, lower = self.bb_cpu.compute(self.close, period=20, std=2.0)
        
        # Vérifications de base
        self.assertEqual(len(upper), self.n)
        self.assertEqual(len(middle), self.n)
        self.assertEqual(len(lower), self.n)
        
        # Vérification NaN au début (insuffisant de données)
        self.assertTrue(np.isnan(upper[0]))
        self.assertTrue(np.isnan(middle[0]))
        self.assertTrue(np.isnan(lower[0]))
        
        # Vérification valeurs valides à la fin
        self.assertFalse(np.isnan(upper[-1]))
        self.assertFalse(np.isnan(middle[-1]))
        self.assertFalse(np.isnan(lower[-1]))
        
        # Vérification ordre: upper >= middle >= lower
        valid_idx = ~np.isnan(upper)
        self.assertTrue(np.all(upper[valid_idx] >= middle[valid_idx]))
        self.assertTrue(np.all(middle[valid_idx] >= lower[valid_idx]))
    
    def test_compute_basic_pandas(self):
        """Test calcul basique avec pandas Series"""
        upper, middle, lower = self.bb_cpu.compute(self.close_series, period=20, std=2.0)
        
        # Mêmes vérifications qu'avec numpy
        self.assertEqual(len(upper), self.n)
        self.assertFalse(np.isnan(upper[-1]))
        
        valid_idx = ~np.isnan(upper)
        self.assertTrue(np.all(upper[valid_idx] >= middle[valid_idx]))
    
    def test_compute_different_periods(self):
        """Test calcul avec différentes périodes"""
        periods = [10, 20, 50]
        
        for period in periods:
            upper, middle, lower = self.bb_cpu.compute(self.close, period=period, std=2.0)
            
            # Nombre de NaN au début doit correspondre à period-1
            nan_count = np.sum(np.isnan(middle))
            self.assertEqual(nan_count, period - 1)
    
    def test_compute_different_std(self):
        """Test calcul avec différents écarts-types"""
        stds = [1.0, 1.5, 2.0, 2.5, 3.0]
        results = []
        
        for std in stds:
            upper, middle, lower = self.bb_cpu.compute(self.close, period=20, std=std)
            results.append((upper, middle, lower, std))
        
        # Vérification que les bandes s'élargissent avec std croissant
        for i in range(1, len(results)):
            prev_upper, prev_middle, prev_lower, prev_std = results[i-1]
            curr_upper, curr_middle, curr_lower, curr_std = results[i]
            
            # Middle band doit être identique (même SMA)
            np.testing.assert_array_almost_equal(prev_middle, curr_middle)
            
            # Bandes doivent s'élargir
            valid_idx = ~np.isnan(curr_upper)
            if np.any(valid_idx):
                # Upper band plus haute
                self.assertTrue(np.all(curr_upper[valid_idx] >= prev_upper[valid_idx]))
                # Lower band plus basse
                self.assertTrue(np.all(curr_lower[valid_idx] <= prev_lower[valid_idx]))
    
    def test_compute_insufficient_data(self):
        """Test avec données insuffisantes"""
        short_data = np.array([100, 101, 102])  # < period=20
        
        with self.assertRaises(ValueError):
            self.bb_cpu.compute(short_data, period=20, std=2.0)
    
    def test_compute_batch_simple(self):
        """Test calcul batch simple"""
        params_list = [
            {'period': 20, 'std': 2.0},
            {'period': 50, 'std': 1.5},
            {'period': 10, 'std': 2.5}
        ]
        
        results = self.bb_cpu.compute_batch(self.close, params_list)
        
        # Vérification des clés
        expected_keys = ['20_2.0', '50_1.5', '10_2.5']
        self.assertEqual(set(results.keys()), set(expected_keys))
        
        # Vérification des résultats
        for key, result in results.items():
            self.assertIsNotNone(result)
            upper, middle, lower = result
            self.assertEqual(len(upper), self.n)
            
            # Validation des résultats
            valid = validate_bollinger_results(upper, middle, lower)
            self.assertTrue(valid)
    
    def test_compute_batch_with_errors(self):
        """Test batch avec paramètres invalides"""
        params_list = [
            {'period': 20, 'std': 2.0},  # Valide
            {'period': 1, 'std': 2.0},   # Invalide: period < 2
            {'period': 2000, 'std': 2.0} # Invalide: period > data size
        ]
        
        results = self.bb_cpu.compute_batch(self.close, params_list)
        
        # Premier résultat doit être valide
        self.assertIsNotNone(results.get('20_2.0'))
        
        # Autres résultats peuvent être None (erreurs)
        # On vérifie juste qu'on n'a pas de crash
        self.assertIsInstance(results, dict)
    
    def test_mathematical_accuracy(self):
        """Test précision mathématique vs calcul manuel"""
        period = 20
        std = 2.0
        
        # Calcul avec notre implémentation
        upper, middle, lower = self.bb_cpu.compute(self.close, period=period, std=std)
        
        # Calcul manuel avec pandas pour comparaison
        close_series = pd.Series(self.close)
        manual_sma = close_series.rolling(window=period).mean()
        manual_std = close_series.rolling(window=period).std(ddof=0)
        manual_upper = manual_sma + (std * manual_std)
        manual_lower = manual_sma - (std * manual_std)
        
        # Comparaison (avec tolérance pour erreurs numériques)
        valid_idx = ~np.isnan(middle)
        np.testing.assert_array_almost_equal(
            middle[valid_idx], 
            np.array(manual_sma.values[valid_idx]), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            upper[valid_idx], 
            np.asarray(manual_upper.values)[valid_idx], 
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            lower[valid_idx], 
            np.asarray(manual_lower.values)[valid_idx], 
            decimal=10
        )


class TestBollingerPublicAPI(unittest.TestCase):
    """Tests des fonctions publiques de l'API"""
    
    def setUp(self):
        """Setup données test"""
        np.random.seed(42)
        self.close = np.random.randn(500) * 10 + 100
    
    def test_compute_bollinger_bands_basic(self):
        """Test API simple compute_bollinger_bands"""
        upper, middle, lower = compute_bollinger_bands(
            self.close, 
            period=20, 
            std=2.0, 
            use_gpu=False  # Force CPU pour éviter problèmes GPU
        )
        
        self.assertEqual(len(upper), len(self.close))
        self.assertEqual(len(middle), len(self.close))
        self.assertEqual(len(lower), len(self.close))
        
        # Validation
        valid = validate_bollinger_results(upper, middle, lower)
        self.assertTrue(valid)
    
    def test_compute_bollinger_batch_basic(self):
        """Test API batch compute_bollinger_batch"""
        params_list = [
            {'period': 20, 'std': 2.0},
            {'period': 30, 'std': 1.5}
        ]
        
        results = compute_bollinger_batch(
            self.close, 
            params_list, 
            use_gpu=False
        )
        
        self.assertEqual(len(results), 2)
        self.assertIn('20_2.0', results)
        self.assertIn('30_1.5', results)
        
        for key, result in results.items():
            if result is not None:
                upper, middle, lower = result
                valid = validate_bollinger_results(upper, middle, lower)
                self.assertTrue(valid)


class TestBollingerValidation(unittest.TestCase):
    """Tests de validation des résultats"""
    
    def test_validate_bollinger_results_valid(self):
        """Test validation avec résultats valides"""
        n = 100
        middle = np.random.randn(n) * 5 + 100
        upper = middle + np.random.rand(n) * 10  # Upper > middle
        lower = middle - np.random.rand(n) * 10  # Lower < middle
        
        valid = validate_bollinger_results(upper, middle, lower)
        self.assertTrue(valid)
    
    def test_validate_bollinger_results_invalid_order(self):
        """Test validation avec ordre invalide"""
        n = 100
        middle = np.random.randn(n) * 5 + 100
        upper = middle - 5  # Upper < middle (invalide)
        lower = middle - 10
        
        valid = validate_bollinger_results(upper, middle, lower)
        self.assertFalse(valid)
    
    def test_validate_bollinger_results_different_lengths(self):
        """Test validation avec longueurs différentes"""
        upper = np.array([105, 106, 107])
        middle = np.array([100, 101])  # Longueur différente
        lower = np.array([95, 96, 97])
        
        valid = validate_bollinger_results(upper, middle, lower)
        self.assertFalse(valid)
    
    def test_validate_bollinger_results_with_nan(self):
        """Test validation avec valeurs NaN"""
        upper = np.array([np.nan, 106, 107])
        middle = np.array([np.nan, 101, 102])
        lower = np.array([np.nan, 96, 97])
        
        # Doit être valide (NaN au début acceptable)
        valid = validate_bollinger_results(upper, middle, lower)
        self.assertTrue(valid)
    
    def test_validate_bollinger_results_all_nan(self):
        """Test validation avec que des NaN"""
        n = 10
        upper = np.full(n, np.nan)
        middle = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        
        # Doit être valide (acceptable si données insuffisantes)
        valid = validate_bollinger_results(upper, middle, lower)
        self.assertTrue(valid)


class TestBollingerPerformance(unittest.TestCase):
    """Tests de performance"""
    
    def test_benchmark_bollinger_performance(self):
        """Test du benchmark de performance"""
        # Test avec petites données pour éviter timeout
        bench_results = benchmark_bollinger_performance(
            data_sizes=[100, 500], 
            n_runs=2
        )
        
        # Vérifications structure résultats
        self.assertIn('cpu_times', bench_results)
        self.assertIn('gpu_times', bench_results)
        self.assertIn('speedups', bench_results)
        self.assertIn('gpu_available', bench_results)
        
        # Vérifications CPU times
        self.assertIn(100, bench_results['cpu_times'])
        self.assertIn(500, bench_results['cpu_times'])
        
        # CPU times doivent être positifs
        self.assertGreater(bench_results['cpu_times'][100], 0)
        self.assertGreater(bench_results['cpu_times'][500], 0)
    
    def test_performance_cpu_vs_manual(self):
        """Test performance CPU vs calcul manuel pandas"""
        np.random.seed(42)
        close = np.random.randn(2000) * 10 + 100
        
        # Notre implémentation
        start_time = time.time()
        upper1, middle1, lower1 = compute_bollinger_bands(close, use_gpu=False)
        our_time = time.time() - start_time
        
        # Calcul manuel pandas
        start_time = time.time()
        close_series = pd.Series(close)
        manual_sma = close_series.rolling(window=20).mean()
        manual_std = close_series.rolling(window=20).std(ddof=0)
        manual_upper = manual_sma + (2.0 * manual_std)
        manual_lower = manual_sma - (2.0 * manual_std)
        manual_time = time.time() - start_time
        
        # Vérification que les résultats sont équivalents
        valid_idx = ~np.isnan(middle1)
        np.testing.assert_array_almost_equal(
            middle1[valid_idx], 
            np.asarray(manual_sma.values)[valid_idx], 
            decimal=10
        )
        
        # Log des performances (pas d'assertion car variable selon machine)
        print(f"\nPerformance Bollinger (n={len(close)}):")
        print(f"  Notre implémentation: {our_time:.4f}s")
        print(f"  Pandas manuel: {manual_time:.4f}s")
        print(f"  Ratio: {manual_time/our_time:.2f}x")


class TestBollingerEdgeCases(unittest.TestCase):
    """Tests des cas limites"""
    
    def setUp(self):
        """Setup"""
        self.settings_cpu = BollingerSettings(use_gpu=False)
        self.bb = BollingerBands(self.settings_cpu)
    
    def test_constant_prices(self):
        """Test avec prix constants"""
        constant_prices = np.full(100, 100.0)  # Tous à 100
        
        upper, middle, lower = self.bb.compute(constant_prices, period=20, std=2.0)
        
        # Avec prix constants, écart-type = 0, donc bandes = middle
        valid_idx = ~np.isnan(middle)
        if np.any(valid_idx):
            np.testing.assert_array_almost_equal(
                upper[valid_idx], 
                middle[valid_idx], 
                decimal=10
            )
            np.testing.assert_array_almost_equal(
                lower[valid_idx], 
                middle[valid_idx], 
                decimal=10
            )
    
    def test_minimal_data(self):
        """Test avec données minimales"""
        # Exactement period points
        minimal_data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])  # 5 points
        period = 5
        
        upper, middle, lower = self.bb.compute(minimal_data, period=period, std=2.0)
        
        # Doit avoir period-1 NaN au début, puis 1 valeur valide
        nan_count = np.sum(np.isnan(middle))
        self.assertEqual(nan_count, period - 1)
        
        # Dernière valeur doit être valide
        self.assertFalse(np.isnan(upper[-1]))
        self.assertFalse(np.isnan(middle[-1]))
        self.assertFalse(np.isnan(lower[-1]))
    
    def test_extreme_volatility(self):
        """Test avec volatilité extrême"""
        # Données très volatiles
        np.random.seed(42)
        extreme_data = np.random.randn(200) * 1000 + 100  # Écart-type très élevé
        
        upper, middle, lower = self.bb.compute(extreme_data, period=20, std=2.0)
        
        # Doit toujours être valide malgré la volatilité
        valid = validate_bollinger_results(upper, middle, lower)
        self.assertTrue(valid)
        
        # Bandes doivent être très larges
        valid_idx = ~np.isnan(upper)
        if np.any(valid_idx):
            band_width = upper[valid_idx] - lower[valid_idx]
            self.assertTrue(np.all(band_width > 0))
    
    def test_very_small_std(self):
        """Test avec écart-type très petit"""
        np.random.seed(42)
        data = np.random.randn(100) * 0.01 + 100  # Très faible volatilité
        
        upper, middle, lower = self.bb.compute(data, period=20, std=0.1)  # Std très petit
        
        valid = validate_bollinger_results(upper, middle, lower)
        self.assertTrue(valid)
        
        # Bandes doivent être très serrées
        valid_idx = ~np.isnan(upper)
        if np.any(valid_idx):
            band_width = upper[valid_idx] - lower[valid_idx]
            # Largeur doit être très faible mais positive
            self.assertTrue(np.all(band_width >= 0))
            self.assertTrue(np.all(band_width < 1.0))  # Très serré


if __name__ == '__main__':
    # Configuration des tests
    unittest.TestCase.maxDiff = None
    
    # Suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajout des classes de tests
    test_classes = [
        TestBollingerSettings,
        TestGPUManager,
        TestBollingerBands,
        TestBollingerPublicAPI,
        TestBollingerValidation,
        TestBollingerPerformance,
        TestBollingerEdgeCases
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
    
    print("🎯 ThreadX Bollinger Bands - Tests unitaires")
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