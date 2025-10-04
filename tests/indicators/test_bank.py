#!/usr/bin/env python3
"""
Tests unitaires pour ThreadX IndicatorBank
==========================================

Tests complets du module bank.py incluant:
- Cache intelligent avec TTL et checksums
- Batch processing et parallélisation
- Registry et mise à jour automatique
- Performance et intégrité
- API publique simplifiée
"""

import unittest
import tempfile
import shutil
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import du module à tester
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from threadx.indicators.bank import (
    IndicatorBank,
    IndicatorSettings,
    CacheManager,
    ensure_indicator,
    force_recompute_indicator,
    batch_ensure_indicators,
    get_bank_stats,
    cleanup_indicators_cache,
    validate_bank_integrity,
    benchmark_bank_performance
)


class TestIndicatorSettings(unittest.TestCase):
    """Tests de la configuration IndicatorSettings"""
    
    def setUp(self):
        """Setup avec répertoire temporaire"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_default_settings(self):
        """Test des valeurs par défaut"""
        settings = IndicatorSettings(cache_dir=self.temp_dir)
        
        self.assertEqual(settings.ttl_seconds, 3600)
        self.assertEqual(settings.batch_threshold, 100)
        self.assertEqual(settings.max_workers, 8)
        self.assertTrue(settings.use_gpu)
        self.assertTrue(settings.auto_registry_update)
        self.assertTrue(settings.checksum_validation)
        self.assertEqual(settings.compression_level, 6)
    
    def test_custom_settings(self):
        """Test configuration personnalisée"""
        settings = IndicatorSettings(
            cache_dir=self.temp_dir,
            ttl_seconds=7200,
            batch_threshold=50,
            max_workers=4,
            use_gpu=False
        )
        
        self.assertEqual(settings.ttl_seconds, 7200)
        self.assertEqual(settings.batch_threshold, 50)
        self.assertEqual(settings.max_workers, 4)
        self.assertFalse(settings.use_gpu)
    
    def test_directory_creation(self):
        """Test création automatique des répertoires"""
        settings = IndicatorSettings(cache_dir=self.temp_dir)
        
        # Vérification des sous-répertoires
        cache_path = Path(self.temp_dir)
        self.assertTrue((cache_path / "bollinger").exists())
        self.assertTrue((cache_path / "atr").exists())
        self.assertTrue((cache_path / "registry").exists())
    
    def test_validation_max_workers(self):
        """Test validation max_workers"""
        with self.assertRaises(ValueError):
            IndicatorSettings(cache_dir=self.temp_dir, max_workers=0)
        
        # Valeur positive doit fonctionner
        settings = IndicatorSettings(cache_dir=self.temp_dir, max_workers=1)
        self.assertEqual(settings.max_workers, 1)


class TestCacheManager(unittest.TestCase):
    """Tests du gestionnaire de cache"""
    
    def setUp(self):
        """Setup avec répertoire temporaire"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        self.settings = IndicatorSettings(cache_dir=self.temp_dir)
        self.cache_manager = CacheManager(self.settings)
    
    def test_generate_cache_key(self):
        """Test génération clé de cache"""
        params = {'period': 20, 'std': 2.0}
        data_hash = 'abcd1234'
        
        key = self.cache_manager._generate_cache_key(
            'bollinger', params, 'BTCUSDC', '15m', data_hash
        )
        
        # Doit contenir tous les éléments
        self.assertIn('bollinger', key)
        self.assertIn('BTCUSDC', key)
        self.assertIn('15m', key)
        self.assertIn('abcd1234', key)
        
        # Doit être reproductible
        key2 = self.cache_manager._generate_cache_key(
            'bollinger', params, 'BTCUSDC', '15m', data_hash
        )
        self.assertEqual(key, key2)
    
    def test_cache_key_sorting(self):
        """Test tri alphabétique des paramètres dans la clé"""
        params1 = {'period': 20, 'std': 2.0, 'method': 'ema'}
        params2 = {'std': 2.0, 'method': 'ema', 'period': 20}  # Ordre différent
        
        key1 = self.cache_manager._generate_cache_key('test', params1)
        key2 = self.cache_manager._generate_cache_key('test', params2)
        
        # Même clé malgré ordre différent
        self.assertEqual(key1, key2)
    
    def test_compute_data_hash(self):
        """Test calcul hash des données"""
        # Numpy array
        data_np = np.array([1.0, 2.0, 3.0])
        hash_np = self.cache_manager._compute_data_hash(data_np)
        self.assertIsInstance(hash_np, str)
        self.assertEqual(len(hash_np), 32)  # MD5 hash
        
        # Pandas Series
        data_pd = pd.Series([1.0, 2.0, 3.0])
        hash_pd = self.cache_manager._compute_data_hash(data_pd)
        
        # Même hash pour mêmes données
        self.assertEqual(hash_np, hash_pd)
        
        # DataFrame OHLCV
        ohlcv = pd.DataFrame({
            'open': [100, 101], 'high': [105, 106],
            'low': [95, 96], 'close': [102, 104], 'volume': [1000, 1500]
        })
        hash_ohlcv = self.cache_manager._compute_data_hash(ohlcv)
        self.assertIsInstance(hash_ohlcv, str)
    
    def test_cache_file_paths(self):
        """Test génération chemins fichiers cache"""
        cache_key = "test_key_123"
        
        cache_file = self.cache_manager._get_cache_filepath(cache_key, "bollinger")
        meta_file = self.cache_manager._get_metadata_filepath(cache_key, "bollinger")
        
        # Vérification structure
        self.assertTrue(str(cache_file).endswith('.parquet'))
        self.assertTrue(str(meta_file).endswith('.meta'))
        self.assertIn('bollinger', str(cache_file))
        self.assertIn(cache_key, str(cache_file))
    
    def test_save_and_load_cache_single_array(self):
        """Test sauvegarde/chargement cache pour array unique (ATR)"""
        cache_key = "atr_test_123"
        result = np.array([1.5, 1.8, 2.1, 1.9, 2.3])
        params = {'period': 14, 'method': 'ema'}
        
        # Sauvegarde
        success = self.cache_manager.save_to_cache(
            cache_key, 'atr', result, params, 'BTCUSDC', '15m'
        )
        self.assertTrue(success)
        
        # Vérification fichiers créés
        cache_file = self.cache_manager._get_cache_filepath(cache_key, 'atr')
        meta_file = self.cache_manager._get_metadata_filepath(cache_key, 'atr')
        self.assertTrue(cache_file.exists())
        self.assertTrue(meta_file.exists())
        
        # Chargement
        loaded_result = self.cache_manager.load_from_cache(cache_key, 'atr')
        self.assertIsNotNone(loaded_result)
        assert loaded_result is not None  # Type narrowing pour LSP
        
        # Vérification données
        np.testing.assert_array_almost_equal(loaded_result, result, decimal=10)
    
    def test_save_and_load_cache_multiple_arrays(self):
        """Test sauvegarde/chargement cache pour multiples arrays (Bollinger)"""
        cache_key = "bb_test_456"
        upper = np.array([105, 107, 103, 108])
        middle = np.array([100, 102, 98, 105])
        lower = np.array([95, 97, 93, 102])
        result = (upper, middle, lower)
        params = {'period': 20, 'std': 2.0}
        
        # Sauvegarde
        success = self.cache_manager.save_to_cache(
            cache_key, 'bollinger', result, params
        )
        self.assertTrue(success)
        
        # Chargement
        loaded_result = self.cache_manager.load_from_cache(cache_key, 'bollinger')
        self.assertIsNotNone(loaded_result)
        assert loaded_result is not None  # Type narrowing pour LSP
        self.assertIsInstance(loaded_result, tuple)
        self.assertEqual(len(loaded_result), 3)
        
        # Vérification données
        loaded_upper, loaded_middle, loaded_lower = loaded_result
        np.testing.assert_array_almost_equal(loaded_upper, upper, decimal=10)
        np.testing.assert_array_almost_equal(loaded_middle, middle, decimal=10)
        np.testing.assert_array_almost_equal(loaded_lower, lower, decimal=10)
    
    def test_cache_validation_ttl(self):
        """Test validation TTL du cache"""
        cache_key = "ttl_test_789"
        result = np.array([1.0, 2.0, 3.0])
        params = {'period': 10}
        
        # Sauvegarde avec TTL court
        old_ttl = self.settings.ttl_seconds
        self.settings.ttl_seconds = 1  # 1 seconde
        
        success = self.cache_manager.save_to_cache(
            cache_key, 'test', result, params
        )
        self.assertTrue(success)
        
        # Immédiatement valide
        valid = self.cache_manager.is_cache_valid(cache_key, 'test')
        self.assertTrue(valid)
        
        # Attendre expiration
        time.sleep(1.5)
        
        # Maintenant expiré
        valid = self.cache_manager.is_cache_valid(cache_key, 'test')
        self.assertFalse(valid)
        
        # Restaurer TTL
        self.settings.ttl_seconds = old_ttl
    
    def test_cache_validation_checksum(self):
        """Test validation checksum du cache"""
        cache_key = "checksum_test_101"
        result = np.array([1.0, 2.0, 3.0])
        params = {'period': 10}
        
        # Sauvegarde avec checksum activé
        self.settings.checksum_validation = True
        success = self.cache_manager.save_to_cache(
            cache_key, 'test', result, params
        )
        self.assertTrue(success)
        
        # Validation OK
        valid = self.cache_manager.is_cache_valid(cache_key, 'test')
        self.assertTrue(valid)
        
        # Corruption simulée du fichier cache
        cache_file = self.cache_manager._get_cache_filepath(cache_key, 'test')
        with open(cache_file, 'ab') as f:
            f.write(b'corruption')  # Ajouter données pour changer checksum
        
        # Maintenant invalide
        valid = self.cache_manager.is_cache_valid(cache_key, 'test')
        self.assertFalse(valid)
    
    def test_cleanup_expired(self):
        """Test nettoyage cache expiré"""
        # Créer plusieurs entrées cache
        for i in range(3):
            cache_key = f"cleanup_test_{i}"
            result = np.array([float(i), float(i+1)])
            self.cache_manager.save_to_cache(
                cache_key, 'test', result, {'period': i+10}
            )
        
        # Vérifier qu'ils existent
        cache_dir = Path(self.temp_dir) / 'test'
        cache_dir.mkdir(exist_ok=True)
        initial_count = len(list(cache_dir.glob('*.meta')))
        
        # Avec TTL normal, rien ne doit être nettoyé
        cleaned = self.cache_manager.cleanup_expired()
        remaining_count = len(list(cache_dir.glob('*.meta')))
        self.assertEqual(remaining_count, initial_count)
        
        # Réduire TTL pour forcer expiration
        old_ttl = self.settings.ttl_seconds
        self.settings.ttl_seconds = 0  # Expire immédiatement
        
        cleaned = self.cache_manager.cleanup_expired()
        self.assertGreater(cleaned, 0)  # Quelque chose a été nettoyé
        
        # Restaurer TTL
        self.settings.ttl_seconds = old_ttl


class TestIndicatorBank(unittest.TestCase):
    """Tests de la classe IndicatorBank"""
    
    def setUp(self):
        """Setup avec répertoire temporaire et données test"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Données OHLCV test
        np.random.seed(42)
        n = 500
        base_price = 100
        returns = np.random.randn(n) * 0.01
        close = base_price * np.cumprod(1 + returns)
        spread = np.random.uniform(0.3, 1.5, n)
        
        self.ohlcv = pd.DataFrame({
            'open': close * (1 - spread/400),
            'high': close * (1 + spread/200),
            'low': close * (1 - spread/200),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        # Settings pour forcer CPU et éviter problèmes GPU
        self.settings = IndicatorSettings(
            cache_dir=self.temp_dir,
            use_gpu=False,
            batch_threshold=5  # Seuil bas pour tester parallélisation
        )
        self.bank = IndicatorBank(self.settings)
    
    def test_bank_initialization(self):
        """Test initialisation IndicatorBank"""
        self.assertIsInstance(self.bank.settings, IndicatorSettings)
        self.assertIsInstance(self.bank.cache_manager, CacheManager)
        self.assertIn('bollinger', self.bank.calculators)
        self.assertIn('atr', self.bank.calculators)
        self.assertEqual(self.bank.stats['cache_hits'], 0)
        self.assertEqual(self.bank.stats['cache_misses'], 0)
    
    def test_ensure_bollinger_first_call(self):
        """Test ensure Bollinger première fois (cache miss)"""
        params = {'period': 20, 'std': 2.0}
        
        result = self.bank.ensure(
            'bollinger', params, self.ohlcv, 
            symbol='TESTBTC', timeframe='15m'
        )
        
        # Vérification résultat
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        
        upper, middle, lower = result
        self.assertEqual(len(upper), len(self.ohlcv))
        
        # Vérification stats (cache miss)
        stats = self.bank.get_stats()
        self.assertEqual(stats['cache_misses'], 1)
        self.assertEqual(stats['computations'], 1)
    
    def test_ensure_bollinger_second_call(self):
        """Test ensure Bollinger deuxième fois (cache hit)"""
        params = {'period': 20, 'std': 2.0}
        
        # Premier appel (cache miss)
        result1 = self.bank.ensure('bollinger', params, self.ohlcv)
        
        # Deuxième appel (cache hit)
        result2 = self.bank.ensure('bollinger', params, self.ohlcv)
        
        # Résultats identiques
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        
        upper1, middle1, lower1 = result1
        upper2, middle2, lower2 = result2
        
        np.testing.assert_array_equal(upper1, upper2)
        np.testing.assert_array_equal(middle1, middle2)
        np.testing.assert_array_equal(lower1, lower2)
        
        # Vérification stats (1 miss, 1 hit)
        stats = self.bank.get_stats()
        self.assertEqual(stats['cache_misses'], 1)
        self.assertEqual(stats['cache_hits'], 1)
    
    def test_ensure_atr_basic(self):
        """Test ensure ATR basique"""
        params = {'period': 14, 'method': 'ema'}
        
        result = self.bank.ensure('atr', params, self.ohlcv)
        
        # Vérification résultat
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.ohlcv))
        
        # ATR doit être >= 0
        valid_idx = ~np.isnan(result)
        self.assertTrue(np.all(result[valid_idx] >= 0))
    
    def test_ensure_invalid_indicator_type(self):
        """Test ensure avec type d'indicateur invalide"""
        with self.assertRaises(ValueError):
            self.bank.ensure('invalid_indicator', {}, self.ohlcv)
    
    def test_ensure_bollinger_missing_close(self):
        """Test ensure Bollinger sans colonne close"""
        # DataFrame sans close
        bad_df = pd.DataFrame({
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'volume': [1000, 1500, 1200]
        })
        
        result = self.bank.ensure('bollinger', {'period': 20, 'std': 2.0}, bad_df)
        self.assertIsNone(result)  # Erreur gérée, retourne None
    
    def test_ensure_atr_missing_ohlc(self):
        """Test ensure ATR sans colonnes OHLC complètes"""
        # DataFrame incomplet
        bad_df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1500, 1200]
        })
        
        result = self.bank.ensure('atr', {'period': 14}, bad_df)
        self.assertIsNone(result)  # Erreur gérée
    
    def test_force_recompute(self):
        """Test force recompute"""
        params = {'period': 20, 'std': 2.0}
        
        # Premier calcul
        result1 = self.bank.ensure('bollinger', params, self.ohlcv)
        initial_computations = self.bank.stats['computations']
        
        # Force recompute
        result2 = self.bank.force_recompute('bollinger', params, self.ohlcv)
        
        # Nouveau calcul effectué
        self.assertEqual(self.bank.stats['computations'], initial_computations + 1)
        
        # Résultats doivent être identiques
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        
        upper1, middle1, lower1 = result1
        upper2, middle2, lower2 = result2
        
        np.testing.assert_array_equal(upper1, upper2)
        np.testing.assert_array_equal(middle1, middle2)
        np.testing.assert_array_equal(lower1, lower2)
    
    def test_batch_ensure_sequential(self):
        """Test batch ensure séquentiel (sous seuil)"""
        params_list = [
            {'period': 20, 'std': 2.0},
            {'period': 30, 'std': 1.5}
        ]  # Seulement 2 paramètres < batch_threshold=5
        
        results = self.bank.batch_ensure('bollinger', params_list, self.ohlcv)
        
        # Vérification résultats
        self.assertEqual(len(results), 2)
        self.assertIn('period=20_std=2.0', results)
        self.assertIn('period=30_std=1.5', results)
        
        for key, result in results.items():
            self.assertIsNotNone(result)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
    
    def test_batch_ensure_parallel(self):
        """Test batch ensure parallèle (au-dessus seuil)"""
        # 6 paramètres > batch_threshold=5 pour déclencher parallélisation
        params_list = [
            {'period': 10, 'std': 1.5},
            {'period': 15, 'std': 2.0},
            {'period': 20, 'std': 2.0},
            {'period': 25, 'std': 1.8},
            {'period': 30, 'std': 2.2},
            {'period': 35, 'std': 1.6}
        ]
        
        results = self.bank.batch_ensure('bollinger', params_list, self.ohlcv)
        
        # Vérification résultats
        self.assertEqual(len(results), 6)
        
        success_count = sum(1 for r in results.values() if r is not None)
        self.assertEqual(success_count, 6)  # Tous doivent réussir
        
        # Vérification stats batch
        stats = self.bank.get_stats()
        self.assertEqual(stats['batch_operations'], 1)
    
    def test_params_to_key(self):
        """Test conversion paramètres vers clé"""
        # Test avec différents types
        params1 = {'period': 20, 'std': 2.0, 'method': 'ema'}
        key1 = self.bank._params_to_key(params1)
        
        # Doit être ordonné alphabétiquement
        self.assertIn('method=ema', key1)
        self.assertIn('period=20', key1)
        self.assertIn('std=2.000', key1)
        
        # Test reproductibilité
        key2 = self.bank._params_to_key(params1)
        self.assertEqual(key1, key2)
        
        # Test ordre différent mais même résultat
        params2 = {'std': 2.0, 'method': 'ema', 'period': 20}
        key3 = self.bank._params_to_key(params2)
        self.assertEqual(key1, key3)
    
    def test_get_stats(self):
        """Test récupération statistiques"""
        # État initial
        stats = self.bank.get_stats()
        self.assertEqual(stats['cache_hits'], 0)
        self.assertEqual(stats['cache_misses'], 0)
        self.assertEqual(stats['total_requests'], 0)
        self.assertEqual(stats['cache_hit_rate_pct'], 0)
        
        # Après quelques opérations
        params = {'period': 20, 'std': 2.0}
        self.bank.ensure('bollinger', params, self.ohlcv)  # Miss
        self.bank.ensure('bollinger', params, self.ohlcv)  # Hit
        
        stats = self.bank.get_stats()
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 1)
        self.assertEqual(stats['total_requests'], 2)
        self.assertEqual(stats['cache_hit_rate_pct'], 50.0)
    
    def test_cleanup_cache(self):
        """Test nettoyage cache de la banque"""
        # Créer quelques entrées
        params_list = [
            {'period': 10, 'std': 1.5},
            {'period': 20, 'std': 2.0}
        ]
        
        for params in params_list:
            self.bank.ensure('bollinger', params, self.ohlcv)
        
        # Avec TTL normal, rien nettoyé
        cleaned = self.bank.cleanup_cache()
        self.assertEqual(cleaned, 0)
        
        # Forcer expiration
        old_ttl = self.bank.settings.ttl_seconds
        self.bank.settings.ttl_seconds = 0
        
        cleaned = self.bank.cleanup_cache()
        self.assertGreaterEqual(cleaned, 0)  # Au moins 0
        
        # Restaurer
        self.bank.settings.ttl_seconds = old_ttl


class TestIndicatorBankPublicAPI(unittest.TestCase):
    """Tests de l'API publique simplifiée"""
    
    def setUp(self):
        """Setup avec données temporaires"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Données OHLCV test
        np.random.seed(42)
        n = 200
        base_price = 100
        returns = np.random.randn(n) * 0.01
        close = base_price * np.cumprod(1 + returns)
        spread = np.random.uniform(0.2, 1.0, n)
        
        self.ohlcv = pd.DataFrame({
            'open': close * (1 - spread/400),
            'high': close * (1 + spread/200),
            'low': close * (1 - spread/200),
            'close': close,
            'volume': np.random.randint(1000, 5000, n)
        })
    
    def test_ensure_indicator_bollinger(self):
        """Test API ensure_indicator pour Bollinger"""
        result = ensure_indicator(
            'bollinger',
            {'period': 20, 'std': 2.0},
            self.ohlcv,
            symbol='TESTBTC',
            timeframe='15m',
            cache_dir=self.temp_dir
        )
        
        self.assertIsNotNone(result)
        assert result is not None  # Type narrowing pour LSP
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        
        upper, middle, lower = result
        self.assertEqual(len(upper), len(self.ohlcv))
    
    def test_ensure_indicator_atr(self):
        """Test API ensure_indicator pour ATR"""
        result = ensure_indicator(
            'atr',
            {'period': 14, 'method': 'ema'},
            self.ohlcv,
            cache_dir=self.temp_dir
        )
        
        self.assertIsNotNone(result)
        assert result is not None  # Type narrowing pour LSP
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.ohlcv))
    
    def test_force_recompute_indicator(self):
        """Test API force_recompute_indicator"""
        params = {'period': 20, 'std': 2.0}
        
        # Premier calcul
        result1 = ensure_indicator('bollinger', params, self.ohlcv, cache_dir=self.temp_dir)
        
        # Force recompute
        result2 = force_recompute_indicator('bollinger', params, self.ohlcv, cache_dir=self.temp_dir)
        
        # Résultats doivent être identiques
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        assert result1 is not None and result2 is not None  # Type narrowing pour LSP
        
        upper1, middle1, lower1 = result1
        upper2, middle2, lower2 = result2
        
        np.testing.assert_array_equal(upper1, upper2)
    
    def test_batch_ensure_indicators(self):
        """Test API batch_ensure_indicators"""
        params_list = [
            {'period': 20, 'std': 2.0},
            {'period': 30, 'std': 1.5}
        ]
        
        results = batch_ensure_indicators(
            'bollinger', params_list, self.ohlcv, cache_dir=self.temp_dir
        )
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        
        for key, result in results.items():
            self.assertIsNotNone(result)
            self.assertIsInstance(result, tuple)
    
    def test_get_bank_stats_api(self):
        """Test API get_bank_stats"""
        # Générer quelques stats
        ensure_indicator('bollinger', {'period': 20, 'std': 2.0}, self.ohlcv, cache_dir=self.temp_dir)
        ensure_indicator('bollinger', {'period': 20, 'std': 2.0}, self.ohlcv, cache_dir=self.temp_dir)  # Cache hit
        
        stats = get_bank_stats(self.temp_dir)
        
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('cache_hit_rate_pct', stats)
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 1)
    
    def test_cleanup_indicators_cache_api(self):
        """Test API cleanup_indicators_cache"""
        # Créer quelques indicateurs
        ensure_indicator('bollinger', {'period': 20, 'std': 2.0}, self.ohlcv, cache_dir=self.temp_dir)
        
        cleaned = cleanup_indicators_cache(self.temp_dir)
        self.assertIsInstance(cleaned, int)
        self.assertGreaterEqual(cleaned, 0)


class TestIndicatorBankUtilities(unittest.TestCase):
    """Tests des utilitaires de validation et benchmark"""
    
    def setUp(self):
        """Setup avec données temporaires"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_validate_bank_integrity_empty(self):
        """Test validation intégrité banque vide"""
        results = validate_bank_integrity(self.temp_dir)
        
        self.assertIn('total_indicators', results)
        self.assertIn('valid_cache', results)
        self.assertEqual(results['total_indicators'], 0)
    
    def test_validate_bank_integrity_with_data(self):
        """Test validation intégrité avec données"""
        # Créer quelques indicateurs
        np.random.seed(42)
        ohlcv = pd.DataFrame({
            'open': [100, 101], 'high': [105, 106],
            'low': [95, 96], 'close': [102, 104], 'volume': [1000, 1500]
        })
        
        ensure_indicator('bollinger', {'period': 2, 'std': 2.0}, ohlcv, cache_dir=self.temp_dir)
        
        results = validate_bank_integrity(self.temp_dir)
        
        self.assertGreater(results['total_indicators'], 0)
        self.assertIn('details', results)
        self.assertIn('bollinger', results['details'])
    
    def test_benchmark_bank_performance_small(self):
        """Test benchmark performance avec petites données"""
        # Test minimal pour éviter timeout
        bench_results = benchmark_bank_performance(
            cache_dir=self.temp_dir,
            n_indicators=4,  # Très petit
            data_size=50     # Très petit
        )
        
        # Vérifications structure
        self.assertIn('setup', bench_results)
        self.assertIn('timings', bench_results)
        self.assertIn('cache_performance', bench_results)
        
        # Setup
        self.assertEqual(bench_results['setup']['n_indicators'], 4)
        self.assertEqual(bench_results['setup']['data_size'], 50)
        
        # Timings
        self.assertIn('cold_cache', bench_results['timings'])
        self.assertIn('warm_cache', bench_results['timings'])
        
        # Performance
        self.assertIn('hit_rate_pct', bench_results['cache_performance'])
        self.assertIn('speedup_warm', bench_results['cache_performance'])


if __name__ == '__main__':
    # Configuration des tests
    unittest.TestCase.maxDiff = None
    
    # Suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajout des classes de tests
    test_classes = [
        TestIndicatorSettings,
        TestCacheManager,
        TestIndicatorBank,
        TestIndicatorBankPublicAPI,
        TestIndicatorBankUtilities
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
    
    print("🏦 ThreadX IndicatorBank - Tests unitaires")
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