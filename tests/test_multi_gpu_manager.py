"""
Tests ThreadX Multi-GPU Manager - Phase 5
==========================================

Test de la distribution de charge multi-GPU avec scénarios:
- 2 GPU (5090 + 2060): répartition 75/25
- 1 GPU: répartition 100/0
- 0 GPU: fallback CPU
- Auto-balance: profiling automatique
- Gestion d'erreurs: OOM, device absent, fonction non vectorisable

Critères d'acceptation:
- Déterminisme: seed=42 → résultats identiques
- Split proportionnel avec correction résidus
- Merge ordonné et cohérent
- Gain performance ≥ 1.5× vs CPU (si GPU)
- Robustesse: toutes erreurs détectées et loggées
"""

import pytest
import numpy as np
import pandas as pd
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

# Import des modules à tester
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from threadx.utils.gpu.multi_gpu import (
    MultiGPUManager, WorkloadChunk, ComputeResult,
    DeviceUnavailableError, GPUMemoryError, ShapeMismatchError,
    NonVectorizableFunctionError, get_default_manager
)

from threadx.utils.gpu.device_manager import DeviceInfo, xp


class TestWorkloadChunk:
    """Tests de la classe WorkloadChunk."""
    
    def test_chunk_creation(self):
        """Test création basique d'un chunk."""
        chunk = WorkloadChunk(
            device_name="5090",
            data_slice=slice(0, 1000),
            start_idx=0,
            end_idx=1000,
            expected_size=1000
        )
        
        assert chunk.device_name == "5090"
        assert chunk.start_idx == 0
        assert chunk.end_idx == 1000
        assert len(chunk) == 1000
        assert chunk.expected_size == 1000
    
    def test_chunk_slice(self):
        """Test des slices variés."""
        chunk = WorkloadChunk("2060", slice(500, 1500), 500, 1500, 1000)
        assert len(chunk) == 1000
        
        # Chunk vide
        empty_chunk = WorkloadChunk("cpu", slice(0, 0), 0, 0, 0)
        assert len(empty_chunk) == 0


class TestComputeResult:
    """Tests de la classe ComputeResult."""
    
    def test_successful_result(self):
        """Test résultat successful."""
        chunk = WorkloadChunk("5090", slice(0, 100), 0, 100, 100)
        result_data = np.array([1, 2, 3])
        
        result = ComputeResult(
            chunk=chunk,
            result=result_data,
            compute_time=0.123,
            device_memory_used=1.5
        )
        
        assert result.success
        assert result.error is None
        assert result.compute_time == 0.123
        assert result.device_memory_used == 1.5
        np.testing.assert_array_equal(result.result, result_data)
    
    def test_failed_result(self):
        """Test résultat avec erreur."""
        chunk = WorkloadChunk("2060", slice(0, 100), 0, 100, 100)
        error = ValueError("Test error")
        
        result = ComputeResult(
            chunk=chunk,
            result=None,
            compute_time=0.05,
            error=error
        )
        
        assert not result.success
        assert result.error == error
        assert result.result is None


class TestMultiGPUManager:
    """Tests principaux du MultiGPUManager."""
    
    @pytest.fixture
    def mock_devices(self):
        """Mock des devices pour tests."""
        gpu_5090 = DeviceInfo(
            device_id=0, name="5090", full_name="RTX 5090",
            memory_total=32 * (1024**3), memory_free=30 * (1024**3),
            compute_capability=(8, 9), is_available=True
        )
        
        gpu_2060 = DeviceInfo(
            device_id=1, name="2060", full_name="RTX 2060",
            memory_total=6 * (1024**3), memory_free=5 * (1024**3),
            compute_capability=(7, 5), is_available=True
        )
        
        cpu_device = DeviceInfo(
            device_id=-1, name="cpu", full_name="CPU Fallback",
            memory_total=0, memory_free=0,
            compute_capability=(0, 0), is_available=True
        )
        
        return [gpu_5090, gpu_2060, cpu_device]
    
    @pytest.fixture
    def mock_single_gpu(self):
        """Mock d'un seul GPU."""
        gpu_5090 = DeviceInfo(
            device_id=0, name="5090", full_name="RTX 5090",
            memory_total=32 * (1024**3), memory_free=30 * (1024**3),
            compute_capability=(8, 9), is_available=True
        )
        
        cpu_device = DeviceInfo(
            device_id=-1, name="cpu", full_name="CPU Fallback",
            memory_total=0, memory_free=0,
            compute_capability=(0, 0), is_available=True
        )
        
        return [gpu_5090, cpu_device]
    
    @pytest.fixture
    def mock_cpu_only(self):
        """Mock CPU uniquement."""
        cpu_device = DeviceInfo(
            device_id=-1, name="cpu", full_name="CPU Fallback",
            memory_total=0, memory_free=0,
            compute_capability=(0, 0), is_available=True
        )
        
        return [cpu_device]
    
    def test_manager_init_dual_gpu(self, mock_devices):
        """Test initialisation avec 2 GPU (5090 + 2060)."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager()
            
            # Vérification balance par défaut 75/25
            assert abs(manager.device_balance["5090"] - 0.75) < 1e-6
            assert abs(manager.device_balance["2060"] - 0.25) < 1e-6
            assert len(manager._gpu_devices) == 2
            assert manager._cpu_device is not None
    
    def test_manager_init_single_gpu(self, mock_single_gpu):
        """Test initialisation avec 1 GPU."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_single_gpu
            
            manager = MultiGPUManager()
            
            # Vérification balance 100% sur 5090
            assert abs(manager.device_balance["5090"] - 1.0) < 1e-6
            assert len(manager._gpu_devices) == 1
    
    def test_manager_init_cpu_only(self, mock_cpu_only):
        """Test initialisation CPU uniquement."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Vérification balance 100% CPU
            assert abs(manager.device_balance["cpu"] - 1.0) < 1e-6
            assert len(manager._gpu_devices) == 0
    
    def test_set_balance_valid(self, mock_devices):
        """Test set_balance avec ratios valides."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager()
            
            # Balance personnalisée
            new_balance = {"5090": 0.8, "2060": 0.2}
            manager.set_balance(new_balance)
            
            assert abs(manager.device_balance["5090"] - 0.8) < 1e-6
            assert abs(manager.device_balance["2060"] - 0.2) < 1e-6
    
    def test_set_balance_normalization(self, mock_devices):
        """Test normalisation automatique des ratios."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager()
            
            # Ratios non normalisés (somme ≠ 1.0)
            new_balance = {"5090": 3.0, "2060": 1.0}  # Somme = 4.0
            manager.set_balance(new_balance)
            
            # Vérification normalisation
            assert abs(manager.device_balance["5090"] - 0.75) < 1e-6  # 3/4
            assert abs(manager.device_balance["2060"] - 0.25) < 1e-6  # 1/4
    
    def test_set_balance_invalid(self, mock_devices):
        """Test validation balance invalide."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager()
            
            # Ratio négatif
            with pytest.raises(ValueError, match="Ratio invalide"):
                manager.set_balance({"5090": -0.5, "2060": 1.5})
            
            # Ratio zéro
            with pytest.raises(ValueError, match="Ratio invalide"):
                manager.set_balance({"5090": 0.0, "2060": 1.0})
            
            # Balance vide
            with pytest.raises(ValueError, match="Balance vide"):
                manager.set_balance({})
    
    def test_split_workload_proportional(self, mock_devices):
        """Test split proportionnel avec correction résidus."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager(device_balance={"5090": 0.75, "2060": 0.25})
            
            # Test avec 1000 échantillons
            chunks = manager._split_workload(1000)
            
            assert len(chunks) == 2
            
            # Vérification tailles approximatives (75% / 25%)
            chunk_5090 = next(c for c in chunks if c.device_name == "5090")
            chunk_2060 = next(c for c in chunks if c.device_name == "2060")
            
            assert chunk_5090.expected_size == 750  # 75% de 1000
            assert chunk_2060.expected_size == 250  # 25% de 1000
            
            # Vérification couverture complète
            total_size = sum(len(c) for c in chunks)
            assert total_size == 1000
            
            # Vérification indices contigus
            assert chunk_5090.start_idx == 0
            assert chunk_5090.end_idx == 750
            assert chunk_2060.start_idx == 750
            assert chunk_2060.end_idx == 1000
    
    def test_split_workload_residue(self, mock_devices):
        """Test gestion des résidus d'arrondi."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager(device_balance={"5090": 0.7, "2060": 0.3})
            
            # Taille avec résidu: 103 échantillons
            chunks = manager._split_workload(103)
            
            assert len(chunks) == 2
            
            # Vérification somme exacte
            total_size = sum(len(c) for c in chunks)
            assert total_size == 103
            
            # Le dernier chunk récupère le résidu
            chunk_5090 = next(c for c in chunks if c.device_name == "5090")
            chunk_2060 = next(c for c in chunks if c.device_name == "2060")
            
            # 70% de 103 = 72.1 → 72, résidu 31 va au 2060
            assert chunk_5090.expected_size == 72
            assert chunk_2060.expected_size == 31
    
    @patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False)
    def test_compute_chunk_cpu_fallback(self, mock_cpu_only):
        """Test calcul sur CPU (fallback)."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Données test
            data = np.array([[1, 2], [3, 4], [5, 6]])
            chunk = WorkloadChunk("cpu", slice(0, 3), 0, 3, 3)
            
            def test_func(x):
                return x.sum(axis=1)
            
            # Calcul
            result = manager._compute_chunk(data, chunk, test_func, seed=42)
            
            assert result.success
            assert result.error is None
            np.testing.assert_array_equal(result.result, [3, 7, 11])  # [1+2, 3+4, 5+6]
            assert result.compute_time > 0
    
    def test_compute_chunk_dataframe(self, mock_cpu_only):
        """Test calcul avec DataFrame."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # DataFrame test
            data = pd.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6]
            })
            chunk = WorkloadChunk("cpu", slice(0, 3), 0, 3, 3)
            
            def test_func(x):
                return x.sum(axis=1)
            
            # Calcul
            result = manager._compute_chunk(data, chunk, test_func, seed=42)
            
            assert result.success
            assert isinstance(result.result, pd.Series)
            pd.testing.assert_series_equal(result.result, pd.Series([5, 7, 9], index=data.index))
    
    def test_compute_chunk_errors(self, mock_cpu_only):
        """Test gestion d'erreurs compute_chunk."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            data = np.array([[1, 2], [3, 4]])
            chunk = WorkloadChunk("cpu", slice(0, 2), 0, 2, 2)
            
            # Fonction qui lève une exception
            def failing_func(x):
                raise ValueError("Test error")
            
            result = manager._compute_chunk(data, chunk, failing_func, seed=42)
            
            assert not result.success
            assert "Test error" in str(result.error)
    
    def test_merge_results_numpy(self, mock_cpu_only):
        """Test merge de résultats numpy."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Création chunks et résultats simulés
            chunk1 = WorkloadChunk("cpu", slice(0, 2), 0, 2, 2)
            chunk2 = WorkloadChunk("cpu", slice(2, 4), 2, 4, 2)
            
            result1 = ComputeResult(chunk1, np.array([1, 2]), 0.1)
            result2 = ComputeResult(chunk2, np.array([3, 4]), 0.1)
            
            # Données originales pour référence type
            original_data = np.array([0, 0, 0, 0])
            
            # Merge
            merged = manager._merge_results([result1, result2], original_data)
            
            np.testing.assert_array_equal(merged, [1, 2, 3, 4])
    
    def test_merge_results_dataframe(self, mock_cpu_only):
        """Test merge de résultats DataFrame."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Indices pour DataFrames
            idx1 = pd.Index([0, 1])
            idx2 = pd.Index([2, 3])
            
            chunk1 = WorkloadChunk("cpu", slice(0, 2), 0, 2, 2)
            chunk2 = WorkloadChunk("cpu", slice(2, 4), 2, 4, 2)
            
            result1 = ComputeResult(chunk1, pd.Series([10, 20], index=idx1), 0.1)
            result2 = ComputeResult(chunk2, pd.Series([30, 40], index=idx2), 0.1)
            
            # DataFrame original pour référence type
            original_data = pd.DataFrame({'col': [0, 0, 0, 0]})
            
            # Merge
            merged = manager._merge_results([result1, result2], original_data)
            
            expected = pd.Series([10, 20, 30, 40], index=pd.Index([0, 1, 2, 3]))
            pd.testing.assert_series_equal(merged, expected)
    
    def test_merge_results_order_invariance(self, mock_cpu_only):
        """Test que le merge est déterministe même si résultats désordonnés."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Chunks dans l'ordre
            chunk1 = WorkloadChunk("cpu", slice(0, 2), 0, 2, 2)
            chunk2 = WorkloadChunk("cpu", slice(2, 4), 2, 4, 2)
            chunk3 = WorkloadChunk("cpu", slice(4, 6), 4, 6, 2)
            
            result1 = ComputeResult(chunk1, np.array([1, 2]), 0.1)
            result2 = ComputeResult(chunk2, np.array([3, 4]), 0.1)
            result3 = ComputeResult(chunk3, np.array([5, 6]), 0.1)
            
            original_data = np.array([0] * 6)
            
            # Merge avec ordre différent (3, 1, 2)
            merged = manager._merge_results([result3, result1, result2], original_data)
            
            # Doit être ordonné par start_idx
            np.testing.assert_array_equal(merged, [1, 2, 3, 4, 5, 6])
    
    @patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False)
    def test_distribute_workload_cpu_simple(self, mock_cpu_only):
        """Test distribution sur CPU uniquement."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Données test
            data = np.random.randn(1000, 5)
            
            def simple_func(x):
                return x.sum(axis=1)
            
            # Distribution
            result = manager.distribute_workload(data, simple_func, seed=42)
            
            # Vérification
            assert result.shape == (1000,)
            assert isinstance(result, np.ndarray)
            
            # Test déterminisme
            result2 = manager.distribute_workload(data, simple_func, seed=42)
            np.testing.assert_array_equal(result, result2)
    
    @patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False)
    def test_distribute_workload_dataframe(self, mock_cpu_only):
        """Test distribution avec DataFrame."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # DataFrame test
            data = pd.DataFrame({
                'a': np.random.randn(500),
                'b': np.random.randn(500)
            })
            
            def df_func(df_chunk):
                return df_chunk['a'] + df_chunk['b']
            
            # Distribution
            result = manager.distribute_workload(data, df_func, seed=42)
            
            # Vérification
            assert len(result) == 500
            assert isinstance(result, pd.Series)
            
            # Vérification index préservé
            pd.testing.assert_index_equal(result.index, data.index)
    
    def test_distribute_workload_deterministic(self, mock_devices):
        """Test déterminisme avec seed fixe."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list, \
             patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False):
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager(device_balance={"cpu": 1.0})
            
            # Fonction avec randomness
            def random_func(x):
                np.random.seed(42)  # Seed interne pour cohérence
                noise = np.random.randn(x.shape[0]) * 0.1
                return x.sum(axis=1) + noise
            
            data = np.random.RandomState(42).randn(100, 3)
            
            # Deux exécutions avec même seed
            result1 = manager.distribute_workload(data, random_func, seed=42)
            result2 = manager.distribute_workload(data, random_func, seed=42)
            
            np.testing.assert_array_almost_equal(result1, result2, decimal=10)
    
    def test_distribute_workload_empty_data(self, mock_cpu_only):
        """Test avec données vides."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Données vides
            empty_data = np.array([]).reshape(0, 3)
            
            def dummy_func(x):
                return x.sum(axis=1)
            
            result = manager.distribute_workload(empty_data, dummy_func, seed=42)
            
            assert len(result) == 0
            assert result.shape == (0,)
    
    def test_distribute_workload_single_chunk(self, mock_cpu_only):
        """Test avec un seul chunk (pas de parallélisme)."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Petites données → un seul chunk
            data = np.array([[1, 2], [3, 4]])
            
            def test_func(x):
                return x.sum(axis=1)
            
            result = manager.distribute_workload(data, test_func, seed=42)
            
            np.testing.assert_array_equal(result, [3, 7])
    
    @patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False)
    def test_profile_auto_balance_cpu(self, mock_cpu_only):
        """Test profiling auto-balance CPU uniquement."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            # Profiling rapide
            ratios = manager.profile_auto_balance(
                sample_size=1000,
                warmup=0,
                runs=2
            )
            
            # Vérification
            assert "cpu" in ratios
            assert abs(ratios["cpu"] - 1.0) < 1e-6
            assert sum(ratios.values()) == pytest.approx(1.0, abs=1e-6)
    
    @patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False)
    def test_profile_auto_balance_multi_device(self, mock_devices):
        """Test profiling avec plusieurs devices (simulé)."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            # Simulation: seulement CPU disponible mais balance multi-device
            mock_list.return_value = [mock_devices[-1]]  # Seulement CPU
            
            manager = MultiGPUManager(device_balance={"cpu": 1.0})
            
            # Mock de distribute_workload pour simuler différentes performances
            original_distribute = manager.distribute_workload
            call_count = 0
            
            def mock_distribute(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # Simulation temps différents selon device
                time.sleep(0.001 if call_count % 2 == 0 else 0.002)
                return original_distribute(*args, **kwargs)
            
            manager.distribute_workload = mock_distribute
            
            # Profiling
            ratios = manager.profile_auto_balance(
                sample_size=100,
                warmup=0,
                runs=1
            )
            
            # Vérification structure
            assert isinstance(ratios, dict)
            assert sum(ratios.values()) == pytest.approx(1.0, abs=1e-6)
    
    def test_synchronize_methods(self, mock_devices):
        """Test méthodes de synchronisation."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list, \
             patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False):
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager()
            
            # Test sync sans GPU (doit être no-op)
            manager.synchronize("nccl")  # Pas d'erreur
            manager.synchronize("cuda")  # Pas d'erreur
            manager.synchronize("auto")  # Pas d'erreur
    
    def test_get_device_stats(self, mock_devices):
        """Test récupération stats devices."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager()
            stats = manager.get_device_stats()
            
            # Vérification structure
            assert "5090" in stats
            assert "2060" in stats
            assert "cpu" in stats
            
            # Vérification contenu
            gpu_stats = stats["5090"]
            assert gpu_stats["device_id"] == 0
            assert gpu_stats["available"] is True
            assert gpu_stats["memory_total_gb"] == 32.0
            assert gpu_stats["current_balance"] == 0.75
    
    def test_error_handling_device_unavailable(self, mock_devices):
        """Test gestion erreur device indisponible."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list, \
             patch('threadx.utils.gpu.multi_gpu.get_device_by_name') as mock_get_device:
            mock_list.return_value = mock_devices
            
            manager = MultiGPUManager()
            
            # Mock device introuvable pendant compute
            mock_get_device.return_value = None
            
            data = np.array([[1, 2]])
            chunk = WorkloadChunk("nonexistent", slice(0, 1), 0, 1, 1)
            
            def dummy_func(x):
                return x.sum(axis=1)
            
            result = manager._compute_chunk(data, chunk, dummy_func, seed=42)
            
            assert not result.success
            assert isinstance(result.error, DeviceUnavailableError)
    
    def test_error_handling_shape_mismatch(self, mock_cpu_only):
        """Test gestion erreur shape mismatch."""
        with patch('threadx.utils.gpu.multi_gpu.list_devices') as mock_list:
            mock_list.return_value = mock_cpu_only
            
            manager = MultiGPUManager()
            
            data = np.array([[1, 2], [3, 4]])  # 2 échantillons
            chunk = WorkloadChunk("cpu", slice(0, 2), 0, 2, 2)
            
            def bad_func(x):
                # Retourne mauvaise longueur
                return np.array([999])  # 1 élément au lieu de 2
            
            result = manager._compute_chunk(data, chunk, bad_func, seed=42)
            
            assert not result.success
            assert isinstance(result.error, ShapeMismatchError)


class TestGlobalManager:
    """Tests du gestionnaire global."""
    
    def test_get_default_manager_singleton(self):
        """Test que get_default_manager retourne le même instance."""
        manager1 = get_default_manager()
        manager2 = get_default_manager()
        
        assert manager1 is manager2
    
    def test_get_default_manager_properties(self):
        """Test propriétés du gestionnaire par défaut."""
        manager = get_default_manager()
        
        assert isinstance(manager, MultiGPUManager)
        assert hasattr(manager, 'device_balance')
        assert hasattr(manager, 'available_devices')


class TestIntegrationScenarios:
    """Tests d'intégration avec scénarios réalistes."""
    
    @patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False)
    def test_realistic_workload_cpu(self):
        """Test workload réaliste sur CPU."""
        # Simulation calcul d'indicateurs techniques
        n_samples = 10000
        n_features = 20
        
        # Données temporelles simulées
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        
        def technical_indicator(x):
            """Simule calcul indicateur technique (ex. moving average)."""
            # Moyenne mobile + écart-type
            rolling_mean = np.convolve(x.flatten(), np.ones(5)/5, mode='same')
            rolling_std = np.convolve((x.flatten() - rolling_mean)**2, np.ones(5)/5, mode='same')**0.5
            
            # Retourne indicateur par échantillon
            result = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                start_idx = i * n_features
                end_idx = (i + 1) * n_features
                result[i] = np.mean(rolling_mean[start_idx:end_idx] + rolling_std[start_idx:end_idx])
            
            return result
        
        # Exécution
        manager = get_default_manager()
        start_time = time.time()
        result = manager.distribute_workload(data, technical_indicator, seed=42)
        elapsed = time.time() - start_time
        
        # Vérifications
        assert result.shape == (n_samples,)
        assert not np.isnan(result).any()
        assert elapsed < 10.0  # Performance raisonnable
        
        # Déterminisme
        result2 = manager.distribute_workload(data, technical_indicator, seed=42)
        np.testing.assert_array_almost_equal(result, result2, decimal=6)
    
    @patch('threadx.utils.gpu.multi_gpu.CUPY_AVAILABLE', False)
    def test_dataframe_financial_simulation(self):
        """Test simulation financière avec DataFrame."""
        # Données OHLCV simulées
        n_days = 5000
        np.random.seed(42)
        
        df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(n_days) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(n_days) * 0.5) + np.abs(np.random.randn(n_days)),
            'low': 100 + np.cumsum(np.random.randn(n_days) * 0.5) - np.abs(np.random.randn(n_days)),
            'close': 100 + np.cumsum(np.random.randn(n_days) * 0.5),
            'volume': np.random.randint(1000, 10000, n_days)
        })
        
        def bollinger_bands(ohlcv_chunk):
            """Calcul simplifié des Bollinger Bands."""
            close_prices = ohlcv_chunk['close'].values
            
            # Moving average (fenêtre 20)
            window = min(20, len(close_prices))
            if window < 2:
                return np.zeros(len(close_prices))
            
            ma = np.convolve(close_prices, np.ones(window)/window, mode='same')
            
            # Standard deviation
            squared_diff = (close_prices - ma) ** 2
            std = np.sqrt(np.convolve(squared_diff, np.ones(window)/window, mode='same'))
            
            # Signal: distance de la close à la bande inférieure
            lower_band = ma - 2 * std
            signal = (close_prices - lower_band) / (2 * std + 1e-8)  # Éviter division par 0
            
            return signal
        
        # Exécution
        manager = get_default_manager()
        signals = manager.distribute_workload(df, bollinger_bands, seed=42)
        
        # Vérifications
        assert len(signals) == n_days
        assert isinstance(signals, pd.Series)
        assert not signals.isna().any()
        pd.testing.assert_index_equal(signals.index, df.index)
    
    def test_performance_benchmark_comparison(self):
        """Test benchmark CPU vs distribution multi-chunk."""
        n_samples = 50000
        n_features = 10
        
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        
        def matrix_ops(x):
            """Opérations matricielles intensives."""
            # Multiplication + inversion + somme
            result = np.sum(x * x.T @ x, axis=1) if x.ndim == 2 else x.sum()
            return result
        
        # Exécution directe NumPy (référence)
        start_time = time.time()
        reference_result = matrix_ops(data)
        numpy_time = time.time() - start_time
        
        # Exécution via MultiGPUManager (CPU, multi-chunk)
        manager = get_default_manager()
        start_time = time.time()
        distributed_result = manager.distribute_workload(data, matrix_ops, seed=42)
        distributed_time = time.time() - start_time
        
        # Vérifications
        np.testing.assert_array_almost_equal(reference_result, distributed_result, decimal=5)
        
        # Log des performances
        print(f"\nPerformance Benchmark:")
        print(f"  NumPy direct: {numpy_time:.3f}s")
        print(f"  Multi-GPU Manager: {distributed_time:.3f}s")
        print(f"  Ratio: {distributed_time/numpy_time:.2f}x")
        
        # Le multi-chunk peut avoir un overhead, mais doit rester raisonnable
        assert distributed_time < numpy_time * 3  # Max 3x slower (overhead acceptable)


if __name__ == "__main__":
    # Exécution des tests
    pytest.main([__file__, "-v", "--tb=short"])