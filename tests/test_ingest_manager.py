"""
Tests pour Data Ingestion Manager - ThreadX
Tests offline, seed=42, avec mocks pour API calls.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from threadx.data.ingest import IngestionManager, download_ohlcv_1m, update_assets_batch
from threadx.data.legacy_adapter import IngestionError, APIError
from threadx.config import Settings

# Fixtures
@pytest.fixture
def test_settings():
    return Settings(
        DATA_ROOT=Path("./test_data"),
        GPU_DEVICES=["cpu"]
    )

@pytest.fixture
def ingestion_manager(test_settings):
    return IngestionManager(test_settings)

@pytest.fixture
def sample_1m_data():
    """DataFrame 1m synthétique pour tests."""
    np.random.seed(42)
    
    # 100 minutes de données (1h40)
    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=pd.Timestamp.now().tz)
    timestamps = pd.date_range(start_time, periods=100, freq='1min', tz='UTC')
    
    data = {
        'open': 50000 + np.random.randn(100) * 100,
        'high': 50100 + np.random.randn(100) * 100,
        'low': 49900 + np.random.randn(100) * 100, 
        'close': 50000 + np.random.randn(100) * 100,
        'volume': np.abs(np.random.randn(100)) * 10
    }
    
    df = pd.DataFrame(data, index=timestamps)
    
    # Correction OHLC logic
    df['high'] = np.maximum.reduce([df['open'], df['high'], df['low'], df['close']])
    df['low'] = np.minimum.reduce([df['open'], df['high'], df['low'], df['close']])
    
    return df.astype(np.float64)

@pytest.fixture
def sample_1m_data_with_gaps():
    """DataFrame 1m avec gaps pour tests."""
    np.random.seed(42)
    
    # Création série continue puis suppression de quelques points
    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=pd.Timestamp.now().tz)
    full_timestamps = pd.date_range(start_time, periods=100, freq='1min', tz='UTC')
    
    # Suppression de quelques minutes pour créer gaps
    gap_indices = [10, 11, 12, 50, 85, 86]  # 2 gaps: 3min + 1min + 2min
    timestamps = full_timestamps.delete(gap_indices)
    
    data = {
        'open': 50000 + np.random.randn(len(timestamps)) * 100,
        'high': 50100 + np.random.randn(len(timestamps)) * 100,
        'low': 49900 + np.random.randn(len(timestamps)) * 100,
        'close': 50000 + np.random.randn(len(timestamps)) * 100,
        'volume': np.abs(np.random.randn(len(timestamps))) * 10
    }
    
    df = pd.DataFrame(data, index=timestamps)
    
    # Correction OHLC logic
    df['high'] = np.maximum.reduce([df['open'], df['high'], df['low'], df['close']])
    df['low'] = np.minimum.reduce([df['open'], df['high'], df['low'], df['close']])
    
    return df.astype(np.float64)


class TestIngestionManager:
    """Tests pour IngestionManager."""
    
    def test_initialization(self, ingestion_manager, test_settings):
        """Test d'initialisation."""
        assert ingestion_manager.settings == test_settings
        assert ingestion_manager.adapter is not None
        
        # Chemins
        expected_raw = Path(test_settings.DATA_ROOT) / "raw" / "1m"
        assert ingestion_manager.raw_1m_path == expected_raw
        
        # Stats
        assert ingestion_manager.session_stats["symbols_processed"] == 0
    
    @patch('threadx.data.ingest.read_frame')
    @patch('threadx.data.ingest.write_frame')
    def test_download_ohlcv_1m_local_data_available(self, mock_write, mock_read, 
                                                   ingestion_manager, sample_1m_data):
        """Test avec données locales disponibles (pas de téléchargement)."""
        # Mock données locales existantes
        mock_read.return_value = sample_1m_data
        
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=pd.Timestamp.now().tz)
        end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=pd.Timestamp.now().tz)
        
        result = ingestion_manager.download_ohlcv_1m("BTCUSDT", start, end)
        
        # Vérifications
        assert not result.empty
        assert len(result) <= len(sample_1m_data)  # Subset de la plage demandée
        assert result.index.min() >= start
        assert result.index.max() <= end
        
        # Aucune écriture (données déjà disponibles)
        mock_write.assert_not_called()
    
    @patch('threadx.data.ingest.read_frame')
    @patch('threadx.data.ingest.write_frame')
    def test_download_ohlcv_1m_missing_data_download(self, mock_write, mock_read,
                                                    ingestion_manager, sample_1m_data):
        """Test avec téléchargement nécessaire."""
        # Mock pas de données locales
        mock_read.side_effect = FileNotFoundError("No local data")
        
        # Mock adapter téléchargement
        with patch.object(ingestion_manager.adapter, 'fetch_klines_1m') as mock_fetch, \
             patch.object(ingestion_manager.adapter, 'json_to_dataframe') as mock_json_to_df:
            
            mock_fetch.return_value = [{"mock": "klines"}]  # Mock raw JSON
            mock_json_to_df.return_value = sample_1m_data
            
            start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=pd.Timestamp.now().tz)
            end = datetime(2024, 1, 1, 1, 40, 0, tzinfo=pd.Timestamp.now().tz)
            
            result = ingestion_manager.download_ohlcv_1m("BTCUSDT", start, end)
            
            # Vérifications
            assert not result.empty
            mock_fetch.assert_called_once_with("BTCUSDT", start, end)
            mock_json_to_df.assert_called_once()
            mock_write.assert_called_once()  # Sauvegarde données téléchargées
    
    def test_resample_from_1m(self, ingestion_manager, sample_1m_data):
        """Test resample depuis 1m truth."""
        # Mock du module resample
        with patch('threadx.data.ingest.resample_from_1m') as mock_resample:
            # Mock result 1h
            resampled_1h = sample_1m_data.resample('1H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            mock_resample.return_value = resampled_1h
            
            result = ingestion_manager.resample_from_1m(sample_1m_data, "1h")
            
            assert not result.empty
            assert len(result) < len(sample_1m_data)  # Moins de barres après resample
            mock_resample.assert_called_once_with(sample_1m_data, "1h")
    
    def test_verify_resample_consistency_ok(self, ingestion_manager, sample_1m_data):
        """Test vérification consistency OK."""
        # Création données 1h cohérentes
        df_1h_expected = sample_1m_data.resample('1H').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        with patch.object(ingestion_manager, 'resample_from_1m') as mock_resample:
            mock_resample.return_value = df_1h_expected
            
            report = ingestion_manager.verify_resample_consistency(
                sample_1m_data, df_1h_expected, "1h"
            )
            
            assert report["ok"] is True
            assert len(report["anomalies"]) == 0
            assert report["stats"]["common_timestamps"] > 0
    
    def test_verify_resample_consistency_anomalies(self, ingestion_manager, sample_1m_data):
        """Test vérification avec anomalies détectées."""
        # Création données 1h avec anomalies
        df_1h_wrong = sample_1m_data.resample('1H').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        # Introduction d'anomalies
        df_1h_wrong['close'] = df_1h_wrong['close'] + 1000  # Écart significatif
        
        with patch.object(ingestion_manager, 'resample_from_1m') as mock_resample:
            # Resample correct pour comparaison
            df_1h_correct = sample_1m_data.resample('1H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            mock_resample.return_value = df_1h_correct
            
            report = ingestion_manager.verify_resample_consistency(
                sample_1m_data, df_1h_wrong, "1h"
            )
            
            assert report["ok"] is False
            assert len(report["anomalies"]) > 0
            assert any("close" in anomaly for anomaly in report["anomalies"])
    
    def test_detect_and_fill_gaps_1m(self, ingestion_manager, sample_1m_data_with_gaps):
        """Test détection et comblement gaps."""
        # Mock adapter methods
        with patch.object(ingestion_manager.adapter, 'detect_gaps_1m') as mock_detect, \
             patch.object(ingestion_manager.adapter, 'fill_gaps_conservative') as mock_fill:
            
            # Mock détection gaps
            mock_gaps = [
                (pd.Timestamp('2024-01-01 00:10:00', tz='UTC'), 
                 pd.Timestamp('2024-01-01 00:12:00', tz='UTC'))
            ]
            mock_detect.side_effect = [mock_gaps, []]  # Avant et après remplissage
            
            # Mock remplissage
            mock_fill.return_value = sample_1m_data_with_gaps  # Simulé rempli
            
            result = ingestion_manager.detect_and_fill_gaps_1m(sample_1m_data_with_gaps)
            
            assert not result.empty
            mock_detect.assert_called()
            mock_fill.assert_called_once()
            assert ingestion_manager.session_stats["gaps_filled"] == 1
    
    @patch('threadx.data.ingest.ThreadPoolExecutor')
    def test_update_assets_batch(self, mock_executor, ingestion_manager, sample_1m_data):
        """Test mise à jour batch."""
        # Mock executor et futures
        mock_future = Mock()
        mock_future.result.return_value = {
            "symbol": "BTCUSDT",
            "success": True,
            "timeframes": [{"tf": "1h", "rows": 2, "status": "ok"}],
            "errors": []
        }
        
        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed
        with patch('threadx.data.ingest.as_completed', return_value=[mock_future]):
            start = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now().tz)
            end = datetime(2024, 1, 2, tzinfo=pd.Timestamp.now().tz)
            
            results = ingestion_manager.update_assets_batch(
                symbols=["BTCUSDT"],
                timeframes=["1h"],
                start=start,
                end=end
            )
            
            assert "summary" in results
            assert "details" in results
            assert "errors" in results
            assert results["summary"]["symbols_requested"] == 1


class TestPublicAPI:
    """Tests pour API publique."""
    
    @patch('threadx.data.ingest.IngestionManager')
    def test_download_ohlcv_1m_api(self, mock_manager_class, sample_1m_data):
        """Test API publique download_ohlcv_1m."""
        mock_manager = Mock()
        mock_manager.download_ohlcv_1m.return_value = sample_1m_data
        mock_manager_class.return_value = mock_manager
        
        start = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now().tz)
        end = datetime(2024, 1, 2, tzinfo=pd.Timestamp.now().tz)
        
        result = download_ohlcv_1m("BTCUSDT", start, end, force=True)
        
        assert not result.empty
        mock_manager.download_ohlcv_1m.assert_called_once_with("BTCUSDT", start, end, force=True)
    
    @patch('threadx.data.ingest.IngestionManager')  
    def test_update_assets_batch_api(self, mock_manager_class):
        """Test API publique update_assets_batch."""
        mock_manager = Mock()
        mock_manager.update_assets_batch.return_value = {"summary": {"total": 1}}
        mock_manager_class.return_value = mock_manager
        
        start = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now().tz)
        end = datetime(2024, 1, 2, tzinfo=pd.Timestamp.now().tz)
        
        result = update_assets_batch(
            symbols=["BTCUSDT"],
            timeframes=["1h"],
            start=start,
            end=end,
            force=True
        )
        
        assert "summary" in result
        mock_manager.update_assets_batch.assert_called_once()


class TestIntegrationSmokeTest:
    """Tests d'intégration smoke (import-only)."""
    
    def test_import_ingest_no_errors(self):
        """Test import modules sans erreur."""
        from threadx.data.ingest import IngestionManager, download_ohlcv_1m, update_assets_batch
        from threadx.data.legacy_adapter import LegacyAdapter
        
        # Vérification classes disponibles
        assert IngestionManager is not None
        assert LegacyAdapter is not None
        assert callable(download_ohlcv_1m)
        assert callable(update_assets_batch)
    
    def test_basic_instantiation(self):
        """Test instantiation basique sans I/O."""
        test_settings = Settings(
            DATA_ROOT=Path("./test_temp"),
            GPU_DEVICES=["cpu"]
        )
        
        # Instantiation sans échec
        manager = IngestionManager(test_settings)
        assert manager is not None
        assert manager.session_stats["symbols_processed"] == 0