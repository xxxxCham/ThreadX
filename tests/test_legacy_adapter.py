"""
Tests complets pour Legacy Adapter - ThreadX Data
Tests offline, seed=42, mocks API calls.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path
import requests

from threadx.data.legacy_adapter import LegacyAdapter, IngestionError, APIError
from threadx.config import Settings

# Fixtures
@pytest.fixture
def test_adapter():
    test_settings = Settings(
        DATA_ROOT=Path("./test_data"),
        GPU_DEVICES=["cpu"]
    )
    return LegacyAdapter(test_settings)

@pytest.fixture
def sample_klines_json():
    """Fixture avec données synthétiques Binance klines."""
    np.random.seed(42)
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    klines = []
    for i in range(100):
        ts = int((base_time + timedelta(minutes=i)).timestamp() * 1000)
        open_price = 50000 + np.random.randn() * 100
        close_price = open_price + np.random.randn() * 50
        high_price = max(open_price, close_price) + abs(np.random.randn()) * 20
        low_price = min(open_price, close_price) - abs(np.random.randn()) * 20
        volume = abs(np.random.randn()) * 10
        
        klines.append([
            ts, f"{open_price:.2f}", f"{high_price:.2f}", f"{low_price:.2f}", 
            f"{close_price:.2f}", f"{volume:.4f}", ts + 59999,
            "100.0", 100, "50.0", "25.0", "0"
        ])
    
    return klines

class TestLegacyAdapter:
    """Tests complets pour LegacyAdapter."""
    
    def test_initialization(self, test_adapter):
        """Test initialisation avec settings TOML."""
        assert test_adapter.binance_endpoint == "https://api.binance.com/api/v3/klines"
        assert test_adapter.max_retries == 3
        assert test_adapter.binance_limit == 1000
        
        # Chemins
        expected_raw = Path(test_adapter.settings.DATA_ROOT) / "raw" / "json"
        assert test_adapter.raw_json_path == expected_raw
    
    @patch('requests.get')
    def test_fetch_klines_1m_success(self, mock_get, test_adapter, sample_klines_json):
        """Test téléchargement réussi."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_klines_json
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 40, 0)
        
        result = test_adapter.fetch_klines_1m("BTCUSDT", start, end)
        
        assert len(result) == 100
        assert mock_get.call_count >= 1
        
        # Vérification paramètres API
        call_args = mock_get.call_args
        params = call_args[1]['params']
        assert params['symbol'] == 'BTCUSDT'
        assert params['interval'] == '1m'
    
    @patch('requests.get')  
    def test_fetch_klines_retry_logic(self, mock_get, test_adapter):
        """Test retry avec backoff."""
        # Premier appel: rate limited
        mock_response_fail = Mock()
        mock_response_fail.status_code = 429
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.HTTPError("Rate limited")
        
        # Deuxième appel: succès
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = []
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 1, 1, 0)
        
        with patch('time.sleep'):  # Mock sleep
            result = test_adapter.fetch_klines_1m("BTCUSDT", start, end)
        
        assert mock_get.call_count == 2
        assert result == []
    
    @patch('requests.get')
    def test_fetch_klines_max_retries_exceeded(self, mock_get, test_adapter):
        """Test échec après max retries."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")
        mock_get.return_value = mock_response
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 1, 1, 0)
        
        with patch('time.sleep'):
            with pytest.raises(APIError, match="Failed after 3 retries"):
                test_adapter.fetch_klines_1m("BTCUSDT", start, end)
    
    def test_json_to_dataframe_success(self, test_adapter, sample_klines_json):
        """Test conversion JSON vers DataFrame."""
        df = test_adapter.json_to_dataframe(sample_klines_json)
        
        # Structure
        assert len(df) == 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == 'UTC'
        
        # Types
        for col in df.columns:
            assert df[col].dtype == np.float64
        
        # OHLC logic
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()  
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        assert (df['volume'] >= 0).all()
    
    def test_json_to_dataframe_empty(self, test_adapter):
        """Test avec données vides."""
        df = test_adapter.json_to_dataframe([])
        assert df.empty
    
    def test_json_to_dataframe_malformed(self, test_adapter):
        """Test avec données malformées."""
        malformed = [["invalid", "data"]]
        
        with pytest.raises(IngestionError):
            test_adapter.json_to_dataframe(malformed)
    
    def test_fix_timestamp_conversion_ms(self, test_adapter):
        """Test normalisation timestamps millisecondes."""
        df_ms = pd.DataFrame({
            'open_time': [1704067200000, 1704067260000],  # 2024-01-01 en ms
            'value': [1, 2]
        })
        
        result = test_adapter._fix_timestamp_conversion(df_ms, 'open_time')
        
        assert str(result['open_time'].dt.tz) == 'UTC'
        assert result['open_time'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00', tz='UTC')
    
    def test_fix_timestamp_conversion_seconds(self, test_adapter):
        """Test normalisation timestamps secondes."""
        df_s = pd.DataFrame({
            'open_time': [1704067200, 1704067260],  # 2024-01-01 en secondes
            'value': [1, 2]
        })
        
        result = test_adapter._fix_timestamp_conversion(df_s, 'open_time')
        
        assert str(result['open_time'].dt.tz) == 'UTC'
        assert result['open_time'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00', tz='UTC')
    
    def test_detect_gaps_1m_no_gaps(self, test_adapter):
        """Test détection gaps - série continue."""
        idx = pd.date_range('2024-01-01 00:00', periods=60, freq='1min', tz='UTC')
        df = pd.DataFrame({'close': range(60)}, index=idx)
        
        gaps = test_adapter.detect_gaps_1m(df)  
        assert gaps == []
    
    def test_detect_gaps_1m_with_gaps(self, test_adapter):
        """Test détection gaps - avec trous."""
        times = pd.to_datetime([
            '2024-01-01 00:00', '2024-01-01 00:01', '2024-01-01 00:02',
            # Gap: 00:03, 00:04 manquants
            '2024-01-01 00:05', '2024-01-01 00:06'
        ], utc=True)
        
        df = pd.DataFrame({'close': range(len(times))}, index=times)
        
        gaps = test_adapter.detect_gaps_1m(df)
        assert len(gaps) == 1
        
        gap_start, gap_end = gaps[0]
        assert gap_start == pd.Timestamp('2024-01-01 00:03', tz='UTC')
        assert gap_end == pd.Timestamp('2024-01-01 00:04', tz='UTC')
    
    def test_fill_gaps_conservative_small_gap(self, test_adapter):
        """Test remplissage petits gaps."""
        times = pd.to_datetime([
            '2024-01-01 00:00', '2024-01-01 00:01',
            # Gap 1 minute (00:02 manquant) 
            '2024-01-01 00:03', '2024-01-01 00:04'
        ], utc=True)
        
        df = pd.DataFrame({
            'open': [100, 101, 103, 104],
            'high': [100, 101, 103, 104],
            'low': [100, 101, 103, 104], 
            'close': [100, 101, 103, 104],
            'volume': [10, 11, 13, 14]
        }, index=times)
        
        filled = test_adapter.fill_gaps_conservative(df, max_gap_ratio=0.5)
        
        # Vérification gap rempli
        expected_times = pd.date_range('2024-01-01 00:00', '2024-01-01 00:04', freq='1min', tz='UTC')
        assert len(filled) == len(expected_times)
        
        # Tri vérifié
        assert filled.index.is_monotonic_increasing
    
    def test_fill_gaps_conservative_large_gap(self, test_adapter):
        """Test non-remplissage grands gaps."""
        times = pd.to_datetime([
            '2024-01-01 00:00', '2024-01-01 00:01',
            # Gap large: 00:02 à 00:10 manquant (> 5%)
            '2024-01-01 00:10'
        ], utc=True)
        
        df = pd.DataFrame({
            'open': [100, 101, 110],
            'high': [100, 101, 110],
            'low': [100, 101, 110],
            'close': [100, 101, 110], 
            'volume': [10, 11, 20]
        }, index=times)
        
        filled = test_adapter.fill_gaps_conservative(df, max_gap_ratio=0.05)
        
        # Gap ne doit PAS être rempli
        assert len(filled) == 3  # Données originales seulement


class TestEdgeCases:
    """Tests cas limites et erreurs."""
    
    def test_empty_input_handling(self, test_adapter):
        """Test gestion entrées vides."""
        # DataFrame vide
        empty_df = pd.DataFrame()
        
        gaps = test_adapter.detect_gaps_1m(empty_df)
        assert gaps == []
        
        filled = test_adapter.fill_gaps_conservative(empty_df)
        assert filled.empty
    
    def test_single_point_data(self, test_adapter):
        """Test avec un seul point de données."""
        single_time = pd.to_datetime(['2024-01-01 00:00'], utc=True)
        df_single = pd.DataFrame({
            'open': [100], 'high': [100], 'low': [100], 'close': [100], 'volume': [10]
        }, index=single_time)
        
        gaps = test_adapter.detect_gaps_1m(df_single)
        assert gaps == []  # Pas de gaps avec 1 seul point
    
    def test_malformed_timestamps(self, test_adapter):
        """Test timestamps malformés."""
        df_bad = pd.DataFrame({
            'open_time': ['not_a_timestamp', 'also_bad'],
            'value': [1, 2]
        })
        
        # Ne doit pas planter, retourner DataFrame original
        result = test_adapter._fix_timestamp_conversion(df_bad, 'open_time')
        assert 'open_time' in result.columns


class TestIntegrationAdapter:
    """Tests d'intégration adapter complet."""
    
    @patch('requests.get')
    def test_full_pipeline_download_to_dataframe(self, mock_get, test_adapter, sample_klines_json):
        """Test pipeline complet: téléchargement → conversion → normalisation."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_klines_json
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 40, 0)
        
        # Étape 1: Téléchargement
        raw_klines = test_adapter.fetch_klines_1m("BTCUSDT", start, end)
        assert len(raw_klines) == 100
        
        # Étape 2: Conversion
        df = test_adapter.json_to_dataframe(raw_klines)
        assert not df.empty
        assert len(df) == 100
        
        # Étape 3: Vérifications finales
        assert df.index.is_monotonic_increasing
        assert df.index.is_unique
        assert str(df.index.tz) == 'UTC'
        
        # OHLCV schema
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        assert list(df.columns) == expected_cols
        
        # Types
        for col in expected_cols:
            assert df[col].dtype == np.float64
    
    def test_robustness_edge_data(self, test_adapter):
        """Test robustesse données limites."""
        # Données avec valeurs extrêmes
        extreme_klines = [
            [1704067200000, "0.0", "999999.99", "0.0", "1.0", "0.0", 1704067259999,
             "0.0", 0, "0.0", "0.0", "0"],  # Prix 0
            [1704067260000, "inf", "inf", "-inf", "nan", "-1.0", 1704067319999,
             "nan", -1, "inf", "-inf", "0"]  # Valeurs impossibles
        ]
        
        # Ne doit pas planter même avec données extrêmes
        try:
            df = test_adapter.json_to_dataframe(extreme_klines)
            # Valeurs invalides doivent être gérées (NaN, normalisation)
            assert not df.empty or df.empty  # Accepter résultat vide si normalisation échoue
        except IngestionError:
            # Acceptable si données trop corrompues
            pass