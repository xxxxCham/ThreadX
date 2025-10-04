"""
Tests ThreadX Data I/O Module - Phase 2        assert isinstance(df_norm.index, pd.DatetimeIndex)
        assert str(df_norm.index.tz) == "UTC"
        assert len(df_norm) == 1000ests complets pour lecture/écriture OHLCV avec validation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

# Import du module à tester
try:
    from threadx.data.io import (
        read_frame, write_frame, normalize_ohlcv,
        DataNotFoundError, FileValidationError, SchemaMismatchError,
        OHLCV_SCHEMA, PANDERA_AVAILABLE
    )
    from threadx.data.synth import make_synth_ohlcv
except ImportError as e:
    pytest.skip(f"Modules ThreadX non disponibles: {e}", allow_module_level=True)


class TestOHLCVNormalization:
    """Tests normalisation OHLCV."""
    
    def test_normalize_basic_ohlcv(self):
        """Test normalisation DataFrame OHLCV de base."""
        # Données brutes avec colonnes correctes
        data = {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0], 
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000.0, 1500.0, 2000.0]
        }
        
        df = pd.DataFrame(data)
        df.index = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        
        df_norm = normalize_ohlcv(df)
        
        # Vérifications
        assert isinstance(df_norm.index, pd.DatetimeIndex)
        assert str(df_norm.index.tz) == "UTC"
        assert len(df_norm) == 3
        assert all(col in df_norm.columns for col in ["open", "high", "low", "close", "volume"])
        assert df_norm.index.is_monotonic_increasing
    
    def test_normalize_with_aliases(self):
        """Test normalisation avec alias colonnes (o/h/l/c)."""
        data = {
            "o": [100.0, 101.0], 
            "h": [105.0, 106.0],
            "l": [99.0, 100.0],
            "c": [104.0, 105.0],
            "vol": [1000.0, 1500.0]
        }
        
        df = pd.DataFrame(data)
        df.index = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        
        df_norm = normalize_ohlcv(df)
        
        # Vérification conversion alias
        assert "open" in df_norm.columns
        assert "high" in df_norm.columns
        assert "volume" in df_norm.columns
        assert "o" not in df_norm.columns
    
    def test_normalize_with_timestamp_column(self):
        """Test normalisation avec colonne timestamp (format JSON)."""
        # Timestamps en millisecondes (format Binance)
        timestamps = [1704067200000, 1704067260000, 1704067320000]  # 01/01/2024 00:00, 00:01, 00:02
        
        data = {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0], 
            "close": [104.0, 105.0, 106.0],
            "volume": [1000.0, 1500.0, 2000.0]
        }
        
        df = pd.DataFrame(data)
        df_norm = normalize_ohlcv(df)
        
        # Vérifications index datetime
        assert isinstance(df_norm.index, pd.DatetimeIndex)
        assert str(df_norm.index.tz) == "UTC"
        assert len(df_norm) == 3
        assert "timestamp" not in df_norm.columns
    
    def test_normalize_invalid_ohlc_logic(self):
        """Test détection incohérences OHLC."""
        data = {
            "open": [100.0],
            "high": [95.0],  # high < open → invalide
            "low": [99.0],
            "close": [104.0],
            "volume": [1000.0]
        }
        
        df = pd.DataFrame(data) 
        df.index = pd.date_range("2024-01-01", periods=1, freq="1min", tz="UTC")
        
        with pytest.raises(SchemaMismatchError, match="OHLC incohérent"):
            normalize_ohlcv(df)
    
    def test_normalize_missing_columns(self):
        """Test erreur colonnes OHLCV manquantes."""
        data = {
            "open": [100.0],
            "high": [105.0],
            # "low" manquant
            "close": [104.0],
            "volume": [1000.0]
        }
        
        df = pd.DataFrame(data)
        
        with pytest.raises(SchemaMismatchError, match="manquantes"):
            normalize_ohlcv(df)
    
    def test_normalize_empty_dataframe(self):
        """Test DataFrame vide."""
        df = pd.DataFrame()
        
        with pytest.raises(SchemaMismatchError, match="vide"):
            normalize_ohlcv(df)


class TestReadFrame:
    """Tests lecture fichiers OHLCV."""
    
    def test_read_parquet_valid(self, tmp_path):
        """Test lecture Parquet valide."""
        # Création données test
        df_test = make_synth_ohlcv(n=100, seed=42)
        
        # Sauvegarde Parquet
        parquet_path = tmp_path / "test.parquet"
        df_test.to_parquet(parquet_path)
        
        # Lecture et validation
        df_read = read_frame(parquet_path, validate=True)
        
        assert len(df_read) == 100
        assert isinstance(df_read.index, pd.DatetimeIndex)
        assert all(col in df_read.columns for col in ["open", "high", "low", "close", "volume"])
    
    def test_read_json_valid(self, tmp_path):
        """Test lecture JSON valide."""
        # Données test avec timestamps
        data = [
            {
                "timestamp": 1704067200000,  # 2024-01-01 00:00:00 UTC
                "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000.0
            },
            {
                "timestamp": 1704067260000,  # 2024-01-01 00:01:00 UTC  
                "open": 104.0, "high": 108.0, "low": 103.0, "close": 107.0, "volume": 1200.0
            }
        ]
        
        # Sauvegarde JSON
        json_path = tmp_path / "test.json"
        with open(json_path, "w") as f:
            json.dump(data, f)
        
        # Lecture
        df_read = read_frame(json_path, validate=True)
        
        assert len(df_read) == 2
        assert isinstance(df_read.index, pd.DatetimeIndex)
        assert str(df_read.index.tz) == "UTC"
    
    def test_read_file_not_found(self, tmp_path):
        """Test fichier inexistant."""
        missing_path = tmp_path / "missing.parquet"
        
        with pytest.raises(DataNotFoundError):
            read_frame(missing_path)
    
    def test_read_empty_file(self, tmp_path):
        """Test fichier vide."""
        empty_path = tmp_path / "empty.parquet"
        empty_path.touch()  # Crée fichier vide
        
        with pytest.raises(FileValidationError, match="vide"):
            read_frame(empty_path)
    
    def test_read_unsupported_format(self, tmp_path):
        """Test format non supporté."""
        txt_path = tmp_path / "test.txt" 
        txt_path.write_text("dummy content")
        
        with pytest.raises(FileValidationError, match="non supporté"):
            read_frame(txt_path)
    
    def test_read_validation_disabled(self, tmp_path):
        """Test lecture sans validation."""
        df_test = make_synth_ohlcv(n=50, seed=123)
        parquet_path = tmp_path / "novalidation.parquet"
        df_test.to_parquet(parquet_path)
        
        df_read = read_frame(parquet_path, validate=False)
        
        assert len(df_read) == 50
        # Pas de levée d'exception même si schéma incorrect potentiellement


class TestWriteFrame:
    """Tests écriture fichiers OHLCV."""
    
    def test_write_parquet_valid(self, tmp_path):
        """Test écriture Parquet valide."""
        df_test = make_synth_ohlcv(n=200, seed=999)
        parquet_path = tmp_path / "write_test.parquet"
        
        # Écriture
        write_frame(df_test, parquet_path)
        
        # Vérifications
        assert parquet_path.exists()
        assert parquet_path.stat().st_size > 0
        
        # Re-lecture pour validation
        df_reread = read_frame(parquet_path)
        assert len(df_reread) == 200
    
    def test_write_json_valid(self, tmp_path):
        """Test écriture JSON valide."""
        df_test = make_synth_ohlcv(n=10, seed=777)  # Petit dataset pour JSON
        json_path = tmp_path / "write_test.json"
        
        # Écriture
        write_frame(df_test, json_path)
        
        # Vérifications
        assert json_path.exists()
        
        # Validation format JSON
        with open(json_path) as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) == 10
        assert all("open" in record for record in data)
    
    def test_write_overwrite_protection(self, tmp_path):
        """Test protection écrasement."""
        df_test = make_synth_ohlcv(n=5, seed=42)
        test_path = tmp_path / "protected.parquet"
        
        # Première écriture
        write_frame(df_test, test_path)
        
        # Tentative écrasement sans overwrite=True
        with pytest.raises(FileValidationError, match="existe"):
            write_frame(df_test, test_path, overwrite=False)
        
        # Écrasement autorisé
        write_frame(df_test, test_path, overwrite=True)  # Ne doit pas lever d'exception
    
    def test_write_creates_directories(self, tmp_path):
        """Test création dossiers parents."""
        df_test = make_synth_ohlcv(n=5, seed=42)
        nested_path = tmp_path / "deep" / "nested" / "path" / "data.parquet"
        
        # Écriture avec création auto dossiers
        write_frame(df_test, nested_path)
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera non disponible")
    def test_write_invalid_schema(self, tmp_path):
        """Test validation schéma avant écriture."""
        # DataFrame invalide
        invalid_data = {
            "open": [100.0, -50.0],  # Prix négatif → invalide
            "high": [105.0, 60.0],
            "low": [99.0, 40.0], 
            "close": [104.0, 55.0],
            "volume": [1000.0, 500.0]
        }
        
        df_invalid = pd.DataFrame(invalid_data)
        df_invalid.index = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        
        test_path = tmp_path / "invalid.parquet"
        
        with pytest.raises(SchemaMismatchError):
            write_frame(df_invalid, test_path)


class TestIntegrationIOWorkflow:
    """Tests intégration workflow I/O complet."""
    
    def test_roundtrip_parquet(self, tmp_path):
        """Test cycle complet: génération → écriture → lecture → validation."""
        # Génération données
        df_original = make_synth_ohlcv(n=500, seed=12345)
        
        # Écriture
        parquet_path = tmp_path / "roundtrip.parquet"
        write_frame(df_original, parquet_path)
        
        # Lecture
        df_roundtrip = read_frame(parquet_path, validate=True)
        
        # Validation identité (approximative pour float)
        assert len(df_roundtrip) == len(df_original)
        assert df_roundtrip.index.equals(df_original.index)
        
        # Vérification valeurs (tolérance float64)
        pd.testing.assert_frame_equal(df_roundtrip, df_original, rtol=1e-10)
    
    def test_roundtrip_json(self, tmp_path):
        """Test cycle JSON avec préservation timestamps UTC."""
        df_original = make_synth_ohlcv(n=20, seed=54321, start="2024-02-15")
        
        json_path = tmp_path / "roundtrip.json"
        write_frame(df_original, json_path)
        
        df_roundtrip = read_frame(json_path, validate=True)
        
        # Vérifications spécifiques JSON
        assert len(df_roundtrip) == 20
        assert str(df_roundtrip.index.tz) == "UTC"  # type: ignore
        
        # Index temporel proche (JSON a moins de précision)
        assert abs((df_roundtrip.index[0] - df_original.index[0]).total_seconds()) < 1
    
    def test_format_autodetection(self, tmp_path):
        """Test auto-détection format par extension."""
        df_test = make_synth_ohlcv(n=30, seed=999)
        
        # Test Parquet
        parquet_path = tmp_path / "auto.parquet"
        write_frame(df_test, parquet_path, fmt=None)  # Auto-détection
        df_parquet = read_frame(parquet_path, fmt=None)
        assert len(df_parquet) == 30
        
        # Test JSON  
        json_path = tmp_path / "auto.json"
        write_frame(df_test, json_path, fmt=None)
        df_json = read_frame(json_path, fmt=None)
        assert len(df_json) == 30