"""
Tests ThreadX Data Resampling Module - Phase 2
Tests complets pour resampling canonique et gestion gaps.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import du module à tester
try:
    from threadx.data.resample import (
        resample_from_1m, resample_batch,
        TimeframeError, GapFillingError,
        _validate_timeframe, _detect_gaps, _smart_ffill
    )
    from threadx.data.synth import make_synth_ohlcv
    from threadx.data.io import SchemaMismatchError
except ImportError as e:
    pytest.skip(f"Modules ThreadX non disponibles: {e}", allow_module_level=True)


class TestTimeframeValidation:
    """Tests validation timeframes."""
    
    def test_valid_timeframes(self):
        """Test timeframes valides."""
        valid_tfs = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        
        for tf in valid_tfs:
            result = _validate_timeframe(tf)
            assert result == tf.lower()
    
    def test_case_insensitive(self):
        """Test insensibilité casse."""
        assert _validate_timeframe("1M") == "1m"
        assert _validate_timeframe("1H") == "1h"
        assert _validate_timeframe("15M") == "15m"
    
    def test_invalid_timeframes(self):
        """Test timeframes invalides."""
        invalid_tfs = ["2s", "90m", "25h", "invalid", ""]
        
        for tf in invalid_tfs:
            with pytest.raises(TimeframeError):
                _validate_timeframe(tf)


class TestGapDetection:
    """Tests détection gaps."""
    
    def test_no_gaps_continuous(self):
        """Test données continues sans gaps."""
        # Données 1m continues
        df_continuous = make_synth_ohlcv(n=60, freq="1min", seed=42)
        
        gap_ratio, n_missing = _detect_gaps(df_continuous, "1min")
        
        assert gap_ratio == 0.0
        assert n_missing == 0
    
    def test_detect_small_gaps(self):
        """Test détection petits gaps (<5%)."""
        # Données avec quelques timestamps manquants
        df_base = make_synth_ohlcv(n=100, freq="1min", seed=123)
        
        # Suppression 3 timestamps aléatoires (3%)
        indices_to_drop = [10, 25, 60]
        df_with_gaps = df_base.drop(df_base.index[indices_to_drop])
        
        gap_ratio, n_missing = _detect_gaps(df_with_gaps, "1min")
        
        assert gap_ratio > 0.0
        assert gap_ratio < 0.05  # <5%
        assert n_missing == 3
    
    def test_detect_large_gaps(self):
        """Test détection gaps importants (>5%)."""
        df_base = make_synth_ohlcv(n=100, freq="1min", seed=456)
        
        # Suppression 10 timestamps (10%)
        indices_to_drop = list(range(20, 30))
        df_with_large_gaps = df_base.drop(df_base.index[indices_to_drop])
        
        gap_ratio, n_missing = _detect_gaps(df_with_large_gaps, "1min")
        
        assert gap_ratio > 0.05  # >5%
        assert n_missing == 10
    
    def test_empty_dataframe_gaps(self):
        """Test DataFrame vide."""
        df_empty = pd.DataFrame()
        
        gap_ratio, n_missing = _detect_gaps(df_empty)
        
        assert gap_ratio == 0.0
        assert n_missing == 0


class TestSmartForwardFill:
    """Tests forward-fill intelligent."""
    
    def test_ffill_small_gaps(self):
        """Test ffill sur petits gaps acceptables."""
        # Données avec gaps <5%
        df_base = make_synth_ohlcv(n=100, freq="1min", seed=789)
        
        # Suppression 3 timestamps (3%)
        df_with_gaps = df_base.drop(df_base.index[[10, 25, 60]])
        original_len = len(df_with_gaps)
        
        # Forward-fill avec seuil 5%
        df_filled = _smart_ffill(df_with_gaps, max_gap_ratio=0.05)
        
        # Vérifications
        assert len(df_filled) >= original_len  # Peut avoir ajouté des lignes
        assert not df_filled.isnull().any().any()  # Pas de NaN
    
    def test_ffill_large_gaps_warning(self):
        """Test ffill conservateur sur gaps importants."""
        df_base = make_synth_ohlcv(n=50, freq="1min", seed=999)
        
        # Suppression 8 timestamps (16% > 5%)
        indices_to_drop = list(range(15, 23))
        df_with_large_gaps = df_base.drop(df_base.index[indices_to_drop])
        
        # Forward-fill avec seuil dépassé
        df_filled = _smart_ffill(df_with_large_gaps, max_gap_ratio=0.05)
        
        # Doit être conservateur (pas de ffill longue distance)
        assert len(df_filled) == len(df_with_large_gaps)  # Pas d'ajout massif
    
    def test_ffill_no_gaps(self):
        """Test ffill sur données continues."""
        df_continuous = make_synth_ohlcv(n=50, freq="1min", seed=111)
        
        df_filled = _smart_ffill(df_continuous, max_gap_ratio=0.05)
        
        # Données inchangées
        pd.testing.assert_frame_equal(df_filled, df_continuous)


class TestResampleFrom1m:
    """Tests resampling canonique 1m → X."""
    
    def test_resample_1m_to_15m(self):
        """Test resampling 1m → 15m avec validation agrégations."""
        # Données 1m (4 heures = 240 minutes → 16 barres 15m)
        df_1m = make_synth_ohlcv(n=240, freq="1min", seed=42)
        
        # Resampling 15m
        df_15m = resample_from_1m(df_1m, "15m")
        
        # Vérifications
        assert len(df_15m) == 16  # 240min / 15min = 16 barres
        assert isinstance(df_15m.index, pd.DatetimeIndex)
        assert all(col in df_15m.columns for col in ["open", "high", "low", "close", "volume"])
        
        # Vérification logique OHLC première barre
        first_15m = df_15m.iloc[0]
        first_15_minutes_1m = df_1m.iloc[:15]
        
        # Agrégations attendues
        assert first_15m["open"] == first_15_minutes_1m["open"].iloc[0]  # Premier open
        assert first_15m["high"] == first_15_minutes_1m["high"].max()    # Max high
        assert first_15m["low"] == first_15_minutes_1m["low"].min()      # Min low
        assert first_15m["close"] == first_15_minutes_1m["close"].iloc[-1]  # Dernier close
        assert abs(first_15m["volume"] - first_15_minutes_1m["volume"].sum()) < 1e-10  # Somme volume
    
    def test_resample_1m_to_1h(self):
        """Test resampling 1m → 1h."""
        # Données 1m (24 heures = 1440 minutes → 24 barres 1h)
        df_1m = make_synth_ohlcv(n=1440, freq="1min", seed=123, start="2024-01-01")
        
        df_1h = resample_from_1m(df_1m, "1h")
        
        # Vérifications
        assert len(df_1h) == 24
        
        # Cohérence temporelle
        expected_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        assert df_1h.index[0] == expected_start
        
        # Espacement 1h
        time_diffs = df_1h.index.to_series().diff().dropna()
        assert all(diff == pd.Timedelta("1H") for diff in time_diffs)
    
    def test_resample_1m_to_4h(self):
        """Test resampling 1m → 4h."""
        # Données 2 jours (48h = 2880min → 12 barres 4h)
        df_1m = make_synth_ohlcv(n=2880, freq="1min", seed=456)
        
        df_4h = resample_from_1m(df_1m, "4h")
        
        assert len(df_4h) == 12  # 2880min / (4*60min) = 12
    
    def test_resample_invalid_timeframe(self):
        """Test erreur timeframe invalide."""
        df_1m = make_synth_ohlcv(n=100, freq="1min", seed=42)
        
        with pytest.raises(TimeframeError):
            resample_from_1m(df_1m, "invalid_tf")
    
    def test_resample_empty_dataframe(self):
        """Test erreur DataFrame vide."""
        df_empty = pd.DataFrame()
        
        with pytest.raises(SchemaMismatchError, match="vide"):
            resample_from_1m(df_empty, "15m")
    
    def test_resample_missing_columns(self):
        """Test erreur colonnes OHLCV manquantes."""
        # DataFrame incomplet
        incomplete_data = {
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            # "low" manquant
            "close": [104.0, 105.0],
            # "volume" manquant
        }
        
        df_incomplete = pd.DataFrame(incomplete_data)
        df_incomplete.index = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        
        with pytest.raises(SchemaMismatchError, match="manquantes"):
            resample_from_1m(df_incomplete, "15m")
    
    def test_resample_with_gaps_small(self):
        """Test resampling avec petits gaps (forward-fill)."""
        df_1m = make_synth_ohlcv(n=120, freq="1min", seed=777)
        
        # Suppression 2 timestamps (1.7% < 5%)
        df_with_gaps = df_1m.drop(df_1m.index[[30, 60]])
        
        # Resampling avec gap filling
        df_resampled = resample_from_1m(df_with_gaps, "15m", gap_ffill_threshold=0.05)
        
        # Doit fonctionner (gaps acceptables)
        assert len(df_resampled) > 0
        assert not df_resampled.isnull().any().any()
    
    def test_resample_with_gaps_large(self):
        """Test resampling avec gaps importants (warning)."""
        df_1m = make_synth_ohlcv(n=100, freq="1min", seed=888)
        
        # Suppression 10 timestamps (10% > 5%)
        indices_to_drop = list(range(40, 50))
        df_with_large_gaps = df_1m.drop(df_1m.index[indices_to_drop])
        
        # Resampling malgré gaps importants
        with pytest.warns(UserWarning):  # Peut générer des warnings
            df_resampled = resample_from_1m(df_with_large_gaps, "15m", gap_ffill_threshold=0.05)
        
        # Doit quand même produire un résultat
        assert len(df_resampled) > 0


class TestResampleBatch:
    """Tests resampling batch."""
    
    def test_batch_sequential_small(self):
        """Test batch séquentiel (peu de symboles)."""
        # Préparation données multi-symboles
        frames_by_symbol = {
            "BTCUSDC": make_synth_ohlcv(n=60, seed=1, base_price=50000),
            "ETHUSDC": make_synth_ohlcv(n=60, seed=2, base_price=3000), 
            "ADAUSDC": make_synth_ohlcv(n=60, seed=3, base_price=1.5)
        }
        
        # Batch avec seuil élevé → mode séquentiel
        results = resample_batch(frames_by_symbol, "15m", batch_threshold=10, parallel=False)
        
        # Vérifications
        assert len(results) == 3
        assert all(symbol in results for symbol in frames_by_symbol.keys())
        assert all(len(df) == 4 for df in results.values())  # 60min / 15min = 4 barres
    
    def test_batch_parallel_large(self):
        """Test batch parallèle (nombreux symboles)."""
        # Génération 15 symboles
        frames_by_symbol = {}
        for i in range(15):
            symbol = f"SYM{i:02d}USDC"
            frames_by_symbol[symbol] = make_synth_ohlcv(n=240, seed=100+i, base_price=100*(i+1))
        
        # Batch avec seuil bas → mode parallèle
        results = resample_batch(frames_by_symbol, "1h", batch_threshold=5, parallel=True, max_workers=4)
        
        # Vérifications
        assert len(results) == 15
        assert all(len(df) == 4 for df in results.values())  # 240min / 60min = 4 barres
        
        # Vérification ordre préservé
        assert list(results.keys()) == list(frames_by_symbol.keys())
    
    def test_batch_empty_input(self):
        """Test batch avec input vide."""
        results = resample_batch({}, "15m")
        
        assert results == {}
    
    def test_batch_invalid_timeframe(self):
        """Test batch avec timeframe invalide."""
        frames_by_symbol = {
            "TESTUSDC": make_synth_ohlcv(n=60, seed=42)
        }
        
        with pytest.raises(TimeframeError):
            resample_batch(frames_by_symbol, "invalid_tf")
    
    def test_batch_with_errors_resilient(self):
        """Test batch résilient aux erreurs individuelles."""
        # Mix données valides/invalides
        frames_by_symbol = {
            "VALIDUSDC": make_synth_ohlcv(n=60, seed=1),
            "INVALIDUSDC": pd.DataFrame(),  # DataFrame vide → erreur
            "ANOTHERUSDC": make_synth_ohlcv(n=60, seed=2)
        }
        
        # Batch ne doit pas crash totalement
        results = resample_batch(frames_by_symbol, "15m", parallel=False)
        
        # Vérifications
        assert len(results) == 3  # Tous les symboles traités
        assert len(results["VALIDUSDC"]) > 0
        assert len(results["INVALIDUSDC"]) == 0  # DataFrame vide pour échec
        assert len(results["ANOTHERUSDC"]) > 0


class TestResamplingEdgeCases:
    """Tests cas limites resampling."""
    
    def test_resample_exact_division(self):
        """Test resampling avec division exacte."""
        # Exactement 1 heure de données 1m
        df_1m = make_synth_ohlcv(n=60, freq="1min", seed=42)
        
        df_1h = resample_from_1m(df_1m, "1h")
        
        assert len(df_1h) == 1  # Une seule barre 1h
    
    def test_resample_partial_last_bar(self):
        """Test resampling avec dernière barre partielle."""
        # 67 minutes de données → 4 barres 15m + 1 partielle (7min)
        df_1m = make_synth_ohlcv(n=67, freq="1min", seed=123)
        
        df_15m = resample_from_1m(df_1m, "15m")
        
        # Vérification dernière barre partielle incluse
        assert len(df_15m) == 5  # 4 complètes + 1 partielle
    
    def test_resample_preserves_timezone(self):
        """Test préservation timezone après resampling."""
        df_1m = make_synth_ohlcv(n=120, freq="1min", seed=456)
        
        # Vérification timezone source
        assert str(df_1m.index.tz) == "UTC"  # type: ignore
        
        df_15m = resample_from_1m(df_1m, "15m")
        
        # Timezone préservée
        assert str(df_15m.index.tz) == "UTC"  # type: ignore
    
    def test_resample_maintains_ohlc_logic(self):
        """Test validation logique OHLC après resampling."""
        df_1m = make_synth_ohlcv(n=180, freq="1min", seed=789)
        
        df_15m = resample_from_1m(df_1m, "15m")
        
        # Vérification cohérence OHLC sur toutes les barres
        for _, row in df_15m.iterrows():
            assert row["low"] <= row["open"]
            assert row["low"] <= row["close"]
            assert row["high"] >= row["open"]
            assert row["high"] >= row["close"]
            assert row["low"] <= row["high"]
            assert row["volume"] >= 0