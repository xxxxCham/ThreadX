"""
Tests ThreadX Synthetic Data Generator - Phase 2
Tests complets pour génération données OHLCV synthétiques.
"""

import pytest
import pandas as pd
import numpy as np

# Import du module à tester
try:
    from threadx.data.synth import (
        make_synth_ohlcv, make_trending_ohlcv, make_volatile_ohlcv,
        validate_synth_determinism, SynthDataError
    )
    from threadx.data.io import OHLCV_SCHEMA, PANDERA_AVAILABLE
except ImportError as e:
    pytest.skip(f"Modules ThreadX non disponibles: {e}", allow_module_level=True)


class TestMakeSynthOHLCV:
    """Tests générateur OHLCV de base."""
    
    def test_basic_generation(self):
        """Test génération de base avec paramètres par défaut."""
        df = make_synth_ohlcv(n=1000, seed=42)
        
        # Vérifications structure
        assert len(df) == 1000
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        
        # Vérifications types
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == "float64"
        
        # Vérifications valeurs positives
        assert (df[["open", "high", "low", "close", "volume"]] > 0).all().all()
    
    def test_custom_parameters(self):
        """Test génération avec paramètres personnalisés."""
        df = make_synth_ohlcv(
            n=500,
            start="2024-06-15",
            freq="5min", 
            seed=123,
            base_price=25000.0,
            volatility=0.05,
            volume_base=500000.0
        )
        
        # Vérifications paramètres appliqués
        assert len(df) == 500
        assert df.index[0].date().isoformat() == "2024-06-15"
        
        # Prix proche de base_price
        price_range = df[["open", "high", "low", "close"]].values.flatten()
        assert np.median(price_range) > 20000  # Proche de 25000
        assert np.median(price_range) < 30000
        
        # Volume proche de volume_base
        assert df["volume"].median() > 100000
        assert df["volume"].median() < 2000000
    
    def test_ohlc_logic_consistency(self):
        """Test cohérence logique OHLC."""
        df = make_synth_ohlcv(n=200, seed=456)
        
        # Vérifications OHLC sur toutes les barres
        for _, row in df.iterrows():
            assert row["low"] <= row["open"], "Low doit être <= Open"
            assert row["low"] <= row["close"], "Low doit être <= Close"
            assert row["high"] >= row["open"], "High doit être >= Open"
            assert row["high"] >= row["close"], "High doit être >= Close"
            assert row["low"] <= row["high"], "Low doit être <= High"
    
    def test_deterministic_generation(self):
        """Test déterminisme avec seed fixe."""
        seed = 789
        n = 100
        
        df1 = make_synth_ohlcv(n=n, seed=seed)
        df2 = make_synth_ohlcv(n=n, seed=seed)
        
        # Doivent être identiques
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_different_seeds_different_results(self):
        """Test seeds différents → résultats différents."""
        n = 50
        
        df1 = make_synth_ohlcv(n=n, seed=111)
        df2 = make_synth_ohlcv(n=n, seed=222)
        
        # Ne doivent pas être identiques
        assert not df1.equals(df2)
        
        # Mais structure identique
        assert df1.shape == df2.shape
        assert df1.columns.equals(df2.columns)
    
    def test_continuous_timestamps(self):
        """Test continuité timestamps."""
        df = make_synth_ohlcv(n=60, freq="1min", seed=42)
        
        # Vérification espacement régulier
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta("1min")
        
        assert all(diff == expected_diff for diff in time_diffs)
    
    def test_volume_variability(self):
        """Test variabilité réaliste du volume."""
        df = make_synth_ohlcv(n=500, seed=42, volume_noise=0.8)
        
        volumes = df["volume"]
        
        # Variabilité significative (pas toutes les valeurs identiques)
        assert volumes.std() > 0
        assert volumes.min() != volumes.max()
        
        # Distribution raisonnable
        cv = volumes.std() / volumes.mean()  # Coefficient de variation
        assert 0.1 < cv < 2.0  # Variabilité entre 10% et 200%


class TestSynthParameterValidation:
    """Tests validation paramètres générateur."""
    
    def test_invalid_n_parameter(self):
        """Test erreur paramètre n invalide."""
        with pytest.raises(SynthDataError, match="n doit être > 0"):
            make_synth_ohlcv(n=0)
        
        with pytest.raises(SynthDataError, match="n doit être > 0"):
            make_synth_ohlcv(n=-10)
    
    def test_invalid_base_price(self):
        """Test erreur base_price invalide."""
        with pytest.raises(SynthDataError, match="base_price doit être > 0"):
            make_synth_ohlcv(n=10, base_price=0)
        
        with pytest.raises(SynthDataError, match="base_price doit être > 0"):
            make_synth_ohlcv(n=10, base_price=-100)
    
    def test_invalid_volatility(self):
        """Test erreur volatility invalide."""
        with pytest.raises(SynthDataError, match="volatility doit être >= 0"):
            make_synth_ohlcv(n=10, volatility=-0.1)
    
    def test_invalid_volume_base(self):
        """Test erreur volume_base invalide."""
        with pytest.raises(SynthDataError, match="volume_base doit être > 0"):
            make_synth_ohlcv(n=10, volume_base=0)
        
        with pytest.raises(SynthDataError, match="volume_base doit être > 0"):
            make_synth_ohlcv(n=10, volume_base=-1000)
    
    def test_edge_case_minimal_volatility(self):
        """Test cas limite volatilité très faible."""
        df = make_synth_ohlcv(n=100, seed=42, volatility=0.001)  # 0.1%
        
        # Doit fonctionner avec prix très stables
        price_range = df[["open", "high", "low", "close"]].values.flatten()
        price_cv = np.std(price_range) / np.mean(price_range)
        
        assert price_cv < 0.1  # Très faible variabilité
    
    def test_edge_case_high_volatility(self):
        """Test cas limite volatilité très élevée."""
        df = make_synth_ohlcv(n=100, seed=42, volatility=0.5)  # 50%
        
        # Doit fonctionner avec prix très volatils
        price_range = df[["open", "high", "low", "close"]].values.flatten()
        price_cv = np.std(price_range) / np.mean(price_range)
        
        assert price_cv > 0.1  # Variabilité significative


class TestMakeTrendingOHLCV:
    """Tests générateur OHLCV avec tendance."""
    
    def test_bullish_trend(self):
        """Test tendance haussière."""
        df = make_trending_ohlcv(n=200, trend_strength=0.5, seed=42)  # +50%
        
        # Vérification structure
        assert len(df) == 200
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Vérification tendance haussière
        first_close = df["close"].iloc[0]
        last_close = df["close"].iloc[-1]
        
        assert last_close > first_close  # Prix final > prix initial
        
        # Vérification amplitude tendance (approximative)
        total_return = (last_close - first_close) / first_close
        assert total_return > 0.2  # Au moins +20% sur la période
    
    def test_bearish_trend(self):
        """Test tendance baissière."""
        df = make_trending_ohlcv(n=150, trend_strength=-0.3, seed=123)  # -30%
        
        # Vérification tendance baissière
        first_close = df["close"].iloc[0]
        last_close = df["close"].iloc[-1]
        
        assert last_close < first_close  # Prix final < prix initial
        
        # Vérification amplitude baisse
        total_return = (last_close - first_close) / first_close
        assert total_return < -0.1  # Au moins -10% sur la période
    
    def test_neutral_trend(self):
        """Test tendance neutre (≈0)."""
        df = make_trending_ohlcv(n=100, trend_strength=0.0, seed=456)
        
        first_close = df["close"].iloc[0]
        last_close = df["close"].iloc[-1]
        
        # Variation minimale
        total_return = abs((last_close - first_close) / first_close)
        assert total_return < 0.1  # Moins de 10% de variation totale
    
    def test_trending_maintains_ohlc_logic(self):
        """Test cohérence OHLC avec tendance."""
        df = make_trending_ohlcv(n=100, trend_strength=0.8, seed=789)
        
        # Vérifications OHLC sur toutes les barres
        for _, row in df.iterrows():
            assert row["low"] <= row["open"]
            assert row["low"] <= row["close"]
            assert row["high"] >= row["open"]
            assert row["high"] >= row["close"]
            assert row["low"] <= row["high"]


class TestMakeVolatileOHLCV:
    """Tests générateur OHLCV avec volatilité variable."""
    
    def test_volatility_spikes(self):
        """Test génération avec pics de volatilité."""
        df = make_volatile_ohlcv(
            n=500, 
            volatility_spikes=5,
            spike_intensity=4.0,
            seed=42
        )
        
        # Vérification structure
        assert len(df) == 500
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Calcul volatilité glissante (returns)
        returns = df["close"].pct_change().abs()
        
        # Doit y avoir des périodes de haute volatilité
        high_vol_periods = (returns > returns.quantile(0.95)).sum()
        assert high_vol_periods >= 5  # Au moins quelques périodes volatiles
    
    def test_spike_intensity_effect(self):
        """Test effet intensité spikes."""
        # Volatilité normale
        df_normal = make_volatile_ohlcv(
            n=200, volatility_spikes=3, spike_intensity=1.5, seed=123
        )
        
        # Volatilité extrême
        df_extreme = make_volatile_ohlcv(
            n=200, volatility_spikes=3, spike_intensity=5.0, seed=123
        )
        
        # Calcul volatilité moyenne
        vol_normal = df_normal["close"].pct_change().abs().mean()
        vol_extreme = df_extreme["close"].pct_change().abs().mean()
        
        assert vol_extreme > vol_normal  # Plus volatile avec spike_intensity élevé
    
    def test_volatile_maintains_ohlc_logic(self):
        """Test cohérence OHLC avec volatilité extrême."""
        df = make_volatile_ohlcv(
            n=100, 
            volatility_spikes=10,
            spike_intensity=6.0,
            seed=999
        )
        
        # Vérifications OHLC même avec volatilité extreme
        for _, row in df.iterrows():
            assert row["low"] <= row["open"]
            assert row["low"] <= row["close"]
            assert row["high"] >= row["open"]
            assert row["high"] >= row["close"]
            assert row["low"] <= row["high"]
            assert row["volume"] > 0
    
    def test_no_spikes_behavior(self):
        """Test comportement sans spikes."""
        df = make_volatile_ohlcv(
            n=100,
            volatility_spikes=0,  # Pas de spikes
            seed=42
        )
        
        # Doit ressembler à génération normale
        assert len(df) == 100
        returns = df["close"].pct_change().abs()
        
        # Volatilité relativement uniforme
        vol_std = returns.std()
        vol_mean = returns.mean()
        
        # Coefficient de variation pas trop élevé
        cv = vol_std / vol_mean if vol_mean > 0 else 0
        assert cv < 3.0  # Variabilité modérée


class TestSynthValidation:
    """Tests validation et conformité schéma."""
    
    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera non disponible")
    def test_synth_schema_compliance(self):
        """Test conformité schéma OHLCV_SCHEMA."""
        df = make_synth_ohlcv(n=100, seed=42)
        
        # Validation Pandera si disponible
        try:
            if OHLCV_SCHEMA is not None:
                OHLCV_SCHEMA.validate(df, lazy=False)
        except Exception as e:
            pytest.fail(f"Données synthétiques non conformes OHLCV_SCHEMA: {e}")
    
    def test_validate_determinism_function(self):
        """Test fonction validation déterminisme."""
        # Test avec seed fixe
        is_deterministic = validate_synth_determinism(seed=42, n=50)
        
        assert is_deterministic is True
    
    def test_multiple_generators_determinism(self):
        """Test déterminisme sur tous les générateurs."""
        seed = 12345
        n = 30
        
        # Test générateur de base
        df1 = make_synth_ohlcv(n=n, seed=seed)
        df2 = make_synth_ohlcv(n=n, seed=seed)
        assert df1.equals(df2)
        
        # Test générateur trending
        df3 = make_trending_ohlcv(n=n, trend_strength=0.2, seed=seed)
        df4 = make_trending_ohlcv(n=n, trend_strength=0.2, seed=seed)
        assert df3.equals(df4)
        
        # Test générateur volatile
        df5 = make_volatile_ohlcv(n=n, volatility_spikes=3, seed=seed)
        df6 = make_volatile_ohlcv(n=n, volatility_spikes=3, seed=seed)
        assert df5.equals(df6)


class TestSynthFrequencySupport:
    """Tests support différentes fréquences temporelles."""
    
    def test_minute_frequencies(self):
        """Test fréquences minutes."""
        freqs = ["1min", "5min", "15min", "30min"]
        
        for freq in freqs:
            df = make_synth_ohlcv(n=20, freq=freq, seed=42)
            
            assert len(df) == 20
            
            # Vérification espacement temporel
            if len(df) > 1:
                time_diff = df.index[1] - df.index[0]
                expected_diff = pd.Timedelta(freq)
                assert time_diff == expected_diff
    
    def test_hour_frequencies(self):
        """Test fréquences heures."""
        freqs = ["1H", "4H", "12H"]
        
        for freq in freqs:
            df = make_synth_ohlcv(n=10, freq=freq, seed=123)
            
            assert len(df) == 10
            
            # Vérification espacement
            if len(df) > 1:
                time_diff = df.index[1] - df.index[0]
                expected_diff = pd.Timedelta(freq)
                assert time_diff == expected_diff
    
    def test_daily_frequency(self):
        """Test fréquence journalière."""
        df = make_synth_ohlcv(n=30, freq="1D", seed=456)
        
        assert len(df) == 30
        
        # Vérification espacement journalier
        if len(df) > 1:
            time_diff = df.index[1] - df.index[0]
            assert time_diff == pd.Timedelta("1D")


class TestSynthEdgeCases:
    """Tests cas limites données synthétiques."""
    
    def test_minimal_dataset(self):
        """Test génération dataset minimal."""
        df = make_synth_ohlcv(n=1, seed=42)
        
        assert len(df) == 1
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Une seule ligne valide
        row = df.iloc[0]
        assert row["low"] <= row["open"]
        assert row["low"] <= row["close"]
        assert row["high"] >= row["open"]
        assert row["high"] >= row["close"]
        assert row["volume"] > 0
    
    def test_large_dataset_performance(self):
        """Test performance sur dataset volumineux."""
        import time
        
        start_time = time.perf_counter()
        
        # Dataset 100k points
        df = make_synth_ohlcv(n=100_000, seed=42)
        
        elapsed = time.perf_counter() - start_time
        
        # Vérifications
        assert len(df) == 100_000
        assert elapsed < 10.0  # Doit être rapide (<10s)
    
    def test_extreme_parameters_stability(self):
        """Test stabilité avec paramètres extrêmes."""
        # Prix très bas
        df_low = make_synth_ohlcv(n=50, base_price=0.001, seed=42)
        assert len(df_low) == 50
        assert (df_low[["open", "high", "low", "close"]] > 0).all().all()
        
        # Prix très élevé  
        df_high = make_synth_ohlcv(n=50, base_price=1_000_000, seed=42)
        assert len(df_high) == 50
        assert (df_high[["open", "high", "low", "close"]] > 0).all().all()
        
        # Volume très faible
        df_low_vol = make_synth_ohlcv(n=50, volume_base=1.0, seed=42)
        assert len(df_low_vol) == 50
        assert (df_low_vol["volume"] > 0).all()
        
        # Volume très élevé
        df_high_vol = make_synth_ohlcv(n=50, volume_base=1_000_000_000, seed=42)
        assert len(df_high_vol) == 50
        assert (df_high_vol["volume"] > 0).all()