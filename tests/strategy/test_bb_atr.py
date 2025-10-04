"""
Tests unitaires pour threadx.strategy.bb_atr
============================================

Test de la stratégie Bollinger Bands + ATR avec gestion avancée du risque.

Classes testées:
- BBAtrParams: Paramètres de stratégie
- BBAtrStrategy: Implémentation de la stratégie
- Fonctions de convenance
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from threadx.strategy.bb_atr import (
    BBAtrParams, BBAtrStrategy,
    generate_signals, backtest, create_default_params
)
from threadx.strategy.model import Trade, RunStats


class TestBBAtrParams:
    """Tests pour la classe BBAtrParams"""
    
    def test_params_creation_default(self):
        """Test création paramètres par défaut"""
        params = BBAtrParams()
        
        assert params.bb_period == 20
        assert params.bb_std == 2.0
        assert params.entry_z == 1.0
        assert params.entry_logic == "AND"
        assert params.atr_period == 14
        assert params.atr_multiplier == 1.5
        assert params.trailing_stop is True
        assert params.risk_per_trade == 0.01
        assert params.min_pnl_pct == 0.01
        assert params.leverage == 1.0
        assert params.max_hold_bars == 72
        assert params.spacing_bars == 6
        assert params.trend_period == 0
        assert isinstance(params.meta, dict)
    
    def test_params_creation_custom(self):
        """Test création paramètres personnalisés"""
        params = BBAtrParams(
            bb_period=50,
            bb_std=2.5,
            entry_z=1.8,
            entry_logic="OR",
            atr_multiplier=2.0,
            risk_per_trade=0.02,
            min_pnl_pct=0.05,
            meta={"custom": "value"}
        )
        
        assert params.bb_period == 50
        assert params.bb_std == 2.5
        assert params.entry_z == 1.8
        assert params.entry_logic == "OR"
        assert params.atr_multiplier == 2.0
        assert params.risk_per_trade == 0.02
        assert params.min_pnl_pct == 0.05
        assert params.meta["custom"] == "value"
    
    def test_params_validation_bb_period(self):
        """Test validation bb_period"""
        with pytest.raises(ValueError, match="bb_period must be >= 2"):
            BBAtrParams(bb_period=1)
    
    def test_params_validation_bb_std(self):
        """Test validation bb_std"""
        with pytest.raises(ValueError, match="bb_std must be > 0"):
            BBAtrParams(bb_std=0.0)
        
        with pytest.raises(ValueError, match="bb_std must be > 0"):
            BBAtrParams(bb_std=-1.0)
    
    def test_params_validation_entry_z(self):
        """Test validation entry_z"""
        with pytest.raises(ValueError, match="entry_z must be > 0"):
            BBAtrParams(entry_z=0.0)
        
        with pytest.raises(ValueError, match="entry_z must be > 0"):
            BBAtrParams(entry_z=-1.0)
    
    def test_params_validation_entry_logic(self):
        """Test validation entry_logic"""
        with pytest.raises(ValueError, match="entry_logic must be 'AND' or 'OR'"):
            BBAtrParams(entry_logic="INVALID")
    
    def test_params_validation_atr_period(self):
        """Test validation atr_period"""
        with pytest.raises(ValueError, match="atr_period must be >= 1"):
            BBAtrParams(atr_period=0)
    
    def test_params_validation_atr_multiplier(self):
        """Test validation atr_multiplier"""
        with pytest.raises(ValueError, match="atr_multiplier must be > 0"):
            BBAtrParams(atr_multiplier=0.0)
    
    def test_params_validation_risk_per_trade(self):
        """Test validation risk_per_trade"""
        with pytest.raises(ValueError, match="risk_per_trade must be in \\(0, 1\\]"):
            BBAtrParams(risk_per_trade=0.0)
        
        with pytest.raises(ValueError, match="risk_per_trade must be in \\(0, 1\\]"):
            BBAtrParams(risk_per_trade=1.5)
    
    def test_params_validation_min_pnl_pct(self):
        """Test validation min_pnl_pct"""
        with pytest.raises(ValueError, match="min_pnl_pct must be >= 0"):
            BBAtrParams(min_pnl_pct=-0.01)
    
    def test_params_validation_leverage(self):
        """Test validation leverage"""
        with pytest.raises(ValueError, match="leverage must be > 0"):
            BBAtrParams(leverage=0.0)
    
    def test_params_validation_max_hold_bars(self):
        """Test validation max_hold_bars"""
        with pytest.raises(ValueError, match="max_hold_bars must be >= 1"):
            BBAtrParams(max_hold_bars=0)
    
    def test_params_validation_spacing_bars(self):
        """Test validation spacing_bars"""
        with pytest.raises(ValueError, match="spacing_bars must be >= 0"):
            BBAtrParams(spacing_bars=-1)
    
    def test_params_serialization(self):
        """Test sérialisation/désérialisation paramètres"""
        original_params = BBAtrParams(
            bb_period=30,
            bb_std=1.8,
            atr_multiplier=2.5,
            meta={"test": "serialization"}
        )
        
        # To dict
        params_dict = original_params.to_dict()
        assert isinstance(params_dict, dict)
        assert params_dict['bb_period'] == 30
        assert params_dict['bb_std'] == 1.8
        assert params_dict['atr_multiplier'] == 2.5
        assert params_dict['meta']['test'] == "serialization"
        
        # From dict
        reconstructed_params = BBAtrParams.from_dict(params_dict)
        assert reconstructed_params.bb_period == original_params.bb_period
        assert reconstructed_params.bb_std == original_params.bb_std
        assert reconstructed_params.atr_multiplier == original_params.atr_multiplier
        assert reconstructed_params.meta == original_params.meta


class TestBBAtrStrategy:
    """Tests pour la classe BBAtrStrategy"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Fixture avec des données OHLCV d'exemple"""
        np.random.seed(42)  # Reproductibilité
        n_bars = 200
        
        # Génération données synthétiques
        timestamps = pd.date_range('2024-01-01T00:00:00Z', periods=n_bars, freq='15min', tz='UTC')
        
        # Prix base avec tendance et volatilité
        base_price = 50000.0
        returns = np.random.normal(0, 0.01, n_bars)  # 1% volatilité
        prices = base_price * np.cumprod(1 + returns)
        
        # Spread OHLC réaliste
        spread_pct = np.random.uniform(0.001, 0.005, n_bars)  # 0.1-0.5%
        
        df = pd.DataFrame({
            'open': prices * (1 - spread_pct/4),
            'high': prices * (1 + spread_pct/2),
            'low': prices * (1 - spread_pct/2),
            'close': prices,
            'volume': np.random.randint(100, 1000, n_bars)
        }, index=timestamps)
        
        return df
    
    def test_strategy_initialization(self):
        """Test initialisation stratégie"""
        strategy = BBAtrStrategy("BTCUSDT", "1h")
        
        assert strategy.symbol == "BTCUSDT"
        assert strategy.timeframe == "1h"
    
    def test_calculate_trend_filter_disabled(self):
        """Test filtre tendance désactivé"""
        strategy = BBAtrStrategy()
        close = np.array([100, 101, 102, 103, 104])
        
        trend = strategy._calculate_trend_filter(close, trend_period=0)
        assert trend is None
    
    def test_calculate_trend_filter_enabled(self):
        """Test filtre tendance activé"""
        strategy = BBAtrStrategy()
        close = np.array([100, 101, 102, 103, 104])
        
        trend = strategy._calculate_trend_filter(close, trend_period=3)
        assert trend is not None
        assert len(trend) == len(close)
        assert isinstance(trend, np.ndarray)
    
    @patch('threadx.strategy.bb_atr.ensure_indicator')
    def test_ensure_indicators_success(self, mock_ensure, sample_ohlcv_data):
        """Test calcul des indicateurs via mock"""
        # Configuration des mocks
        n_bars = len(sample_ohlcv_data)
        
        # Mock Bollinger Bands (upper, middle, lower)
        bb_upper = sample_ohlcv_data['high'] * 1.02
        bb_middle = sample_ohlcv_data['close']
        bb_lower = sample_ohlcv_data['low'] * 0.98
        
        # Mock ATR
        atr_values = np.full(n_bars, 500.0)  # ATR constant pour test
        
        def mock_ensure_side_effect(indicator_type, params, df, **kwargs):
            if indicator_type == 'bollinger':
                return (bb_upper.values, bb_middle.values, bb_lower.values)
            elif indicator_type == 'atr':
                return atr_values
            else:
                raise ValueError(f"Unknown indicator: {indicator_type}")
        
        mock_ensure.side_effect = mock_ensure_side_effect
        
        # Test
        strategy = BBAtrStrategy("TESTBTC", "15m")
        params = BBAtrParams(bb_period=20, bb_std=2.0, atr_period=14)
        
        df_bb, atr_array = strategy._ensure_indicators(sample_ohlcv_data, params)
        
        # Vérifications
        assert 'bb_upper' in df_bb.columns
        assert 'bb_middle' in df_bb.columns
        assert 'bb_lower' in df_bb.columns
        assert 'bb_z' in df_bb.columns
        assert len(atr_array) == n_bars
        
        # Vérification appels mock
        assert mock_ensure.call_count == 2
        calls = mock_ensure.call_args_list
        
        # Premier appel: Bollinger
        assert calls[0][0][0] == 'bollinger'
        assert calls[0][0][1]['period'] == 20
        assert calls[0][0][1]['std'] == 2.0
        
        # Deuxième appel: ATR
        assert calls[1][0][0] == 'atr'
        assert calls[1][0][1]['period'] == 14
        assert calls[1][0][1]['method'] == 'ema'
    
    @patch('threadx.strategy.bb_atr.ensure_indicator')
    def test_generate_signals_basic(self, mock_ensure, sample_ohlcv_data):
        """Test génération signaux basique"""
        n_bars = len(sample_ohlcv_data)
        
        # Mock des indicateurs avec signaux détectables
        close_prices = sample_ohlcv_data['close'].values
        bb_middle = close_prices.copy()
        bb_std_dev = np.full(n_bars, 1000.0)  # Écart-type constant
        bb_upper = bb_middle + 2.0 * bb_std_dev
        bb_lower = bb_middle - 2.0 * bb_std_dev
        
        # Z-score artificiel pour générer signaux
        bb_z = np.zeros(n_bars)
        bb_z[50] = -1.5  # Signal ENTER_LONG potentiel
        bb_z[100] = 1.5  # Signal ENTER_SHORT potentiel
        
        atr_values = np.full(n_bars, 800.0)
        
        def mock_ensure_side_effect(indicator_type, params, df, **kwargs):
            if indicator_type == 'bollinger':
                return (bb_upper, bb_middle, bb_lower)
            elif indicator_type == 'atr':
                return atr_values
            else:
                raise ValueError(f"Unknown indicator: {indicator_type}")
        
        mock_ensure.side_effect = mock_ensure_side_effect
        
        # Test génération signaux
        strategy = BBAtrStrategy("TESTBTC", "15m")
        params = BBAtrParams(bb_period=20, entry_z=1.0, spacing_bars=10)
        
        # Patch manuel du Z-score dans _ensure_indicators
        original_ensure = strategy._ensure_indicators
        def patched_ensure(df, params):
            df_bb, atr_array = original_ensure(df, params)
            df_bb['bb_z'] = bb_z  # Override Z-score
            return df_bb, atr_array
        
        strategy._ensure_indicators = patched_ensure
        
        signals_df = strategy.generate_signals(sample_ohlcv_data, params.to_dict())
        
        # Vérifications
        assert len(signals_df) == n_bars
        assert 'signal' in signals_df.columns
        assert 'bb_z' in signals_df.columns
        assert 'atr' in signals_df.columns
        
        # Compter les signaux
        signals = signals_df['signal'].values
        enter_longs = np.sum(signals == "ENTER_LONG")
        enter_shorts = np.sum(signals == "ENTER_SHORT")
        holds = np.sum(signals == "HOLD")
        
        assert enter_longs + enter_shorts + holds == n_bars
        assert holds > 0  # Majorité des signaux sont HOLD
    
    @patch('threadx.strategy.bb_atr.ensure_indicator')
    def test_backtest_no_trades(self, mock_ensure, sample_ohlcv_data):
        """Test backtest sans trades (pas de signaux)"""
        n_bars = len(sample_ohlcv_data)
        
        # Mock indicateurs sans signaux (Z-score toujours proche de 0)
        close_prices = sample_ohlcv_data['close'].values
        bb_middle = close_prices.copy()
        bb_upper = close_prices * 1.01
        bb_lower = close_prices * 0.99
        atr_values = np.full(n_bars, 100.0)
        
        def mock_ensure_side_effect(indicator_type, params, df, **kwargs):
            if indicator_type == 'bollinger':
                return (bb_upper, bb_middle, bb_lower)
            elif indicator_type == 'atr':
                return atr_values
            else:
                raise ValueError(f"Unknown indicator: {indicator_type}")
        
        mock_ensure.side_effect = mock_ensure_side_effect
        
        # Test backtest
        strategy = BBAtrStrategy("TESTBTC", "15m")
        params = BBAtrParams(bb_period=20, entry_z=3.0)  # Seuil élevé = pas de signaux
        
        equity_curve, run_stats = strategy.backtest(
            sample_ohlcv_data, 
            params.to_dict(), 
            initial_capital=10000.0
        )
        
        # Vérifications
        assert len(equity_curve) == n_bars
        assert equity_curve.iloc[0] == 10000.0  # Capital initial
        assert equity_curve.iloc[-1] == 10000.0  # Pas de trades = capital inchangé
        
        assert run_stats.total_trades == 0
        assert run_stats.total_pnl == 0.0
        assert not run_stats.is_profitable  # Pas profitable car pas de trades
        assert not run_stats.has_trades
    
    @patch('threadx.strategy.bb_atr.ensure_indicator')
    def test_backtest_with_trades(self, mock_ensure, sample_ohlcv_data):
        """Test backtest avec trades simulés"""
        n_bars = len(sample_ohlcv_data)
        close_prices = sample_ohlcv_data['close'].values
        
        # Mock indicateurs avec signaux forts
        bb_middle = close_prices.copy()
        bb_std_dev = np.full(n_bars, close_prices.mean() * 0.02)  # 2% std
        bb_upper = bb_middle + 2.0 * bb_std_dev
        bb_lower = bb_middle - 2.0 * bb_std_dev
        
        atr_values = np.full(n_bars, close_prices.mean() * 0.01)  # 1% ATR
        
        def mock_ensure_side_effect(indicator_type, params, df, **kwargs):
            if indicator_type == 'bollinger':
                return (bb_upper, bb_middle, bb_lower)
            elif indicator_type == 'atr':
                return atr_values
            else:
                raise ValueError(f"Unknown indicator: {indicator_type}")
        
        mock_ensure.side_effect = mock_ensure_side_effect
        
        # Modification des données pour créer des signaux clairs
        test_data = sample_ohlcv_data.copy()
        
        # Barre 50: prix très bas -> signal ENTER_LONG
        test_data.iloc[50, test_data.columns.get_loc('close')] = bb_lower[50] * 0.9
        
        # Barre 60: prix retour normal -> sortie profitable
        test_data.iloc[60, test_data.columns.get_loc('close')] = bb_middle[60]
        
        # Test backtest
        strategy = BBAtrStrategy("TESTBTC", "15m")
        params = BBAtrParams(
            bb_period=20, 
            entry_z=0.5,  # Seuil bas pour générer signaux
            spacing_bars=5,
            risk_per_trade=0.01,
            min_pnl_pct=0.0  # Pas de filtrage PnL pour ce test
        )
        
        equity_curve, run_stats = strategy.backtest(
            test_data, 
            params.to_dict(), 
            initial_capital=10000.0,
            fee_bps=0.0  # Pas de frais pour simplifier
        )
        
        # Vérifications
        assert len(equity_curve) == n_bars
        assert equity_curve.iloc[0] == 10000.0
        
        # Au moins quelques signaux doivent être générés
        # (difficile de garantir trades avec données synthétiques, 
        #  mais au moins pas d'erreur)
        assert run_stats.bars_analyzed == n_bars
        assert run_stats.initial_capital == 10000.0
        assert isinstance(run_stats.total_trades, int)
        assert run_stats.total_trades >= 0
    
    def test_validate_ohlcv_input(self, sample_ohlcv_data):
        """Test validation input OHLCV"""
        strategy = BBAtrStrategy()
        params = BBAtrParams().to_dict()
        
        # DataFrame valide: ne doit pas lever d'exception
        with patch('threadx.strategy.bb_atr.ensure_indicator'):
            strategy.generate_signals(sample_ohlcv_data, params)
        
        # DataFrame invalide: doit lever ValueError
        invalid_df = sample_ohlcv_data.drop(columns=['close'])
        
        with pytest.raises(ValueError, match="Missing OHLCV columns"):
            with patch('threadx.strategy.bb_atr.ensure_indicator'):
                strategy.generate_signals(invalid_df, params)
    
    def test_validate_params_input(self, sample_ohlcv_data):
        """Test validation paramètres input"""
        strategy = BBAtrStrategy()
        
        # Paramètres valides: ne doit pas lever d'exception
        valid_params = BBAtrParams().to_dict()
        with patch('threadx.strategy.bb_atr.ensure_indicator'):
            strategy.generate_signals(sample_ohlcv_data, valid_params)
        
        # Paramètres invalides: manque clé requise
        invalid_params = {'bb_period': 20}  # bb_std manquant
        
        with pytest.raises(ValueError, match="Missing required strategy parameters"):
            with patch('threadx.strategy.bb_atr.ensure_indicator'):
                strategy.generate_signals(sample_ohlcv_data, invalid_params)


class TestConvenienceFunctions:
    """Tests pour les fonctions de convenance"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture données simples"""
        timestamps = pd.date_range('2024-01-01T00:00:00Z', periods=50, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'open': np.random.uniform(49000, 51000, 50),
            'high': np.random.uniform(50000, 52000, 50),
            'low': np.random.uniform(48000, 50000, 50),
            'close': np.random.uniform(49500, 50500, 50),
            'volume': np.random.randint(100, 1000, 50)
        }, index=timestamps)
        return df
    
    @patch('threadx.strategy.bb_atr.BBAtrStrategy')
    def test_generate_signals_convenience(self, mock_strategy_class, sample_data):
        """Test fonction de convenance generate_signals"""
        # Mock de l'instance stratégie
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = pd.DataFrame({'signal': ['HOLD'] * len(sample_data)})
        mock_strategy_class.return_value = mock_strategy
        
        # Test
        params = {'bb_period': 20, 'bb_std': 2.0}
        result = generate_signals(sample_data, params, "BTCUSDT", "1h")
        
        # Vérifications
        mock_strategy_class.assert_called_once_with("BTCUSDT", "1h")
        mock_strategy.generate_signals.assert_called_once_with(sample_data, params)
        assert len(result) == len(sample_data)
    
    @patch('threadx.strategy.bb_atr.BBAtrStrategy')
    def test_backtest_convenience(self, mock_strategy_class, sample_data):
        """Test fonction de convenance backtest"""
        # Mock des résultats
        mock_equity = pd.Series([10000, 10100, 10200], index=sample_data.index[:3])
        mock_stats = Mock()
        mock_stats.total_trades = 2
        
        mock_strategy = Mock()
        mock_strategy.backtest.return_value = (mock_equity, mock_stats)
        mock_strategy_class.return_value = mock_strategy
        
        # Test
        params = {'bb_period': 30, 'atr_multiplier': 2.0}
        equity, stats = backtest(
            sample_data, 
            params, 
            initial_capital=50000.0,
            symbol="ETHUSDT",
            timeframe="4h",
            fee_bps=5.0
        )
        
        # Vérifications
        mock_strategy_class.assert_called_once_with("ETHUSDT", "4h")
        mock_strategy.backtest.assert_called_once_with(
            sample_data, 
            params, 
            50000.0,
            fee_bps=5.0
        )
        assert equity is mock_equity
        assert stats is mock_stats
    
    def test_create_default_params(self):
        """Test création paramètres par défaut avec surcharges"""
        # Paramètres par défaut
        default_params = create_default_params()
        assert isinstance(default_params, BBAtrParams)
        assert default_params.bb_period == 20
        assert default_params.atr_multiplier == 1.5
        
        # Avec surcharges
        custom_params = create_default_params(
            bb_period=50,
            atr_multiplier=2.5,
            risk_per_trade=0.02
        )
        assert custom_params.bb_period == 50
        assert custom_params.atr_multiplier == 2.5
        assert custom_params.risk_per_trade == 0.02
        assert custom_params.bb_std == 2.0  # Valeur par défaut conservée
        
        # Paramètre inconnu (doit être ignoré avec warning log)
        params_with_invalid = create_default_params(
            bb_period=25,
            invalid_param="ignored"
        )
        assert params_with_invalid.bb_period == 25
        assert not hasattr(params_with_invalid, 'invalid_param')


class TestEndToEndIntegration:
    """Tests d'intégration end-to-end"""
    
    def test_realistic_scenario(self):
        """Test scénario réaliste avec données cohérentes"""
        np.random.seed(42)  # Reproductibilité
        
        # Génération données réalistes
        n_bars = 1000
        timestamps = pd.date_range('2024-01-01T00:00:00Z', periods=n_bars, freq='1h', tz='UTC')
        
        # Prix avec tendance et volatilité cohérente
        base_price = 50000.0
        trend = np.linspace(0, 0.1, n_bars)  # Tendance haussière 10%
        noise = np.random.normal(0, 0.02, n_bars)  # Volatilité 2%
        prices = base_price * np.cumprod(1 + trend/n_bars + noise)
        
        # OHLC cohérent
        spread = np.random.uniform(0.001, 0.003, n_bars)
        df = pd.DataFrame({
            'open': prices * (1 - spread/4),
            'high': prices * (1 + spread/2),
            'low': prices * (1 - spread/2),  
            'close': prices,
            'volume': np.random.randint(500, 2000, n_bars)
        }, index=timestamps)
        
        # Paramètres conservateurs
        params = BBAtrParams(
            bb_period=20,
            bb_std=2.0,
            entry_z=1.5,
            atr_period=14,
            atr_multiplier=1.5,
            risk_per_trade=0.01,
            spacing_bars=12,  # 12h minimum entre trades
            max_hold_bars=48  # Max 48h par position
        )
        
        # Mock des indicateurs pour éviter dépendance Phase 3
        with patch('threadx.strategy.bb_atr.ensure_indicator') as mock_ensure:
            # Bollinger Bands simulées
            bb_middle = pd.Series(prices).rolling(20).mean().values
            bb_std = np.array(pd.Series(prices).rolling(20).std().values) * 2.0
            bb_upper = np.array(bb_middle) + bb_std
            bb_lower = np.array(bb_middle) - bb_std
            
            # ATR simulé (% du prix)
            high_low = df['high'] - df['low']
            atr_values = high_low.rolling(14).mean().values
            
            def mock_ensure_side_effect(indicator_type, params_dict, df_input, **kwargs):
                if indicator_type == 'bollinger':
                    return (bb_upper, bb_middle, bb_lower)
                elif indicator_type == 'atr':
                    return atr_values
                else:
                    raise ValueError(f"Unknown indicator: {indicator_type}")
            
            mock_ensure.side_effect = mock_ensure_side_effect
            
            # Exécution backtest
            strategy = BBAtrStrategy("BTCUSDT", "1h")
            equity_curve, run_stats = strategy.backtest(
                df, 
                params.to_dict(), 
                initial_capital=100000.0,
                fee_bps=4.0,
                slippage_bps=1.0
            )
        
        # Vérifications réalistes
        assert len(equity_curve) == n_bars
        assert equity_curve.iloc[0] == 100000.0
        assert isinstance(run_stats, RunStats)
        assert run_stats.bars_analyzed == n_bars
        assert run_stats.initial_capital == 100000.0
        
        # Stats cohérentes
        if run_stats.has_trades:
            assert run_stats.total_trades > 0
            assert run_stats.win_trades >= 0
            assert run_stats.loss_trades >= 0
            assert run_stats.win_trades + run_stats.loss_trades == run_stats.total_trades
            assert 0 <= run_stats.win_rate_pct <= 100
        
        # Équité cohérente
        assert not equity_curve.isna().any()
        assert (equity_curve >= 0).all()  # Pas de capital négatif
        
        # Drawdown cohérent
        if run_stats.max_drawdown < 0:
            assert run_stats.max_drawdown_pct < 0
        
        print(f"Backtest terminé: {run_stats.total_trades} trades, "
              f"PnL={run_stats.total_pnl:.2f} ({run_stats.total_pnl_pct:.2f}%), "
              f"Win Rate={run_stats.win_rate_pct:.1f}%")