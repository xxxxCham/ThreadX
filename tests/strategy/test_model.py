"""
Tests unitaires pour threadx.strategy.model
==========================================

Test des types de données et utilitaires du module model.

Classes testées:
- Trade: Structure de transaction
- RunStats: Statistiques de performance
- JSON serialization/désérialization
- Validation utilities
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from threadx.strategy.model import (
    Trade, TradeDict, RunStats, RunStatsDict,
    ThreadXJSONEncoder, save_run_results, load_run_results,
    validate_ohlcv_dataframe, validate_strategy_params
)


class TestTrade:
    """Tests pour la classe Trade"""
    
    def test_trade_creation_valid(self):
        """Test création trade valide"""
        trade = Trade(
            side="LONG",
            qty=1.5,
            entry_price=50000.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=48000.0,
            meta={"test": "value"}
        )
        
        assert trade.side == "LONG"
        assert trade.qty == 1.5
        assert trade.entry_price == 50000.0
        assert trade.stop == 48000.0
        assert trade.is_open()
        assert trade.is_long()
        assert not trade.is_short()
        assert trade.meta["test"] == "value"
    
    def test_trade_validation_invalid_side(self):
        """Test validation side invalide"""
        with pytest.raises(ValueError, match="Side must be 'LONG' or 'SHORT'"):
            Trade(
                side="INVALID",
                qty=1.0,
                entry_price=100.0,
                entry_time="2024-01-15T10:30:00Z",
                stop=95.0
            )
    
    def test_trade_validation_invalid_qty(self):
        """Test validation quantité invalide"""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Trade(
                side="LONG",
                qty=-1.0,
                entry_price=100.0,
                entry_time="2024-01-15T10:30:00Z",
                stop=95.0
            )
    
    def test_trade_validation_invalid_price(self):
        """Test validation prix invalide"""
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Trade(
                side="LONG",
                qty=1.0,
                entry_price=-100.0,
                entry_time="2024-01-15T10:30:00Z",
                stop=95.0
            )
    
    def test_trade_validation_invalid_timestamp(self):
        """Test validation timestamp invalide"""
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            Trade(
                side="LONG",
                qty=1.0,
                entry_price=100.0,
                entry_time="invalid_timestamp",
                stop=95.0
            )
    
    def test_calculate_unrealized_pnl_long(self):
        """Test calcul PnL non réalisé position longue"""
        trade = Trade(
            side="LONG",
            qty=2.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=95.0,
            fees_paid=5.0
        )
        
        # Prix montant: profit
        pnl = trade.calculate_unrealized_pnl(110.0)
        assert pnl == 15.0  # (110 - 100) * 2 - 5
        
        # Prix descendant: perte
        pnl = trade.calculate_unrealized_pnl(90.0)
        assert pnl == -25.0  # (90 - 100) * 2 - 5
    
    def test_calculate_unrealized_pnl_short(self):
        """Test calcul PnL non réalisé position courte"""
        trade = Trade(
            side="SHORT",
            qty=1.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=105.0,
            fees_paid=2.0
        )
        
        # Prix descendant: profit
        pnl = trade.calculate_unrealized_pnl(90.0)
        assert pnl == 8.0  # (100 - 90) * 1 - 2
        
        # Prix montant: perte
        pnl = trade.calculate_unrealized_pnl(110.0)
        assert pnl == -12.0  # (100 - 110) * 1 - 2
    
    def test_close_trade_long(self):
        """Test fermeture trade long"""
        trade = Trade(
            side="LONG",
            qty=1.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=95.0,
            fees_paid=1.0
        )
        
        trade.close_trade(
            exit_price=110.0,
            exit_time="2024-01-15T11:30:00Z",
            exit_fees=1.1
        )
        
        assert not trade.is_open()
        assert trade.exit_price == 110.0
        assert trade.pnl_realized == 7.9  # (110 - 100) * 1 - 2.1
        assert trade.pnl_unrealized is None
        assert trade.fees_paid == 2.1
    
    def test_should_stop_loss(self):
        """Test détection stop loss"""
        # Position longue
        long_trade = Trade(
            side="LONG",
            qty=1.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=95.0
        )
        
        assert not long_trade.should_stop_loss(96.0)  # Au dessus du stop
        assert long_trade.should_stop_loss(95.0)     # Exactement au stop
        assert long_trade.should_stop_loss(90.0)     # En dessous du stop
        
        # Position courte
        short_trade = Trade(
            side="SHORT",
            qty=1.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=105.0
        )
        
        assert not short_trade.should_stop_loss(104.0)  # En dessous du stop
        assert short_trade.should_stop_loss(105.0)      # Exactement au stop
        assert short_trade.should_stop_loss(110.0)      # Au dessus du stop
    
    def test_should_take_profit(self):
        """Test détection take profit"""
        # Position longue avec take profit
        long_trade = Trade(
            side="LONG",
            qty=1.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=95.0,
            take_profit=120.0
        )
        
        assert not long_trade.should_take_profit(115.0)  # En dessous du TP
        assert long_trade.should_take_profit(120.0)      # Exactement au TP
        assert long_trade.should_take_profit(125.0)      # Au dessus du TP
        
        # Position courte avec take profit
        short_trade = Trade(
            side="SHORT",
            qty=1.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=105.0,
            take_profit=80.0
        )
        
        assert not short_trade.should_take_profit(85.0)  # Au dessus du TP
        assert short_trade.should_take_profit(80.0)      # Exactement au TP
        assert short_trade.should_take_profit(75.0)      # En dessous du TP
    
    def test_trade_duration_minutes(self):
        """Test calcul durée trade"""
        trade = Trade(
            side="LONG",
            qty=1.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=95.0
        )
        
        # Trade ouvert: pas de durée
        assert trade.duration_minutes() is None
        
        # Trade fermé: calcul durée
        trade.close_trade(110.0, "2024-01-15T11:30:00Z")
        assert trade.duration_minutes() == 60.0  # 1 heure
    
    def test_roi_percent(self):
        """Test calcul ROI"""
        trade = Trade(
            side="LONG",
            qty=2.0,
            entry_price=100.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=95.0
        )
        
        # Trade ouvert: pas de ROI
        assert trade.roi_percent() is None
        
        # Trade fermé: calcul ROI
        trade.close_trade(110.0, "2024-01-15T11:30:00Z")
        pnl_val = trade.pnl_realized if trade.pnl_realized is not None else 0.0
        expected_roi = (pnl_val / (100.0 * 2.0)) * 100.0
        roi_result = trade.roi_percent()
        roi_val = roi_result if roi_result is not None else 0.0
        assert abs(roi_val - expected_roi) < 0.01
    
    def test_trade_serialization(self):
        """Test sérialisation/désérialisation Trade"""
        original_trade = Trade(
            side="LONG",
            qty=1.5,
            entry_price=50000.0,
            entry_time="2024-01-15T10:30:00Z",
            stop=48000.0,
            take_profit=55000.0,
            meta={"bb_z": -2.1, "atr": 1200.5}
        )
        
        # To dict
        trade_dict = original_trade.to_dict()
        assert isinstance(trade_dict, dict)
        assert trade_dict['side'] == "LONG"
        assert trade_dict['qty'] == 1.5
        if 'meta' in trade_dict and trade_dict['meta']:
            assert trade_dict['meta']['bb_z'] == -2.1
        
        # From dict
        reconstructed_trade = Trade.from_dict(trade_dict)
        assert reconstructed_trade.side == original_trade.side
        assert reconstructed_trade.qty == original_trade.qty
        assert reconstructed_trade.entry_price == original_trade.entry_price
        assert reconstructed_trade.meta == original_trade.meta


class TestRunStats:
    """Tests pour la classe RunStats"""
    
    def test_runstats_creation(self):
        """Test création RunStats"""
        stats = RunStats(
            final_equity=12000.0,
            initial_capital=10000.0,
            total_pnl=2000.0,
            max_drawdown=-500.0,
            max_drawdown_pct=-5.0,
            total_trades=10,
            win_trades=6,
            loss_trades=4,
            win_rate_pct=60.0,
            total_fees_paid=150.0,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-31T23:59:59Z",
            bars_analyzed=1000
        )
        
        assert stats.total_pnl_pct == 20.0  # (2000 / 10000) * 100
        assert stats.is_profitable
        assert stats.has_trades
    
    def test_runstats_properties(self):
        """Test propriétés calculées RunStats"""
        stats = RunStats(
            final_equity=8000.0,
            initial_capital=10000.0,
            total_pnl=-2000.0,
            max_drawdown=-1000.0,
            max_drawdown_pct=-10.0,
            total_trades=5,
            win_trades=2,
            loss_trades=3,
            win_rate_pct=40.0,
            total_fees_paid=100.0,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-31T23:59:59Z",
            bars_analyzed=500,
            avg_win=500.0,
            avg_loss=-400.0
        )
        
        assert stats.total_pnl_pct == -20.0
        assert not stats.is_profitable
        assert stats.has_trades
        
        # Risk reward ratio
        rr = stats.risk_reward_ratio()
        assert rr is not None
        assert abs(rr - 1.25) < 0.01  # 500 / 400
    
    def test_runstats_expectancy(self):
        """Test calcul expectancy"""
        stats = RunStats(
            final_equity=11000.0,
            initial_capital=10000.0,
            total_pnl=1000.0,
            max_drawdown=-200.0,
            max_drawdown_pct=-2.0,
            total_trades=10,
            win_trades=6,
            loss_trades=4,
            win_rate_pct=60.0,
            total_fees_paid=50.0,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-31T23:59:59Z",
            bars_analyzed=1000,
            avg_win=300.0,
            avg_loss=-100.0
        )
        
        expectancy = stats.expectancy()
        expected = (0.6 * 300.0) + (0.4 * -100.0)  # 180 - 40 = 140
        expectancy_val = expectancy if expectancy is not None else 0.0
        assert abs(expectancy_val - expected) < 0.01
    
    def test_runstats_from_trades_and_equity(self):
        """Test création RunStats depuis trades et equity"""
        # Création trades exemple
        trades = [
            Trade(
                side="LONG", qty=1.0, entry_price=100.0,
                entry_time="2024-01-01T10:00:00Z", stop=95.0,
                exit_price=110.0, exit_time="2024-01-01T11:00:00Z",
                pnl_realized=9.0, fees_paid=1.0
            ),
            Trade(
                side="SHORT", qty=1.0, entry_price=200.0,
                entry_time="2024-01-01T12:00:00Z", stop=205.0,
                exit_price=190.0, exit_time="2024-01-01T13:00:00Z",
                pnl_realized=8.0, fees_paid=2.0
            )
        ]
        
        # Courbe d'équité exemple
        timestamps = pd.date_range('2024-01-01T10:00:00Z', periods=4, freq='1h', tz='UTC')
        equity_curve = pd.Series([10000, 10009, 10017, 10017], index=timestamps)
        
        stats = RunStats.from_trades_and_equity(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=10000
        )
        
        assert stats.total_trades == 2
        assert stats.win_trades == 2
        assert stats.loss_trades == 0
        assert stats.win_rate_pct == 100.0
        assert stats.total_pnl == 17.0
        assert stats.total_fees_paid == 3.0
        assert stats.is_profitable
    
    def test_runstats_serialization(self):
        """Test sérialisation/désérialisation RunStats"""
        original_stats = RunStats(
            final_equity=15000.0,
            initial_capital=10000.0,
            total_pnl=5000.0,
            max_drawdown=-800.0,
            max_drawdown_pct=-8.0,
            total_trades=20,
            win_trades=12,
            loss_trades=8,
            win_rate_pct=60.0,
            total_fees_paid=200.0,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-03-31T23:59:59Z",
            bars_analyzed=2000,
            sharpe_ratio=1.5,
            sortino_ratio=2.1,
            meta={"strategy": "test"}
        )
        
        # To dict
        stats_dict = original_stats.to_dict()
        assert stats_dict['total_pnl_pct'] == 50.0
        if 'meta' in stats_dict and stats_dict['meta']:
            assert stats_dict['meta']['strategy'] == "test"
        
        # From dict
        reconstructed_stats = RunStats.from_dict(stats_dict)
        assert reconstructed_stats.final_equity == original_stats.final_equity
        assert reconstructed_stats.sharpe_ratio == original_stats.sharpe_ratio
        assert reconstructed_stats.meta == original_stats.meta


class TestJSONSerialization:
    """Tests pour la sérialisation JSON"""
    
    def test_threadx_json_encoder(self):
        """Test encodeur JSON ThreadX"""
        trade = Trade(
            side="LONG", qty=1.0, entry_price=100.0,
            entry_time="2024-01-01T10:00:00Z", stop=95.0
        )
        
        stats = RunStats(
            final_equity=11000.0, initial_capital=10000.0, total_pnl=1000.0,
            max_drawdown=-100.0, max_drawdown_pct=-1.0,
            total_trades=1, win_trades=1, loss_trades=0, win_rate_pct=100.0,
            total_fees_paid=0.0, start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T23:59:59Z", bars_analyzed=100
        )
        
        data = {
            'trade': trade,
            'stats': stats,
            'numpy_array': np.array([1.0, 2.0, 3.0]),
            'numpy_float': np.float64(3.14),
            'numpy_int': np.int64(42)
        }
        
        # Test sérialisation
        json_str = json.dumps(data, cls=ThreadXJSONEncoder, indent=2)
        assert isinstance(json_str, str)
        
        # Test désérialisation
        parsed = json.loads(json_str)
        assert parsed['trade']['side'] == "LONG"
        assert parsed['stats']['total_pnl'] == 1000.0
        assert parsed['numpy_array'] == [1.0, 2.0, 3.0]
        assert parsed['numpy_float'] == 3.14
        assert parsed['numpy_int'] == 42
    
    def test_save_load_run_results(self):
        """Test sauvegarde/chargement résultats run"""
        # Données exemple
        trades = [
            Trade(
                side="LONG", qty=1.0, entry_price=100.0,
                entry_time="2024-01-01T10:00:00Z", stop=95.0,
                exit_price=105.0, exit_time="2024-01-01T11:00:00Z",
                pnl_realized=4.0, fees_paid=1.0
            )
        ]
        
        stats = RunStats(
            final_equity=10004.0, initial_capital=10000.0, total_pnl=4.0,
            max_drawdown=0.0, max_drawdown_pct=0.0,
            total_trades=1, win_trades=1, loss_trades=0, win_rate_pct=100.0,
            total_fees_paid=1.0, start_time="2024-01-01T10:00:00Z",
            end_time="2024-01-01T11:00:00Z", bars_analyzed=2
        )
        
        timestamps = pd.date_range('2024-01-01T10:00:00Z', periods=2, freq='1h', tz='UTC')
        equity_curve = pd.Series([10000.0, 10004.0], index=timestamps)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_results.json"
            
            # Sauvegarde
            metadata = {"test": "value", "param": 42}
            save_run_results(trades, stats, equity_curve, file_path, metadata)
            
            assert file_path.exists()
            
            # Chargement
            loaded_trades, loaded_stats, loaded_equity = load_run_results(file_path)
            
            # Vérifications
            assert len(loaded_trades) == 1
            assert loaded_trades[0].side == "LONG"
            assert loaded_trades[0].pnl_realized == 4.0
            
            assert loaded_stats.total_pnl == 4.0
            assert loaded_stats.total_trades == 1
            
            assert len(loaded_equity) == 2
            assert loaded_equity.iloc[0] == 10000.0
            assert loaded_equity.iloc[1] == 10004.0


class TestValidation:
    """Tests pour les fonctions de validation"""
    
    def test_validate_ohlcv_dataframe_valid(self):
        """Test validation DataFrame OHLCV valide"""
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=timestamps)
        
        # Ne doit pas lever d'exception
        validate_ohlcv_dataframe(df)
    
    def test_validate_ohlcv_dataframe_empty(self):
        """Test validation DataFrame vide"""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is None or empty"):
            validate_ohlcv_dataframe(df)
    
    def test_validate_ohlcv_dataframe_missing_columns(self):
        """Test validation colonnes manquantes"""
        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            # 'low' manquante
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=timestamps)
        
        with pytest.raises(ValueError, match="Missing OHLCV columns"):
            validate_ohlcv_dataframe(df)
    
    def test_validate_ohlcv_dataframe_invalid_index(self):
        """Test validation index non-datetime"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=[0, 1, 2])  # Index entier au lieu de datetime
        
        with pytest.raises(ValueError, match="DataFrame index must be DatetimeIndex"):
            validate_ohlcv_dataframe(df)
    
    def test_validate_strategy_params_valid(self):
        """Test validation paramètres stratégie valides"""
        params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'entry_z': 1.5,
            'extra_param': 'ignored'
        }
        required_keys = ['bb_period', 'bb_std', 'entry_z']
        
        # Ne doit pas lever d'exception
        validate_strategy_params(params, required_keys)
    
    def test_validate_strategy_params_missing_keys(self):
        """Test validation paramètres manquants"""
        params = {
            'bb_period': 20,
            # 'bb_std' manquant
            'entry_z': 1.5
        }
        required_keys = ['bb_period', 'bb_std', 'entry_z']
        
        with pytest.raises(ValueError, match="Missing required strategy parameters"):
            validate_strategy_params(params, required_keys)
    
    def test_validate_strategy_params_not_dict(self):
        """Test validation paramètres non-dict"""
        from typing import cast, Any
        params = cast(Any, "not_a_dict")  # Cast pour éviter l'erreur de type
        required_keys = ['bb_period']
        
        with pytest.raises(ValueError, match="Strategy params must be a dictionary"):
            validate_strategy_params(params, required_keys)