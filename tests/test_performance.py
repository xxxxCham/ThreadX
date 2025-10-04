#!/usr/bin/env python3
"""
ThreadX Phase 6 - Performance Metrics Tests
==========================================

Comprehensive unit tests for performance metrics module.

Features:
- Deterministic testing with seed=42
- Formula validation with known expected values
- Edge case testing (NaN/inf, empty data, zero trades)
- GPU vs CPU parity testing (when CuPy available)
- Visualization testing with file I/O verification
- Robust error handling validation

Test Categories:
1. Core Metrics: equity_curve, max_drawdown, drawdown_series
2. Risk Metrics: sharpe_ratio, sortino_ratio with annualization
3. Trade Metrics: profit_factor, win_rate, expectancy
4. Integration: summarize with comprehensive aggregation
5. Visualization: plot_drawdown with file generation
6. Edge Cases: empty data, invalid inputs, extreme values
7. GPU Parity: CPU vs GPU result consistency (if available)

Coverage Target: >95% with focus on formula correctness and error paths.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any
import warnings

import numpy as np
import pandas as pd

# Add ThreadX to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ThreadX performance module
from threadx.backtest.performance import (
    equity_curve, max_drawdown, drawdown_series,
    sharpe_ratio, sortino_ratio, profit_factor,
    win_rate, expectancy, summarize, plot_drawdown,
    HAS_CUPY, xp
)

# GPU support detection
if HAS_CUPY:
    import cupy as cp


class TestPerformanceMetrics(unittest.TestCase):
    """Test suite for ThreadX Performance Metrics module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with deterministic random state."""
        # Set deterministic seeds for reproducible tests
        np.random.seed(42)
        if HAS_CUPY:
            cp.random.seed(42)
        
        # Suppress matplotlib warnings in test environment
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        print(f"ThreadX Performance Tests initialized (GPU={'available' if HAS_CUPY else 'unavailable'})")
    
    def setUp(self):
        """Set up individual test with fresh random state."""
        np.random.seed(42)  # Reset for each test
        if HAS_CUPY:
            cp.random.seed(42)
    
    def _create_synthetic_returns(self, n_periods: int = 100, 
                                 mean_return: float = 0.001,
                                 volatility: float = 0.02) -> pd.Series:
        """Create synthetic returns series for testing."""
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='D')
        returns = np.random.normal(mean_return, volatility, n_periods)
        return pd.Series(returns, index=dates, name='returns')
    
    def _create_synthetic_trades(self, n_trades: int = 50) -> pd.DataFrame:
        """Create synthetic trades DataFrame for testing."""
        dates = pd.date_range('2024-01-01', periods=n_trades, freq='D')
        
        # Generate realistic trade data
        sides = np.random.choice(['LONG', 'SHORT'], n_trades)
        entry_prices = 100 + np.random.normal(0, 10, n_trades)
        
        # Create correlated exit prices (60% win rate target)
        win_mask = np.random.random(n_trades) < 0.6
        exit_prices = entry_prices.copy()
        
        # Winning trades: 1-5% gains
        exit_prices[win_mask] *= (1 + np.random.uniform(0.01, 0.05, win_mask.sum()))
        
        # Losing trades: 0.5-3% losses
        exit_prices[~win_mask] *= (1 - np.random.uniform(0.005, 0.03, (~win_mask).sum()))
        
        quantities = np.random.uniform(10, 100, n_trades)
        pnl = (exit_prices - entry_prices) * quantities
        returns = (exit_prices / entry_prices - 1.0)
        
        return pd.DataFrame({
            'side': sides,
            'entry_time': dates,
            'exit_time': dates + pd.Timedelta(hours=1),
            'entry_price': entry_prices,
            'exit_price': exit_prices,
            'qty': quantities,
            'pnl': pnl,
            'ret': returns
        })


class TestEquityCurve(TestPerformanceMetrics):
    """Test equity curve calculation."""
    
    def test_equity_curve_basic(self):
        """Test basic equity curve calculation with known values."""
        # Simple test case with known result
        returns = pd.Series([0.1, -0.05, 0.02], 
                           index=pd.date_range('2024-01-01', periods=3))
        initial_capital = 1000.0
        
        equity = equity_curve(returns, initial_capital)
        
        # Manual calculation for verification
        expected_values = [
            1000.0 * 1.1,          # 1100.0 after 10% gain
            1100.0 * 0.95,         # 1045.0 after 5% loss
            1045.0 * 1.02          # 1065.9 after 2% gain
        ]
        
        self.assertEqual(len(equity), 3)
        np.testing.assert_array_almost_equal(equity.values, expected_values, decimal=6)
        
        # Test final value matches cumulative calculation
        expected_final = initial_capital * np.prod(1 + returns.values)
        self.assertAlmostEqual(equity.iloc[-1], expected_final, places=6)
    
    def test_equity_curve_empty_returns(self):
        """Test equity curve with empty returns series."""
        empty_returns = pd.Series([], dtype=float)
        equity = equity_curve(empty_returns, 1000.0)
        
        self.assertTrue(equity.empty)
        self.assertEqual(len(equity), 0)
    
    def test_equity_curve_invalid_capital(self):
        """Test equity curve with invalid initial capital."""
        returns = pd.Series([0.01, 0.02])
        
        with self.assertRaises(ValueError):
            equity_curve(returns, 0.0)
        
        with self.assertRaises(ValueError):
            equity_curve(returns, -1000.0)
    
    def test_equity_curve_nan_handling(self):
        """Test equity curve with NaN values in returns."""
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, -0.01])
        equity = equity_curve(returns, 1000.0)
        
        # Should have 3 valid values (dropped 2 NaN)
        self.assertEqual(len(equity), 3)
        
        # Verify calculation on cleaned data
        clean_returns = [0.01, 0.02, -0.01]
        expected_final = 1000.0 * np.prod(1 + np.array(clean_returns))
        self.assertAlmostEqual(equity.iloc[-1], expected_final, places=6)
    
    def test_equity_curve_inf_handling(self):
        """Test equity curve with infinite values (clipped to reasonable bounds)."""
        returns = pd.Series([0.01, np.inf, -np.inf, 0.02])
        equity = equity_curve(returns, 1000.0)
        
        # Should complete without error (inf values clipped)
        self.assertEqual(len(equity), 4)
        self.assertTrue(np.isfinite(equity.values).all())


class TestDrawdownMetrics(TestPerformanceMetrics):
    """Test drawdown-related calculations."""
    
    def test_drawdown_series_basic(self):
        """Test drawdown series calculation with known pattern."""
        # Equity pattern: peak at 1200, trough at 800, recovery to 1100
        equity = pd.Series([1000, 1200, 1000, 800, 900, 1100],
                          index=pd.date_range('2024-01-01', periods=6))
        
        dd = drawdown_series(equity)
        
        # Expected drawdowns from running peaks
        expected_dd = [
            0.0,      # At initial level
            0.0,      # New peak
            -1/6,     # 1000/1200 - 1 = -1/6
            -1/3,     # 800/1200 - 1 = -1/3
            -0.25,    # 900/1200 - 1 = -0.25
            -1/12     # 1100/1200 - 1 = -1/12
        ]
        
        np.testing.assert_array_almost_equal(dd.values, expected_dd, decimal=6)
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        equity = pd.Series([1000, 1200, 800, 1100])  # 33.33% drawdown from 1200 to 800
        
        max_dd = max_drawdown(equity)
        
        expected_max_dd = (800 / 1200) - 1.0  # -1/3 = -0.3333...
        self.assertAlmostEqual(max_dd, expected_max_dd, places=6)
        
        # Verify consistency with drawdown_series
        dd_series = drawdown_series(equity)
        self.assertAlmostEqual(max_dd, dd_series.min(), places=6)
    
    def test_drawdown_empty_series(self):
        """Test drawdown with empty equity series."""
        empty_equity = pd.Series([], dtype=float)
        
        dd_series = drawdown_series(empty_equity)
        max_dd = max_drawdown(empty_equity)
        
        self.assertTrue(dd_series.empty)
        self.assertEqual(max_dd, 0.0)
    
    def test_drawdown_always_increasing(self):
        """Test drawdown with always-increasing equity (no drawdown)."""
        equity = pd.Series([1000, 1100, 1200, 1300])
        
        dd_series = drawdown_series(equity)
        max_dd = max_drawdown(equity)
        
        # All drawdown values should be 0 (always at peak)
        np.testing.assert_array_almost_equal(dd_series.values, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(max_dd, 0.0)
    
    def test_drawdown_plateau(self):
        """Test drawdown with long plateau periods."""
        equity = pd.Series([1000, 1000, 1000, 1200, 1200, 1200, 800, 800])
        
        dd_series = drawdown_series(equity)
        max_dd = max_drawdown(equity)
        
        # Maximum drawdown should be from 1200 to 800 = -1/3
        expected_max_dd = (800 / 1200) - 1.0
        self.assertAlmostEqual(max_dd, expected_max_dd, places=6)
    
    def test_drawdown_negative_equity_handling(self):
        """Test drawdown handling with negative equity values."""
        equity_with_negatives = pd.Series([1000, 500, -100, 200])
        
        # Should handle gracefully with warnings
        max_dd = max_drawdown(equity_with_negatives)
        
        # Should filter out negative values and calculate on remaining
        self.assertTrue(np.isfinite(max_dd))


class TestRiskMetrics(TestPerformanceMetrics):
    """Test risk-adjusted return metrics."""
    
    def test_sharpe_ratio_basic(self):
        """Test Sharpe ratio with known values."""
        # Create returns with known statistics
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 10)  # 50 periods
        
        # Manual calculation for verification
        risk_free_annual = 0.02
        periods_per_year = 252  # Trading days
        risk_free_period = risk_free_annual / periods_per_year
        
        excess_returns = returns - risk_free_period
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std(ddof=1)
        expected_sharpe = (mean_excess * np.sqrt(periods_per_year)) / std_excess
        
        calculated_sharpe = sharpe_ratio(returns, risk_free=risk_free_annual, 
                                       periods_per_year=periods_per_year)
        
        self.assertAlmostEqual(calculated_sharpe, expected_sharpe, places=6)
    
    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility (constant returns)."""
        constant_returns = pd.Series([0.01] * 100)  # Same return every period
        
        sharpe = sharpe_ratio(constant_returns, risk_free=0.0)
        
        # Should return 0.0 for zero volatility case
        self.assertEqual(sharpe, 0.0)
    
    def test_sortino_ratio_basic(self):
        """Test Sortino ratio calculation."""
        # Create returns with mix of positive and negative
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01, -0.02, 0.025])
        
        risk_free_annual = 0.01
        periods_per_year = 365
        
        sortino = sortino_ratio(returns, risk_free=risk_free_annual, 
                               periods_per_year=periods_per_year)
        
        # Manual verification
        risk_free_period = risk_free_annual / periods_per_year
        excess_returns = returns - risk_free_period
        mean_excess = excess_returns.mean()
        
        # Downside returns only
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std(ddof=1)
        expected_sortino = (mean_excess * np.sqrt(periods_per_year)) / downside_std
        
        self.assertAlmostEqual(sortino, expected_sortino, places=6)
    
    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio with no negative returns."""
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.025])
        
        sortino = sortino_ratio(positive_returns, risk_free=0.0)
        
        # Should return infinity for positive mean with no downside
        self.assertTrue(np.isinf(sortino) and sortino > 0)
    
    def test_risk_metrics_empty_returns(self):
        """Test risk metrics with empty returns."""
        empty_returns = pd.Series([], dtype=float)
        
        sharpe = sharpe_ratio(empty_returns)
        sortino = sortino_ratio(empty_returns)
        
        self.assertEqual(sharpe, 0.0)
        self.assertEqual(sortino, 0.0)
    
    def test_risk_metrics_nan_handling(self):
        """Test risk metrics with NaN values."""
        returns_with_nan = pd.Series([0.01, np.nan, 0.02, np.nan, -0.01])
        
        sharpe = sharpe_ratio(returns_with_nan)
        sortino = sortino_ratio(returns_with_nan)
        
        # Should handle NaN by dropping them
        self.assertTrue(np.isfinite(sharpe))
        self.assertTrue(np.isfinite(sortino))


class TestTradeMetrics(TestPerformanceMetrics):
    """Test trade-based performance metrics."""
    
    def test_profit_factor_basic(self):
        """Test profit factor with known values."""
        # Create trades with known profit factor
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -80, 150, -30]  # Gross profit: 450, Gross loss: 160
        })
        
        pf = profit_factor(trades)
        expected_pf = 450 / 160  # 2.8125
        
        self.assertAlmostEqual(pf, expected_pf, places=6)
    
    def test_profit_factor_no_losses(self):
        """Test profit factor with only winning trades."""
        winning_trades = pd.DataFrame({'pnl': [100, 200, 150]})
        
        pf = profit_factor(winning_trades)
        
        # Should return infinity (no losses)
        self.assertTrue(np.isinf(pf) and pf > 0)
    
    def test_profit_factor_no_wins(self):
        """Test profit factor with only losing trades."""
        losing_trades = pd.DataFrame({'pnl': [-100, -50, -200]})
        
        pf = profit_factor(losing_trades)
        
        # Should return 0.0 (no profits)
        self.assertEqual(pf, 0.0)
    
    def test_win_rate_basic(self):
        """Test win rate calculation."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 75, -20, 150]  # 4 wins out of 7 trades
        })
        
        wr = win_rate(trades)
        expected_wr = 4 / 7
        
        self.assertAlmostEqual(wr, expected_wr, places=6)
    
    def test_win_rate_with_ret_column(self):
        """Test win rate using 'ret' column when 'pnl' unavailable."""
        trades = pd.DataFrame({
            'ret': [0.05, -0.02, 0.08, -0.01, 0.03]  # 3 wins out of 5
        })
        
        wr = win_rate(trades)
        expected_wr = 3 / 5
        
        self.assertAlmostEqual(wr, expected_wr, places=6)
    
    def test_expectancy_basic(self):
        """Test expectancy calculation with known values."""
        # Trades: 3 wins (avg 100), 2 losses (avg -40)
        trades = pd.DataFrame({
            'pnl': [80, -30, 120, -50, 100]
        })
        
        expectancy_val = expectancy(trades)
        
        # Manual calculation
        wins = [80, 120, 100]  # avg = 100
        losses = [-30, -50]    # avg = -40 (absolute)
        win_rate_val = 3/5
        loss_rate = 2/5
        expected_expectancy = (100 * win_rate_val) - (40 * loss_rate)  # 60 - 16 = 44
        
        self.assertAlmostEqual(expectancy_val, expected_expectancy, places=6)
    
    def test_expectancy_zero_trades(self):
        """Test expectancy with empty trades."""
        empty_trades = pd.DataFrame({'pnl': []})
        
        exp = expectancy(empty_trades)
        
        self.assertEqual(exp, 0.0)
    
    def test_trade_metrics_missing_columns(self):
        """Test trade metrics with missing required columns."""
        invalid_trades = pd.DataFrame({'price': [100, 200, 150]})
        
        with self.assertRaises(ValueError):
            profit_factor(invalid_trades)
        
        with self.assertRaises(ValueError):
            win_rate(invalid_trades)
        
        with self.assertRaises(ValueError):
            expectancy(invalid_trades)


class TestSummarizeIntegration(TestPerformanceMetrics):
    """Test comprehensive performance summary integration."""
    
    def test_summarize_complete(self):
        """Test complete summarize function with realistic data."""
        # Create synthetic data
        returns = self._create_synthetic_returns(n_periods=252)  # 1 year daily
        trades = self._create_synthetic_trades(n_trades=20)
        initial_capital = 10000.0
        
        summary = summarize(trades, returns, initial_capital, 
                          risk_free=0.02, periods_per_year=252)
        
        # Verify all expected keys are present
        expected_keys = {
            'final_equity', 'pnl', 'total_return', 'cagr', 'sharpe', 'sortino',
            'max_drawdown', 'profit_factor', 'win_rate', 'expectancy',
            'total_trades', 'win_trades', 'loss_trades', 'avg_win', 'avg_loss',
            'largest_win', 'largest_loss', 'duration_days', 'annual_volatility'
        }
        
        self.assertEqual(set(summary.keys()), expected_keys)
        
        # Verify data types and reasonable ranges
        self.assertIsInstance(summary['final_equity'], (int, float))
        self.assertIsInstance(summary['total_trades'], int)
        self.assertEqual(summary['total_trades'], len(trades))
        
        # Verify mathematical relationships
        self.assertAlmostEqual(
            summary['pnl'], 
            summary['final_equity'] - initial_capital, 
            places=2
        )
        
        self.assertEqual(
            summary['total_trades'],
            summary['win_trades'] + summary['loss_trades']
        )
        
        # Verify win rate consistency
        if summary['total_trades'] > 0:
            calculated_win_rate = summary['win_trades'] / summary['total_trades']
            self.assertAlmostEqual(summary['win_rate'], calculated_win_rate, places=6)
    
    def test_summarize_empty_data(self):
        """Test summarize with empty data (edge case)."""
        empty_returns = pd.Series([], dtype=float)
        empty_trades = pd.DataFrame({'pnl': []})
        
        summary = summarize(empty_trades, empty_returns, 10000.0)
        
        # Should return safe defaults
        self.assertEqual(summary['final_equity'], 10000.0)
        self.assertEqual(summary['total_trades'], 0)
        self.assertEqual(summary['win_rate'], 0.0)
        self.assertEqual(summary['pnl'], 0.0)
    
    def test_summarize_only_returns(self):
        """Test summarize with returns but no trades."""
        returns = self._create_synthetic_returns(n_periods=100)
        empty_trades = pd.DataFrame({'pnl': []})
        
        summary = summarize(empty_trades, returns, 10000.0)
        
        # Should calculate equity-based metrics but not trade metrics
        self.assertNotEqual(summary['final_equity'], 10000.0)  # Should change with returns
        self.assertEqual(summary['total_trades'], 0)
        self.assertNotEqual(summary['sharpe'], 0.0)  # Should have risk metrics
    
    def test_summarize_only_trades(self):
        """Test summarize with trades but no returns."""
        empty_returns = pd.Series([], dtype=float)
        trades = self._create_synthetic_trades(n_trades=15)
        
        summary = summarize(trades, empty_returns, 10000.0)
        
        # Should calculate trade metrics but not return-based metrics
        self.assertEqual(summary['final_equity'], 10000.0)  # No change without returns
        self.assertNotEqual(summary['total_trades'], 0)
        self.assertNotEqual(summary['win_rate'], 0.0)  # Should have trade metrics
        self.assertEqual(summary['sharpe'], 0.0)  # No returns for risk metrics


class TestVisualization(TestPerformanceMetrics):
    """Test drawdown plot generation."""
    
    def test_plot_drawdown_save(self):
        """Test drawdown plot saving to file."""
        equity = self._create_synthetic_returns(100).cumsum() + 10000
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_drawdown.png"
            
            result_path = plot_drawdown(equity, save_path=save_path)
            
            # Verify file was created
            self.assertEqual(result_path, save_path)
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 1000)  # At least 1KB
    
    def test_plot_drawdown_no_save(self):
        """Test drawdown plot without saving."""
        equity = pd.Series([10000, 10500, 9500, 11000])
        
        result_path = plot_drawdown(equity, save_path=None)
        
        self.assertIsNone(result_path)
    
    def test_plot_drawdown_empty_equity(self):
        """Test drawdown plot with empty equity series."""
        empty_equity = pd.Series([], dtype=float)
        
        result_path = plot_drawdown(empty_equity, save_path=None)
        
        self.assertIsNone(result_path)
    
    def test_plot_drawdown_directory_creation(self):
        """Test that plot_drawdown creates parent directories."""
        equity = pd.Series([10000, 10500, 9500, 11000], 
                          index=pd.date_range('2024-01-01', periods=4))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Nested path that doesn't exist
            save_path = Path(temp_dir) / "reports" / "plots" / "drawdown.png"
            
            result_path = plot_drawdown(equity, save_path=save_path)
            
            self.assertEqual(result_path, save_path)
            self.assertTrue(save_path.exists())
            self.assertTrue(save_path.parent.exists())


class TestGPUParity(TestPerformanceMetrics):
    """Test GPU vs CPU result consistency."""
    
    @unittest.skipUnless(HAS_CUPY, "CuPy not available")
    def test_gpu_cpu_equity_curve_parity(self):
        """Test that GPU and CPU produce identical equity curves."""
        returns = self._create_synthetic_returns(n_periods=100000)  # Large enough for GPU
        initial_capital = 10000.0
        
        # Force CPU calculation
        with unittest.mock.patch('threadx.backtest.performance.HAS_CUPY', False):
            equity_cpu = equity_curve(returns, initial_capital)
        
        # Force GPU calculation
        equity_gpu = equity_curve(returns, initial_capital)
        
        # Results should be nearly identical (allowing for floating point precision)
        np.testing.assert_allclose(
            equity_cpu.values, equity_gpu.values, 
            rtol=1e-7, atol=1e-9
        )
    
    @unittest.skipUnless(HAS_CUPY, "CuPy not available") 
    def test_gpu_cpu_drawdown_parity(self):
        """Test GPU vs CPU drawdown calculation parity."""
        equity = pd.Series(
            10000 + np.cumsum(np.random.normal(0, 100, 75000)),  # Large series
            index=pd.date_range('2024-01-01', periods=75000, freq='min')
        )
        
        # CPU calculation
        with unittest.mock.patch('threadx.backtest.performance.HAS_CUPY', False):
            dd_cpu = drawdown_series(equity)
            max_dd_cpu = max_drawdown(equity)
        
        # GPU calculation
        dd_gpu = drawdown_series(equity)
        max_dd_gpu = max_drawdown(equity)
        
        # Verify parity
        np.testing.assert_allclose(dd_cpu.values, dd_gpu.values, rtol=1e-7, atol=1e-9)
        self.assertAlmostEqual(max_dd_cpu, max_dd_gpu, places=9)
    
    @unittest.skipUnless(HAS_CUPY, "CuPy not available")
    def test_gpu_cpu_risk_metrics_parity(self):
        """Test GPU vs CPU risk metrics parity."""
        returns = self._create_synthetic_returns(n_periods=25000)  # GPU threshold
        
        # CPU calculation
        with unittest.mock.patch('threadx.backtest.performance.HAS_CUPY', False):
            sharpe_cpu = sharpe_ratio(returns, risk_free=0.02, periods_per_year=365)
            sortino_cpu = sortino_ratio(returns, risk_free=0.02, periods_per_year=365)
        
        # GPU calculation
        sharpe_gpu = sharpe_ratio(returns, risk_free=0.02, periods_per_year=365)
        sortino_gpu = sortino_ratio(returns, risk_free=0.02, periods_per_year=365)
        
        # Verify parity
        self.assertAlmostEqual(sharpe_cpu, sharpe_gpu, places=7)
        self.assertAlmostEqual(sortino_cpu, sortino_gpu, places=7)


class TestEdgeCases(TestPerformanceMetrics):
    """Test edge cases and error handling."""
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        # Extreme returns
        extreme_returns = pd.Series([10.0, -0.99, 50.0, -0.5])  # 1000%, -99%, 5000%, -50%
        
        # Should handle without crashing
        equity = equity_curve(extreme_returns, 1000.0)
        sharpe = sharpe_ratio(extreme_returns)
        
        self.assertTrue(np.isfinite(equity.values).all())
        self.assertTrue(np.isfinite(sharpe))
    
    def test_irregular_timestamps(self):
        """Test performance with irregular timestamp spacing."""
        # Irregular timestamps (weekends, holidays skipped)
        irregular_dates = pd.to_datetime([
            '2024-01-01', '2024-01-03', '2024-01-08', '2024-01-15', '2024-01-16'
        ])
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01], index=irregular_dates)
        
        equity = equity_curve(returns, 10000.0)
        summary = summarize(pd.DataFrame({'pnl': []}), returns, 10000.0)
        
        # Should handle irregular spacing
        self.assertEqual(len(equity), 5)
        self.assertTrue(summary['duration_days'] > 0)
    
    def test_single_datapoint(self):
        """Test metrics with single data point."""
        single_return = pd.Series([0.05], index=[pd.Timestamp('2024-01-01')])
        single_trade = pd.DataFrame({'pnl': [100.0]})
        
        equity = equity_curve(single_return, 1000.0)
        summary = summarize(single_trade, single_return, 1000.0)
        
        # Should handle gracefully
        self.assertEqual(len(equity), 1)
        self.assertEqual(summary['total_trades'], 1)
        self.assertEqual(summary['win_rate'], 1.0)  # Single profitable trade
    
    def test_very_small_values(self):
        """Test handling of very small numerical values."""
        tiny_returns = pd.Series([1e-10, -1e-10, 1e-12] * 100)
        
        sharpe = sharpe_ratio(tiny_returns)
        equity = equity_curve(tiny_returns, 1000.0)
        
        # Should handle without numerical issues
        self.assertTrue(np.isfinite(sharpe))
        self.assertTrue(np.isfinite(equity.values).all())


def run_performance_validation():
    """Run comprehensive performance validation suite."""
    print("ThreadX Phase 6 - Performance Metrics Validation")
    print("=" * 55)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEquityCurve,
        TestDrawdownMetrics,
        TestRiskMetrics,
        TestTradeMetrics,
        TestSummarizeIntegration,
        TestVisualization,
        TestGPUParity,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestClass(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'='*55}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    # Performance metrics validation
    if result.testsRun > 0 and len(result.failures) == 0 and len(result.errors) == 0:
        print(f"\n🎉 ALL TESTS PASSED - ThreadX Phase 6 Performance Metrics VALIDATED!")
        return 0
    else:
        print(f"\n❌ Some tests failed - Phase 6 needs corrections")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_performance_validation())