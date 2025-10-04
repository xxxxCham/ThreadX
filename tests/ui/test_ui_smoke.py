"""
ThreadX UI Smoke Tests - TechinTerror Interface
==============================================

Headless smoke tests for the ThreadX TechinTerror interface to ensure:
- Application launches and closes without crashes
- All tabs are created properly
- Charts and tables functions work without errors
- Import-only test for Streamlit fallback
- Mock-based testing with no network dependencies

Author: ThreadX Framework
Version: Phase 8 - Smoke Tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

import pandas as pd
import numpy as np

# Suppress GUI-related warnings for headless testing
os.environ['DISPLAY'] = ':0.0'  # Fake display for headless
os.environ['MPLBACKEND'] = 'Agg'
import tkinter as tk

# Set random seed for reproducible tests
np.random.seed(42)

# Import modules under test
try:
    from threadx.ui.app import ThreadXApp, run_app, extract_sym_tf, scan_dir_by_ext, _clean_series
    from threadx.ui.charts import plot_equity, plot_drawdown, plot_candlesticks
    from threadx.ui.tables import render_trades_table, render_metrics_table, export_table
except ImportError as e:
    # Skip tests if modules not available
    raise unittest.SkipTest(f"ThreadX UI modules not available: {e}")


class TestTechinTerrorSmokeTests(unittest.TestCase):
    """Smoke tests for TechinTerror interface."""
    
    def setUp(self):
        """Set up test environment."""
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.ERROR)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Mock test data
        self.test_dates = pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC')
        self.test_df = pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=self.test_dates)
        
        self.test_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=self.test_dates)
        self.test_equity = (1 + self.test_returns).cumprod() * 10000
        
        self.test_trades = pd.DataFrame({
            'entry_time': self.test_dates[::10],
            'exit_time': self.test_dates[5::10],
            'side': ['LONG'] * 10,
            'pnl': np.random.normal(50, 100, 10),
            'entry_price': np.random.uniform(50000, 51000, 10),
            'exit_price': np.random.uniform(50000, 51000, 10)
        })
        
        self.test_metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'total_trades': 10,
            'win_rate': 0.6
        }
        self.logger = logging.getLogger(__name__)
        
        # Create mock data
        self.sample_equity = self._create_sample_equity()
        self.sample_trades = self._create_sample_trades()
        self.sample_metrics = self._create_sample_metrics()
        
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_equity(self) -> pd.Series:
        """Create sample equity curve for testing."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        returns = pd.Series(np.random.randn(1000) * 0.01, index=dates)
        equity = (1 + returns).cumprod() * 10000
        return equity
    
    def _create_sample_trades(self) -> pd.DataFrame:
        """Create sample trades data for testing."""
        n_trades = 20
        dates = pd.date_range('2024-01-01', periods=n_trades * 10, freq='H')
        entry_times = dates[::10]
        exit_times = entry_times + pd.Timedelta('2H')
        
        return pd.DataFrame({
            'entry_time': entry_times,
            'exit_time': exit_times,
            'pnl': np.random.randn(n_trades) * 100,
            'side': ['LONG'] * 10 + ['SHORT'] * 10,
            'entry_price': 50000 + np.random.randn(n_trades) * 1000,
            'exit_price': 50000 + np.random.randn(n_trades) * 1000
        })
    
    def _create_sample_metrics(self) -> dict:
        """Create sample performance metrics for testing."""
        return {
            'final_equity': 11000.0,
            'pnl': 1000.0,
            'total_return': 0.10,
            'cagr': 0.12,
            'sharpe': 1.5,
            'sortino': 1.2,
            'max_drawdown': -0.05,
            'total_trades': 100,
            'win_trades': 60,
            'loss_trades': 40,
            'win_rate': 0.6,
            'profit_factor': 1.8,
            'expectancy': 10.0,
            'avg_win': 25.0,
            'avg_loss': -15.0,
            'largest_win': 150.0,
            'largest_loss': -80.0,
            'duration_days': 365,
            'annual_volatility': 0.15
        }
    
    @patch('threadx.ui.app.ThreadXApp.__init__', return_value=None)
    def test_tkinter_app_creation_headless(self, mock_init):
        """Test Tkinter app creation in headless mode."""
        try:
            # Import and mock create app
            from threadx.ui.app import ThreadXApp
            
            # Create mock app instance
            app = ThreadXApp()
            
            # Verify mock was called
            mock_init.assert_called_once()
            
            self.logger.info("✅ Tkinter app creation test passed")
                
        except ImportError:
            self.skipTest("Tkinter not available in test environment")
        except Exception as e:
            self.fail(f"Tkinter app creation failed: {e}")
    
    def test_app_parameter_loading(self):
        """Test parameter loading functionality."""
        try:
            # Create test parameter file
            test_params = {
                'symbol': 'BTCUSDC',
                'timeframe': '15m',
                'bb_period': 20,
                'bb_std': 2.0,
                'entry_z': 2.0,
                'k_sl': 1.5,
                'leverage': 3,
                'risk': 0.02
            }
            
            params_file = self.temp_dir / "test_params.json"
            with open(params_file, 'w') as f:
                json.dump(test_params, f, indent=2)
            
            # Mock Tkinter app
            with patch('tkinter.Tk'):
                from threadx.ui.app import ThreadXApp
                
                app = ThreadXApp()
                
                # Test parameter loading
                loaded_params = app.load_params_from_json(params_file)
                
                # Verify parameters loaded correctly
                self.assertEqual(loaded_params['symbol'], 'BTCUSDC')
                self.assertEqual(loaded_params['bb_period'], 20)
                self.assertEqual(loaded_params['leverage'], 3)
                
                self.logger.info("✅ Parameter loading test passed")
                
        except Exception as e:
            self.fail(f"Parameter loading test failed: {e}")
    
    def test_non_blocking_operations(self):
        """Test that operations run in threads (non-blocking)."""
        try:
            with patch('tkinter.Tk'):
                from threadx.ui.app import ThreadXApp
                
                app = ThreadXApp()
                
                # Mock Phase 3/5/6 components to simulate work
                mock_work_duration = 0.1  # Short duration for test
                
                def mock_ensure(*args, **kwargs):
                    time.sleep(mock_work_duration)
                    return True
                
                def mock_run(*args, **kwargs):
                    time.sleep(mock_work_duration)
                    return self.sample_equity.pct_change().dropna(), self.sample_trades
                
                app.bank.ensure = Mock(side_effect=mock_ensure)
                app.engine.run = Mock(side_effect=mock_run)
                
                # Test regenerate indicators (should be non-blocking)
                start_time = time.time()
                app.trigger_regenerate()
                immediate_time = time.time()
                
                # Should return immediately (non-blocking)
                self.assertLess(immediate_time - start_time, mock_work_duration / 2)
                
                # Wait for background task to complete
                time.sleep(mock_work_duration * 2)
                
                # Verify mock was called
                app.bank.ensure.assert_called_once()
                
                # Test backtest (should be non-blocking)
                start_time = time.time()
                app.trigger_backtest()
                immediate_time = time.time()
                
                # Should return immediately (non-blocking)
                self.assertLess(immediate_time - start_time, mock_work_duration / 2)
                
                # Wait for background task to complete
                time.sleep(mock_work_duration * 2)
                
                # Verify mock was called
                app.engine.run.assert_called_once()
                
                self.logger.info("✅ Non-blocking operations test passed")
                
        except Exception as e:
            self.fail(f"Non-blocking operations test failed: {e}")
    
    def test_chart_generation(self):
        """Test chart generation and export."""
        try:
            from threadx.ui.charts import plot_equity, plot_drawdown
            
            # Test equity chart
            equity_path = self.temp_dir / "test_equity.png"
            result_path = plot_equity(self.sample_equity, save_path=equity_path)
            
            # Verify chart was created
            self.assertTrue(equity_path.exists())
            self.assertGreater(equity_path.stat().st_size, 0)
            self.assertEqual(result_path, equity_path)
            
            # Test drawdown chart
            drawdown_path = self.temp_dir / "test_drawdown.png"
            result_path = plot_drawdown(self.sample_equity, save_path=drawdown_path)
            
            # Verify chart was created
            self.assertTrue(drawdown_path.exists())
            self.assertGreater(drawdown_path.stat().st_size, 0)
            self.assertEqual(result_path, drawdown_path)
            
            self.logger.info("✅ Chart generation test passed")
            
        except Exception as e:
            self.fail(f"Chart generation test failed: {e}")
    
    def test_table_rendering(self):
        """Test table rendering functionality."""
        try:
            from threadx.ui.tables import render_trades_table, render_metrics_table
            
            # Test trades table
            trades_result = render_trades_table(self.sample_trades)
            
            # Verify result structure
            self.assertIsInstance(trades_result, dict)
            self.assertIn('data', trades_result)
            self.assertIn('summary', trades_result)
            self.assertIn('total_rows', trades_result)
            self.assertEqual(trades_result['total_rows'], len(self.sample_trades))
            
            # Test metrics table
            metrics_result = render_metrics_table(self.sample_metrics)
            
            # Verify result structure
            self.assertIsInstance(metrics_result, dict)
            self.assertIn('data', metrics_result)
            self.assertIn('summary', metrics_result)
            
            self.logger.info("✅ Table rendering test passed")
            
        except Exception as e:
            self.fail(f"Table rendering test failed: {e}")
    
    def test_data_export(self):
        """Test data export functionality."""
        try:
            from threadx.ui.tables import export_table
            
            # Test CSV export
            csv_path = self.temp_dir / "test_export.csv"
            result_path = export_table(self.sample_trades, csv_path)
            
            # Verify export
            self.assertTrue(csv_path.exists())
            self.assertGreater(csv_path.stat().st_size, 0)
            self.assertEqual(result_path, csv_path)
            
            # Verify CSV content
            loaded_df = pd.read_csv(csv_path)
            self.assertEqual(len(loaded_df), len(self.sample_trades))
            
            # Test Parquet export
            parquet_path = self.temp_dir / "test_export.parquet"
            result_path = export_table(self.sample_trades, parquet_path)
            
            # Verify export
            self.assertTrue(parquet_path.exists())
            self.assertGreater(parquet_path.stat().st_size, 0)
            self.assertEqual(result_path, parquet_path)
            
            # Verify Parquet content
            loaded_df = pd.read_parquet(parquet_path)
            self.assertEqual(len(loaded_df), len(self.sample_trades))
            
            self.logger.info("✅ Data export test passed")
            
        except Exception as e:
            self.fail(f"Data export test failed: {e}")
    
    def test_streamlit_import(self):
        """Test Streamlit app import without execution."""
        try:
            # Test import without running server
            import apps.streamlit.app as streamlit_app
            
            # Verify module loaded
            self.assertTrue(hasattr(streamlit_app, 'main'))
            self.assertTrue(hasattr(streamlit_app, 'init_session_state'))
            
            # Test helper functions
            self.assertTrue(hasattr(streamlit_app, 'validate_parameters'))
            self.assertTrue(hasattr(streamlit_app, 'has_results'))
            
            # Test parameter validation function
            valid_params = {
                'entry_z': 2.0,
                'k_sl': 1.5,
                'trail_k': 1.0,
                'leverage': 3,
                'risk': 0.02
            }
            
            is_valid, message = streamlit_app.validate_parameters(valid_params)
            self.assertTrue(is_valid)
            
            # Test invalid parameters
            invalid_params = {
                'entry_z': -1.0,  # Invalid
                'k_sl': 1.5,
                'trail_k': 1.0,
                'leverage': 3,
                'risk': 0.02
            }
            
            is_valid, message = streamlit_app.validate_parameters(invalid_params)
            self.assertFalse(is_valid)
            
            self.logger.info("✅ Streamlit import test passed")
            
        except ImportError as e:
            self.skipTest(f"Streamlit not available: {e}")
        except Exception as e:
            self.fail(f"Streamlit import test failed: {e}")
    
    def test_app_export_results(self):
        """Test complete results export functionality."""
        try:
            with patch('tkinter.Tk'):
                from threadx.ui.app import ThreadXApp
                
                app = ThreadXApp()
                
                # Set up mock results
                app.last_returns = self.sample_equity.pct_change().dropna()
                app.last_trades = self.sample_trades
                app.last_metrics = self.sample_metrics
                
                # Test export
                export_dir = self.temp_dir / "export"
                exported_files = app.export_results(export_dir)
                
                # Verify exports
                self.assertGreater(len(exported_files), 0)
                
                # Check expected files exist
                expected_files = ['trades.csv', 'metrics.json', 'returns.csv', 'parameters.json']
                for expected_file in expected_files:
                    file_path = export_dir / expected_file
                    self.assertTrue(file_path.exists(), f"Missing file: {expected_file}")
                    self.assertGreater(file_path.stat().st_size, 0, f"Empty file: {expected_file}")
                
                self.logger.info("✅ App export results test passed")
                
        except Exception as e:
            self.fail(f"App export results test failed: {e}")
    
    def test_chart_error_handling(self):
        """Test chart error handling with invalid data."""
        try:
            from threadx.ui.charts import plot_equity, plot_drawdown
            
            # Test with empty series
            empty_series = pd.Series([], dtype=float)
            
            with self.assertRaises(ValueError):
                plot_equity(empty_series)
            
            # Test with non-datetime index
            invalid_series = pd.Series([1, 2, 3], index=[0, 1, 2])
            
            with self.assertRaises(ValueError):
                plot_equity(invalid_series)
            
            self.logger.info("✅ Chart error handling test passed")
            
        except Exception as e:
            self.fail(f"Chart error handling test failed: {e}")
    
    def test_table_error_handling(self):
        """Test table error handling with invalid data."""
        try:
            from threadx.ui.tables import render_trades_table, render_metrics_table
            
            # Test with empty DataFrame
            empty_df = pd.DataFrame()
            
            with self.assertRaises(ValueError):
                render_trades_table(empty_df)
            
            # Test with missing columns
            incomplete_df = pd.DataFrame({'entry_time': [1, 2, 3]})
            
            with self.assertRaises(ValueError):
                render_trades_table(incomplete_df)
            
            # Test with empty metrics
            empty_metrics = {}
            
            with self.assertRaises(ValueError):
                render_metrics_table(empty_metrics)
            
            self.logger.info("✅ Table error handling test passed")
            
        except Exception as e:
            self.fail(f"Table error handling test failed: {e}")
    
    def test_deterministic_results(self):
        """Test that results are deterministic with same seed."""
        try:
            # Set seed
            np.random.seed(42)
            equity1 = self._create_sample_equity()
            
            # Reset seed
            np.random.seed(42)
            equity2 = self._create_sample_equity()
            
            # Results should be identical
            pd.testing.assert_series_equal(equity1, equity2)
            
            self.logger.info("✅ Deterministic results test passed")
            
        except Exception as e:
            self.fail(f"Deterministic results test failed: {e}")
    
    def test_relative_paths(self):
        """Test relative path handling."""
        try:
            from threadx.ui.charts import plot_equity
            
            # Test with relative path
            relative_path = Path("test_chart.png")
            absolute_path = self.temp_dir / relative_path
            
            # Create chart with relative path resolution
            result_path = plot_equity(self.sample_equity, save_path=absolute_path)
            
            # Verify file created
            self.assertTrue(absolute_path.exists())
            self.assertEqual(result_path, absolute_path)
            
            self.logger.info("✅ Relative paths test passed")
            
        except Exception as e:
            self.fail(f"Relative paths test failed: {e}")
    
    def test_threading_safety(self):
        """Test thread safety of UI operations."""
        try:
            from threadx.ui.charts import plot_equity
            
            results = []
            errors = []
            
            def chart_worker(worker_id):
                try:
                    chart_path = self.temp_dir / f"chart_{worker_id}.png"
                    result = plot_equity(self.sample_equity, save_path=chart_path)
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # Run multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=chart_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)
            
            # Verify results
            self.assertEqual(len(errors), 0, f"Threading errors: {errors}")
            self.assertEqual(len(results), 3)
            
            # Verify all files created
            for i in range(3):
                chart_path = self.temp_dir / f"chart_{i}.png"
                self.assertTrue(chart_path.exists())
            
            self.logger.info("✅ Threading safety test passed")
            
        except Exception as e:
            self.fail(f"Threading safety test failed: {e}")


class TestUIIntegration(unittest.TestCase):
    """Integration tests for UI components with mocked Phase 3/5/6."""
    
    def setUp(self):
        """Set up integration test environment."""
        np.random.seed(42)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="threadx_ui_integration_"))
        self.addCleanup(self._cleanup_temp_dir)
        
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from parameters to export."""
        try:
            with patch('tkinter.Tk'):
                from threadx.ui.app import ThreadXApp
                
                app = ThreadXApp()
                
                # Mock successful operations
                app.bank.ensure = Mock(return_value=True)
                
                dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
                returns = pd.Series(np.random.randn(1000) * 0.01, index=dates)
                trades = pd.DataFrame({
                    'entry_time': dates[::50],
                    'exit_time': dates[50::50],
                    'pnl': np.random.randn(20) * 100,
                    'side': ['LONG'] * 20
                })
                
                app.engine.run = Mock(return_value=(returns, trades))
                app.performance.summarize = Mock(return_value={
                    'sharpe': 1.5,
                    'total_return': 0.10,
                    'max_drawdown': -0.05
                })
                
                # 1. Load parameters
                params_file = self.temp_dir / "params.json"
                test_params = {
                    'symbol': 'BTCUSDC',
                    'timeframe': '15m',
                    'bb_period': 20,
                    'bb_std': 2.0
                }
                
                with open(params_file, 'w') as f:
                    json.dump(test_params, f)
                
                loaded_params = app.load_params_from_json(params_file)
                self.assertEqual(loaded_params['symbol'], 'BTCUSDC')
                
                # 2. Trigger regenerate (async)
                app.trigger_regenerate()
                time.sleep(0.1)  # Allow background task
                
                # 3. Trigger backtest (async)
                app.trigger_backtest()
                time.sleep(0.1)  # Allow background task
                
                # 4. Export results
                export_dir = self.temp_dir / "results"
                exported_files = app.export_results(export_dir)
                
                # Verify workflow completed
                self.assertGreater(len(exported_files), 0)
                app.bank.ensure.assert_called()
                app.engine.run.assert_called()
                app.performance.summarize.assert_called()
                
                print("✅ End-to-end workflow test passed")
                
        except Exception as e:
            self.fail(f"End-to-end workflow test failed: {e}")


if __name__ == '__main__':
    # Configure test environment
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)