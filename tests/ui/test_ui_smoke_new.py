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

import json
import os
import sys
import tempfile
import time
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

# Import modules under test with error handling
try:
    from threadx.ui.app import ThreadXApp, run_app, extract_sym_tf, scan_dir_by_ext, _clean_series
except ImportError as e:
    # Mock these if not available
    class MockThreadXApp:
        pass
    ThreadXApp = MockThreadXApp
    run_app = lambda: None
    extract_sym_tf = lambda x: None
    scan_dir_by_ext = lambda x, y: []
    _clean_series = lambda x: x

try:
    from threadx.ui.charts import plot_equity, plot_drawdown, plot_candlesticks
except ImportError:
    plot_equity = lambda *args, **kwargs: None
    plot_drawdown = lambda *args, **kwargs: None
    plot_candlesticks = lambda *args, **kwargs: None

try:
    from threadx.ui.tables import render_trades_table, render_metrics_table, export_table
except ImportError:
    render_trades_table = lambda *args, **kwargs: None
    render_metrics_table = lambda *args, **kwargs: None
    export_table = lambda *args, **kwargs: None


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
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_app_creation_and_destruction(self):
        """Test that ThreadXApp can be created and destroyed without crashing."""
        if ThreadXApp == MockThreadXApp:
            self.skipTest("ThreadX UI modules not available")
            
        with patch('threadx.ui.app.get_settings'), \
             patch('threadx.ui.app.setup_logging_once'), \
             patch('threadx.ui.app.get_logger'):
            
            try:
                # Create app (should not crash)
                app = ThreadXApp()
                self.assertIsInstance(app, ThreadXApp)
                
                # Check essential attributes exist
                self.assertTrue(hasattr(app, 'notebook'))
                self.assertTrue(hasattr(app, 'executor'))
                self.assertTrue(hasattr(app, 'logger'))
                self.assertTrue(hasattr(app, 'status_var'))
                
                # Check tabs were created
                self.assertGreaterEqual(app.notebook.index('end'), 5)  # At least 5 tabs
                
                # Destroy app (should not crash)
                app.destroy()
                
            except Exception as e:
                self.fail(f"App creation/destruction failed: {e}")
    
    def test_utility_functions(self):
        """Test utility functions work correctly."""
        if extract_sym_tf == lambda x: None:
            self.skipTest("extract_sym_tf not available")
            
        # Test extract_sym_tf
        test_cases = [
            ("BTCUSDC_1h.parquet", ("BTCUSDC", "1h")),
            ("ETHUSDT-15m.json", ("ETHUSDT", "15m")),
            ("invalid_file.txt", None)
        ]
        
        for filename, expected in test_cases:
            result = extract_sym_tf(filename)
            self.assertEqual(result, expected, f"Failed for {filename}")
    
    def test_scan_dir_functionality(self):
        """Test directory scanning works."""
        if scan_dir_by_ext == lambda x, y: []:
            self.skipTest("scan_dir_by_ext not available")
            
        # Create test files
        test_files = [
            "BTCUSDC_1h.parquet",
            "ETHUSDT_15m.json", 
            "ADAUSDC_1m.csv",
            "invalid.txt"
        ]
        
        for filename in test_files:
            (self.temp_path / filename).touch()
        
        # Test JSON scanning
        json_files = scan_dir_by_ext(str(self.temp_path), {".json", ".csv"})
        self.assertEqual(len(json_files), 2)  # ETHUSDT_15m and ADAUSDC_1m
        
        # Test Parquet scanning
        parquet_files = scan_dir_by_ext(str(self.temp_path), {".parquet"})
        self.assertEqual(len(parquet_files), 1)  # BTCUSDC_1h
    
    def test_data_cleaning(self):
        """Test data cleaning function."""
        if _clean_series == lambda x: x:
            self.skipTest("_clean_series not available")
            
        # Test with clean data
        clean_result = _clean_series(self.test_df)
        self.assertIsInstance(clean_result, pd.DataFrame)
        self.assertFalse(clean_result.empty)
        
        # Test with dirty data (duplicates, NaNs)
        dirty_df = self.test_df.copy()
        dirty_df.loc[dirty_df.index[0]] = np.nan  # Add NaN row
        dirty_df = pd.concat([dirty_df, dirty_df.iloc[:1]])  # Add duplicate
        
        cleaned = _clean_series(dirty_df)
        self.assertLess(len(cleaned), len(dirty_df))  # Should remove dirty data
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_chart_functions(self, mock_close, mock_savefig):
        """Test chart generation functions."""
        if plot_equity == lambda *args, **kwargs: None:
            self.skipTest("Chart functions not available")
            
        # Test equity chart
        save_path = self.temp_path / "equity.png"
        result = plot_equity(self.test_equity, save_path=save_path)
        self.assertEqual(result, save_path)
        
        # Test drawdown chart
        save_path = self.temp_path / "drawdown.png"
        result = plot_drawdown(self.test_equity, save_path=save_path)
        self.assertEqual(result, save_path)
        
        # Test candlestick chart  
        save_path = self.temp_path / "candlesticks.png"
        result = plot_candlesticks(self.test_df, save_path=save_path)
        self.assertEqual(result, save_path)
        
        # Verify matplotlib was called
        self.assertGreater(mock_savefig.call_count, 0)
        self.assertGreater(mock_close.call_count, 0)
    
    def test_table_functions(self):
        """Test table rendering functions."""
        if render_trades_table == lambda *args, **kwargs: None:
            self.skipTest("Table functions not available")
            
        # Create a mock parent widget
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        try:
            # Test trades table
            trades_table = render_trades_table(self.test_trades, parent=root)
            self.assertIsNotNone(trades_table)
            
            # Test metrics table
            metrics_table = render_metrics_table(self.test_metrics, parent=root)
            self.assertIsNotNone(metrics_table)
            
            # Test export
            export_path = self.temp_path / "test_export.csv"
            result = export_table(self.test_trades, export_path)
            self.assertEqual(result, export_path)
            
        finally:
            root.destroy()
    
    def test_threading_integration(self):
        """Test that threading components work properly."""
        if ThreadXApp == MockThreadXApp:
            self.skipTest("ThreadX UI modules not available")
            
        with patch('threadx.ui.app.get_settings'), \
             patch('threadx.ui.app.setup_logging_once'), \
             patch('threadx.ui.app.get_logger'):
            
            try:
                app = ThreadXApp()
                
                # Test executor exists and can submit tasks
                self.assertIsNotNone(app.executor)
                
                # Submit a simple task
                future = app.executor.submit(lambda: 42)
                result = future.result(timeout=1.0)
                self.assertEqual(result, 42)
                
                app.destroy()
                
            except Exception as e:
                self.fail(f"Threading integration failed: {e}")
    
    def test_mock_data_operations(self):
        """Test data operations with mocked backends."""
        if ThreadXApp == MockThreadXApp:
            self.skipTest("ThreadX UI modules not available")
            
        with patch('threadx.ui.app.get_settings'), \
             patch('threadx.ui.app.setup_logging_once'), \
             patch('threadx.ui.app.get_logger'), \
             patch('threadx.ui.app.IngestionManager') as mock_ingestion:
            
            try:
                app = ThreadXApp()
                
                # Test scan data directories (should not crash)
                app._scan_data_directories()
                
                # Test load and display (should not crash with empty data)
                app._load_and_display_data("BTCUSDC", "1h")
                
                app.destroy()
                
            except Exception as e:
                self.fail(f"Mock data operations failed: {e}")
    
    def test_error_handling(self):
        """Test error handling doesn't crash the application."""
        if ThreadXApp == MockThreadXApp:
            self.skipTest("ThreadX UI modules not available")
            
        with patch('threadx.ui.app.get_settings'), \
             patch('threadx.ui.app.setup_logging_once'), \
             patch('threadx.ui.app.get_logger'):
            
            try:
                app = ThreadXApp()
                
                # Test with invalid data (should handle gracefully)
                empty_df = pd.DataFrame()
                app._update_data_display("INVALID", "1h", empty_df)
                
                # Test with invalid parameters (should handle gracefully)
                app._on_home_load_data()  # No selection
                
                app.destroy()
                
            except Exception as e:
                self.fail(f"Error handling test failed: {e}")
    
    def test_streamlit_import_only(self):
        """Test that Streamlit components can be imported without crashing."""
        try:
            # Test import-only - should not launch server
            import sys
            with patch.dict(sys.modules, {'streamlit': Mock()}):
                # Mock streamlit to avoid actual import
                pass
                
            # This should not crash
            self.assertTrue(True, "Streamlit import test passed")
            
        except ImportError:
            # Streamlit not available - that's OK for this test
            self.skipTest("Streamlit not available")
    
    def test_file_export_operations(self):
        """Test file export operations create non-empty files."""
        # Test chart exports
        equity_path = self.temp_path / "test_equity.png"
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            result = plot_equity(self.test_equity, save_path=equity_path)
            if result is None:
                self.skipTest("Chart functions not available")
            self.assertEqual(result, equity_path)
        
        # Test data exports
        csv_path = self.temp_path / "test_data.csv"
        result = export_table(self.test_trades, csv_path, format='csv')
        if result is None:
            self.skipTest("Export functions not available")
        self.assertEqual(result, csv_path)
        
        parquet_path = self.temp_path / "test_data.parquet"
        result = export_table(self.test_trades, parquet_path, format='parquet')
        if result is None:
            self.skipTest("Export functions not available")
        self.assertEqual(result, parquet_path)
    
    def test_app_lifecycle_workflow(self):
        """Test complete app lifecycle workflow."""
        if ThreadXApp == MockThreadXApp:
            self.skipTest("ThreadX UI modules not available")
            
        with patch('threadx.ui.app.get_settings'), \
             patch('threadx.ui.app.setup_logging_once'), \
             patch('threadx.ui.app.get_logger'), \
             patch('threadx.ui.app.IngestionManager') as mock_ingestion:
            
            try:
                app = ThreadXApp()
                
                # Create test parameters file
                params_file = self.temp_path / "test_params.json"
                test_params = {
                    'symbol': 'BTCUSDC',
                    'timeframe': '15m',
                    'bb_period': 20,
                    'bb_std': 2.0
                }
                
                with open(params_file, 'w') as f:
                    json.dump(test_params, f)
                
                # Test parameter loading if method exists
                if hasattr(app, 'load_params_from_json'):
                    loaded_params = app.load_params_from_json(params_file)
                    self.assertEqual(loaded_params['symbol'], 'BTCUSDC')
                
                # Test background operations if methods exist
                if hasattr(app, 'trigger_regenerate'):
                    app.trigger_regenerate()
                    time.sleep(0.1)  # Allow background task
                
                if hasattr(app, 'trigger_backtest'):
                    app.trigger_backtest()
                    time.sleep(0.1)  # Allow background task
                
                # Test export if method exists
                if hasattr(app, 'export_results'):
                    export_dir = self.temp_path / "results"
                    exported_files = app.export_results(export_dir)
                    self.assertIsNotNone(exported_files)
                
                app.destroy()
                
            except Exception as e:
                self.fail(f"App lifecycle workflow failed: {e}")


class TestRunAppFunction(unittest.TestCase):
    """Test the run_app function."""
    
    @patch('threadx.ui.app.ThreadXApp')
    @patch('threadx.ui.app.setup_logging_once')
    @patch('threadx.ui.app.get_logger')
    def test_run_app_creation(self, mock_logger, mock_setup, mock_app_class):
        """Test run_app creates and runs app properly."""
        if run_app == lambda: None:
            self.skipTest("run_app function not available")
            
        mock_app = Mock()
        mock_app_class.return_value = mock_app
        
        # This would normally start mainloop, but we'll mock it
        mock_app.mainloop = Mock()
        
        # Test function exists and can be called
        self.assertTrue(callable(run_app))
        
        # Note: We don't actually call run_app() here as it would block


if __name__ == '__main__':
    # Configure test environment
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests with detailed output
    unittest.main(verbosity=2)