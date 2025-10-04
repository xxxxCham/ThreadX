"""
ThreadX Sweep & Logging Tests - Phase 7
========================================

Comprehensive tests for parameter sweep engine and logging system.

Tests cover:
- Parallel grid execution with deterministic results (seed=42)
- Append-only Parquet storage with file locks
- Best results tracking and sorting
- Error handling and robustness
- GPU/CPU compatibility
- Logging configuration and rotation

Author: ThreadX Framework
Version: Phase 7 - Sweep & Logging
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import shutil

import numpy as np
import pandas as pd
import pytest

# Set up test environment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.backtest.sweep import (
    run_grid, append_run_history, update_best_by_run, 
    load_run_history, validate_param_grid, make_run_id
)
from threadx.utils.log import get_logger, setup_logging_once


class TestSweepLogging(unittest.TestCase):
    """Comprehensive tests for sweep engine and logging."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp(prefix="threadx_test_"))
        self.addCleanup(self._cleanup_test_dir)
        
        # Set up deterministic test data
        np.random.seed(42)
        self.test_df = self._create_test_data()
        
        # Set up logging
        setup_logging_once()
        self.logger = get_logger(__name__)
        
    def _cleanup_test_dir(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_data(self, n_periods: int = 1000) -> pd.DataFrame:
        """Create synthetic OHLCV data for testing."""
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='15min')
        
        # Realistic price simulation
        base_price = 50000.0
        returns = np.cumsum(np.random.randn(n_periods) * 0.001)
        close_prices = base_price * np.exp(returns)
        
        # OHLC generation
        volatility = close_prices * 0.005
        high_prices = close_prices + np.abs(np.random.randn(n_periods)) * volatility
        low_prices = close_prices - np.abs(np.random.randn(n_periods)) * volatility
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        volume = np.random.randint(1000, 10000, n_periods)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    def _create_mock_engine(self, fail_rate: float = 0.0):
        """Create mock engine function for testing."""
        def mock_engine(df, params, **kwargs):
            # Simulate failure rate
            if np.random.random() < fail_rate:
                raise ValueError(f"Simulated failure for params: {params}")
            
            # Generate mock results based on parameters
            n_periods = len(df)
            volatility = 0.01 * params.get('bb_std', 2.0)  # Scale with bb_std
            returns = pd.Series(
                np.random.randn(n_periods) * volatility,
                index=df.index
            )
            
            # Mock trades
            n_trades = max(1, int(n_periods / 100))  # ~1% of periods
            trade_times = np.random.choice(df.index, n_trades, replace=False)
            trade_pnls = np.random.randn(n_trades) * 100 * params.get('bb_std', 2.0)
            
            trades = pd.DataFrame({
                'entry_time': trade_times,
                'exit_time': trade_times + pd.Timedelta('1h'),
                'pnl': trade_pnls,
                'side': ['LONG'] * n_trades
            })
            
            return returns, trades
        
        return mock_engine
    
    def test_nominal_grid_execution(self):
        """Test nominal parameter grid execution."""
        # Create parameter grid (≥10 combinations)
        param_grid = []
        for bb_period in [14, 20, 26]:
            for bb_std in [1.8, 2.0, 2.2]:
                for entry_z in [1.5, 2.0]:
                    param_grid.append({
                        'bb_period': bb_period,
                        'bb_std': bb_std,
                        'entry_z': entry_z,
                        'leverage': 3,
                        'risk': 0.02
                    })
        
        self.assertGreaterEqual(len(param_grid), 10, "Need at least 10 parameter combinations")
        
        # Mock engine function
        engine_func = self._create_mock_engine()
        
        # Time single-threaded execution
        start_time = time.time()
        results_single = run_grid(
            df=self.test_df,
            param_grid=param_grid[:5],  # Smaller subset for timing
            engine_func=engine_func,
            symbol="TESTBTC",
            timeframe="15m",
            max_workers=1,
            seed=42
        )
        single_time = time.time() - start_time
        
        # Time multi-threaded execution
        start_time = time.time()
        results_multi = run_grid(
            df=self.test_df,
            param_grid=param_grid[:5],
            engine_func=engine_func,
            symbol="TESTBTC",
            timeframe="15m",
            max_workers=4,
            seed=42
        )
        multi_time = time.time() - start_time
        
        # Verify results structure
        self.assertEqual(len(results_multi), 5)
        
        required_columns = [
            'run_id', 'timestamp', 'symbol', 'timeframe', 'params_json',
            'success', 'final_equity', 'pnl', 'sharpe', 'total_trades'
        ]
        for col in required_columns:
            self.assertIn(col, results_multi.columns, f"Missing column: {col}")
        
        # Verify deterministic results (same seed should give same results)
        results_repeat = run_grid(
            df=self.test_df,
            param_grid=param_grid[:3],
            engine_func=engine_func,
            symbol="TESTBTC",
            timeframe="15m",
            max_workers=4,
            seed=42
        )
        
        # Check run_id determinism
        self.assertEqual(results_multi['run_id'].iloc[0], results_repeat['run_id'].iloc[0])
        
        # Performance check (multi-threaded should be faster or comparable)
        # Allow for some variability in timing
        self.logger.info(f"Single-threaded: {single_time:.3f}s, Multi-threaded: {multi_time:.3f}s")
        self.assertLessEqual(multi_time, single_time * 2.0, "Multi-threaded should not be much slower")
    
    def test_append_only_history(self):
        """Test append-only history functionality."""
        history_path = self.test_dir / "test_runs.parquet"
        
        # Create first batch of results
        results1 = pd.DataFrame({
            'run_id': ['run_001', 'run_002'],
            'task_id': ['task_001', 'task_002'],
            'symbol': ['BTCUSDC', 'ETHUSD'],
            'timeframe': ['15m', '1h'],
            'params_json': ['{"bb_std": 2.0}', '{"bb_std": 1.8}'],
            'success': [True, True],
            'sharpe': [1.5, 0.8],
            'total_trades': [100, 50],
            'timestamp': [pd.Timestamp.now(tz='UTC')] * 2
        })
        
        # First append
        append_run_history(results1, history_path)
        self.assertTrue(history_path.exists())
        
        loaded1 = pd.read_parquet(history_path)
        self.assertEqual(len(loaded1), 2)
        
        # Create second batch
        results2 = pd.DataFrame({
            'run_id': ['run_003', 'run_004'],
            'task_id': ['task_003', 'task_004'],
            'symbol': ['ADAUSD', 'SOLUSD'],
            'timeframe': ['30m', '2h'],
            'params_json': ['{"bb_std": 2.5}', '{"bb_std": 1.5}'],
            'success': [True, False],
            'sharpe': [2.1, 0.0],
            'total_trades': [75, 0],
            'timestamp': [pd.Timestamp.now(tz='UTC')] * 2
        })
        
        # Second append (should increase size)
        original_size = history_path.stat().st_size
        append_run_history(results2, history_path)
        new_size = history_path.stat().st_size
        self.assertGreater(new_size, original_size, "File size should increase after append")
        
        loaded2 = pd.read_parquet(history_path)
        self.assertEqual(len(loaded2), 4, "Should have 4 total records after append")
        
        # Test duplicate handling (same task_id should not duplicate)
        results_duplicate = results1.copy()
        append_run_history(results_duplicate, history_path)
        
        loaded3 = pd.read_parquet(history_path)
        self.assertEqual(len(loaded3), 4, "Duplicates should not increase record count")
    
    def test_file_locking(self):
        """Test file locking during concurrent access."""
        history_path = self.test_dir / "locked_runs.parquet"
        
        # Create test data
        results = pd.DataFrame({
            'run_id': ['concurrent_001'],
            'task_id': ['task_concurrent_001'],
            'symbol': ['TESTCOIN'],
            'success': [True],
            'sharpe': [1.0],
            'timestamp': [pd.Timestamp.now(tz='UTC')]
        })
        
        # Test that append_run_history handles locking properly
        # (This is more of a smoke test since true concurrency testing is complex)
        try:
            append_run_history(results, history_path)
            self.assertTrue(history_path.exists())
            
            # Verify file is readable after locking operations
            loaded = pd.read_parquet(history_path)
            self.assertEqual(len(loaded), 1)
            
        except Exception as e:
            self.fail(f"File locking operations failed: {e}")
    
    def test_best_by_run_sorting(self):
        """Test best results sorting and stability."""
        history_path = self.test_dir / "sort_test_runs.parquet"
        best_path = self.test_dir / "sort_test_best.parquet"
        
        # Create history with multiple runs and tasks
        history_data = []
        
        # Run 1: Multiple tasks with different Sharpe ratios
        for i, sharpe in enumerate([1.5, 2.1, 1.8]):
            history_data.append({
                'run_id': 'run_001',
                'task_id': f'task_001_{i:03d}',
                'symbol': 'BTCUSDC',
                'timeframe': '15m',
                'params_json': f'{{"bb_std": {1.8 + i * 0.1}}}',
                'success': True,
                'sharpe': sharpe,
                'sortino': sharpe * 1.1,
                'cagr': sharpe * 10,
                'win_rate': 0.6,
                'profit_factor': 1.5,
                'max_drawdown': -0.1,
                'total_trades': 100,
                'final_equity': 11000 + i * 500,
                'timestamp': pd.Timestamp.now(tz='UTC')
            })
        
        # Run 2: Different performance
        for i, sharpe in enumerate([1.2, 2.1, 1.4]):  # Same max sharpe as run 1 for tie-breaking test
            history_data.append({
                'run_id': 'run_002',
                'task_id': f'task_002_{i:03d}',
                'symbol': 'ETHUSD',
                'timeframe': '1h',
                'params_json': f'{{"bb_std": {2.0 + i * 0.1}}}',
                'success': True,
                'sharpe': sharpe,
                'sortino': sharpe * 1.1,
                'cagr': sharpe * 10,
                'win_rate': 0.65,
                'profit_factor': 1.6,
                'max_drawdown': -0.08,
                'total_trades': 80,
                'final_equity': 10800 + i * 300,
                'timestamp': pd.Timestamp.now(tz='UTC')
            })
        
        history_df = pd.DataFrame(history_data)
        history_df.to_parquet(history_path, index=False)
        
        # Update best results
        update_best_by_run(history_path, best_path, sort_by='sharpe')
        
        self.assertTrue(best_path.exists())
        best_df = pd.read_parquet(best_path)
        
        # Should have 2 runs
        self.assertEqual(len(best_df), 2)
        
        # Verify sorting (descending by Sharpe)
        sharpe_values = best_df['best_sharpe'].tolist()
        self.assertEqual(sharpe_values, sorted(sharpe_values, reverse=True))
        
        # Verify tie-breaking by run_id (stable sort)
        tied_runs = best_df[best_df['best_sharpe'] == 2.1]
        if len(tied_runs) > 1:
            run_ids = tied_runs['run_id'].tolist()
            self.assertEqual(run_ids, sorted(run_ids))
        
        # Verify best selection within each run
        run_001_best = best_df[best_df['run_id'] == 'run_001'].iloc[0]
        self.assertEqual(run_001_best['best_sharpe'], 2.1)  # Highest in run_001
        
        run_002_best = best_df[best_df['run_id'] == 'run_002'].iloc[0]
        self.assertEqual(run_002_best['best_sharpe'], 2.1)  # Highest in run_002
        
        # Test different sort metrics
        update_best_by_run(history_path, best_path, sort_by='max_drawdown')
        best_dd_df = pd.read_parquet(best_path)
        
        # Max drawdown should be sorted ascending (less negative is better)
        dd_values = best_dd_df['sort_value'].tolist()
        self.assertEqual(dd_values, sorted(dd_values))  # Ascending order
    
    def test_error_handling_robustness(self):
        """Test error handling in sweep execution."""
        # Create parameter grid
        param_grid = [
            {'bb_period': 20, 'bb_std': 2.0, 'fail': False},
            {'bb_period': 14, 'bb_std': 1.8, 'fail': True},   # This will fail
            {'bb_period': 26, 'bb_std': 2.2, 'fail': False},
            {'bb_period': 21, 'bb_std': 1.9, 'fail': True},   # This will fail
        ]
        
        # Mock engine that fails on certain parameters
        def failing_engine(df, params, **kwargs):
            if params.get('fail', False):
                raise RuntimeError(f"Intentional failure for testing: {params}")
            
            # Return normal results
            returns = pd.Series([0.001, -0.002, 0.003], index=df.index[:3])
            trades = pd.DataFrame({
                'entry_time': [df.index[0]],
                'exit_time': [df.index[1]],
                'pnl': [100.0],
                'side': ['LONG']
            })
            return returns, trades
        
        # Execute sweep with failures
        results = run_grid(
            df=self.test_df,
            param_grid=param_grid,
            engine_func=failing_engine,
            symbol="FAILTEST",
            timeframe="15m",
            max_workers=2,
            seed=42
        )
        
        # Should have all 4 results (2 successful, 2 failed)
        self.assertEqual(len(results), 4)
        
        # Check success/failure distribution
        success_count = results['success'].sum()
        failure_count = (~results['success']).sum()
        
        self.assertEqual(success_count, 2, "Should have 2 successful tasks")
        self.assertEqual(failure_count, 2, "Should have 2 failed tasks")
        
        # Failed tasks should have error messages
        failed_results = results[~results['success']]
        for _, row in failed_results.iterrows():
            self.assertIsNotNone(row['error'])
            self.assertIn("Intentional failure", row['error'])
        
        # Successful tasks should not have errors
        successful_results = results[results['success']]
        for _, row in successful_results.iterrows():
            self.assertTrue(pd.isna(row['error']) or row['error'] == '')
        
        # Verify Parquet remains readable after errors
        history_path = self.test_dir / "error_test_history.parquet"
        append_run_history(results, history_path)
        
        loaded = pd.read_parquet(history_path)
        self.assertEqual(len(loaded), 4)
        self.assertTrue(all(col in loaded.columns for col in ['success', 'error']))
    
    def test_gpu_cpu_compatibility(self):
        """Test GPU/CPU compatibility through engine delegation."""
        param_grid = [
            {'bb_period': 20, 'bb_std': 2.0},
            {'bb_period': 14, 'bb_std': 1.8}
        ]
        
        # Mock engine that handles use_gpu parameter
        def gpu_aware_engine(df, params, use_gpu=False, **kwargs):
            # Simulate different behavior based on GPU flag
            multiplier = 1.1 if use_gpu else 1.0
            
            returns = pd.Series(
                np.random.randn(len(df)) * 0.01 * multiplier,
                index=df.index
            )
            trades = pd.DataFrame({
                'entry_time': [df.index[0]],
                'exit_time': [df.index[10]],
                'pnl': [100.0 * multiplier],
                'side': ['LONG']
            })
            return returns, trades
        
        # Test with GPU enabled
        try:
            results_gpu = run_grid(
                df=self.test_df,
                param_grid=param_grid,
                engine_func=gpu_aware_engine,
                symbol="GPUTEST",
                timeframe="15m",
                use_gpu=True,
                max_workers=2,
                seed=42
            )
            
            self.assertEqual(len(results_gpu), 2)
            self.assertTrue(all(results_gpu['success']))
            
        except Exception as e:
            self.logger.warning(f"GPU test failed (expected if no GPU): {e}")
        
        # Test with CPU fallback
        results_cpu = run_grid(
            df=self.test_df,
            param_grid=param_grid,
            engine_func=gpu_aware_engine,
            symbol="CPUTEST",
            timeframe="15m",
            use_gpu=False,
            max_workers=2,
            seed=42
        )
        
        self.assertEqual(len(results_cpu), 2)
        self.assertTrue(all(results_cpu['success']))
    
    def test_logging_configuration(self):
        """Test logging configuration and rotation setup."""
        # Test logger creation
        test_logger = get_logger("test.sweep.engine")
        self.assertIsNotNone(test_logger)
        
        # Test custom log directory
        custom_log_dir = self.test_dir / "custom_logs"
        custom_logger = get_logger(
            "test.custom",
            log_dir=custom_log_dir,
            level="DEBUG"
        )
        
        # Generate some log messages
        custom_logger.info("Test info message")
        custom_logger.debug("Test debug message")
        custom_logger.warning("Test warning message")
        
        # Check if log directory was created
        self.assertTrue(custom_log_dir.exists())
        
        # Test rotation configuration (simulate by checking handler settings)
        handlers = custom_logger.handlers
        file_handlers = [h for h in handlers if hasattr(h, 'maxBytes')]
        
        if file_handlers:
            file_handler = file_handlers[0]
            self.assertEqual(file_handler.maxBytes, 10 * 1024 * 1024)  # 10MB
            self.assertEqual(file_handler.backupCount, 5)
    
    def test_validation_functions(self):
        """Test parameter validation and utility functions."""
        # Test validate_param_grid
        valid_grid = [
            {'bb_period': 20, 'bb_std': 2.0},
            {'bb_period': 14, 'bb_std': 1.8}
        ]
        
        normalized = validate_param_grid(valid_grid)
        self.assertEqual(len(normalized), 2)
        self.assertIsInstance(normalized[0], dict)
        
        # Test with dataclass (mock)
        from dataclasses import dataclass
        
        @dataclass
        class TestParams:
            bb_period: int
            bb_std: float
        
        dataclass_grid = [
            TestParams(bb_period=20, bb_std=2.0),
            {'bb_period': 14, 'bb_std': 1.8}
        ]
        
        normalized_mixed = validate_param_grid(dataclass_grid)
        self.assertEqual(len(normalized_mixed), 2)
        self.assertTrue(all(isinstance(p, dict) for p in normalized_mixed))
        
        # Test invalid grid
        with self.assertRaises(ValueError):
            validate_param_grid([])  # Empty grid
        
        with self.assertRaises(ValueError):
            validate_param_grid([None])  # Invalid type
        
        # Test make_run_id determinism
        run_id1 = make_run_id(42, {'symbol': 'BTC', 'timeframe': '15m'})
        run_id2 = make_run_id(42, {'symbol': 'BTC', 'timeframe': '15m'})
        self.assertEqual(run_id1, run_id2, "Same inputs should produce same run_id")
        
        run_id3 = make_run_id(43, {'symbol': 'BTC', 'timeframe': '15m'})
        self.assertNotEqual(run_id1, run_id3, "Different seeds should produce different run_ids")
    
    def test_io_and_relative_paths(self):
        """Test I/O operations with relative paths."""
        # Test relative path handling
        relative_history = Path("test_runs.parquet")
        full_path = self.test_dir / relative_history
        
        results = pd.DataFrame({
            'run_id': ['io_test_001'],
            'task_id': ['task_io_001'],
            'symbol': ['IOTEST'],
            'success': [True],
            'sharpe': [1.5],
            'timestamp': [pd.Timestamp.now(tz='UTC')]
        })
        
        # Test append with relative path resolution
        append_run_history(results, full_path)
        self.assertTrue(full_path.exists())
        
        # Test load
        loaded = load_run_history(full_path)
        self.assertEqual(len(loaded), 1)
        
        # Test load non-existent file
        empty_df = load_run_history(self.test_dir / "nonexistent.parquet")
        self.assertTrue(empty_df.empty)
    
    def test_checkpointing_and_resume(self):
        """Test checkpoint/resume functionality."""
        checkpoint_path = self.test_dir / "checkpoint_test.parquet"
        
        param_grid = [
            {'bb_period': 20, 'bb_std': 2.0, 'id': 1}, 
            {'bb_period': 14, 'bb_std': 1.8, 'id': 2},
            {'bb_period': 26, 'bb_std': 2.2, 'id': 3}
        ]
        
        engine_func = self._create_mock_engine()
        
        # First run with checkpoint
        results1 = run_grid(
            df=self.test_df,
            param_grid=param_grid,
            engine_func=engine_func,
            symbol="CHECKPOINT",
            timeframe="15m",
            checkpoint_path=checkpoint_path,
            max_workers=2,
            seed=42
        )
        
        self.assertEqual(len(results1), 3)
        self.assertTrue(checkpoint_path.exists())
        
        # Simulate resuming from checkpoint by running again
        # (In real scenario, would be a different process/session)
        results2 = run_grid(
            df=self.test_df,
            param_grid=param_grid,
            engine_func=engine_func,
            symbol="CHECKPOINT",
            timeframe="15m",
            checkpoint_path=checkpoint_path,
            max_workers=2,
            seed=42
        )
        
        # Should get same results (all tasks already completed)
        self.assertEqual(len(results2), 3)
        pd.testing.assert_frame_equal(
            results1.sort_values('task_id').reset_index(drop=True),
            results2.sort_values('task_id').reset_index(drop=True)
        )


if __name__ == '__main__':
    # Set up test logging
    setup_logging_once()
    
    # Run tests
    unittest.main(verbosity=2)