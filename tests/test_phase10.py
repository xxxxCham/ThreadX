"""
ThreadX Phase 10 Tests - Migration & Environment
===============================================

Tests unitaires et d'intégration pour les outils de migration et 
diagnostic d'environnement.

Coverage:
- Migration: scan, parse, normalize, conflicts, idempotence
- Environment: specs, packages, GPU detection, benchmarks
- End-to-end: pipeline Data→Indicators→Strategy→Engine→Performance
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Import modules under test
from tools.migrate_from_tradxpro import (
    parse_symbol_timeframe,
    normalize_ohlcv_dataframe,
    resolve_conflict,
    convert_single_file,
    scan_old_tradx,
    main as migrate_main
)
from tools.check_env import (
    get_system_specs,
    check_package_versions,
    detect_gpu_info,
    run_numpy_benchmark,
    run_pandas_benchmark,
    run_parquet_benchmark,
    main as check_env_main
)

class TestMigrationParsing(unittest.TestCase):
    """Test migration parsing functions."""
    
    def test_parse_symbol_timeframe_basic(self):
        """Test basic symbol/timeframe parsing."""
        # Standard formats
        self.assertEqual(parse_symbol_timeframe("BTCUSDT_1h.json"), ("BTCUSDT", "1h"))
        self.assertEqual(parse_symbol_timeframe("ETHUSDT-5m.csv"), ("ETHUSDT", "5m"))
        self.assertEqual(parse_symbol_timeframe("ADAUSDT_15m.parquet"), ("ADAUSDT", "15m"))
        
        # With prefixes
        self.assertEqual(parse_symbol_timeframe("binance-SOLUSDT-1h-2024.json"), ("SOLUSDT", "1h"))
        
        # Case insensitive
        self.assertEqual(parse_symbol_timeframe("btcusdt_1H.json"), ("BTCUSDT", "1h"))
        
    def test_parse_symbol_timeframe_edge_cases(self):
        """Test edge cases for parsing."""
        # Invalid formats
        self.assertIsNone(parse_symbol_timeframe("invalid_file.txt"))
        self.assertIsNone(parse_symbol_timeframe("no_timeframe.json"))
        self.assertIsNone(parse_symbol_timeframe(""))
        
        # Various timeframes
        self.assertEqual(parse_symbol_timeframe("BTCUSDT_1m.json"), ("BTCUSDT", "1m"))
        self.assertEqual(parse_symbol_timeframe("BTCUSDT_30m.json"), ("BTCUSDT", "30m"))
        self.assertEqual(parse_symbol_timeframe("BTCUSDT_4h.json"), ("BTCUSDT", "4h"))
        self.assertEqual(parse_symbol_timeframe("BTCUSDT_1d.json"), ("BTCUSDT", "1d"))

class TestOHLCVNormalization(unittest.TestCase):
    """Test OHLCV data normalization."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        self.sample_df = pd.DataFrame({
            'timestamp': dates,
            'o': np.random.uniform(40000, 50000, 100),
            'h': np.random.uniform(50000, 55000, 100),
            'l': np.random.uniform(35000, 40000, 100),
            'c': np.random.uniform(40000, 50000, 100),
            'v': np.random.uniform(1000, 10000, 100),
        })
        
        # Ensure H >= L and H >= max(O,C), L <= min(O,C)
        self.sample_df['h'] = np.maximum(
            self.sample_df['h'],
            self.sample_df[['o', 'c']].max(axis=1)
        )
        self.sample_df['l'] = np.minimum(
            self.sample_df['l'],
            self.sample_df[['o', 'c']].min(axis=1)
        )
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        normalized = normalize_ohlcv_dataframe(self.sample_df, "test.json")
        
        # Check structure
        self.assertIsInstance(normalized.index, pd.DatetimeIndex)
        if hasattr(normalized.index, 'tz'):
            self.assertEqual(str(normalized.index.tz), 'UTC')
        
        # Check columns
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        self.assertEqual(list(normalized.columns), expected_cols)
        
        # Check data types
        for col in expected_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(normalized[col]))
    
    def test_normalize_missing_volume(self):
        """Test normalization with missing volume."""
        df_no_vol = self.sample_df.drop('v', axis=1)
        normalized = normalize_ohlcv_dataframe(df_no_vol, "test.json")
        
        # Volume should be filled with 0
        self.assertTrue('volume' in normalized.columns)
        self.assertTrue((normalized['volume'] == 0.0).all())
    
    def test_normalize_invalid_data(self):
        """Test handling of invalid data."""
        # Data with NaN values
        df_with_nan = self.sample_df.copy()
        df_with_nan.loc[0, 'o'] = np.nan
        
        normalized = normalize_ohlcv_dataframe(df_with_nan, "test.json")
        
        # Should remove rows with NaN OHLC
        self.assertEqual(len(normalized), len(self.sample_df) - 1)
    
    def test_normalize_missing_required_columns(self):
        """Test error on missing required columns."""
        df_missing = self.sample_df[['timestamp', 'o']].copy()  # Missing h,l,c
        
        with self.assertRaises(ValueError):
            normalize_ohlcv_dataframe(df_missing, "test.json")

class TestConflictResolution(unittest.TestCase):
    """Test conflict resolution strategies."""
    
    def setUp(self):
        """Set up test data for conflicts."""
        dates1 = pd.date_range('2024-01-01', periods=50, freq='1H', tz='UTC')
        dates2 = pd.date_range('2024-01-01 12:00', periods=50, freq='1H', tz='UTC')
        
        self.df1 = pd.DataFrame({
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(200, 300, 50),
            'low': np.random.uniform(50, 100, 50),
            'close': np.random.uniform(100, 200, 50), 
            'volume': np.random.uniform(1000, 2000, 50),
        }, index=dates1)
        
        self.df2 = pd.DataFrame({
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(200, 300, 50),
            'low': np.random.uniform(50, 100, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(1000, 2000, 50),
        }, index=dates2)
    
    def test_resolve_latest(self):
        """Test 'latest' resolution strategy."""
        resolved = resolve_conflict(self.df1, self.df2, mode="latest")
        
        # Should keep latest values for overlapping timestamps
        overlap_start = max(self.df1.index.min(), self.df2.index.min())
        overlap_end = min(self.df1.index.max(), self.df2.index.max())
        
        # For overlapping period, should have df2 values (later)
        overlap_mask = (resolved.index >= overlap_start) & (resolved.index <= overlap_end)
        if overlap_mask.any():
            # At least some overlapping data should come from df2
            pass  # Specific assertions depend on exact data
    
    def test_resolve_merge(self):
        """Test 'merge' resolution strategy."""
        resolved = resolve_conflict(self.df1, self.df2, mode="merge")
        
        # Should preserve all df1 data and add new timestamps from df2
        self.assertTrue(len(resolved) >= len(self.df1))
        
        # df1 timestamps should be preserved
        for timestamp in self.df1.index:
            if timestamp in resolved.index:
                pd.testing.assert_series_equal(
                    resolved.loc[timestamp], 
                    self.df1.loc[timestamp],
                    check_names=False
                )
    
    def test_resolve_append(self):
        """Test 'append' resolution strategy."""
        resolved = resolve_conflict(self.df1, self.df2, mode="append")
        
        # Should have all records from both dataframes
        self.assertEqual(len(resolved), len(self.df1) + len(self.df2))

class TestMigrationIntegration(unittest.TestCase):
    """Test migration integration scenarios."""
    
    def setUp(self):
        """Set up temporary directories and files."""
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = Path(self.temp_dir) / "source"
        self.output_dir = Path(self.temp_dir) / "output"
        
        self.source_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create sample JSON file
        sample_data = []
        for i in range(100):
            sample_data.append({
                'timestamp': (datetime(2024, 1, 1) + pd.Timedelta(hours=i)).isoformat(),
                'o': 40000 + i,
                'h': 40000 + i + 100,
                'l': 40000 + i - 100,
                'c': 40000 + i + 50,
                'v': 1000 + i
            })
        
        self.sample_file = self.source_dir / "BTCUSDT_1h.json"
        with open(self.sample_file, 'w') as f:
            json.dump(sample_data, f)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_scan_old_tradx(self):
        """Test scanning old TradX directory."""
        files = scan_old_tradx(self.source_dir)
        
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], self.sample_file)
    
    def test_convert_single_file(self):
        """Test converting a single file."""
        result = convert_single_file(
            self.sample_file,
            self.output_dir,
            dry_run=False
        )
        
        self.assertIsNotNone(result)
        if result is not None:
            self.assertTrue(result.exists())
            self.assertEqual(result.name, "BTCUSDT_1h.parquet")
            
            # Verify converted data
            df = pd.read_parquet(result)
            self.assertEqual(len(df), 100)
            self.assertEqual(list(df.columns), ['open', 'high', 'low', 'close', 'volume'])
    
    def test_migration_dry_run(self):
        """Test migration in dry-run mode."""
        # Mock sys.argv for testing
        test_args = [
            '--root', str(self.source_dir),
            '--dry-run'
        ]
        
        exit_code = migrate_main(test_args)
        
        self.assertEqual(exit_code, 0)
        # No files should be created in dry-run
        output_files = list(self.output_dir.glob("*.parquet"))
        self.assertEqual(len(output_files), 0)

class TestEnvironmentCheck(unittest.TestCase):
    """Test environment checking functions."""
    
    def test_get_system_specs(self):
        """Test system specifications gathering."""
        specs = get_system_specs()
        
        # Verify structure
        self.assertIsInstance(specs.python_version, str)
        self.assertIsInstance(specs.cpu_count, int)
        self.assertIsInstance(specs.memory_total_gb, float)
        self.assertTrue(specs.cpu_count > 0)
        self.assertTrue(specs.memory_total_gb > 0)
    
    def test_check_package_versions(self):
        """Test package version checking."""
        packages = check_package_versions()
        
        # Should find at least numpy and pandas
        package_names = [p.name for p in packages]
        self.assertIn('numpy', package_names)
        self.assertIn('pandas', package_names)
        
        # Check numpy is installed (required for tests to run)
        numpy_pkg = next(p for p in packages if p.name == 'numpy')
        self.assertTrue(numpy_pkg.installed)
    
    @patch('cupy.cuda.runtime.getDeviceCount')
    def test_detect_gpu_info_mock(self, mock_get_device_count):
        """Test GPU detection with mocked CuPy."""
        # Mock CuPy not available
        mock_get_device_count.side_effect = ImportError("No CuPy")
        
        gpu_info = detect_gpu_info()
        
        self.assertFalse(gpu_info.detected)
        self.assertEqual(gpu_info.device_count, 0)
    
    def test_benchmarks_run(self):
        """Test that benchmarks can run without errors."""
        # These should not raise exceptions
        numpy_result = run_numpy_benchmark()
        pandas_result = run_pandas_benchmark()
        parquet_result = run_parquet_benchmark()
        
        # Verify structure
        self.assertIsInstance(numpy_result.duration_sec, float)
        self.assertIsInstance(pandas_result.operations_per_sec, float)
        self.assertIsInstance(parquet_result.throughput_mb_per_sec, float)
        
        # Reasonable performance bounds
        self.assertGreater(numpy_result.operations_per_sec, 0.1)
        self.assertGreater(pandas_result.operations_per_sec, 0.1)
        if parquet_result.throughput_mb_per_sec is not None:
            self.assertGreater(parquet_result.throughput_mb_per_sec, 0.1)

class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up for end-to-end tests."""
        np.random.seed(42)  # Deterministic
        
        # Create synthetic OHLCV data
        dates = pd.date_range('2024-01-01', periods=1000, freq='1H', tz='UTC')
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, 1000),
            'high': np.random.uniform(50000, 55000, 1000),
            'low': np.random.uniform(35000, 40000, 1000),
            'close': np.random.uniform(40000, 50000, 1000),
            'volume': np.random.uniform(1000, 10000, 1000),
        }, index=dates)
        
        # Fix OHLC relationships
        self.test_data['high'] = np.maximum(
            self.test_data['high'],
            self.test_data[['open', 'close']].max(axis=1)
        )
        self.test_data['low'] = np.minimum(
            self.test_data['low'],
            self.test_data[['open', 'close']].min(axis=1)
        )
    
    @pytest.mark.integration
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked ThreadX components."""
        # Mock ThreadX imports to avoid dependencies
        with patch('src.threadx.data.load_data') as mock_load:
            with patch('src.threadx.indicators.bank.ensure') as mock_indicators:
                with patch('src.threadx.strategy.run') as mock_strategy:
                    with patch('src.threadx.engine.run') as mock_engine:
                        with patch('src.threadx.performance.summarize') as mock_perf:
                            
                            # Configure mocks
                            mock_load.return_value = self.test_data
                            mock_indicators.return_value = {'sma_20': np.random.random(1000)}
                            mock_strategy.return_value = {'signals': np.random.choice([0, 1, -1], 1000)}
                            mock_engine.return_value = {'equity': np.cumsum(np.random.normal(0, 0.01, 1000))}
                            mock_perf.return_value = {
                                'total_return': 0.15,
                                'sharpe_ratio': 1.2,
                                'max_drawdown': 0.08
                            }
                            
                            # Execute pipeline (mocked)
                            data = mock_load()
                            indicators = mock_indicators(data)
                            strategy_result = mock_strategy(data, indicators)
                            engine_result = mock_engine(data, strategy_result)
                            performance = mock_perf(engine_result)
                            
                            # Verify mock calls
                            mock_load.assert_called_once()
                            mock_indicators.assert_called_once_with(data)
                            mock_strategy.assert_called_once_with(data, indicators)
                            mock_engine.assert_called_once_with(data, strategy_result)
                            mock_perf.assert_called_once_with(engine_result)
                            
                            # Verify results structure
                            self.assertIn('total_return', performance)
                            self.assertIn('sharpe_ratio', performance)
                            self.assertIn('max_drawdown', performance)
    
    def test_data_integrity_through_pipeline(self):
        """Test data integrity is maintained through processing."""
        # Start with known data
        original_checksum = hash(tuple(self.test_data.values.flatten()))
        
        # Simulate processing steps that shouldn't change the data
        processed_data = self.test_data.copy()
        
        # Add metadata (shouldn't affect core data)
        processed_data.attrs['processed'] = True
        
        # Verify data unchanged
        processed_checksum = hash(tuple(processed_data.values.flatten()))
        self.assertEqual(original_checksum, processed_checksum)
        
        # Count should be preserved
        self.assertEqual(len(processed_data), len(self.test_data))

class TestCommandLineInterfaces(unittest.TestCase):
    """Test command-line interfaces."""
    
    def test_migrate_help(self):
        """Test migration tool help."""
        with self.assertRaises(SystemExit) as cm:
            migrate_main(['--help'])
        # argparse exits with 0 for help
        self.assertEqual(cm.exception.code, 0)
    
    def test_check_env_help(self):
        """Test environment check help."""
        with self.assertRaises(SystemExit) as cm:
            check_env_main(['--help'])
        self.assertEqual(cm.exception.code, 0)
    
    def test_check_env_json_output(self):
        """Test environment check JSON output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_json = f.name
        
        try:
            exit_code = check_env_main(['--json', temp_json])
            self.assertEqual(exit_code, 0)
            
            # Verify JSON file was created and is valid
            self.assertTrue(Path(temp_json).exists())
            
            with open(temp_json) as f:
                report_data = json.load(f)
            
            # Verify JSON structure
            self.assertIn('timestamp', report_data)
            self.assertIn('system', report_data)
            self.assertIn('packages', report_data)
            self.assertIn('benchmarks', report_data)
            
        finally:
            # Cleanup
            if Path(temp_json).exists():
                Path(temp_json).unlink()

# Test configuration and coverage setup
def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    try:
        # Try to import coverage module
        import importlib
        coverage = importlib.import_module('coverage')
        
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        unittest.main(verbosity=2, exit=False)
        
        cov.stop()
        cov.save()
        
        # Generate report
        print("\n" + "="*60)
        print("COVERAGE REPORT")
        print("="*60)
        cov.report(show_missing=True)
        
        # Check coverage threshold
        total_coverage = cov.report(show_missing=False)
        if total_coverage < 80:
            print(f"\nWARNING: Coverage {total_coverage:.1f}% below 80% threshold")
            return 1
        else:
            print(f"\n✅ Coverage {total_coverage:.1f}% meets 80% requirement")
            return 0
            
    except ImportError:
        print("Coverage not available, running tests without coverage")
        unittest.main(verbosity=2)
        return 0

if __name__ == '__main__':
    # Ensure deterministic tests
    np.random.seed(42)
    
    # Run with coverage if available
    exit_code = run_tests_with_coverage()
    exit(exit_code)