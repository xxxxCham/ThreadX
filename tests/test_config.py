"""
Test Suite for ThreadX Configuration - Phase 1
Comprehensive tests for TOML-based configuration system.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import toml

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from threadx.config import (
    Settings,
    TOMLConfigLoader,
    ConfigurationError,
    PathValidationError,
    load_settings,
    get_settings,
    print_config
)


class TestTOMLConfigLoader(unittest.TestCase):
    """Test cases for TOMLConfigLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "paths": {
                "data_root": "./test_data",
                "raw_json": "{data_root}/raw/json",
                "processed": "{data_root}/processed"
            },
            "gpu": {
                "devices": ["5090", "2060"],
                "load_balance": {"5090": 0.75, "2060": 0.25},
                "memory_threshold": 0.8,
                "enable_gpu": True
            },
            "performance": {
                "target_tasks_per_min": 2500,
                "vectorization_batch_size": 10000,
                "cache_ttl_sec": 3600
            },
            "trading": {
                "supported_timeframes": ["1m", "5m", "1h"],
                "default_timeframe": "1h",
                "base_currency": "USDT"
            },
            "logging": {
                "level": "INFO"
            },
            "security": {
                "allow_absolute_paths": False
            }
        }
    
    def test_load_config_success(self):
        """Test successful configuration loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(self.test_config, f)
            temp_file = f.name
        
        try:
            loader = TOMLConfigLoader(temp_file)
            settings = loader.load_config()
            
            self.assertIsInstance(settings, Settings)
            self.assertEqual(settings.DATA_ROOT, "./test_data")
            self.assertEqual(settings.GPU_DEVICES, ["5090", "2060"])
            self.assertEqual(settings.TARGET_TASKS_PER_MIN, 2500)
            
        finally:
            os.unlink(temp_file)
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with missing file."""
        loader = TOMLConfigLoader("nonexistent.toml")
        
        with self.assertRaises(ConfigurationError) as cm:
            loader.load_config()
        
        self.assertIn("Configuration file not found", str(cm.exception))
    
    def test_load_config_invalid_toml(self):
        """Test configuration loading with invalid TOML syntax."""
        invalid_toml = "[paths\ndata_root = ./data"  # Missing closing bracket
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(invalid_toml)
            temp_file = f.name
        
        try:
            loader = TOMLConfigLoader(temp_file)
            
            with self.assertRaises(ConfigurationError) as cm:
                loader.load_config()
            
            self.assertIn("Invalid TOML syntax", str(cm.exception))
            
        finally:
            os.unlink(temp_file)
    
    def test_cli_overrides(self):
        """Test CLI argument overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(self.test_config, f)
            temp_file = f.name
        
        try:
            loader = TOMLConfigLoader(temp_file)
            cli_overrides = {
                "data_root": "./override_data",
                "log_level": "DEBUG"
            }
            
            settings = loader.load_config(cli_overrides)
            
            self.assertEqual(settings.DATA_ROOT, "./override_data")
            self.assertEqual(settings.LOG_LEVEL, "DEBUG")
            
        finally:
            os.unlink(temp_file)
    
    def test_path_validation_absolute_paths_disallowed(self):
        """Test path validation rejects absolute paths when not allowed."""
        config_with_absolute = self.test_config.copy()
        config_with_absolute["paths"]["data_root"] = "/absolute/path"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(config_with_absolute, f)
            temp_file = f.name
        
        try:
            loader = TOMLConfigLoader(temp_file)
            
            with self.assertRaises(PathValidationError) as cm:
                loader.load_config()
            
            self.assertIn("Absolute path not allowed", str(cm.exception))
            
        finally:
            os.unlink(temp_file)
    
    def test_path_template_resolution(self):
        """Test path template resolution with data_root."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(self.test_config, f)
            temp_file = f.name
        
        try:
            loader = TOMLConfigLoader(temp_file)
            settings = loader.load_config()
            
            # Check that path templates are resolved
            self.assertEqual(settings.RAW_JSON, "./test_data/raw/json")
            self.assertEqual(settings.PROCESSED, "./test_data/processed")
            
        finally:
            os.unlink(temp_file)
    
    def test_gpu_config_validation(self):
        """Test GPU configuration validation."""
        # Test invalid load balance (doesn't sum to 1.0)
        invalid_config = self.test_config.copy()
        invalid_config["gpu"]["load_balance"] = {"5090": 0.5, "2060": 0.3}  # Sum = 0.8
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(invalid_config, f)
            temp_file = f.name
        
        try:
            loader = TOMLConfigLoader(temp_file)
            
            with self.assertRaises(ConfigurationError) as cm:
                loader.load_config()
            
            self.assertIn("load balance ratios must sum to 1.0", str(cm.exception))
            
        finally:
            os.unlink(temp_file)
    
    def test_performance_config_validation(self):
        """Test performance configuration validation."""
        # Test negative value
        invalid_config = self.test_config.copy()
        invalid_config["performance"]["target_tasks_per_min"] = -1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(invalid_config, f)
            temp_file = f.name
        
        try:
            loader = TOMLConfigLoader(temp_file)
            
            with self.assertRaises(ConfigurationError) as cm:
                loader.load_config()
            
            self.assertIn("must be positive", str(cm.exception))
            
        finally:
            os.unlink(temp_file)


class TestSettings(unittest.TestCase):
    """Test cases for Settings dataclass."""
    
    def test_settings_creation(self):
        """Test Settings dataclass creation with defaults."""
        settings = Settings()
        
        # Test default values
        self.assertEqual(settings.DATA_ROOT, "./data")
        self.assertEqual(settings.GPU_DEVICES, ["5090", "2060"])
        self.assertEqual(settings.TARGET_TASKS_PER_MIN, 2500)
        self.assertTrue(settings.ENABLE_GPU)
        self.assertEqual(settings.SUPPORTED_TF, ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"))
    
    def test_settings_immutability(self):
        """Test Settings dataclass is frozen (immutable)."""
        settings = Settings()
        
        with self.assertRaises(AttributeError):
            settings.DATA_ROOT = "./new_data"


class TestConfigurationFunctions(unittest.TestCase):
    """Test cases for configuration utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "paths": {"data_root": "./test_data"},
            "gpu": {"devices": ["5090"], "load_balance": {"5090": 1.0}},
            "performance": {"target_tasks_per_min": 1000},
            "trading": {"supported_timeframes": ["1h"]}
        }
    
    def test_load_settings_with_file(self):
        """Test load_settings function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(self.test_config, f)
            temp_file = f.name
        
        try:
            settings = load_settings(temp_file, cli_args=[])
            
            self.assertIsInstance(settings, Settings)
            self.assertEqual(settings.DATA_ROOT, "./test_data")
            
        finally:
            os.unlink(temp_file)
    
    def test_load_settings_cli_args(self):
        """Test load_settings with CLI arguments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(self.test_config, f)
            temp_file = f.name
        
        try:
            cli_args = ["--data-root", "./cli_override", "--log-level", "DEBUG"]
            settings = load_settings(temp_file, cli_args)
            
            self.assertEqual(settings.DATA_ROOT, "./cli_override")
            self.assertEqual(settings.LOG_LEVEL, "DEBUG")
            
        finally:
            os.unlink(temp_file)
    
    def test_get_settings_singleton(self):
        """Test get_settings singleton behavior."""
        # Mock load_settings to avoid file dependency
        with patch('threadx.config.settings.load_settings') as mock_load:
            mock_settings = Settings(DATA_ROOT="./mock_data")
            mock_load.return_value = mock_settings
            
            # Clear global settings
            import threadx.config.settings as settings_module
            settings_module._settings = None
            
            # First call should load settings
            settings1 = get_settings()
            self.assertEqual(mock_load.call_count, 1)
            
            # Second call should reuse cached settings
            settings2 = get_settings()
            self.assertEqual(mock_load.call_count, 1)
            self.assertIs(settings1, settings2)
            
            # Force reload should reload
            settings3 = get_settings(force_reload=True)
            self.assertEqual(mock_load.call_count, 2)
    
    def test_print_config(self):
        """Test print_config function."""
        settings = Settings(DATA_ROOT="./test", GPU_DEVICES=["test_gpu"])
        
        # Capture stdout
        from io import StringIO
        import sys
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            print_config(settings)
            output = captured_output.getvalue()
            
            self.assertIn("ThreadX Configuration", output)
            self.assertIn("Data Root: ./test", output)
            self.assertIn("Devices: ['test_gpu']", output)
            
        finally:
            sys.stdout = sys.__stdout__


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration system."""
    
    def test_end_to_end_configuration_loading(self):
        """Test complete configuration loading workflow."""
        # Create a comprehensive test configuration
        full_config = {
            "paths": {
                "data_root": "./integration_test_data",
                "raw_json": "{data_root}/raw",
                "processed": "{data_root}/processed"
            },
            "gpu": {
                "devices": ["5090", "2060"],
                "load_balance": {"5090": 0.8, "2060": 0.2},
                "memory_threshold": 0.75,
                "enable_gpu": True
            },
            "performance": {
                "target_tasks_per_min": 3000,
                "vectorization_batch_size": 50000,
                "cache_ttl_sec": 7200,
                "max_workers": 8
            },
            "trading": {
                "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "default_timeframe": "4h",
                "base_currency": "BTC",
                "fee_rate": 0.0015
            },
            "backtesting": {
                "initial_capital": 50000.0,
                "max_positions": 20
            },
            "logging": {
                "level": "WARNING",
                "max_file_size_mb": 50
            },
            "security": {
                "read_only_data": False,
                "allow_absolute_paths": False
            },
            "monte_carlo": {
                "default_simulations": 50000,
                "seed": 123
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(full_config, f)
            temp_file = f.name
        
        try:
            # Test loading without CLI overrides
            settings = load_settings(temp_file, cli_args=[])
            
            # Verify all sections loaded correctly
            self.assertEqual(settings.DATA_ROOT, "./integration_test_data")
            self.assertEqual(settings.RAW_JSON, "./integration_test_data/raw")
            self.assertEqual(settings.GPU_DEVICES, ["5090", "2060"])
            self.assertEqual(settings.LOAD_BALANCE, {"5090": 0.8, "2060": 0.2})
            self.assertEqual(settings.TARGET_TASKS_PER_MIN, 3000)
            self.assertEqual(settings.BASE_CURRENCY, "BTC")
            self.assertEqual(settings.INITIAL_CAPITAL, 50000.0)
            self.assertEqual(settings.LOG_LEVEL, "WARNING")
            self.assertFalse(settings.READ_ONLY_DATA)
            self.assertEqual(settings.DEFAULT_SIMULATIONS, 50000)
            
            # Test with CLI overrides
            cli_args = ["--data-root", "./cli_override", "--enable-gpu"]
            settings_with_cli = load_settings(temp_file, cli_args)
            
            self.assertEqual(settings_with_cli.DATA_ROOT, "./cli_override")
            self.assertTrue(settings_with_cli.ENABLE_GPU)
            
        finally:
            os.unlink(temp_file)
    
    def test_configuration_error_handling(self):
        """Test error handling in configuration system."""
        # Test missing required sections
        incomplete_config = {
            "paths": {"data_root": "./test"}
            # Missing gpu, performance, trading sections
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(incomplete_config, f)
            temp_file = f.name
        
        try:
            with self.assertRaises(ConfigurationError) as cm:
                load_settings(temp_file, cli_args=[])
            
            self.assertIn("Missing required configuration section", str(cm.exception))
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)