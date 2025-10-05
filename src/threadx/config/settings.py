"""
ThreadX Configuration Module - Phase 1
Settings and configuration management with TOML-only approach.
No environment variables used - pure TOML configuration.
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import toml


@dataclass(frozen=True)
    # Override config file if specified
    if args and args.config:
        config_file = args.config
    
    # Prepare CLI overrides
    cli_overrides = {}
    if args:
        if args.data_root:
            cli_overrides["data_root"] = args.data_root
        if args.log_level:
            cli_overrides["log_level"] = args.log_level
        if args.enable_gpu:
            cli_overrides["gpu_enabled"] = True
        if args.disable_gpu:
            cli_overrides["gpu_enabled"] = False:
    """
    Centralized settings dataclass for ThreadX.
    All configuration loaded from TOML, no environment variables.
    """
    
    # Paths Configuration
    DATA_ROOT: str = "./data"
    RAW_JSON: str = "{data_root}/raw/json"
    PROCESSED: str = "{data_root}/processed"
    INDICATORS: str = "{data_root}/indicators"
    RUNS: str = "{data_root}/runs"
    LOGS: str = "./logs"
    CACHE: str = "./cache"
    CONFIG: str = "./config"
    
    # GPU Configuration
    GPU_DEVICES: List[str] = field(default_factory=lambda: ["5090", "2060"])
    LOAD_BALANCE: Dict[str, float] = field(default_factory=lambda: {"5090": 0.75, "2060": 0.25})
    MEMORY_THRESHOLD: float = 0.8
    AUTO_FALLBACK: bool = True
    ENABLE_GPU: bool = True
    
    # Performance Configuration
    TARGET_TASKS_PER_MIN: int = 2500
    VECTORIZATION_BATCH_SIZE: int = 10000
    CACHE_TTL_SEC: int = 3600
    MAX_WORKERS: int = 4
    MEMORY_LIMIT_MB: int = 8192
    
    # Trading Configuration
    SUPPORTED_TF: tuple = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")
    DEFAULT_TIMEFRAME: str = "1h"
    BASE_CURRENCY: str = "USDT"
    FEE_RATE: float = 0.001
    SLIPPAGE_RATE: float = 0.0005
    
    # Backtesting Configuration
    INITIAL_CAPITAL: float = 10000.0
    MAX_POSITIONS: int = 10
    POSITION_SIZE: float = 0.1
    STOP_LOSS: float = 0.02
    TAKE_PROFIT: float = 0.04
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 100
    MAX_FILES: int = 10
    LOG_ROTATE: bool = True
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Configuration
    READ_ONLY_DATA: bool = True
    VALIDATE_PATHS: bool = True
    ALLOW_ABSOLUTE_PATHS: bool = False
    MAX_FILE_SIZE_MB: int = 1000
    
    # Monte Carlo Configuration
    DEFAULT_SIMULATIONS: int = 10000
    MAX_SIMULATIONS: int = 1000000
    DEFAULT_STEPS: int = 252
    MC_SEED: int = 42
    CONFIDENCE_LEVELS: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Cache Configuration
    CACHE_ENABLE: bool = True
    CACHE_MAX_SIZE_MB: int = 2048
    CACHE_TTL_SECONDS: int = 3600
    CACHE_COMPRESSION: bool = True
    CACHE_STRATEGY: str = "LRU"


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class PathValidationError(Exception):
    """Exception raised for path validation errors."""
    pass


class TOMLConfigLoader:
    """
    TOML Configuration Loader for ThreadX.
    Handles loading, validation, and CLI overrides.
    """
    
    def __init__(self, config_file: str = "paths.toml"):
        self.config_file = Path(config_file)
        self.config_data: Dict[str, Any] = {}
        self._validated_paths: Dict[str, str] = {}
        
    def load_config(self, cli_overrides: Optional[Dict[str, Any]] = None) -> Settings:
        """
        Load configuration from TOML file with optional CLI overrides.
        
        Args:
            cli_overrides: Dictionary of CLI argument overrides
            
        Returns:
            Settings dataclass instance
            
        Raises:
            ConfigurationError: If configuration loading fails
            PathValidationError: If path validation fails
        """
        try:
            # Load TOML file
            if self.config_file.exists():
                self.config_data = toml.load(self.config_file)
            else:
                raise ConfigurationError(f"Configuration file not found: {self.config_file}")
            
            # Apply CLI overrides
            if cli_overrides:
                self._apply_cli_overrides(cli_overrides)
            
            # Validate configuration
            self._validate_config()
            
            # Create Settings instance
            return self._create_settings()
            
        except toml.TomlDecodeError as e:
            raise ConfigurationError(f"Invalid TOML syntax in {self.config_file}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _apply_cli_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply CLI argument overrides to configuration."""
        for key, value in overrides.items():
            if key == "data_root":
                self.config_data.setdefault("paths", {})["data_root"] = value
            elif key == "log_level":
                self.config_data.setdefault("logging", {})["level"] = value
            elif key == "enable_gpu":
                self.config_data.setdefault("gpu", {})["enable_gpu"] = value
            # Add more CLI overrides as needed
    
    def _validate_config(self) -> None:
        """Validate configuration data and paths."""
        # Validate required sections
        required_sections = ["paths", "gpu", "performance", "trading"]
        for section in required_sections:
            if section not in self.config_data:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate paths
        self._validate_paths()
        
        # Validate GPU configuration
        self._validate_gpu_config()
        
        # Validate performance settings
        self._validate_performance_config()
    
    def _validate_paths(self) -> None:
        """Validate path configuration and resolve path templates."""
        paths_config = self.config_data.get("paths", {})
        security_config = self.config_data.get("security", {})
        
        # Check for absolute paths if not allowed
        if not security_config.get("allow_absolute_paths", False):
            for key, path in paths_config.items():
                if isinstance(path, str) and os.path.isabs(path):
                    raise PathValidationError(f"Absolute path not allowed for {key}: {path}")
        
        # Resolve path templates
        data_root = paths_config.get("data_root", "./data")
        self._validated_paths["data_root"] = data_root
        
        for key, path_template in paths_config.items():
            if isinstance(path_template, str) and "{data_root}" in path_template:
                resolved_path = path_template.format(data_root=data_root)
                self._validated_paths[key] = resolved_path
            else:
                self._validated_paths[key] = path_template
    
    def _validate_gpu_config(self) -> None:
        """Validate GPU configuration."""
        gpu_config = self.config_data.get("gpu", {})
        
        # Validate load balance ratios sum to ~1.0
        load_balance = gpu_config.get("load_balance", {})
        if load_balance:
            total_balance = sum(load_balance.values())
            if not (0.99 <= total_balance <= 1.01):  # Allow small floating point errors
                raise ConfigurationError(f"GPU load balance ratios must sum to 1.0, got {total_balance}")
        
        # Validate memory threshold
        memory_threshold = gpu_config.get("memory_threshold", 0.8)
        if not (0.1 <= memory_threshold <= 1.0):
            raise ConfigurationError(f"GPU memory threshold must be between 0.1 and 1.0, got {memory_threshold}")
    
    def _validate_performance_config(self) -> None:
        """Validate performance configuration."""
        perf_config = self.config_data.get("performance", {})
        
        # Validate positive values
        positive_values = ["target_tasks_per_min", "vectorization_batch_size", "cache_ttl_sec", "max_workers"]
        for key in positive_values:
            value = perf_config.get(key, 1)
            if value <= 0:
                raise ConfigurationError(f"Performance setting {key} must be positive, got {value}")
    
    def _create_settings(self) -> Settings:
        """Create Settings dataclass from loaded configuration."""
        paths = self.config_data.get("paths", {})
        gpu = self.config_data.get("gpu", {})
        performance = self.config_data.get("performance", {})
        trading = self.config_data.get("trading", {})
        backtesting = self.config_data.get("backtesting", {})
        logging = self.config_data.get("logging", {})
        security = self.config_data.get("security", {})
        monte_carlo = self.config_data.get("monte_carlo", {})
        cache = self.config_data.get("cache", {})
        
        return Settings(
            # Paths - use validated paths
            DATA_ROOT=self._validated_paths.get("data_root", "./data"),
            RAW_JSON=self._validated_paths.get("raw_json", "{data_root}/raw/json"),
            PROCESSED=self._validated_paths.get("processed", "{data_root}/processed"),
            INDICATORS=self._validated_paths.get("indicators", "{data_root}/indicators"),
            RUNS=self._validated_paths.get("runs", "{data_root}/runs"),
            LOGS=self._validated_paths.get("logs", "./logs"),
            CACHE=self._validated_paths.get("cache", "./cache"),
            CONFIG=self._validated_paths.get("config", "./config"),
            
            # GPU
            GPU_DEVICES=gpu.get("devices", ["5090", "2060"]),
            LOAD_BALANCE=gpu.get("load_balance", {"5090": 0.75, "2060": 0.25}),
            MEMORY_THRESHOLD=gpu.get("memory_threshold", 0.8),
            AUTO_FALLBACK=gpu.get("auto_fallback", True),
            ENABLE_GPU=gpu.get("enable_gpu", True),
            
            # Performance
            TARGET_TASKS_PER_MIN=performance.get("target_tasks_per_min", 2500),
            VECTORIZATION_BATCH_SIZE=performance.get("vectorization_batch_size", 10000),
            CACHE_TTL_SEC=performance.get("cache_ttl_sec", 3600),
            MAX_WORKERS=performance.get("max_workers", 4),
            MEMORY_LIMIT_MB=performance.get("memory_limit_mb", 8192),
            
            # Trading
            SUPPORTED_TF=tuple(trading.get("supported_timeframes", ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"])),
            DEFAULT_TIMEFRAME=trading.get("default_timeframe", "1h"),
            BASE_CURRENCY=trading.get("base_currency", "USDT"),
            FEE_RATE=trading.get("fee_rate", 0.001),
            SLIPPAGE_RATE=trading.get("slippage_rate", 0.0005),
            
            # Backtesting
            INITIAL_CAPITAL=backtesting.get("initial_capital", 10000.0),
            MAX_POSITIONS=backtesting.get("max_positions", 10),
            POSITION_SIZE=backtesting.get("position_size", 0.1),
            STOP_LOSS=backtesting.get("stop_loss", 0.02),
            TAKE_PROFIT=backtesting.get("take_profit", 0.04),
            
            # Logging
            LOG_LEVEL=logging.get("level", "INFO"),
            MAX_FILE_SIZE_MB=logging.get("max_file_size_mb", 100),
            MAX_FILES=logging.get("max_files", 10),
            LOG_ROTATE=logging.get("rotate", True),
            LOG_FORMAT=logging.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            
            # Security
            READ_ONLY_DATA=security.get("read_only_data", True),
            VALIDATE_PATHS=security.get("validate_paths", True),
            ALLOW_ABSOLUTE_PATHS=security.get("allow_absolute_paths", False),
            
            # Monte Carlo
            DEFAULT_SIMULATIONS=monte_carlo.get("default_simulations", 10000),
            MAX_SIMULATIONS=monte_carlo.get("max_simulations", 1000000),
            DEFAULT_STEPS=monte_carlo.get("default_steps", 252),
            MC_SEED=monte_carlo.get("seed", 42),
            CONFIDENCE_LEVELS=monte_carlo.get("confidence_levels", [0.95, 0.99]),
            
            # Cache
            CACHE_ENABLE=cache.get("enable", True),
            CACHE_MAX_SIZE_MB=cache.get("max_size_mb", 2048),
            CACHE_TTL_SECONDS=cache.get("ttl_seconds", 3600),
            CACHE_COMPRESSION=cache.get("compression", True),
            CACHE_STRATEGY=cache.get("strategy", "LRU")
        )


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for configuration overrides."""
    parser = argparse.ArgumentParser(
        description="ThreadX Configuration Management",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="paths.toml",
        help="Path to TOML configuration file (default: paths.toml)"
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        help="Override data root path"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override log level"
    )
    
    parser.add_argument(
        "--enable-gpu",
        action="store_true",
        help="Enable GPU acceleration"
    )
    
    parser.add_argument(
        "--disable-gpu",
        action="store_true", 
        help="Disable GPU acceleration"
    )
    
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print loaded configuration and exit"
    )
    
    return parser


def load_settings(config_file: str = "paths.toml", cli_args: Optional[List[str]] = None) -> Settings:
    """
    Main function to load ThreadX settings.
    
    Args:
        config_file: Path to TOML configuration file
        cli_args: CLI arguments for testing (if None, uses sys.argv)
        
    Returns:
        Settings dataclass instance
        
    Raises:
        ConfigurationError: If configuration loading fails
    """
    # Parse CLI arguments seulement si explicitement demandé
    args = None
    if cli_args is not None:
        parser = create_cli_parser()
        args = parser.parse_args(cli_args)
    elif '--config' in sys.argv or '--enable-gpu' in sys.argv or '--disable-gpu' in sys.argv:
        # Parse seulement si des arguments ThreadX sont présents
        parser = create_cli_parser()
        try:
            args = parser.parse_args()
        except SystemExit:
            # Ignore les erreurs de parsing (ex: args PyTest)
            args = None
    
    # Override config file if specified
    if args and args.config:
        config_file = args.config
    
    # Prepare CLI overrides
    cli_overrides = {}
    if args:
        if args.data_root:
            cli_overrides["data_root"] = args.data_root
        if args.log_level:
            cli_overrides["log_level"] = args.log_level
        if args.enable_gpu:
            cli_overrides["enable_gpu"] = True
        if args.disable_gpu:
            cli_overrides["enable_gpu"] = False
    
    # Load configuration
    loader = TOMLConfigLoader(config_file)
    settings = loader.load_config(cli_overrides)
    
    # Print configuration if requested
    if args and args.print_config:
        print_config(settings)
        sys.exit(0)
    
    return settings


def print_config(settings: Settings) -> None:
    """Print current configuration in a readable format."""
    print("ThreadX Configuration")
    print("=" * 50)
    
    # Paths
    print("\n[PATHS]")
    print(f"Data Root: {settings.DATA_ROOT}")
    print(f"Raw JSON: {settings.RAW_JSON}")
    print(f"Processed: {settings.PROCESSED}")
    print(f"Indicators: {settings.INDICATORS}")
    print(f"Runs: {settings.RUNS}")
    print(f"Logs: {settings.LOGS}")
    print(f"Cache: {settings.CACHE}")
    
    # GPU
    print("\n[GPU]")
    print(f"Devices: {settings.GPU_DEVICES}")
    print(f"Load Balance: {settings.LOAD_BALANCE}")
    print(f"Memory Threshold: {settings.MEMORY_THRESHOLD}")
    print(f"Auto Fallback: {settings.AUTO_FALLBACK}")
    print(f"GPU Enabled: {settings.ENABLE_GPU}")
    
    # Performance
    print("\n[PERFORMANCE]")
    print(f"Target Tasks/Min: {settings.TARGET_TASKS_PER_MIN}")
    print(f"Batch Size: {settings.VECTORIZATION_BATCH_SIZE}")
    print(f"Cache TTL: {settings.CACHE_TTL_SEC}s")
    print(f"Max Workers: {settings.MAX_WORKERS}")
    
    # Trading
    print("\n[TRADING]")
    print(f"Supported Timeframes: {settings.SUPPORTED_TF}")
    print(f"Default Timeframe: {settings.DEFAULT_TIMEFRAME}")
    print(f"Base Currency: {settings.BASE_CURRENCY}")
    print(f"Fee Rate: {settings.FEE_RATE}")
    
    # Security
    print("\n[SECURITY]")
    print(f"Read Only Data: {settings.READ_ONLY_DATA}")
    print(f"Validate Paths: {settings.VALIDATE_PATHS}")
    print(f"Allow Absolute Paths: {settings.ALLOW_ABSOLUTE_PATHS}")


# Global settings instance (loaded lazily)
_settings: Optional[Settings] = None


def get_settings(force_reload: bool = False) -> Settings:
    """
    Get global settings instance (singleton pattern).
    
    Args:
        force_reload: Force reload configuration from file
        
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None or force_reload:
        _settings = load_settings()
    
    return _settings


# Export main classes and functions
__all__ = [
    "Settings",
    "TOMLConfigLoader", 
    "ConfigurationError",
    "PathValidationError",
    "load_settings",
    "get_settings",
    "print_config",
    "create_cli_parser"
]


if __name__ == "__main__":
    # CLI interface for configuration management
    try:
        settings = load_settings()
        print("✅ Configuration loaded successfully!")
        print_config(settings)
    except (ConfigurationError, PathValidationError) as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
