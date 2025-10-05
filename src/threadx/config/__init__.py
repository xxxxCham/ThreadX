"""
ThreadX Configuration Package - Phase 1
Configuration and paths management for ThreadX framework.
"""

from pathlib import Path
from typing import Any, Dict, Union

from .settings import (
    Settings,
    TOMLConfigLoader,
    ConfigurationError,
    PathValidationError,
    load_settings,
    get_settings,
    print_config,
)
from .loaders import (
    TOMLConfigLoader as _DictLoader,
    ConfigurationError as _LoaderConfigurationError,
)


def load_config_dict(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a TOML configuration file and return its content as a dictionary."""

    resolved_path = Path(config_path)

    try:
        loader = _DictLoader(resolved_path)
    except _LoaderConfigurationError as exc:
        raise ConfigurationError(str(exc)) from exc

    config_data = loader.config_data

    if not isinstance(config_data, dict):
        raise ConfigurationError(
            f"Configuration file {resolved_path} did not return a dictionary."
        )

    return dict(config_data)


__version__ = "1.0.0"
__author__ = "ThreadX Team"

__all__ = [
    "Settings",
    "TOMLConfigLoader",
    "ConfigurationError",
    "PathValidationError",
    "load_settings",
    "get_settings",
    "print_config",
    "load_config_dict",
]