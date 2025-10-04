"""
ThreadX Configuration Package - Phase 1
Configuration and paths management for ThreadX framework.
"""

from .settings import (
    Settings,
    TOMLConfigLoader,
    ConfigurationError,
    PathValidationError,
    load_settings,
    get_settings,
    print_config
)

__version__ = "1.0.0"
__author__ = "ThreadX Team"

__all__ = [
    "Settings",
    "TOMLConfigLoader",
    "ConfigurationError", 
    "PathValidationError",
    "load_settings",
    "get_settings",
    "print_config"
]