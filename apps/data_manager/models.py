# apps/data_manager/models.py
"""Data models for data manager - minimal shims for tests."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DataQuality:
    """Data quality metrics."""

    score: float = 0.0
    issues: int = 0
    metadata: Optional[Dict[str, Any]] = None


__all__ = ["DataQuality"]
