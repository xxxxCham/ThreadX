# apps/data_manager/discovery/local_scanner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional
import os

__all__ = [
    "scan_local_data",
    "validate_dataset_entry",
    "LocalDataScanner",
    "create_demo_catalog",
]


def scan_local_data(
    paths: Optional[Iterable[str]] = None,
    patterns: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Shim minimal: retourne des entrées normalisées, éventuellement vide.
    """
    results: List[Dict[str, Any]] = []
    for p in paths or []:
        if os.path.exists(p):
            results.append({
                "path": os.path.abspath(p),
                "ok": True,
                "meta": {}
            })
    return results


def validate_dataset_entry(entry: Dict[str, Any]) -> bool:
    return isinstance(entry, dict) and "path" in entry


def create_demo_catalog() -> Dict[str, Any]:
    """Return a demo catalog for tests."""
    return {
        "datasets": [],
        "total": 0,
        "timestamp": "2025-10-12T00:00:00Z",
    }


@dataclass
class LocalDataScanner:
    paths: List[str]
    patterns: Optional[List[str]] = None

    def scan(self) -> List[Dict[str, Any]]:
        return scan_local_data(self.paths, self.patterns)

    @staticmethod
    def validate(entry: Dict[str, Any]) -> bool:
        return validate_dataset_entry(entry)
