from __future__ import annotations

"""Local data discovery shim for apps.data_manager

Provides a LocalDataScanner dataclass with a scan() method. Tests import
the symbol directly; the implementation here is intentionally minimal and
deterministic.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class LocalDataScanner:
    """Minimal scanner used by tests.

    Attributes:
        root_path: optional base path to scan (not used for real IO here)
    """

    root_path: Optional[str] = None

    def scan(self) -> List[Dict[str, Any]]:
        """Return a deterministic list of discovered dataset metadata.

        The real project performs filesystem inspection; for tests we return
        a stable fixture-like structure.
        """

        return [
            {
                "name": "example_dataset",
                "path": self.root_path or "./data/example.parquet",
                "metadata": {"rows": 1024},
            },
        ]

    @staticmethod
    def validate(entry: Dict[str, Any]) -> bool:
        return bool(entry.get("name") and entry.get("path"))


__all__ = ["LocalDataScanner"]


def create_demo_catalog(root_path: str | None = None) -> dict:
    """Return a small demo catalog used by tests."""

    scanner = LocalDataScanner(root_path)
    entries = scanner.scan()
    return {e["name"]: e for e in entries}


__all__.append("create_demo_catalog")
