from __future__ import annotations
from typing import List, Dict, Any


def scan_local_data(*args, **kwargs) -> List[Dict[str, Any]]:
    # Shim minimal: retourne une liste vide au bon format
    return []


def validate_dataset_entry(entry: Dict[str, Any]) -> bool:
    return isinstance(entry, dict)
