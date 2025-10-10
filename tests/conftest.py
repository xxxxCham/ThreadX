"""Configuration de test commune pour assurer les imports absolus."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / "src"
if _SRC_PATH.is_dir():
    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
