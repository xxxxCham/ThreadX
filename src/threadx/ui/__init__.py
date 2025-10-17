"""
ThreadX UI Module - Dash Components & Layout
============================================

Module UI pour ThreadX Dashboard.
Expose les composants Dash réutilisables et le layout principal.

Exports:
    - create_layout: Fonction principale layout Dashboard

Usage:
    from threadx.ui.layout import create_layout
    app.layout = create_layout(bridge)

Structure:
    ui/
      ├─ layout.py         (Layout principal Dash - P4)
      ├─ callbacks.py      (P7 - Callbacks)
      └─ components/
          ├─ data_manager.py      (P5)
          ├─ indicators_panel.py  (P5)
          ├─ backtest_panel.py    (P6)
          └─ optimization_panel.py (P6)

Author: ThreadX Framework
Version: Prompt 4 - Layout Principal
"""

# Dash layout (P4)
from .layout import create_layout

__all__ = [
    "create_layout",
]

__version__ = "0.3.0"  # Dash-only integration
