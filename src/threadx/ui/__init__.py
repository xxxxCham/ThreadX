"""
ThreadX UI Module - Dash Components & Layout
============================================

Module UI pour ThreadX Dashboard.
Expose les composants Dash réutilisables et le layout principal.

Exports:
    - create_layout: Fonction principale layout Dashboard
    - Legacy Tkinter/Streamlit (deprecated, voir apps/)

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

# Legacy components (deprecated, kept for compatibility)
try:
    from .app import ThreadXApp, run_app
    from .charts import plot_equity, plot_drawdown, altair_equity
    from .tables import (
        export_table,
        render_metrics_table,
        render_trades_table,
    )
except ImportError:
    # Legacy modules optionnels
    pass

__all__ = [
    # Dash (P4)
    "create_layout",
    # Legacy (compatibility)
    "ThreadXApp",
    "run_app",
    "plot_equity",
    "plot_drawdown",
    "altair_equity",
    "render_trades_table",
    "render_metrics_table",
    "export_table",
]

__version__ = "0.2.0"  # P4 Dash integration
