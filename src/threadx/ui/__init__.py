"""
ThreadX UI Components - Phase 8
===============================

Composants d'interface utilisateur pour ThreadX :
- Application Tkinter principale (Windows-first)
- Composants graphiques (Matplotlib/Altair)
- Composants tabulaires avec export
- Fallback Streamlit

Author: ThreadX Framework
Version: Phase 8 - UI Components
"""

from .app import ThreadXApp, run_app
from .charts import plot_equity, plot_drawdown, altair_equity
from .tables import render_trades_table, render_metrics_table, export_table

__all__ = [
    'ThreadXApp',
    'run_app',
    'plot_equity',
    'plot_drawdown', 
    'altair_equity',
    'render_trades_table',
    'render_metrics_table',
    'export_table'
]