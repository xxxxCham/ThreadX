"""
ThreadX UI Components
====================

Composants d'interface utilisateur pour ThreadX :
- Application Tkinter principale (Windows-first)
- Interface Streamlit (fallback web)
- Composants graphiques (Matplotlib/Altair)
- Composants tabulaires avec export

Author: ThreadX Team
Version: Phase A - Cleanup & Refactoring
"""

# Composants d'interface utilisateur
from .app import ThreadXApp, run_app
from .charts import plot_equity, plot_drawdown, altair_equity
from .tables import render_trades_table, render_metrics_table, export_table

# Points d'entrée unifiés
from . import tkinter
from . import streamlit

__all__ = [
    # Application
    "ThreadXApp",
    "run_app",
    # Points d'entrée
    "tkinter",
    "streamlit",
    # Composants graphiques
    "plot_equity",
    "plot_drawdown",
    "altair_equity",
    # Composants tabulaires
    "render_trades_table",
    "render_metrics_table",
    "export_table" "render_metrics_table",
    "export_table",
]
