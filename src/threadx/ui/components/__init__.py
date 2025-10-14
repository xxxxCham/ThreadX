"""
ThreadX UI Components Package
=============================

Composants Dash modulaires pour l'interface utilisateur ThreadX.

Modules disponibles:
    - data_manager: Panel de gestion des données
    - indicators_panel: Panel de configuration des indicateurs
    - backtest_panel: Panel de backtesting
    - optimization_panel: Panel d'optimisation

Author: ThreadX Framework
Version: Prompt 6 - Composants Backtest + Optimization
"""

from threadx.ui.components.backtest_panel import create_backtest_panel
from threadx.ui.components.data_manager import create_data_manager_panel
from threadx.ui.components.indicators_panel import (
    create_indicators_panel
)
from threadx.ui.components.optimization_panel import (
    create_optimization_panel
)

__all__ = [
    "create_data_manager_panel",
    "create_indicators_panel",
    "create_backtest_panel",
    "create_optimization_panel",
]
