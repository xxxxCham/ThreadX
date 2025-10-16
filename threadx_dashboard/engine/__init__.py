"""
ThreadX Calculation Engine
==========================

Moteur de calcul pur pour ThreadX.
Cette couche contient uniquement la logique métier et les calculs,
sans aucune dépendance vers l'interface utilisateur.

Modules:
- backtest_engine: Logique de backtesting
- data_processor: Traitement et validation des données

Note: Pour les indicateurs, utiliser src/threadx/indicators/ (référence unique)
"""

from .backtest_engine import BacktestEngine
from .data_processor import DataProcessor

# ✅ MIGRATION: Indicateurs déplacés vers src/threadx/indicators/
# Utiliser: from threadx.indicators.indicators_np import ema_np, rsi_np, etc.
# Ou: from threadx.indicators.engine import enrich_indicators

__all__ = ["BacktestEngine", "DataProcessor"]
