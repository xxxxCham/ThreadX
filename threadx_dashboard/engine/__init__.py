"""
ThreadX Calculation Engine
==========================

Moteur de calcul pur pour ThreadX.
Cette couche contient uniquement la logique métier et les calculs,
sans aucune dépendance vers l'interface utilisateur.

Modules:
- backtest_engine: Logique de backtesting
- indicators: Calcul des indicateurs techniques
- data_processor: Traitement et validation des données
"""

from .backtest_engine import BacktestEngine
from .indicators import IndicatorCalculator
from .data_processor import DataProcessor

__all__ = ["BacktestEngine", "IndicatorCalculator", "DataProcessor"]
