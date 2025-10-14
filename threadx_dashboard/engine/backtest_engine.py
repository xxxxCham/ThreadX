"""
Moteur de Backtesting ThreadX
============================

Moteur de calcul pur pour l'exécution de backtests.
Cette classe contient uniquement la logique métier sans dépendances UI.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Types d'ordres de trading"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Statuts des ordres"""

    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


@dataclass
class TradingOrder:
    """Représente un ordre de trading"""

    date: str
    order_type: OrderType
    symbol: str
    quantity: float
    price: float
    status: OrderStatus = OrderStatus.PENDING
    execution_price: Optional[float] = None
    execution_date: Optional[str] = None


@dataclass
class BacktestResult:
    """Résultat d'un backtest"""

    initial_capital: float
    final_capital: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    equity_curve: pd.Series
    trade_history: List[TradingOrder]
    daily_returns: pd.Series
    metadata: Dict[str, Any]


class BacktestEngine:
    """
    Moteur de backtesting pur - logique métier uniquement

    Cette classe exécute les backtests selon une stratégie donnée,
    sans aucune dépendance vers l'interface utilisateur.
    """

    def __init__(self, initial_capital: float = 10000):
        """
        Initialise le moteur de backtest

        Args:
            initial_capital: Capital initial pour le backtest
        """
        self.initial_capital = initial_capital
        self.reset()

    def reset(self):
        """Remet à zéro l'état du moteur"""
        self.current_capital = self.initial_capital
        self.positions = {}  # symbol -> quantity
        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []

    def execute_backtest(
        self,
        price_data: pd.DataFrame,
        trading_signals: List[TradingOrder],
        transaction_cost: float = 0.001,
    ) -> BacktestResult:
        """
        Execute un backtest complet

        Args:
            price_data: DataFrame avec colonnes [date, symbol, open, high, low, close, volume]
            trading_signals: Liste des signaux de trading à exécuter
            transaction_cost: Coût de transaction (pourcentage)

        Returns:
            BacktestResult: Résultats complets du backtest
        """
        self.reset()

        # Valider les données d'entrée
        self._validate_input_data(price_data, trading_signals)

        # Indexer les prix par date
        price_index = self._create_price_index(price_data)

        # Trier les signaux par date
        signals_sorted = sorted(
            trading_signals, key=lambda x: datetime.strptime(x.date, "%Y-%m-%d")
        )

        # Exécuter le backtest jour par jour
        dates = sorted(price_index.keys())
        signal_index = 0

        for date in dates:
            # Exécuter les signaux du jour
            while (
                signal_index < len(signals_sorted)
                and signals_sorted[signal_index].date == date
            ):

                signal = signals_sorted[signal_index]
                self._execute_order(signal, price_index[date], transaction_cost)
                signal_index += 1

            # Calculer l'équité du jour
            daily_equity = self._calculate_daily_equity(date, price_index[date])
            self.equity_curve.append(
                {
                    "date": date,
                    "equity": daily_equity,
                    "cash": self.current_capital,
                    "positions_value": daily_equity - self.current_capital,
                }
            )

            # Calculer le rendement quotidien
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2]["equity"]
                daily_return = (daily_equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)

        # Calculer les métriques finales
        return self._calculate_final_metrics(price_data)

    def _validate_input_data(
        self, price_data: pd.DataFrame, trading_signals: List[TradingOrder]
    ):
        """Valide les données d'entrée"""
        required_columns = ["date", "symbol", "open", "high", "low", "close"]

        for col in required_columns:
            if col not in price_data.columns:
                raise ValueError(f"Colonne manquante dans price_data: {col}")

        if price_data.empty:
            raise ValueError("price_data ne peut pas être vide")

        # Valider les signaux
        for signal in trading_signals:
            if not isinstance(signal, TradingOrder):
                raise TypeError("Tous les signaux doivent être des TradingOrder")

    def _create_price_index(self, price_data: pd.DataFrame) -> Dict[str, Dict]:
        """Crée un index des prix par date et symbole"""
        price_index = {}

        for _, row in price_data.iterrows():
            date = (
                row["date"]
                if isinstance(row["date"], str)
                else row["date"].strftime("%Y-%m-%d")
            )
            symbol = row["symbol"]

            if date not in price_index:
                price_index[date] = {}

            price_index[date][symbol] = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
            }

        return price_index

    def _execute_order(
        self,
        order: TradingOrder,
        daily_prices: Dict[str, Dict],
        transaction_cost: float,
    ):
        """Exécute un ordre de trading"""
        symbol = order.symbol

        if symbol not in daily_prices:
            order.status = OrderStatus.CANCELLED
            return

        # Prix d'exécution (utilise le prix de clôture par simplicité)
        execution_price = daily_prices[symbol]["close"]
        trade_value = order.quantity * execution_price
        cost = trade_value * transaction_cost

        if order.order_type == OrderType.BUY:
            # Vérifier si on a assez de cash
            total_cost = trade_value + cost
            if self.current_capital >= total_cost:
                self.current_capital -= total_cost

                # Ajouter à la position
                if symbol not in self.positions:
                    self.positions[symbol] = 0
                self.positions[symbol] += order.quantity

                # Marquer l'ordre comme exécuté
                order.status = OrderStatus.EXECUTED
                order.execution_price = execution_price
                order.execution_date = order.date

                self.trade_history.append(order)
            else:
                order.status = OrderStatus.CANCELLED

        elif order.order_type == OrderType.SELL:
            # Vérifier si on a assez de positions
            current_position = self.positions.get(symbol, 0)

            if current_position >= order.quantity:
                self.current_capital += trade_value - cost
                self.positions[symbol] -= order.quantity

                # Nettoyer les positions vides
                if self.positions[symbol] == 0:
                    del self.positions[symbol]

                # Marquer l'ordre comme exécuté
                order.status = OrderStatus.EXECUTED
                order.execution_price = execution_price
                order.execution_date = order.date

                self.trade_history.append(order)
            else:
                order.status = OrderStatus.CANCELLED

    def _calculate_daily_equity(
        self, date: str, daily_prices: Dict[str, Dict]
    ) -> float:
        """Calcule l'équité totale pour une journée donnée"""
        positions_value = 0

        for symbol, quantity in self.positions.items():
            if symbol in daily_prices:
                price = daily_prices[symbol]["close"]
                positions_value += quantity * price

        return self.current_capital + positions_value

    def _calculate_final_metrics(self, price_data: pd.DataFrame) -> BacktestResult:
        """Calcule les métriques finales du backtest"""
        if not self.equity_curve:
            raise ValueError("Aucune donnée d'équité disponible")

        # Convertir en Series pour les calculs
        equity_series = pd.Series([e["equity"] for e in self.equity_curve])

        # Métriques de base
        final_capital = equity_series.iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # Drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()

        # Sharpe ratio
        if len(self.daily_returns) > 1:
            daily_returns_series = pd.Series(self.daily_returns)
            sharpe_ratio = (
                daily_returns_series.mean() / daily_returns_series.std() * np.sqrt(252)
                if daily_returns_series.std() != 0
                else 0
            )
        else:
            sharpe_ratio = 0

        # Statistiques des trades
        executed_trades = [
            t for t in self.trade_history if t.status == OrderStatus.EXECUTED
        ]

        # Calculer les gains/pertes par paire buy/sell
        winning_trades = 0
        losing_trades = 0
        wins = []
        losses = []

        # Logique simplifiée : comparer les trades buy/sell successifs
        buy_trades = [t for t in executed_trades if t.order_type == OrderType.BUY]
        sell_trades = [t for t in executed_trades if t.order_type == OrderType.SELL]

        for i, buy_trade in enumerate(buy_trades):
            if i < len(sell_trades):
                sell_trade = sell_trades[i]
                profit = (sell_trade.execution_price - buy_trade.execution_price) * min(
                    buy_trade.quantity, sell_trade.quantity
                )

                if profit > 0:
                    winning_trades += 1
                    wins.append(profit)
                else:
                    losing_trades += 1
                    losses.append(profit)

        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity_series,
            trade_history=executed_trades,
            daily_returns=pd.Series(self.daily_returns),
            metadata={
                "start_date": self.equity_curve[0]["date"],
                "end_date": self.equity_curve[-1]["date"],
                "total_days": len(self.equity_curve),
            },
        )
