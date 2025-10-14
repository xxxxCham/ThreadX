"""
Couche de Services ThreadX
==========================

Services d'orchestration pour les opérations de backtesting.
Cette couche fait le lien entre l'interface utilisateur et les moteurs de calcul.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
from dataclasses import asdict

from ..engine.backtest_engine import (
    BacktestEngine,
    TradingOrder,
    BacktestResult,
    OrderType,
    OrderStatus,
)
from ..engine.indicators import IndicatorCalculator, IndicatorResult
from ..engine.data_processor import DataProcessor, ProcessedData, DataQuality


class BacktestService:
    """
    Service de backtesting - orchestration des calculs

    Cette classe orchestre les opérations de backtesting sans contenir
    de logique métier, qui reste dans les classes du moteur.
    """

    def __init__(self):
        """Initialise le service de backtesting"""
        self.backtest_engine = BacktestEngine()
        self.indicator_calculator = IndicatorCalculator()
        self.data_processor = DataProcessor()

    def prepare_data_for_backtest(
        self, raw_data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Tuple[bool, ProcessedData]:
        """
        Prépare et valide les données pour le backtesting

        Args:
            raw_data: Données brutes de marché
            symbol: Symbole de l'actif

        Returns:
            Tuple[bool, ProcessedData]: Succès et données traitées
        """
        try:
            processed_data = self.data_processor.process_market_data(
                raw_data=raw_data, symbol=symbol, auto_clean=True
            )

            # Vérifier la qualité minimale requise
            min_quality = DataQuality.ACCEPTABLE
            is_suitable = (
                processed_data.validation_result.quality.value >= min_quality.value
                and processed_data.validation_result.is_valid
            )

            return is_suitable, processed_data

        except Exception as e:
            # En cas d'erreur, retourner des données vides avec échec
            empty_result = ProcessedData(
                raw_data=raw_data,
                cleaned_data=pd.DataFrame(),
                validation_result=None,
                processing_metadata={"error": str(e)},
            )
            return False, empty_result

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: Dict[str, Any],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
    ) -> BacktestResult:
        """
        Exécute un backtest complet

        Args:
            data: Données de marché nettoyées
            strategy_params: Paramètres de stratégie
            initial_capital: Capital initial
            commission_rate: Taux de commission

        Returns:
            BacktestResult: Résultats du backtest
        """
        # Configuration du moteur
        self.backtest_engine.set_initial_capital(initial_capital)
        self.backtest_engine.set_commission_rate(commission_rate)

        # Exécution du backtest
        return self.backtest_engine.run_backtest(data, strategy_params)

    def calculate_indicators(
        self, data: pd.DataFrame, indicators_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, IndicatorResult]:
        """
        Calcule plusieurs indicateurs techniques

        Args:
            data: Données de marché
            indicators_config: Configuration des indicateurs

        Returns:
            Dict[str, IndicatorResult]: Résultats des indicateurs
        """
        results = {}

        for indicator_name, config in indicators_config.items():
            try:
                method_name = f"calculate_{indicator_name.lower()}"
                if hasattr(self.indicator_calculator, method_name):
                    method = getattr(self.indicator_calculator, method_name)
                    result = method(data, **config)
                    results[indicator_name] = result
            except Exception as e:
                # En cas d'erreur, créer un résultat d'échec
                results[indicator_name] = IndicatorResult(
                    name=indicator_name,
                    values=pd.Series(dtype=float),
                    metadata={"error": str(e), "success": False},
                )

        return results

    def generate_trading_signals(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, IndicatorResult],
        signal_rules: Dict[str, Any],
    ) -> pd.Series:
        """
        Génère des signaux de trading basés sur les indicateurs

        Args:
            data: Données de marché
            indicators: Indicateurs calculés
            signal_rules: Règles de génération de signaux

        Returns:
            pd.Series: Signaux (1=achat, -1=vente, 0=neutre)
        """
        signals = pd.Series(0, index=data.index, name="signals")

        # Exemple de logique simple pour démonstration
        if "sma" in indicators and "rsi" in indicators:
            sma_result = indicators["sma"]
            rsi_result = indicators["rsi"]

            if len(sma_result.values) > 0 and len(rsi_result.values) > 0:
                # Signal d'achat: prix > SMA et RSI < 30 (survendu)
                buy_condition = (data["close"] > sma_result.values) & (
                    rsi_result.values < 30
                )

                # Signal de vente: prix < SMA et RSI > 70 (suracheté)
                sell_condition = (data["close"] < sma_result.values) & (
                    rsi_result.values > 70
                )

                signals.loc[buy_condition] = 1
                signals.loc[sell_condition] = -1

        return signals

    def get_backtest_summary(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Génère un résumé lisible du backtest

        Args:
            result: Résultats du backtest

        Returns:
            Dict[str, Any]: Résumé formaté
        """
        return {
            "performance": {
                "total_return": result.total_return,
                "total_return_pct": result.total_return * 100,
                "annualized_return": result.annualized_return,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "win_rate": result.win_rate,
            },
            "trading": {
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "avg_trade_return": result.avg_trade_return,
            },
            "portfolio": {
                "final_portfolio_value": result.final_portfolio_value,
                "initial_capital": result.initial_capital,
                "total_fees_paid": result.total_fees_paid,
            },
            "metadata": result.metadata,
        }


class DataService:
    """
    Service de gestion des données - orchestration des opérations sur les données
    """

    def __init__(self):
        """Initialise le service de données"""
        self.data_processor = DataProcessor()

    def load_and_validate_data(
        self, file_path: str, symbol: Optional[str] = None
    ) -> Tuple[bool, ProcessedData]:
        """
        Charge et valide un fichier de données

        Args:
            file_path: Chemin vers le fichier
            symbol: Symbole de l'actif (optionnel)

        Returns:
            Tuple[bool, ProcessedData]: Succès et données traitées
        """
        try:
            # Charger les données selon l'extension
            if file_path.endswith(".csv"):
                raw_data = pd.read_csv(file_path)
            elif file_path.endswith(".parquet"):
                raw_data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Format de fichier non supporté: {file_path}")

            # Traiter et valider
            processed_data = self.data_processor.process_market_data(
                raw_data=raw_data, symbol=symbol or "UNKNOWN", auto_clean=True
            )

            return processed_data.validation_result.is_valid, processed_data

        except Exception as e:
            empty_result = ProcessedData(
                raw_data=pd.DataFrame(),
                cleaned_data=pd.DataFrame(),
                validation_result=None,
                processing_metadata={"error": str(e)},
            )
            return False, empty_result

    def generate_demo_data(
        self, symbol: str = "DEMO", days: int = 252, base_price: float = 100.0
    ) -> pd.DataFrame:
        """
        Génère des données de démonstration

        Args:
            symbol: Symbole de l'actif
            days: Nombre de jours
            base_price: Prix de base

        Returns:
            pd.DataFrame: Données de démonstration
        """
        import numpy as np
        from datetime import timedelta

        # Générer des dates
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days), periods=days, freq="D"
        )

        # Générer des prix avec marche aléatoire
        np.random.seed(42)  # Pour la reproductibilité
        returns = np.random.normal(0.001, 0.02, days)  # Retours quotidiens
        prices = [base_price]

        for i in range(1, days):
            price = prices[-1] * (1 + returns[i])
            prices.append(max(price, 0.01))  # Prix minimum pour éviter les négatifs

        # Créer les données OHLC
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Générer O, H, L basés sur le cours de clôture
            volatility = 0.02

            open_price = close_price * (1 + np.random.normal(0, volatility / 4))
            high_price = max(open_price, close_price) * (
                1 + abs(np.random.normal(0, volatility / 2))
            )
            low_price = min(open_price, close_price) * (
                1 - abs(np.random.normal(0, volatility / 2))
            )

            # Volume aléatoire
            volume = int(np.random.uniform(1000000, 5000000))

            data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume,
                }
            )

        return pd.DataFrame(data)

    def resample_timeframe(
        self, data: pd.DataFrame, target_timeframe: str
    ) -> pd.DataFrame:
        """
        Rééchantillonne les données à une fréquence différente

        Args:
            data: Données source
            target_timeframe: Fréquence cible

        Returns:
            pd.DataFrame: Données rééchantillonnées
        """
        return self.data_processor.resample_data(
            data=data, timeframe=target_timeframe, date_column="date"
        )


class ChartService:
    """
    Service de préparation des données pour les graphiques
    """

    def prepare_price_chart_data(
        self,
        data: pd.DataFrame,
        indicators: Optional[Dict[str, IndicatorResult]] = None,
    ) -> Dict[str, Any]:
        """
        Prépare les données pour le graphique de prix

        Args:
            data: Données de marché
            indicators: Indicateurs techniques (optionnel)

        Returns:
            Dict[str, Any]: Données formatées pour Plotly
        """
        chart_data = {
            "dates": data["date"].tolist(),
            "ohlc": {
                "open": data["open"].tolist(),
                "high": data["high"].tolist(),
                "low": data["low"].tolist(),
                "close": data["close"].tolist(),
            },
            "volume": data["volume"].tolist(),
            "indicators": {},
        }

        # Ajouter les indicateurs si fournis
        if indicators:
            for name, result in indicators.items():
                if len(result.values) > 0:
                    chart_data["indicators"][name] = {
                        "values": result.values.tolist(),
                        "metadata": result.metadata,
                    }

        return chart_data

    def prepare_portfolio_chart_data(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Prépare les données pour le graphique de portefeuille

        Args:
            result: Résultats du backtest

        Returns:
            Dict[str, Any]: Données formatées pour le graphique
        """
        portfolio_history = result.portfolio_history

        return {
            "dates": [entry["date"] for entry in portfolio_history],
            "values": [entry["total_value"] for entry in portfolio_history],
            "cash": [entry["cash"] for entry in portfolio_history],
            "positions_value": [
                entry["positions_value"] for entry in portfolio_history
            ],
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
        }

    def prepare_trades_chart_data(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Prépare les données pour le graphique des trades

        Args:
            result: Résultats du backtest

        Returns:
            Dict[str, Any]: Données des trades formatées
        """
        trades = result.executed_orders

        buy_trades = [trade for trade in trades if trade.order_type == OrderType.BUY]
        sell_trades = [trade for trade in trades if trade.order_type == OrderType.SELL]

        return {
            "buy_trades": {
                "dates": [trade.timestamp for trade in buy_trades],
                "prices": [trade.executed_price for trade in buy_trades],
                "quantities": [trade.quantity for trade in buy_trades],
            },
            "sell_trades": {
                "dates": [trade.timestamp for trade in sell_trades],
                "prices": [trade.executed_price for trade in sell_trades],
                "quantities": [trade.quantity for trade in sell_trades],
            },
            "total_trades": len(trades),
            "win_rate": result.win_rate,
        }
