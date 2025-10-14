"""
Calculateur d'Indicateurs Techniques ThreadX
============================================

Moteur de calcul pur pour les indicateurs techniques.
Cette classe contient uniquement les formules de calcul,
sans aucune dépendance vers l'interface utilisateur.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class IndicatorType(Enum):
    """Types d'indicateurs disponibles"""

    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average
    RSI = "rsi"  # Relative Strength Index
    MACD = "macd"  # Moving Average Convergence Divergence
    BOLLINGER = "bollinger"  # Bollinger Bands
    ATR = "atr"  # Average True Range
    STOCHASTIC = "stochastic"  # Stochastic Oscillator
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class IndicatorResult:
    """Résultat du calcul d'un indicateur"""

    name: str
    indicator_type: IndicatorType
    values: Union[pd.Series, Dict[str, pd.Series]]
    parameters: Dict[str, any]
    metadata: Dict[str, any]


class IndicatorCalculator:
    """
    Calculateur d'indicateurs techniques pur - logique métier uniquement

    Cette classe calcule les indicateurs selon les formules standards,
    sans aucune dépendance vers l'interface utilisateur.
    """

    def __init__(self):
        """Initialise le calculateur d'indicateurs"""
        self.available_indicators = {
            IndicatorType.SMA: self._calculate_sma,
            IndicatorType.EMA: self._calculate_ema,
            IndicatorType.RSI: self._calculate_rsi,
            IndicatorType.MACD: self._calculate_macd,
            IndicatorType.BOLLINGER: self._calculate_bollinger,
            IndicatorType.ATR: self._calculate_atr,
            IndicatorType.STOCHASTIC: self._calculate_stochastic,
            IndicatorType.SUPPORT_RESISTANCE: self._calculate_support_resistance,
        }

    def calculate_indicator(
        self, indicator_type: IndicatorType, price_data: pd.DataFrame, **kwargs
    ) -> IndicatorResult:
        """
        Calcule un indicateur technique

        Args:
            indicator_type: Type d'indicateur à calculer
            price_data: DataFrame avec colonnes OHLCV
            **kwargs: Paramètres spécifiques à l'indicateur

        Returns:
            IndicatorResult: Résultat du calcul
        """
        if indicator_type not in self.available_indicators:
            raise ValueError(f"Indicateur non supporté: {indicator_type}")

        # Valider les données d'entrée
        self._validate_price_data(price_data)

        # Calculer l'indicateur
        calculator_func = self.available_indicators[indicator_type]
        return calculator_func(price_data, **kwargs)

    def calculate_multiple_indicators(
        self, indicators_config: List[Dict], price_data: pd.DataFrame
    ) -> Dict[str, IndicatorResult]:
        """
        Calcule plusieurs indicateurs en une fois

        Args:
            indicators_config: Liste de configs [{type, name, params}, ...]
            price_data: DataFrame avec colonnes OHLCV

        Returns:
            Dict[str, IndicatorResult]: Résultats indexés par nom
        """
        results = {}

        for config in indicators_config:
            indicator_type = IndicatorType(config["type"])
            name = config.get("name", f"{indicator_type.value}")
            params = config.get("params", {})

            result = self.calculate_indicator(indicator_type, price_data, **params)
            result.name = name
            results[name] = result

        return results

    def _validate_price_data(self, price_data: pd.DataFrame):
        """Valide les données de prix"""
        required_columns = ["close"]

        for col in required_columns:
            if col not in price_data.columns:
                raise ValueError(f"Colonne manquante: {col}")

        if price_data.empty:
            raise ValueError("price_data ne peut pas être vide")

    # =========================================================================
    # CALCULATEURS D'INDICATEURS INDIVIDUELS
    # =========================================================================

    def _calculate_sma(
        self, price_data: pd.DataFrame, period: int = 20, price_col: str = "close"
    ) -> IndicatorResult:
        """Calcule la Simple Moving Average"""
        sma_values = price_data[price_col].rolling(window=period).mean()

        return IndicatorResult(
            name=f"SMA_{period}",
            indicator_type=IndicatorType.SMA,
            values=sma_values,
            parameters={"period": period, "price_col": price_col},
            metadata={"formula": "Simple Moving Average"},
        )

    def _calculate_ema(
        self, price_data: pd.DataFrame, period: int = 20, price_col: str = "close"
    ) -> IndicatorResult:
        """Calcule l'Exponential Moving Average"""
        ema_values = price_data[price_col].ewm(span=period).mean()

        return IndicatorResult(
            name=f"EMA_{period}",
            indicator_type=IndicatorType.EMA,
            values=ema_values,
            parameters={"period": period, "price_col": price_col},
            metadata={"formula": "Exponential Moving Average"},
        )

    def _calculate_rsi(
        self, price_data: pd.DataFrame, period: int = 14, price_col: str = "close"
    ) -> IndicatorResult:
        """Calcule le Relative Strength Index"""
        prices = price_data[price_col]
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return IndicatorResult(
            name=f"RSI_{period}",
            indicator_type=IndicatorType.RSI,
            values=rsi,
            parameters={"period": period, "price_col": price_col},
            metadata={"formula": "RSI = 100 - (100 / (1 + RS))"},
        )

    def _calculate_macd(
        self,
        price_data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_col: str = "close",
    ) -> IndicatorResult:
        """Calcule le MACD (Moving Average Convergence Divergence)"""
        prices = price_data[price_col]

        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        return IndicatorResult(
            name=f"MACD_{fast_period}_{slow_period}_{signal_period}",
            indicator_type=IndicatorType.MACD,
            values={"macd": macd_line, "signal": signal_line, "histogram": histogram},
            parameters={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "price_col": price_col,
            },
            metadata={"formula": "MACD = EMA_fast - EMA_slow"},
        )

    def _calculate_bollinger(
        self,
        price_data: pd.DataFrame,
        period: int = 20,
        std_multiplier: float = 2.0,
        price_col: str = "close",
    ) -> IndicatorResult:
        """Calcule les Bollinger Bands"""
        prices = price_data[price_col]

        middle_band = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()

        upper_band = middle_band + (std_dev * std_multiplier)
        lower_band = middle_band - (std_dev * std_multiplier)

        return IndicatorResult(
            name=f"BB_{period}_{std_multiplier}",
            indicator_type=IndicatorType.BOLLINGER,
            values={"upper": upper_band, "middle": middle_band, "lower": lower_band},
            parameters={
                "period": period,
                "std_multiplier": std_multiplier,
                "price_col": price_col,
            },
            metadata={"formula": "BB = SMA ± (StdDev * multiplier)"},
        )

    def _calculate_atr(
        self, price_data: pd.DataFrame, period: int = 14
    ) -> IndicatorResult:
        """Calcule l'Average True Range"""
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in price_data.columns:
                raise ValueError(f"Colonne manquante pour ATR: {col}")

        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return IndicatorResult(
            name=f"ATR_{period}",
            indicator_type=IndicatorType.ATR,
            values=atr,
            parameters={"period": period},
            metadata={"formula": "ATR = RMA(TrueRange, period)"},
        )

    def _calculate_stochastic(
        self, price_data: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> IndicatorResult:
        """Calcule l'oscillateur Stochastique"""
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in price_data.columns:
                raise ValueError(f"Colonne manquante pour Stochastic: {col}")

        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return IndicatorResult(
            name=f"STOCH_{k_period}_{d_period}",
            indicator_type=IndicatorType.STOCHASTIC,
            values={"k": k_percent, "d": d_percent},
            parameters={"k_period": k_period, "d_period": d_period},
            metadata={
                "formula": "%K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)"
            },
        )

    def _calculate_support_resistance(
        self, price_data: pd.DataFrame, window: int = 20, num_levels: int = 3
    ) -> IndicatorResult:
        """Calcule les niveaux de support et résistance"""
        prices = price_data["close"]

        # Identifier les pics et creux locaux
        local_maxima = []
        local_minima = []

        for i in range(window, len(prices) - window):
            # Vérifier si c'est un maximum local
            if prices.iloc[i] == prices.iloc[i - window : i + window + 1].max():
                local_maxima.append(prices.iloc[i])

            # Vérifier si c'est un minimum local
            if prices.iloc[i] == prices.iloc[i - window : i + window + 1].min():
                local_minima.append(prices.iloc[i])

        # Calculer les niveaux significatifs
        resistance_levels = []
        support_levels = []

        if local_maxima:
            # Grouper les niveaux proches et prendre les plus significatifs
            maxima_sorted = sorted(local_maxima, reverse=True)
            resistance_levels = maxima_sorted[:num_levels]

        if local_minima:
            minima_sorted = sorted(local_minima)
            support_levels = minima_sorted[:num_levels]

        return IndicatorResult(
            name=f"SR_{window}_{num_levels}",
            indicator_type=IndicatorType.SUPPORT_RESISTANCE,
            values={
                "resistance": resistance_levels,
                "support": support_levels,
                "local_maxima": local_maxima,
                "local_minima": local_minima,
            },
            parameters={"window": window, "num_levels": num_levels},
            metadata={"formula": "Local extrema detection with window analysis"},
        )
