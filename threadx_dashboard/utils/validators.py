"""
Validateurs de données pour ThreadX Dashboard
============================================

Ce module contient les fonctions de validation des paramètres
de backtesting et des données de trading.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, date
import pandas as pd


def validate_symbol(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Valide un symbole de trading.

    Args:
        symbol: Symbole à valider (ex: 'BTCUSDT')

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not symbol or not isinstance(symbol, str):
        return False, "Le symbole ne peut pas être vide"

    symbol = symbol.upper().strip()

    # Vérification du format basique
    if not re.match(r"^[A-Z0-9]{3,12}$", symbol):
        return False, "Format de symbole invalide (3-12 caractères alphanumériques)"

    return True, None


def validate_timeframe(timeframe: str) -> Tuple[bool, Optional[str]]:
    """
    Valide un timeframe de trading.

    Args:
        timeframe: Timeframe à valider (ex: '1h', '4h', '1d')

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    valid_timeframes = [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    ]

    if not timeframe or timeframe not in valid_timeframes:
        return (
            False,
            f"Timeframe invalide. Valeurs supportées: {', '.join(valid_timeframes)}",
        )

    return True, None


def validate_date_range(
    start_date: Union[str, date, datetime], end_date: Union[str, date, datetime]
) -> Tuple[bool, Optional[str]]:
    """
    Valide une plage de dates.

    Args:
        start_date: Date de début
        end_date: Date de fin

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        if start_date >= end_date:
            return False, "La date de début doit être antérieure à la date de fin"

        # Vérifier que les dates ne sont pas dans le futur
        today = date.today()
        if start_date > today or end_date > today:
            return False, "Les dates ne peuvent pas être dans le futur"

        # Vérifier la plage maximale (2 ans)
        if (end_date - start_date).days > 730:
            return False, "La plage de dates ne peut pas dépasser 2 ans"

        return True, None

    except Exception as e:
        return False, f"Format de date invalide: {str(e)}"


def validate_backtest_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valide les paramètres de backtesting.

    Args:
        params: Dictionnaire des paramètres

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []

    # Capital initial
    initial_capital = params.get("initial_capital", 0)
    if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
        errors.append("Le capital initial doit être un nombre positif")
    elif initial_capital < 100:
        errors.append("Le capital initial doit être d'au moins 100")
    elif initial_capital > 1000000:
        errors.append("Le capital initial ne peut pas dépasser 1,000,000")

    # Commission
    commission = params.get("commission", 0)
    if not isinstance(commission, (int, float)) or commission < 0:
        errors.append("La commission doit être un nombre positif ou zéro")
    elif commission > 0.1:  # 10%
        errors.append("La commission ne peut pas dépasser 10%")

    # Slippage
    slippage = params.get("slippage", 0)
    if not isinstance(slippage, (int, float)) or slippage < 0:
        errors.append("Le slippage doit être un nombre positif ou zéro")
    elif slippage > 0.05:  # 5%
        errors.append("Le slippage ne peut pas dépasser 5%")

    # Nombre maximum de positions
    max_positions = params.get("max_positions", 1)
    if not isinstance(max_positions, int) or max_positions < 1:
        errors.append("Le nombre maximum de positions doit être un entier positif")
    elif max_positions > 10:
        errors.append("Le nombre maximum de positions ne peut pas dépasser 10")

    return len(errors) == 0, errors


def validate_strategy_params(
    strategy_name: str, params: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Valide les paramètres d'une stratégie spécifique.

    Args:
        strategy_name: Nom de la stratégie
        params: Paramètres de la stratégie

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []

    if strategy_name == "bb_atr":
        # Bollinger Bands + ATR strategy
        bb_period = params.get("bb_period", 20)
        if not isinstance(bb_period, int) or bb_period < 5 or bb_period > 100:
            errors.append("La période des Bollinger Bands doit être entre 5 et 100")

        bb_std = params.get("bb_std", 2.0)
        if not isinstance(bb_std, (int, float)) or bb_std < 0.5 or bb_std > 5.0:
            errors.append("L'écart-type des Bollinger Bands doit être entre 0.5 et 5.0")

        atr_period = params.get("atr_period", 14)
        if not isinstance(atr_period, int) or atr_period < 5 or atr_period > 50:
            errors.append("La période de l'ATR doit être entre 5 et 50")

        atr_multiplier = params.get("atr_multiplier", 2.0)
        if (
            not isinstance(atr_multiplier, (int, float))
            or atr_multiplier < 0.5
            or atr_multiplier > 10.0
        ):
            errors.append("Le multiplicateur ATR doit être entre 0.5 et 10.0")

    elif strategy_name == "rsi_ma":
        # RSI + Moving Average strategy
        rsi_period = params.get("rsi_period", 14)
        if not isinstance(rsi_period, int) or rsi_period < 5 or rsi_period > 50:
            errors.append("La période du RSI doit être entre 5 et 50")

        rsi_oversold = params.get("rsi_oversold", 30)
        if (
            not isinstance(rsi_oversold, (int, float))
            or rsi_oversold < 10
            or rsi_oversold > 40
        ):
            errors.append("Le niveau de survente RSI doit être entre 10 et 40")

        rsi_overbought = params.get("rsi_overbought", 70)
        if (
            not isinstance(rsi_overbought, (int, float))
            or rsi_overbought < 60
            or rsi_overbought > 90
        ):
            errors.append("Le niveau de surachat RSI doit être entre 60 et 90")

        if rsi_oversold >= rsi_overbought:
            errors.append(
                "Le niveau de survente doit être inférieur au niveau de surachat"
            )

        ma_period = params.get("ma_period", 50)
        if not isinstance(ma_period, int) or ma_period < 10 or ma_period > 200:
            errors.append("La période de la moyenne mobile doit être entre 10 et 200")

    return len(errors) == 0, errors


def validate_file_upload(
    filename: str, content: bytes, max_size_mb: int = 10
) -> Tuple[bool, Optional[str]]:
    """
    Valide un fichier uploadé.

    Args:
        filename: Nom du fichier
        content: Contenu du fichier en bytes
        max_size_mb: Taille maximale en MB

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not filename:
        return False, "Nom de fichier manquant"

    # Vérifier l'extension
    allowed_extensions = [".csv", ".json", ".xlsx", ".parquet"]
    file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if file_ext not in allowed_extensions:
        return (
            False,
            f"Extension non supportée. Extensions autorisées: {', '.join(allowed_extensions)}",
        )

    # Vérifier la taille
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_size_mb:
        return (
            False,
            f"Fichier trop volumineux ({size_mb:.1f}MB). Taille maximale: {max_size_mb}MB",
        )

    return True, None


def sanitize_input(value: Any, input_type: str = "string") -> Any:
    """
    Nettoie et sécurise une valeur d'entrée.

    Args:
        value: Valeur à nettoyer
        input_type: Type attendu ('string', 'number', 'boolean')

    Returns:
        Any: Valeur nettoyée
    """
    if value is None:
        return None

    if input_type == "string":
        if not isinstance(value, str):
            value = str(value)
        # Supprimer les caractères dangereux
        value = re.sub(r'[<>"\']', "", value)
        return value.strip()[:500]  # Limite à 500 caractères

    elif input_type == "number":
        try:
            if isinstance(value, str):
                value = value.replace(",", ".")  # Gérer les décimales avec virgule
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    elif input_type == "boolean":
        if isinstance(value, str):
            return value.lower() in ["true", "1", "yes", "on"]
        return bool(value)

    return value
