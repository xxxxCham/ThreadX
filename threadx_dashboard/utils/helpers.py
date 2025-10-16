"""
Fonctions d'aide et utilitaires pour ThreadX Dashboard
=====================================================

Ce module contient des fonctions utilitaires génériques
utilisées dans toute l'application.

RÈGLE ARCHITECTURE: Aucun calcul métier ici.
Tous les calculs pandas/numpy doivent passer par Bridge.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from config import THEME, LOG_FILE, LOG_FORMAT, LOG_LEVEL

# Import Bridge pour déléguer calculs métier
from threadx.bridge import MetricsController


def setup_logging() -> logging.Logger:
    """
    Configure le système de logging pour l'application.

    Returns:
        logging.Logger: Logger configuré
    """
    # Créer le répertoire des logs s'il n'existe pas
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configuration du logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logger = logging.getLogger("threadx_dashboard")
    logger.info("Système de logging initialisé")

    return logger


def format_currency(value: float, currency: str = "USD", precision: int = 2) -> str:
    """
    Formate une valeur monétaire.

    Args:
        value: Valeur à formater
        currency: Devise (USD, EUR, etc.)
        precision: Nombre de décimales

    Returns:
        str: Valeur formatée
    """
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """
    Formate un pourcentage.

    Args:
        value: Valeur à formater (0.1 = 10%)
        precision: Nombre de décimales

    Returns:
        str: Pourcentage formaté
    """
    return f"{value*100:.{precision}f}%"


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calcule rendements - DÉLÈGUE À BRIDGE.

    Args:
        prices: Série de prix

    Returns:
        pd.Series: Série de rendements
    """
    # ANCIEN CODE (INTERDIT): return prices.pct_change().dropna()

    # NOUVEAU: Déléguer au Bridge
    metrics_controller = MetricsController()
    result = metrics_controller.calculate_returns(prices.tolist())
    return pd.Series(result["returns"], index=prices.index[1:])


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calcule ratio de Sharpe - DÉLÈGUE À BRIDGE.

    Args:
        returns: Série de rendements
        risk_free_rate: Taux sans risque annuel

    Returns:
        float: Ratio de Sharpe
    """
    # ANCIEN CODE (INTERDIT):
    # if returns.empty or returns.std() == 0:
    #     return 0.0
    # excess_returns = returns.mean() * 252 - risk_free_rate
    # volatility = returns.std() * (252**0.5)
    # return excess_returns / volatility if volatility != 0 else 0.0

    # NOUVEAU: Déléguer au Bridge
    metrics_controller = MetricsController()
    return metrics_controller.calculate_sharpe_ratio(
        returns=returns.tolist(), risk_free_rate=risk_free_rate
    )


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calcule drawdown maximum - DÉLÈGUE À BRIDGE.

    Args:
        equity_curve: Courbe d'équité

    Returns:
        float: Drawdown maximum (négatif)
    """
    # ANCIEN CODE (INTERDIT):
    # if equity_curve.empty:
    #     return 0.0
    # peak = equity_curve.expanding().max()
    # drawdown = (equity_curve - peak) / peak
    # return drawdown.min()

    # NOUVEAU: Déléguer au Bridge
    if equity_curve.empty:
        return 0.0

    metrics_controller = MetricsController()
    result = metrics_controller.calculate_max_drawdown(equity_curve.tolist())
    return result["max_drawdown"]


def get_color_palette(n_colors: int) -> List[str]:
    """
    Récupère une palette de couleurs pour les graphiques.

    Args:
        n_colors: Nombre de couleurs nécessaires

    Returns:
        List[str]: Liste de couleurs hex
    """
    if n_colors <= len(qualitative.Plotly):
        return qualitative.Plotly[:n_colors]

    # Répéter les couleurs si nécessaire
    colors = qualitative.Plotly * ((n_colors // len(qualitative.Plotly)) + 1)
    return colors[:n_colors]


def create_plotly_theme() -> Dict[str, Any]:
    """
    Crée un thème Plotly basé sur la configuration.

    Returns:
        Dict: Configuration du thème
    """
    return {
        "layout": {
            "paper_bgcolor": THEME["primary_bg"],
            "plot_bgcolor": THEME["secondary_bg"],
            "font": {"color": THEME["text_primary"]},
            "xaxis": {
                "gridcolor": THEME["border_color"],
                "linecolor": THEME["border_color"],
                "tickcolor": THEME["text_secondary"],
            },
            "yaxis": {
                "gridcolor": THEME["border_color"],
                "linecolor": THEME["border_color"],
                "tickcolor": THEME["text_secondary"],
            },
            "colorway": [
                THEME["accent_primary"],
                THEME["success"],
                THEME["warning"],
                THEME["danger"],
            ],
        }
    }


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Charge du JSON de manière sécurisée.

    Args:
        json_str: Chaîne JSON
        default: Valeur par défaut en cas d'erreur

    Returns:
        Any: Objet Python ou valeur par défaut
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Division sécurisée (évite la division par zéro).

    Args:
        numerator: Numérateur
        denominator: Dénominateur
        default: Valeur par défaut si division par zéro

    Returns:
        float: Résultat de la division ou valeur par défaut
    """
    return numerator / denominator if denominator != 0 else default


def timestamp_to_datetime(timestamp: Union[int, float, str]) -> datetime:
    """
    Convertit un timestamp en datetime.

    Args:
        timestamp: Timestamp (unix ou string)

    Returns:
        datetime: Objet datetime
    """
    if isinstance(timestamp, str):
        return pd.to_datetime(timestamp)

    # Gérer les timestamps en millisecondes
    if timestamp > 1e10:
        timestamp = timestamp / 1000

    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Aplatit un dictionnaire imbriqué.

    Args:
        d: Dictionnaire à aplatir
        parent_key: Clé parent pour la récursion
        sep: Séparateur pour les clés

    Returns:
        Dict: Dictionnaire aplati
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Récupère la taille d'un fichier en MB.

    Args:
        file_path: Chemin du fichier

    Returns:
        float: Taille en MB
    """
    try:
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Tronque une chaîne de caractères.

    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter si tronqué

    Returns:
        str: Texte tronqué
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def generate_session_id() -> str:
    """
    Génère un ID de session unique.

    Returns:
        str: ID de session
    """
    import uuid

    return str(uuid.uuid4())


def is_market_open(timestamp: Optional[datetime] = None) -> bool:
    """
    Vérifie si le marché crypto est ouvert (24/7 pour les cryptos).

    Args:
        timestamp: Timestamp à vérifier (défaut: maintenant)

    Returns:
        bool: True (crypto marché toujours ouvert)
    """
    # Les marchés crypto sont ouverts 24/7
    return True


def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Valide qu'un dictionnaire contient tous les champs requis.

    Args:
        data: Données à valider
        required_fields: Liste des champs requis

    Returns:
        bool: True si tous les champs sont présents
    """
    return all(field in data for field in required_fields)


# =============================================================================
# FONCTIONS POUR BACKTESTING ET GRAPHIQUES
# =============================================================================


def filter_data_by_timeframe(data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
    """
    Filtre les données selon la période sélectionnée

    Args:
        data: Dictionnaire de données de backtest
        timeframe: '1D', '1W', '1M', '3M', '6M', 'ALL'

    Returns:
        Données filtrées selon le timeframe
    """
    if timeframe == "ALL":
        return data

    # Calculer la date de début selon le timeframe
    end_date = datetime.now()
    if timeframe == "1D":
        start_date = end_date - pd.Timedelta(days=1)
    elif timeframe == "1W":
        start_date = end_date - pd.Timedelta(weeks=1)
    elif timeframe == "1M":
        start_date = end_date - pd.Timedelta(days=30)
    elif timeframe == "3M":
        start_date = end_date - pd.Timedelta(days=90)
    elif timeframe == "6M":
        start_date = end_date - pd.Timedelta(days=180)
    else:
        return data

    # Copier les données pour ne pas modifier l'original
    filtered_data = data.copy()

    # Filtrer toutes les séries temporelles
    for key in ["price_history", "volume", "portfolio", "buy_hold"]:
        if key in filtered_data and "dates" in filtered_data[key]:
            dates = pd.to_datetime(filtered_data[key]["dates"])
            mask = (dates >= start_date) & (dates <= end_date)

            # Filtrer chaque champ de données
            filtered_data[key] = {
                field: [
                    val
                    for i, val in enumerate(values)
                    if i < len(mask) and mask.iloc[i]
                ]
                for field, values in filtered_data[key].items()
                if isinstance(values, list)
            }

    # Filtrer les signaux d'achat/vente
    for signal_type in ["buy_signals", "sell_signals"]:
        if signal_type in filtered_data:
            filtered_data[signal_type] = [
                signal
                for signal in filtered_data[signal_type]
                if start_date <= pd.to_datetime(signal["date"]) <= end_date
            ]

    return filtered_data


def calculate_drawdown(equity_curve: pd.Series) -> tuple:
    """
    Calcule le drawdown courant et maximum

    Args:
        equity_curve: Série des valeurs d'équité

    Returns:
        Tuple (current_dd, max_dd, dd_start_date, dd_end_date)
    """
    if len(equity_curve) == 0:
        return 0.0, 0.0, None, None

    # Calculer le pic roulant
    peak = equity_curve.expanding().max()

    # Calculer le drawdown en pourcentage
    drawdown = (equity_curve - peak) / peak

    # Drawdown courant et maximum
    current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0
    max_dd = drawdown.min()

    # Trouver les dates de début et fin du drawdown max
    max_dd_idx = drawdown.idxmin()

    # Chercher le début du drawdown (dernier pic avant le minimum)
    dd_start_idx = None
    for i in range(max_dd_idx, -1, -1):
        if i < len(drawdown) and drawdown.iloc[i] == 0:  # Nouveau pic
            dd_start_idx = equity_curve.index[i]
            break

    # Chercher la fin du drawdown (retour au pic après le minimum)
    dd_end_idx = None
    for i in range(max_dd_idx, len(drawdown)):
        if drawdown.iloc[i] >= -0.01:  # Retour proche du pic (1% de tolérance)
            dd_end_idx = equity_curve.index[i]
            break

    return current_dd, max_dd, dd_start_idx, dd_end_idx


def calculate_support_resistance(prices: pd.Series, window: int = 20) -> tuple:
    """
    Calcule les niveaux de support et résistance

    Args:
        prices: Série des prix
        window: Fenêtre pour identifier les pics/creux

    Returns:
        Tuple (support_level, resistance_level)
    """
    if len(prices) < window * 2:
        return None, None

    # Identifier les pics et creux locaux
    local_mins = []
    local_maxs = []

    for i in range(window, len(prices) - window):
        # Vérifier si c'est un minimum local
        window_prices = prices.iloc[i - window : i + window + 1]
        if prices.iloc[i] == window_prices.min():
            local_mins.append(prices.iloc[i])

        # Vérifier si c'est un maximum local
        if prices.iloc[i] == window_prices.max():
            local_maxs.append(prices.iloc[i])

    # Calculer les niveaux de support et résistance
    import numpy as np

    support = np.median(local_mins) if local_mins else None
    resistance = np.median(local_maxs) if local_maxs else None

    return support, resistance


def validate_backtest_data(data: Dict[str, Any]) -> bool:
    """
    Valide la structure des données de backtest

    Args:
        data: Données de backtest à valider

    Returns:
        True si valide, False sinon
    """
    required_keys = ["asset_name", "price_history", "portfolio"]

    # Vérifier les clés obligatoires
    for key in required_keys:
        if key not in data:
            return False

    # Vérifier la structure price_history
    price_history = data["price_history"]
    required_price_fields = ["dates", "close"]

    for field in required_price_fields:
        if field not in price_history or not price_history[field]:
            return False

    # Vérifier que les listes ont la même longueur
    dates_len = len(price_history["dates"])
    for field in ["close", "open", "high", "low"]:
        if field in price_history and len(price_history[field]) != dates_len:
            return False

    # Vérifier la structure portfolio
    portfolio = data["portfolio"]
    if "equity" not in portfolio or not portfolio["equity"]:
        return False

    return True
