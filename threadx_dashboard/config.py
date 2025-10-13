"""
Configuration globale pour ThreadX Dashboard
===========================================

Ce module contient tous les paramètres de configuration pour l'application
Dash de backtesting de trading.
"""

import os
from pathlib import Path

# Répertoire racine de l'application
BASE_DIR = Path(__file__).parent

# =============================================================================
# THÈME ET COULEURS
# =============================================================================

THEME = {
    "primary_bg": "#1a1a1a",  # Fond principal sombre
    "secondary_bg": "#242424",  # Fond secondaire
    "tertiary_bg": "#2a2a2a",  # Fond tertiaire
    "text_primary": "#ffffff",  # Texte blanc
    "text_secondary": "#b0b0b0",  # Texte gris
    "accent_primary": "#00d4ff",  # Cyan principal
    "accent_secondary": "#00a8cc",  # Cyan sombre
    "success": "#00ff00",  # Vert succès
    "danger": "#ff4444",  # Rouge erreur
    "warning": "#ffaa00",  # Orange warning
    "border_color": "#404040",  # Couleur bordure
}

# Palettes pour les graphiques
CHART_COLORS = [
    "#00d4ff",
    "#ff6b6b",
    "#4ecdc4",
    "#45b7d1",
    "#f9ca24",
    "#f0932b",
    "#eb4d4b",
    "#6c5ce7",
]

# =============================================================================
# AUTHENTIFICATION
# =============================================================================

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "True").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "False").lower() == "true"
DEFAULT_USERNAME = os.getenv("DEFAULT_USERNAME", "admin")
DEFAULT_PASSWORD = os.getenv("DEFAULT_PASSWORD", "admin123")  # À changer en production
SECRET_KEY = os.getenv(
    "SECRET_KEY", "threadx-dashboard-secret-key-change-in-production"
)
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "86400"))  # 24 heures

# =============================================================================
# API ET DONNÉES
# =============================================================================

DATA_SOURCE = os.getenv("DATA_SOURCE", "Yahoo Finance")
CACHE_DURATION = int(os.getenv("CACHE_DURATION", "3600"))  # 1 heure en secondes
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Configuration des données
DATA_CONFIG = {
    "default_symbol": "BTCUSDT",
    "default_timeframe": "1h",
    "max_history_days": 365,
    "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "supported_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"],
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "dashboard.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# SERVEUR
# =============================================================================

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8050"))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# =============================================================================
# LAYOUTS ET COMPOSANTS
# =============================================================================

LAYOUT_CONFIG = {
    "navbar_height": "60px",
    "sidebar_width": "250px",
    "sidebar_width_collapsed": "80px",
    "footer_height": "40px",
    "content_padding": "20px",
    "card_border_radius": "8px",
    "transition_duration": "0.3s",
}

# =============================================================================
# BACKTESTING
# =============================================================================

BACKTEST_CONFIG = {
    "initial_capital": 10000,
    "commission": 0.001,  # 0.1%
    "slippage": 0.0005,  # 0.05%
    "max_positions": 1,
    "default_strategy": "bb_atr",
    "risk_free_rate": 0.02,  # 2% annuel
}

# =============================================================================
# FICHIERS STATIQUES
# =============================================================================

ASSETS_PATH = BASE_DIR / "assets"
LOGO_PATH = ASSETS_PATH / "logo.png"
FAVICON_PATH = ASSETS_PATH / "favicon.ico"
CSS_PATH = ASSETS_PATH / "style.css"

# =============================================================================
# FONCTION D'EXPORT
# =============================================================================


def get_config():
    """
    Retourne un dictionnaire avec toute la configuration.

    Returns:
        dict: Configuration complète de l'application
    """
    return {
        "theme": THEME,
        "auth": {
            "enabled": AUTH_ENABLED,
            "demo_mode": DEMO_MODE,
            "session_timeout": SESSION_TIMEOUT,
        },
        "data": DATA_CONFIG,
        "layout": LAYOUT_CONFIG,
        "backtest": BACKTEST_CONFIG,
        "server": {"host": HOST, "port": PORT, "debug": DEBUG},
    }
