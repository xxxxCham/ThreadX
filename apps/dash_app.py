"""
ThreadX Dash Application - Main Entry Point
===========================================

Application Dash principale pour ThreadX.
Charge le layout modulaire, configure le serveur, et expose
l'interface web sur le port configuré (default: 8050).

Architecture:
    apps/dash_app.py (CE FICHIER)
         ↓
    src/threadx/ui/layout.py (Layout statique)
         ↓
    src/threadx/ui/components/* (P5-P6)
         ↓
    src/threadx/ui/callbacks.py (P7)

Usage:
    # Depuis racine ThreadX
    python apps/dash_app.py

    # Avec port custom
    $env:THREADX_DASH_PORT=8888
    python apps/dash_app.py

Configuration:
    - Port : Variable THREADX_DASH_PORT (fallback 8050)
    - Theme : Bootstrap DARKLY
    - Debug : False (production-ready)

Author: ThreadX Framework
Version: Prompt 4 - Layout Principal
"""

import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc

# Ajouter src au PYTHONPATH si nécessaire
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from threadx.ui.layout import create_layout

# Optional: Import Bridge pour signature future (P7)
# Pas d'appel métier ici, juste passage instance
try:
    from threadx.bridge import ThreadXBridge

    bridge = ThreadXBridge(max_workers=4)
except ImportError:
    # Bridge pas encore implémenté ou tests isolés
    bridge = None


# Configuration application
PORT = int(os.environ.get("THREADX_DASH_PORT", 8050))
DEBUG = os.environ.get("THREADX_DASH_DEBUG", "false").lower() == "true"

# Initialiser Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="ThreadX Dashboard",
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0",
        }
    ],
)

# Serveur Flask sous-jacent (pour déploiement production)
server = app.server

# Charger layout principal
app.layout = create_layout(bridge)


if __name__ == "__main__":
    print("=" * 60)
    print("ThreadX Dash Dashboard")
    print("=" * 60)
    print(f"Server starting on: http://127.0.0.1:{PORT}")
    print(f"Debug mode: {DEBUG}")
    print(f"Theme: Bootstrap DARKLY")
    if bridge:
        print(f"Bridge: Initialized ({bridge.config.max_workers} workers)")
    else:
        print("Bridge: Not available (will be connected in P7)")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    app.run_server(debug=DEBUG, port=PORT, host="127.0.0.1")
