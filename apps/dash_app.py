"""
ThreadX Dash Application - Main Entry Point
===========================================

Application Dash principale pour ThreadX.
Charge le layout modulaire, configure le serveur et expose
l'interface web sur le port configuré (default: 8050).

Architecture:
    apps/dash_app.py (ce fichier)
        ↓
    src/threadx/ui/layout.py (layout Dash)
        ↓
    src/threadx/ui/components/* (panels UI)
        ↓
    src/threadx/ui/callbacks.py (Bridge & logique async)

Usage:
    # Depuis la racine du repo
    python apps/dash_app.py

    # Avec port custom
    $env:THREADX_DASH_PORT=8888
    python apps/dash_app.py

Configuration:
    - Port : variable THREADX_DASH_PORT (fallback 8050)
    - Theme : Bootstrap DARKLY
    - Debug : False par défaut (production ready)

Version: Dash UI consolidée
"""

from __future__ import annotations

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

# Bridge et callbacks (optionnels en environnement réduit)
try:
    from threadx.bridge import ThreadXBridge
    from threadx.ui.callbacks import register_callbacks

    bridge = ThreadXBridge(max_workers=4)
except ImportError:
    bridge = None
    register_callbacks = None

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

# Layout principal (Dash uniquement)
app.layout = create_layout(bridge)

# Enregistrer callbacks si Bridge disponible
if bridge and register_callbacks:
    register_callbacks(app, bridge)
    print("Callbacks: registered (Bridge active)")
else:
    print("Callbacks: skipped (Bridge unavailable)")


if __name__ == "__main__":
    print("=" * 60)
    print("ThreadX Dash Dashboard")
    print("=" * 60)
    print(f"Server starting on: http://127.0.0.1:{PORT}")
    print(f"Debug mode: {DEBUG}")
    print(f"Theme: Bootstrap DARKLY")
    if bridge:
        print(f"Bridge: initialized ({bridge.config.max_workers} worker(s))")
    else:
        print("Bridge: not available (callbacks disabled)")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=DEBUG, port=PORT, host="127.0.0.1")
