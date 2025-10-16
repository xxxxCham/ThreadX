"""
ThreadX UI Layout - Structure Principale Dashboard
==================================================

Layout statique Dash pour ThreadX Dashboard.
Architecture 4 onglets modulaire avec composants P5-P6 intégrés.

Onglets:
    1. Data Creation & Management - Téléchargement Binance, validation UDFI,
       registry, mise à jour indicateurs batch
    2. Indicators - Build cache indicateurs techniques
    3. Backtest - Exécution stratégies + résultats
    4. Optimization - Parameter sweeps + heatmaps

Design:
    - Theme: Bootstrap DARKLY
    - Responsive: Breakpoints md/lg
    - IDs déterministes: data-*, ind-*, bt-*, opt-*

Usage:
    from threadx.ui.layout import create_layout
    app.layout = create_layout(bridge)

Author: ThreadX Framework
Version: Prompt 10 - Data Creation & Management + Layout Principal
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

# Import composants P5-P6
from threadx.ui.components.data_manager import create_data_manager_panel
from threadx.ui.components.indicators_panel import create_indicators_panel
from threadx.ui.components.backtest_panel import create_backtest_panel
from threadx.ui.components.optimization_panel import create_optimization_panel


def create_layout(bridge=None):
    """
    Créer layout principal Dashboard ThreadX.

    Args:
        bridge: ThreadXBridge instance (optionnel, pour P7).
                Pas utilisé ici mais signature prête pour callbacks.

    Returns:
        dbc.Container: Layout complet avec 4 onglets.

    Structure:
        Container (plein écran)
          ├─ Header (titre + sous-titre)
          ├─ Tabs (Data, Indicators, Backtest, Optimization)
          │   ├─ Tab 1: Data Manager
          │   │   └─ Row: Settings (gauche) + Results (droite)
          │   ├─ Tab 2: Indicators
          │   │   └─ Row: Settings + Results
          │   ├─ Tab 3: Backtest
          │   │   └─ Row: Settings + Results
          │   └─ Tab 4: Optimization
          │       └─ Row: Settings + Results
          └─ Footer (optionnel)
    """

    return dbc.Container(
        fluid=True,
        className="bg-dark text-light min-vh-100 p-0",
        children=[
            # ═════════════════════════════════════════════════════
            # HIDDEN STORES - Task IDs for Polling (P7)
            # ═════════════════════════════════════════════════════
            dcc.Store(id="data-task-store", data=None),
            dcc.Store(id="indicators-task-store", data=None),
            dcc.Store(id="bt-task-store", data=None),
            dcc.Store(id="opt-task-store", data=None),
            # ═════════════════════════════════════════════════════
            # GLOBAL POLLING INTERVAL - 500ms (P7)
            # ═════════════════════════════════════════════════════
            dcc.Interval(
                id="global-interval",
                interval=500,  # 500ms
                n_intervals=0,
                disabled=False,
            ),
            # ═════════════════════════════════════════════════════
            # HEADER - Titre + Sous-titre
            # ═════════════════════════════════════════════════════
            dbc.Navbar(
                dbc.Container(
                    fluid=True,
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H2(
                                        "ThreadX Dashboard",
                                        className="mb-0 text-light",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    html.P(
                                        "Backtesting Framework - "
                                        "GPU-Accelerated Trading Analysis",
                                        className="mb-0 text-muted " "small",
                                    ),
                                    width="auto",
                                ),
                            ],
                            align="center",
                            className="g-3",
                        ),
                    ],
                ),
                color="dark",
                dark=True,
                className="mb-4",
            ),
            # ═════════════════════════════════════════════════════
            # TABS - 4 Onglets Principaux
            # ═════════════════════════════════════════════════════
            dbc.Container(
                fluid=True,
                children=[
                    dcc.Tabs(
                        id="main-tabs",
                        value="tab-data",
                        className="bg-dark",
                        children=[
                            # ─────────────────────────────────────
                            # TAB 1: Data Creation & Management
                            # ─────────────────────────────────────
                            dcc.Tab(
                                label="Data Creation & Management",
                                value="tab-data",
                                className="bg-dark text-light",
                                selected_className="bg-primary",
                                children=[
                                    html.Div(
                                        className="p-0",
                                        children=[
                                            create_data_manager_panel(),
                                        ],
                                    )
                                ],
                            ),
                            # ─────────────────────────────────────
                            # TAB 2: Indicators
                            # ─────────────────────────────────────
                            dcc.Tab(
                                label="Indicators",
                                value="tab-indicators",
                                className="bg-dark text-light",
                                selected_className="bg-primary",
                                children=[
                                    html.Div(
                                        className="p-4",
                                        children=[
                                            html.H3(
                                                "Technical Indicators",
                                                className="text-light mb-1",
                                            ),
                                            html.P(
                                                "Build and cache indicator calculations",
                                                className="text-muted mb-4",
                                            ),
                                            create_indicators_panel(),
                                        ],
                                    )
                                ],
                            ),
                            # ─────────────────────────────────────
                            # TAB 3: Backtest
                            # ─────────────────────────────────────
                            dcc.Tab(
                                label="Backtest",
                                value="tab-backtest",
                                className="bg-dark text-light",
                                selected_className="bg-primary",
                                children=[
                                    html.Div(
                                        className="p-4",
                                        children=[
                                            html.H3(
                                                "Strategy Backtesting",
                                                className="text-light mb-1",
                                            ),
                                            html.P(
                                                "Run backtests and analyze performance metrics",
                                                className="text-muted mb-4",
                                            ),
                                            create_backtest_panel(),
                                        ],
                                    )
                                ],
                            ),
                            # ─────────────────────────────────────
                            # TAB 4: Optimization
                            # ─────────────────────────────────────
                            dcc.Tab(
                                label="Optimization",
                                value="tab-optimization",
                                className="bg-dark text-light",
                                selected_className="bg-primary",
                                children=[
                                    html.Div(
                                        className="p-4",
                                        children=[
                                            html.H3(
                                                "Parameter Optimization",
                                                className="text-light mb-1",
                                            ),
                                            html.P(
                                                "Parameter sweeps and optimization heatmaps",
                                                className="text-muted mb-4",
                                            ),
                                            create_optimization_panel(),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    )
                ],
            ),
            # ═════════════════════════════════════════════════════
            # FOOTER (Optionnel - Version + Crédits)
            # ═════════════════════════════════════════════════════
            html.Footer(
                dbc.Container(
                    fluid=True,
                    children=[
                        html.Hr(className="border-secondary"),
                        html.P(
                            "ThreadX v0.1.0 - Built with Dash & Bootstrap",
                            className="text-center text-muted small mb-0",
                        ),
                    ],
                ),
                className="mt-5 pb-3",
            ),
        ],
    )


def _create_tab_layout(tab_id, title, subtitle):
    """
    Créer layout standard pour un onglet.

    Pattern répété pour chaque tab:
        - Titre + Sous-titre
        - Row responsive (Settings gauche, Results droite)
        - Placeholders pour composants futurs (P5-P6)

    Args:
        tab_id: Préfixe ID unique (ex: 'data', 'ind', 'bt', 'opt').
        title: Titre affiché (ex: "Data Management").
        subtitle: Description courte (ex: "Upload and validate data").

    Returns:
        html.Div: Layout tab avec zones Settings/Results.
    """

    return html.Div(
        className="p-4",
        children=[
            # Titre Tab
            html.H3(title, className="text-light mb-1"),
            html.P(subtitle, className="text-muted mb-4"),
            # Grille Responsive (Settings + Results)
            dbc.Row(
                [
                    # ─────────────────────────────────────────────
                    # SETTINGS PANEL (Gauche, 1/3 largeur)
                    # ─────────────────────────────────────────────
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "Settings",
                                        className="bg-secondary text-light",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id=f"{tab_id}-settings-pane",
                                                children=[
                                                    html.P(
                                                        "Settings panel "
                                                        "(Placeholder for P5-P6)",
                                                        className=(
                                                            "text-muted " "fst-italic"
                                                        ),
                                                    ),
                                                ],
                                            ),
                                        ],
                                        className="bg-dark",
                                    ),
                                ],
                                className="bg-dark border-secondary mb-3",
                            ),
                        ],
                        md=4,
                        lg=3,
                        className="mb-3 mb-md-0",
                    ),
                    # ─────────────────────────────────────────────
                    # RESULTS PANEL (Droite, 2/3 largeur)
                    # ─────────────────────────────────────────────
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "Results",
                                        className="bg-secondary text-light",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id=f"{tab_id}-results-pane",
                                                children=[
                                                    html.P(
                                                        (
                                                            "Results panel "
                                                            "(Placeholder for "
                                                            "P5-P6)"
                                                        ),
                                                        className=(
                                                            "text-muted " "fst-italic"
                                                        ),
                                                    ),
                                                ],
                                            ),
                                        ],
                                        className="bg-dark",
                                    ),
                                ],
                                className="bg-dark border-secondary",
                            ),
                        ],
                        md=8,
                        lg=9,
                    ),
                ],
                className="g-3",
            ),
        ],
    )
