"""
ThreadX UI - Optimization Panel Component
==========================================

Composant Dash pour configuration et résultats d'optimisation
de paramètres (parameter sweeps). Fournit interface pour grille
de paramètres et visualisation des résultats (top params, heatmap).

IDs Exposés (pour callbacks P7):
    Inputs: opt-strategy, opt-symbol, opt-period-min,
            opt-period-max, opt-period-step, opt-std-min,
            opt-std-max, opt-std-step, opt-run-btn
    Outputs: opt-results-table, opt-heatmap, opt-loading,
             opt-status

Author: ThreadX Framework
Version: Prompt 6 - Composants Backtest + Optimization
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.development.base_component import Component


def create_optimization_panel() -> Component:
    """
    Create optimization sweep configuration and results panel.

    Provides UI for configuring parameter grid, running sweeps
    via Bridge (P7), and displaying results (top parameters table,
    heatmap visualization). All outputs are empty placeholders.

    Returns:
        Component: Complete optimization panel with config/results.

    Structure:
        Row (responsive)
          ├─ Col (md=4): Configuration Panel
          │   ├─ Strategy/Symbol dropdowns
          │   ├─ Parameter grid (min/max/step)
          │   └─ Run button
          └─ Col (md=8): Results Panel
              ├─ Tabs (Top Results, Heatmap)
              └─ Loading spinner
    """

    # Configuration panel (left column)
    config_panel = dbc.Col(
        md=4,
        style={"borderRight": "1px solid #444", "paddingRight": "20px"},
        children=[
            html.H4("Sweep Configuration", className="text-light mb-3"),
            # Strategy Selection
            html.Label("Strategy", className="text-light mb-2"),
            dcc.Dropdown(
                id="opt-strategy",
                options=[
                    {"label": "EMA Crossover", "value": "ema_crossover"},
                    {"label": "Bollinger Reversion", "value": "bollinger_reversion"},
                    {"label": "RSI Momentum", "value": "rsi_momentum"},
                    {"label": "MACD Trend", "value": "macd_trend"},
                ],
                value="bollinger_reversion",
                placeholder="Select strategy...",
                className="mb-3 bg-dark",
            ),
            # Symbol Selection
            html.Label("Symbol", className="text-light mb-2"),
            dcc.Dropdown(
                id="opt-symbol",
                options=[],
                placeholder="Select symbol... (from data registry)",
                className="mb-3 bg-dark",
            ),
            # Timeframe Selection
            html.Label("Timeframe", className="text-light mb-2"),
            dcc.Dropdown(
                id="opt-timeframe",
                options=[
                    {"label": "1 minute", "value": "1m"},
                    {"label": "5 minutes", "value": "5m"},
                    {"label": "15 minutes", "value": "15m"},
                    {"label": "1 hour", "value": "1h"},
                    {"label": "4 hours", "value": "4h"},
                    {"label": "1 day", "value": "1d"},
                ],
                value="1h",
                placeholder="Select timeframe...",
                className="mb-3 bg-dark",
            ),
            html.Hr(className="border-secondary my-3"),
            # Parameter Grid
            html.H5("Parameter Grid", className="text-light mb-3"),
            # Period Grid
            html.H6("Period Range", className="text-light small mb-2"),
            dbc.Row(
                className="g-2 mb-3",
                children=[
                    dbc.Col(
                        md=4,
                        children=[
                            html.Label("Min", className="text-light small mb-1"),
                            dcc.Input(
                                id="opt-period-min",
                                type="number",
                                value=10,
                                min=1,
                                max=500,
                                className=("form-control bg-dark text-light"),
                            ),
                        ],
                    ),
                    dbc.Col(
                        md=4,
                        children=[
                            html.Label("Max", className="text-light small mb-1"),
                            dcc.Input(
                                id="opt-period-max",
                                type="number",
                                value=50,
                                min=1,
                                max=500,
                                className=("form-control bg-dark text-light"),
                            ),
                        ],
                    ),
                    dbc.Col(
                        md=4,
                        children=[
                            html.Label("Step", className="text-light small mb-1"),
                            dcc.Input(
                                id="opt-period-step",
                                type="number",
                                value=5,
                                min=1,
                                max=100,
                                className=("form-control bg-dark text-light"),
                            ),
                        ],
                    ),
                ],
            ),
            # Std Deviation Grid
            html.H6("Standard Deviation Range", className="text-light small mb-2"),
            dbc.Row(
                className="g-2 mb-3",
                children=[
                    dbc.Col(
                        md=4,
                        children=[
                            html.Label("Min", className="text-light small mb-1"),
                            dcc.Input(
                                id="opt-std-min",
                                type="number",
                                value=1.0,
                                min=0.1,
                                max=10.0,
                                step=0.1,
                                className=("form-control bg-dark text-light"),
                            ),
                        ],
                    ),
                    dbc.Col(
                        md=4,
                        children=[
                            html.Label("Max", className="text-light small mb-1"),
                            dcc.Input(
                                id="opt-std-max",
                                type="number",
                                value=3.0,
                                min=0.1,
                                max=10.0,
                                step=0.1,
                                className=("form-control bg-dark text-light"),
                            ),
                        ],
                    ),
                    dbc.Col(
                        md=4,
                        children=[
                            html.Label("Step", className="text-light small mb-1"),
                            dcc.Input(
                                id="opt-std-step",
                                type="number",
                                value=0.5,
                                min=0.1,
                                max=5.0,
                                step=0.1,
                                className=("form-control bg-dark text-light"),
                            ),
                        ],
                    ),
                ],
            ),
            # Run Button
            dbc.Button(
                "Run Sweep",
                id="opt-run-btn",
                color="warning",
                className="w-100 mt-3",
                n_clicks=0,
            ),
            # Estimated Combinations
            html.Div(
                id="opt-combinations-info",
                className="text-center text-muted mt-3 small",
                children="Estimated combinations: ~48",
            ),
        ],
    )

    # Results panel (right column)
    results_panel = dbc.Col(
        md=8,
        style={"paddingLeft": "20px"},
        children=[
            html.H4("Results", className="text-light mb-3"),
            dbc.Tabs(
                id="opt-tabs",
                active_tab="top-results",
                children=[
                    # Top Results Tab
                    dbc.Tab(
                        label="Top Results",
                        tab_id="top-results",
                        children=[
                            dcc.Loading(
                                id="opt-results-loading",
                                type="circle",
                                children=[
                                    html.Div(
                                        id="opt-results-table",
                                        className="mt-3",
                                        children=[
                                            html.Div(
                                                className=(
                                                    "text-center " "text-muted mt-5"
                                                ),
                                                children=[
                                                    html.I(
                                                        className=(
                                                            "bi bi-trophy "
                                                            "display-4 mb-3"
                                                        )
                                                    ),
                                                    html.P(
                                                        (
                                                            "No optimization "
                                                            "results yet"
                                                        ),
                                                        className="mb-0",
                                                    ),
                                                    html.P(
                                                        (
                                                            "Run a sweep to "
                                                            "see top params"
                                                        ),
                                                        className=(
                                                            "small " "text-muted"
                                                        ),
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Heatmap Tab
                    dbc.Tab(
                        label="Heatmap",
                        tab_id="heatmap",
                        children=[
                            dcc.Loading(
                                id="opt-heatmap-loading",
                                type="circle",
                                children=[
                                    html.Div(
                                        className="mt-3",
                                        children=[
                                            dcc.Graph(
                                                id="opt-heatmap",
                                                figure=go.Figure(
                                                    layout=go.Layout(
                                                        template=("plotly_dark"),
                                                        title=("No sweep " "run yet"),
                                                        height=500,
                                                        xaxis_title=("Period"),
                                                        yaxis_title=("Std Dev"),
                                                    )
                                                ),
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # Main panel
    return html.Div(
        className="p-4 bg-dark",
        children=[
            html.H4("Parameter Optimization & Sweep", className="text-light mb-1"),
            html.P(
                "Configure parameter grid and run optimization sweeps",
                className="text-muted mb-4",
            ),
            dbc.Row(className="g-3", children=[config_panel, results_panel]),
            # Global Loading Status
            html.Hr(className="border-secondary my-3"),
            dcc.Loading(
                id="opt-loading",
                type="default",
                children=html.Div(
                    id="opt-status",
                    className="text-center text-muted",
                    children="Ready to run sweep",
                ),
            ),
        ],
    )
