"""
ThreadX UI - Indicators Panel Component
========================================

Composant Dash pour configuration et gestion des indicateurs techniques.
Fournit interface pour calcul, caching, et monitoring des indicateurs.

IDs Exposés (pour callbacks P7):
    Inputs: indicators-symbol, indicators-timeframe, ema-period,
            rsi-period, bollinger-period, bollinger-std,
            build-indicators-btn
    Outputs: indicators-cache-body, indicators-alert,
             indicators-loading

Author: ThreadX Framework
Version: Prompt 5 - Composants Data + Indicators
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_indicators_panel():
    """
    Create indicators configuration panel component.

    Returns:
        html.Div: Complete indicators panel with forms and tables.
    """

    # Configuration card (left column)
    config_card = dbc.Card(
        className="bg-dark border-secondary h-100",
        children=[
            dbc.CardHeader("Configuration", className="bg-secondary text-light"),
            dbc.CardBody(
                className="bg-dark",
                children=[
                    # Symbol Dropdown
                    html.Label("Symbol", className="text-light mb-2"),
                    dcc.Dropdown(
                        id="indicators-symbol",
                        options=[],
                        placeholder=("Select symbol... (loaded from registry)"),
                        className="mb-3 bg-dark",
                    ),
                    # Timeframe Dropdown
                    html.Label("Timeframe", className="text-light mb-2"),
                    dcc.Dropdown(
                        id="indicators-timeframe",
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
                    html.Hr(className="border-secondary"),
                    # EMA Parameters
                    html.H6("EMA Parameters", className="text-light mb-2 mt-3"),
                    html.Label("Period", className="text-light small mb-1"),
                    dcc.Input(
                        id="ema-period",
                        type="number",
                        value=20,
                        min=1,
                        max=500,
                        className=("form-control bg-dark text-light mb-3"),
                    ),
                    # RSI Parameters
                    html.H6("RSI Parameters", className="text-light mb-2"),
                    html.Label("Period", className="text-light small mb-1"),
                    dcc.Input(
                        id="rsi-period",
                        type="number",
                        value=14,
                        min=1,
                        max=100,
                        className=("form-control bg-dark text-light mb-3"),
                    ),
                    # Bollinger Parameters
                    html.H6("Bollinger Bands Parameters", className="text-light mb-2"),
                    html.Label("Period", className="text-light small mb-1"),
                    dcc.Input(
                        id="bollinger-period",
                        type="number",
                        value=20,
                        min=1,
                        max=100,
                        className=("form-control bg-dark text-light mb-3"),
                    ),
                    html.Label("Standard Deviation", className="text-light small mb-1"),
                    dcc.Input(
                        id="bollinger-std",
                        type="number",
                        value=2.0,
                        min=0.1,
                        max=5.0,
                        step=0.1,
                        className=("form-control bg-dark text-light mb-3"),
                    ),
                    # Build Button
                    dbc.Button(
                        "Build Indicators Cache",
                        id="build-indicators-btn",
                        color="success",
                        className="w-100 mt-4",
                        n_clicks=0,
                    ),
                ],
            ),
        ],
    )

    # Results card (right column)
    results_card = dbc.Card(
        className="bg-dark border-secondary h-100",
        children=[
            dbc.CardHeader("Cache Status", className="bg-secondary text-light"),
            dbc.CardBody(
                className="bg-dark",
                children=[
                    # Alert Messages
                    dbc.Alert(
                        id="indicators-alert",
                        is_open=False,
                        dismissable=True,
                        className="mb-3",
                    ),
                    # Loading Wrapper
                    dcc.Loading(
                        id="indicators-loading",
                        type="circle",
                        children=[
                            # Cache Status Table
                            dbc.Table(
                                id="indicators-cache-table",
                                striped=True,
                                bordered=True,
                                hover=True,
                                color="dark",
                                className="mb-0",
                                children=[
                                    html.Thead(
                                        children=[
                                            html.Tr(
                                                [
                                                    html.Th("Indicator"),
                                                    html.Th("Parameters"),
                                                    html.Th("Status"),
                                                    html.Th("Size"),
                                                ]
                                            )
                                        ]
                                    ),
                                    html.Tbody(
                                        id="indicators-cache-body",
                                        children=[],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Empty State
                    html.Div(
                        id="indicators-empty-state",
                        className="text-center text-muted mt-4",
                        children=[
                            html.I(className=("bi bi-graph-up display-4 mb-3")),
                            html.P(
                                "No indicators cached yet",
                                className="mb-0",
                            ),
                            html.P(
                                "Configure params and click Build",
                                className="small text-muted",
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
            html.H4("Indicators Configuration", className="text-light mb-1"),
            html.P(
                "Configure and build technical indicators cache",
                className="text-muted mb-4",
            ),
            dbc.Row(
                className="g-3",
                children=[
                    dbc.Col(config_card, md=6),
                    dbc.Col(results_card, md=6),
                ],
            ),
        ],
    )
