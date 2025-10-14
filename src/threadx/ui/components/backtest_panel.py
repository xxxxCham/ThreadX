"""
ThreadX UI - Backtest Panel Component
======================================

Composant Dash pour configuration et résultats de backtesting.
Fournit interface pour stratégies, paramètres, et visualisation
des résultats (equity curve, drawdown, trades, metrics).

IDs Exposés (pour callbacks P7):
    Inputs: bt-strategy, bt-symbol, bt-timeframe, bt-period,
            bt-std, bt-run-btn
    Outputs: bt-equity-graph, bt-drawdown-graph, bt-trades-table,
             bt-metrics-table, bt-loading, bt-status

Author: ThreadX Framework
Version: Prompt 6 - Composants Backtest + Optimization
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.development.base_component import Component


def create_backtest_panel() -> Component:
    """
    Create backtest configuration and results panel component.

    Provides UI for configuring backtest parameters, running backtests
    via Bridge (P7), and displaying results (equity curve, drawdown,
    trades table, metrics). All graphs are empty placeholders.

    Returns:
        Component: Complete backtest panel with config and results.

    Structure:
        Row (responsive)
          ├─ Col (md=4): Configuration Panel
          │   ├─ Strategy dropdown
          │   ├─ Symbol/Timeframe dropdowns
          │   ├─ Strategy params (period, std)
          │   └─ Run button
          └─ Col (md=8): Results Panel
              ├─ Tabs (Charts, Trades, Metrics)
              └─ Loading spinner
    """

    # Configuration panel (left column)
    config_panel = dbc.Col(
        md=4,
        style={"borderRight": "1px solid #444", "paddingRight": "20px"},
        children=[
            html.H4("Backtest Configuration", className="text-light mb-3"),
            # Strategy Selection
            html.Label("Strategy", className="text-light mb-2"),
            dcc.Dropdown(
                id="bt-strategy",
                options=[
                    {"label": "EMA Crossover", "value": "ema_crossover"},
                    {"label": "Bollinger Reversion", "value": "bollinger_reversion"},
                    {"label": "RSI Momentum", "value": "rsi_momentum"},
                    {"label": "MACD Trend", "value": "macd_trend"},
                ],
                value="ema_crossover",
                placeholder="Select strategy...",
                className="mb-3 bg-dark",
            ),
            # Symbol Selection
            html.Label("Symbol", className="text-light mb-2"),
            dcc.Dropdown(
                id="bt-symbol",
                options=[],
                placeholder="Select symbol... (from data registry)",
                className="mb-3 bg-dark",
            ),
            # Timeframe Selection
            html.Label("Timeframe", className="text-light mb-2"),
            dcc.Dropdown(
                id="bt-timeframe",
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
            # Strategy Parameters
            html.H5("Strategy Parameters", className="text-light mb-3"),
            html.Label("Period", className="text-light small mb-1"),
            dcc.Input(
                id="bt-period",
                type="number",
                value=20,
                min=1,
                max=500,
                className="form-control bg-dark text-light mb-3",
            ),
            html.Label("Standard Deviation", className="text-light small mb-1"),
            dcc.Input(
                id="bt-std",
                type="number",
                value=2.0,
                min=0.1,
                max=5.0,
                step=0.1,
                className="form-control bg-dark text-light mb-3",
            ),
            html.Label("Initial Capital", className="text-light small mb-1"),
            dcc.Input(
                id="bt-initial-capital",
                type="number",
                value=10000,
                min=100,
                max=1000000,
                className="form-control bg-dark text-light mb-3",
            ),
            html.Label("Commission (%)", className="text-light small mb-1"),
            dcc.Input(
                id="bt-commission",
                type="number",
                value=0.1,
                min=0,
                max=10,
                step=0.01,
                className="form-control bg-dark text-light mb-3",
            ),
            # Run Button
            dbc.Button(
                "Run Backtest",
                id="bt-run-btn",
                color="success",
                className="w-100 mt-3",
                n_clicks=0,
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
                id="bt-tabs",
                active_tab="charts",
                children=[
                    # Charts Tab
                    dbc.Tab(
                        label="Charts",
                        tab_id="charts",
                        children=[
                            dcc.Loading(
                                id="bt-charts-loading",
                                type="circle",
                                children=[
                                    html.Div(
                                        className="mb-4",
                                        children=[
                                            html.H6(
                                                "Equity Curve",
                                                className=("text-light mt-3"),
                                            ),
                                            dcc.Graph(
                                                id="bt-equity-graph",
                                                figure=go.Figure(
                                                    layout=go.Layout(
                                                        template="plotly_dark",
                                                        title=(
                                                            "No backtest " "run yet"
                                                        ),
                                                        height=300,
                                                    )
                                                ),
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="mb-4",
                                        children=[
                                            html.H6("Drawdown", className="text-light"),
                                            dcc.Graph(
                                                id="bt-drawdown-graph",
                                                figure=go.Figure(
                                                    layout=go.Layout(
                                                        template=("plotly_dark"),
                                                        title=(
                                                            "No backtest " "run yet"
                                                        ),
                                                        height=300,
                                                    )
                                                ),
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Trades Tab
                    dbc.Tab(
                        label="Trades",
                        tab_id="trades",
                        children=[
                            dcc.Loading(
                                id="bt-trades-loading",
                                type="circle",
                                children=[
                                    html.Div(
                                        id="bt-trades-table",
                                        className="mt-3",
                                        children=[
                                            html.Div(
                                                className=(
                                                    "text-center " "text-muted mt-5"
                                                ),
                                                children=[
                                                    html.I(
                                                        className=(
                                                            "bi bi-list-ul "
                                                            "display-4 mb-3"
                                                        )
                                                    ),
                                                    html.P(
                                                        "No trades yet",
                                                        className="mb-0",
                                                    ),
                                                    html.P(
                                                        (
                                                            "Run a backtest "
                                                            "to see trades"
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
                    # Metrics Tab
                    dbc.Tab(
                        label="Metrics",
                        tab_id="metrics",
                        children=[
                            dcc.Loading(
                                id="bt-metrics-loading",
                                type="circle",
                                children=[
                                    html.Div(
                                        id="bt-metrics-table",
                                        className="mt-3",
                                        children=[
                                            html.Div(
                                                className=(
                                                    "text-center " "text-muted mt-5"
                                                ),
                                                children=[
                                                    html.I(
                                                        className=(
                                                            "bi "
                                                            "bi-bar-chart "
                                                            "display-4 mb-3"
                                                        )
                                                    ),
                                                    html.P(
                                                        "No metrics yet",
                                                        className="mb-0",
                                                    ),
                                                    html.P(
                                                        (
                                                            "Run a backtest "
                                                            "to see metrics"
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
                ],
            ),
        ],
    )

    # Main panel
    return html.Div(
        className="p-4 bg-dark",
        children=[
            html.H4("Backtest Configuration & Results", className="text-light mb-1"),
            html.P(
                "Configure and run backtests for trading strategies",
                className="text-muted mb-4",
            ),
            dbc.Row(className="g-3", children=[config_panel, results_panel]),
            # Global Loading Status
            html.Hr(className="border-secondary my-3"),
            dcc.Loading(
                id="bt-loading",
                type="default",
                children=html.Div(
                    id="bt-status",
                    className="text-center text-muted",
                    children="Ready to run backtest",
                ),
            ),
        ],
    )
