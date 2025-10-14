"""
ThreadX UI - Data Manager Component
===================================

Composant Dash pour gestion des données de marché.
Fournit interface pour upload, validation, et registry des datasets.

IDs Exposés (pour callbacks P7):
    Inputs: data-upload, data-source, data-symbol, data-timeframe,
            validate-data-btn
    Outputs: data-registry-table, data-alert, data-loading

Author: ThreadX Framework
Version: Prompt 5 - Composants Data + Indicators
"""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


def create_data_manager_panel():
    """
    Create data management panel component.

    Returns:
        html.Div: Complete data manager panel with forms and tables.
    """

    # Configuration card (left column)
    config_card = dbc.Card(
        className="bg-dark border-secondary h-100",
        children=[
            dbc.CardHeader("Configuration", className="bg-secondary text-light"),
            dbc.CardBody(
                className="bg-dark",
                children=[
                    # Upload Section
                    html.Label("Upload Data File", className="text-light mb-2"),
                    dcc.Upload(
                        id="data-upload",
                        children=html.Div(
                            className=(
                                "text-center p-3 border "
                                "border-dashed border-secondary rounded"
                            ),
                            children=[
                                html.I(className="bi bi-upload me-2"),
                                "Drag and Drop or ",
                                html.A("Select File", className="text-primary"),
                                html.Div(
                                    "CSV, Parquet", className="small text-muted mt-1"
                                ),
                            ],
                        ),
                        multiple=False,
                        accept=".csv,.parquet",
                        className="mb-3",
                    ),
                    html.Hr(className="border-secondary"),
                    # Source Dropdown
                    html.Label("Data Source", className="text-light mb-2"),
                    dcc.Dropdown(
                        id="data-source",
                        options=[
                            {"label": "Yahoo Finance", "value": "yahoo"},
                            {"label": "Local File", "value": "local"},
                            {"label": "Binance", "value": "binance"},
                            {"label": "Custom CSV", "value": "custom"},
                        ],
                        value="local",
                        placeholder="Select data source...",
                        className="mb-3 bg-dark",
                    ),
                    # Symbol Input
                    html.Label("Symbol", className="text-light mb-2"),
                    dcc.Input(
                        id="data-symbol",
                        type="text",
                        placeholder="e.g., BTCUSDT",
                        value="",
                        className=("form-control bg-dark text-light mb-3"),
                    ),
                    # Timeframe Dropdown
                    html.Label("Timeframe", className="text-light mb-2"),
                    dcc.Dropdown(
                        id="data-timeframe",
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
                    # Validate Button
                    dbc.Button(
                        "Validate Data",
                        id="validate-data-btn",
                        color="primary",
                        className="w-100 mt-3",
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
            dbc.CardHeader("Data Registry", className="bg-secondary text-light"),
            dbc.CardBody(
                className="bg-dark",
                children=[
                    # Alert Messages
                    dbc.Alert(
                        id="data-alert",
                        is_open=False,
                        dismissable=True,
                        className="mb-3",
                    ),
                    # Loading Wrapper
                    dcc.Loading(
                        id="data-loading",
                        type="circle",
                        children=[
                            # Registry Table
                            dash_table.DataTable(
                                id="data-registry-table",
                                columns=[
                                    {"name": "Symbol", "id": "symbol"},
                                    {"name": "Timeframe", "id": "tf"},
                                    {"name": "Rows", "id": "rows"},
                                    {"name": "Status", "id": "status"},
                                    {"name": "Quality", "id": "quality"},
                                ],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "10px",
                                    "backgroundColor": "#212529",
                                    "color": "white",
                                },
                                style_header={
                                    "backgroundColor": "#343a40",
                                    "fontWeight": "bold",
                                    "color": "white",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"row_index": "odd"},
                                        "backgroundColor": "#2c3034",
                                    }
                                ],
                            ),
                        ],
                    ),
                    # Empty State
                    html.Div(
                        id="data-empty-state",
                        className="text-center text-muted mt-4",
                        children=[
                            html.I(className=("bi bi-database display-4 mb-3")),
                            html.P(
                                "No datasets validated yet",
                                className="mb-0",
                            ),
                            html.P(
                                "Upload a file and click Validate Data",
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
            html.H4("Data Management", className="text-light mb-1"),
            html.P(
                "Upload, validate, and manage market data sources",
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
