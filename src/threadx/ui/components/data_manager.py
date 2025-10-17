"""
ThreadX UI - Data Creation & Management Component
==================================================

Composant Dash pour création et gestion de la banque de données OHLCV.

Fonctionnalités:
- Téléchargement Binance (Single Symbol / Top 100 / Groups)
- Validation UDFI stricte
- Sauvegarde Parquet + Registry (avec checksums)
- Mise à jour indicateurs en batch
- Persistance sélections globales (réutilisation autres onglets)

IDs Exposés (pour callbacks):
    Inputs:
        - data-source (Dropdown: single/top/group)
        - data-symbol (Input: symbole pour mode single)
        - data-group-select (Dropdown: L1/DeFi/L2/Stable pour mode group)
        - data-timeframe (Dropdown: 1m/5m/15m/1h/4h/1d)
        - data-start-date (DatePickerSingle)
        - data-end-date (DatePickerSingle)
        - download-data-btn (Button: télécharger)
        - data-indicators-select (Dropdown multi: RSI/MACD/BB/etc.)
        - update-indicators-btn (Button: MAJ indicateurs)

    Outputs:
        - data-alert (Alert: messages succès/erreur)
        - data-loading (Loading: indicateur activité)
        - data-registry-table (DataTable: registry datasets)
        - data-preview-graph (Graph: candlestick preview)

    Stores:
        - data-global-store (Store: persistance sélections globales)

Author: ThreadX Framework
Version: Prompt 10 - Data Creation & Management
"""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
from datetime import datetime, timedelta


def create_data_manager_panel():
    """
    Create Data Creation & Management panel.

    Returns:
        html.Div: Complete panel avec configuration, preview, et registry.
    """

    # ===== Colonne Gauche: Configuration =====
    config_card = dbc.Card(
        className="bg-dark border-secondary h-100",
        children=[
            dbc.CardHeader(
                "Data Source & Configuration", className="bg-secondary text-light"
            ),
            dbc.CardBody(
                className="bg-dark",
                children=[
                    # === Mode Source ===
                    html.Label("Source Mode", className="text-light mb-2 fw-bold"),
                    dcc.Dropdown(
                        id="data-source",
                        options=[
                            {"label": "🎯 Single Symbol", "value": "single"},
                            {
                                "label": "📊 Top 100 (Market Cap + Volume)",
                                "value": "top",
                            },
                            {"label": "🏷️  Group (L1/DeFi/L2/Stable)", "value": "group"},
                        ],
                        value="single",
                        placeholder="Select source mode...",
                        className="mb-3",
                        style={"color": "#000"},
                    ),
                    # === Input Symbol (mode single) ===
                    html.Div(
                        id="data-symbol-container",
                        children=[
                            html.Label("Symbol", className="text-light mb-2"),
                            dcc.Input(
                                id="data-symbol",
                                type="text",
                                placeholder="e.g., BTCUSDC",
                                value="BTCUSDC",
                                className="form-control bg-dark text-light mb-3",
                            ),
                        ],
                    ),
                    # === Dropdown Group (mode group) ===
                    html.Div(
                        id="data-group-container",
                        style={"display": "none"},
                        children=[
                            html.Label("Group", className="text-light mb-2"),
                            dcc.Dropdown(
                                id="data-group-select",
                                options=[
                                    {"label": "L1 (BTC, ETH, SOL, ADA)", "value": "L1"},
                                    {
                                        "label": "DeFi (UNI, AAVE, LINK, DOT)",
                                        "value": "DeFi",
                                    },
                                    {"label": "L2 (MATIC, ARB, OP)", "value": "L2"},
                                    {
                                        "label": "Stable (EUR, FDUSD, USDE)",
                                        "value": "Stable",
                                    },
                                ],
                                value="L1",
                                placeholder="Select group...",
                                className="mb-3",
                                style={"color": "#000"},
                            ),
                        ],
                    ),
                    html.Hr(className="border-secondary my-3"),
                    # === Timeframe ===
                    html.Label("Timeframe", className="text-light mb-2 fw-bold"),
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
                        className="mb-3",
                        style={"color": "#000"},
                    ),
                    # === Upload dataset (optionnel) ===
                    html.Label("Upload Dataset", className="text-light mb-2 fw-bold"),
                    dcc.Upload(
                        id="data-upload",
                        children=html.Div(
                            [
                                html.I(className="bi bi-upload me-2"),
                                "Glissez un fichier CSV/Parquet ici ou cliquez pour sélectionner",
                            ],
                            className="text-center text-muted small",
                        ),
                        className="mb-3 bg-dark text-light border border-secondary rounded",
                        style={
                            "width": "100%",
                            "padding": "18px",
                            "borderStyle": "dashed",
                            "cursor": "pointer",
                        },
                    ),
                    # === Plage Dates ===
                    html.Label("Date Range", className="text-light mb-2 fw-bold"),
                    html.Div(
                        className="row g-2 mb-3",
                        children=[
                            html.Div(
                                className="col-6",
                                children=[
                                    html.Label(
                                        "Start", className="text-muted small mb-1"
                                    ),
                                    dcc.DatePickerSingle(
                                        id="data-start-date",
                                        date=(
                                            datetime.now() - timedelta(days=30)
                                        ).date(),
                                        display_format="YYYY-MM-DD",
                                        className="w-100",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="col-6",
                                children=[
                                    html.Label(
                                        "End", className="text-muted small mb-1"
                                    ),
                                    dcc.DatePickerSingle(
                                        id="data-end-date",
                                        date=datetime.now().date(),
                                        display_format="YYYY-MM-DD",
                                        className="w-100",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Hr(className="border-secondary my-3"),
                    # === Bouton Télécharger ===
                    dbc.Button(
                        [
                            html.I(className="bi bi-download me-2"),
                            "Download & Validate Data",
                        ],
                        id="download-data-btn",
                        color="primary",
                        className="w-100 mb-3",
                        n_clicks=0,
                    ),
                    dbc.Button(
                        [
                            html.I(className="bi bi-clipboard-check me-2"),
                            "Validate Registry Only",
                        ],
                        id="validate-data-btn",
                        color="secondary",
                        outline=True,
                        className="w-100 mb-3",
                        n_clicks=0,
                    ),
                    # === Section Mise à Jour Indicateurs ===
                    html.Hr(className="border-secondary my-3"),
                    html.Label(
                        "Update Indicators (Batch)", className="text-light mb-2 fw-bold"
                    ),
                    dcc.Dropdown(
                        id="data-indicators-select",
                        options=[
                            {"label": "RSI (14)", "value": "rsi_14"},
                            {"label": "MACD (12,26,9)", "value": "macd"},
                            {"label": "Bollinger Bands (20,2)", "value": "bb_20_2"},
                            {"label": "SMA (20)", "value": "sma_20"},
                            {"label": "EMA (50)", "value": "ema_50"},
                            {"label": "ATR (14)", "value": "atr_14"},
                        ],
                        value=["rsi_14", "macd", "bb_20_2"],
                        multi=True,
                        placeholder="Select indicators...",
                        className="mb-3",
                        style={"color": "#000"},
                    ),
                    dbc.Button(
                        [
                            html.I(className="bi bi-calculator me-2"),
                            "Update Indicators",
                        ],
                        id="update-indicators-btn",
                        color="success",
                        className="w-100",
                        n_clicks=0,
                    ),
                ],
            ),
        ],
    )

    # ===== Colonne Droite: Registry + Preview =====
    results_card = dbc.Card(
        className="bg-dark border-secondary h-100",
        children=[
            dbc.CardHeader(
                "Data Registry & Preview", className="bg-secondary text-light"
            ),
            dbc.CardBody(
                className="bg-dark",
                children=[
                    # === Alert Messages ===
                    dbc.Alert(
                        id="data-alert",
                        is_open=False,
                        dismissable=True,
                        className="mb-3",
                    ),
                    # === Loading Wrapper ===
                    dcc.Loading(
                        id="data-loading",
                        type="circle",
                        children=[
                            # === Registry Table ===
                            html.Div(
                                className="mb-4",
                                children=[
                                    html.H6("Registry", className="text-light mb-2"),
                                    dash_table.DataTable(
                                        id="data-registry-table",
                                        columns=[
                                            {"name": "Symbol", "id": "symbol"},
                                            {"name": "Timeframe", "id": "timeframe"},
                                            {"name": "Rows", "id": "rows"},
                                            {"name": "Start", "id": "start"},
                                            {"name": "End", "id": "end"},
                                            {"name": "Checksum", "id": "checksum"},
                                        ],
                                        data=[],
                                        page_size=10,
                                        style_table={
                                            "overflowX": "auto",
                                            "maxHeight": "300px",
                                            "overflowY": "auto",
                                        },
                                        style_cell={
                                            "textAlign": "left",
                                            "padding": "8px",
                                            "backgroundColor": "#212529",
                                            "color": "white",
                                            "fontSize": "0.85rem",
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
                            # === Preview Graph ===
                            html.Div(
                                children=[
                                    html.H6(
                                        "OHLCV Preview", className="text-light mb-2"
                                    ),
                                    dcc.Graph(
                                        id="data-preview-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "400px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # ===== Stores (persistance globale) =====
    global_store = dcc.Store(
        id="data-global-store",
        storage_type="session",
        data={
            "symbols": [],
            "timeframe": "1h",
            "start_date": None,
            "end_date": None,
            "last_downloaded": None,
        },
    )

    # ===== Panel Principal =====
    return html.Div(
        className="p-4 bg-dark",
        children=[
            html.H4("Data Creation & Management", className="text-light mb-1"),
            html.P(
                "Download OHLCV from Binance, validate UDFI, save to registry, "
                "and update indicators in batch",
                className="text-muted mb-4",
            ),
            dbc.Row(
                className="g-3",
                children=[
                    dbc.Col(config_card, md=5),
                    dbc.Col(results_card, md=7),
                ],
            ),
            global_store,
        ],
    )
