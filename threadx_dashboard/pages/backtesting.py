"""
ThreadX Dashboard - Page de Backtesting
Affiche les graphiques et résultats du backtesting avec interactions complètes
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from components.charts import ChartsManager
from config import THEME


# Initialiser ChartsManager
charts_manager = ChartsManager(THEME)


def get_layout():
    """Layout principal de la page backtesting"""
    return dbc.Container(
        [
            # SECTION 1: Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1(
                                "Backtesting Analysis", className="mb-2 text-primary"
                            ),
                            html.P(
                                "Price & Signals | Volume | Portfolio Balance",
                                className="text-secondary mb-0",
                            ),
                            html.Hr(className="mb-4"),
                        ]
                    )
                ],
                className="mb-4 mt-4",
            ),
            # SECTION 2: Loading state et alertes
            dcc.Loading(
                id="loading-charts",
                type="default",
                color=THEME.get("accent", "#00d4ff"),
                children=[
                    dbc.Alert(
                        [
                            html.I(className="fas fa-info-circle me-2"),
                            "No backtest data available. Configure parameters in Settings and click Submit.",
                        ],
                        id="no-data-alert",
                        color="info",
                        dismissable=True,
                        is_open=True,
                        style={"display": "block"},
                    )
                ],
            ),
            # SECTION 3: Price & Signals Chart
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H5(
                                                [
                                                    html.I(
                                                        className="fas fa-chart-line me-2"
                                                    ),
                                                    "Price & Signals",
                                                ],
                                                className="mb-0 text-white",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dcc.Dropdown(
                                                id="chart-timeframe",
                                                options=[
                                                    {"label": "1 Day", "value": "1D"},
                                                    {"label": "1 Week", "value": "1W"},
                                                    {"label": "1 Month", "value": "1M"},
                                                    {
                                                        "label": "3 Months",
                                                        "value": "3M",
                                                    },
                                                    {
                                                        "label": "All Time",
                                                        "value": "ALL",
                                                    },
                                                ],
                                                value="ALL",
                                                clearable=False,
                                                className="bg-dark text-white",
                                                style={
                                                    "backgroundColor": THEME.get(
                                                        "secondary_bg", "#242424"
                                                    ),
                                                    "color": THEME.get(
                                                        "text_primary", "#ffffff"
                                                    ),
                                                },
                                            ),
                                        ],
                                        width=6,
                                        className="text-end",
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dcc.Graph(
                                id="price-signals-graph",
                                style={"height": "450px"},
                                config={
                                    "responsive": True,
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": [
                                        "pan2d",
                                        "lasso2d",
                                        "select2d",
                                    ],
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "price_signals_chart",
                                        "height": 450,
                                        "width": 1200,
                                        "scale": 1,
                                    },
                                },
                            ),
                            # Légende des signaux
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "▼", className="text-success me-1"
                                            ),
                                            html.Span(
                                                "Buy Signal", className="me-3 small"
                                            ),
                                            html.Span(
                                                "▲", className="text-danger me-1"
                                            ),
                                            html.Span(
                                                "Sell Signal", className="me-3 small"
                                            ),
                                            html.Span(
                                                "─", className="text-warning me-1"
                                            ),
                                            html.Span(
                                                "Support", className="me-3 small"
                                            ),
                                            html.Span(
                                                "─", className="text-success me-1"
                                            ),
                                            html.Span(
                                                "Resistance", className="me-3 small"
                                            ),
                                        ],
                                        className="d-flex align-items-center justify-content-center",
                                    )
                                ],
                                className="text-secondary mt-2 p-2 rounded",
                                style={
                                    "backgroundColor": THEME.get(
                                        "tertiary_bg", "#2a2a2a"
                                    )
                                },
                            ),
                        ]
                    )
                ],
                className="mb-4",
                color="secondary",
                outline=True,
            ),
            # SECTION 4: Volume Chart
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5(
                                [
                                    html.I(className="fas fa-chart-bar me-2"),
                                    "Trading Volume",
                                ],
                                className="mb-3 text-white",
                            ),
                            dcc.Graph(
                                id="volume-graph",
                                style={"height": "250px"},
                                config={
                                    "responsive": True,
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": [
                                        "pan2d",
                                        "lasso2d",
                                        "select2d",
                                    ],
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "volume_chart",
                                        "height": 250,
                                        "width": 1200,
                                        "scale": 1,
                                    },
                                },
                            ),
                        ]
                    )
                ],
                className="mb-4",
                color="secondary",
                outline=True,
            ),
            # SECTION 5: Portfolio Balance Chart
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H5(
                                                [
                                                    html.I(
                                                        className="fas fa-wallet me-2"
                                                    ),
                                                    "Portfolio Balance",
                                                ],
                                                className="mb-0 text-white",
                                            ),
                                            html.Small(
                                                "Strategy vs Buy & Hold",
                                                className="text-secondary",
                                            ),
                                        ],
                                        width=8,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        "1M",
                                                        id="btn-1m",
                                                        size="sm",
                                                        color="secondary",
                                                        outline=True,
                                                    ),
                                                    dbc.Button(
                                                        "3M",
                                                        id="btn-3m",
                                                        size="sm",
                                                        color="secondary",
                                                        outline=True,
                                                    ),
                                                    dbc.Button(
                                                        "6M",
                                                        id="btn-6m",
                                                        size="sm",
                                                        color="secondary",
                                                        outline=True,
                                                    ),
                                                    dbc.Button(
                                                        "All",
                                                        id="btn-all",
                                                        size="sm",
                                                        color="info",
                                                    ),
                                                ],
                                                className="w-100",
                                            )
                                        ],
                                        width=4,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dcc.Graph(
                                id="portfolio-graph",
                                style={"height": "350px"},
                                config={
                                    "responsive": True,
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": [
                                        "pan2d",
                                        "lasso2d",
                                        "select2d",
                                    ],
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "portfolio_chart",
                                        "height": 350,
                                        "width": 1200,
                                        "scale": 1,
                                    },
                                },
                            ),
                            # Metrics row
                            html.Hr(className="mt-4 mb-3"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.I(
                                                                className="fas fa-dollar-sign me-2"
                                                            ),
                                                            "Final Equity",
                                                        ],
                                                        className="text-secondary small mb-1",
                                                    ),
                                                    html.Div(
                                                        "$0",
                                                        id="metric-final-equity",
                                                        className="h4 text-success mb-0",
                                                    ),
                                                ],
                                                className="text-center p-3 rounded",
                                                style={
                                                    "backgroundColor": THEME.get(
                                                        "tertiary_bg", "#2a2a2a"
                                                    )
                                                },
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.I(
                                                                className="fas fa-chart-line-down me-2"
                                                            ),
                                                            "Max Drawdown",
                                                        ],
                                                        className="text-secondary small mb-1",
                                                    ),
                                                    html.Div(
                                                        "0%",
                                                        id="metric-max-drawdown",
                                                        className="h4 text-danger mb-0",
                                                    ),
                                                ],
                                                className="text-center p-3 rounded",
                                                style={
                                                    "backgroundColor": THEME.get(
                                                        "tertiary_bg", "#2a2a2a"
                                                    )
                                                },
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.I(
                                                                className="fas fa-percentage me-2"
                                                            ),
                                                            "Total Return",
                                                        ],
                                                        className="text-secondary small mb-1",
                                                    ),
                                                    html.Div(
                                                        "0%",
                                                        id="metric-total-return",
                                                        className="h4 text-success mb-0",
                                                    ),
                                                ],
                                                className="text-center p-3 rounded",
                                                style={
                                                    "backgroundColor": THEME.get(
                                                        "tertiary_bg", "#2a2a2a"
                                                    )
                                                },
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.I(
                                                                className="fas fa-chart-area me-2"
                                                            ),
                                                            "Sharpe Ratio",
                                                        ],
                                                        className="text-secondary small mb-1",
                                                    ),
                                                    html.Div(
                                                        "0.00",
                                                        id="metric-sharpe-ratio",
                                                        className="h4 text-info mb-0",
                                                    ),
                                                ],
                                                className="text-center p-3 rounded",
                                                style={
                                                    "backgroundColor": THEME.get(
                                                        "tertiary_bg", "#2a2a2a"
                                                    )
                                                },
                                            ),
                                        ],
                                        width=3,
                                    ),
                                ],
                                className="g-3",
                            ),
                        ]
                    )
                ],
                className="mb-4",
                color="secondary",
                outline=True,
            ),
            # SECTION 6: Detailed Statistics Table
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5(
                                [
                                    html.I(className="fas fa-table me-2"),
                                    "Detailed Statistics",
                                ],
                                className="mb-3 text-white",
                            ),
                            html.Div(id="detailed-stats-table"),
                        ]
                    )
                ],
                className="mb-4",
                color="secondary",
                outline=True,
            ),
            # Hidden stores pour passer les données entre callbacks
            dcc.Store(id="backtest-data-store"),
            dcc.Store(id="filtered-data-store"),
        ],
        fluid=True,
        className="py-4",
    )


# Layout principal
layout = get_layout()


# =============================================================================
# CALLBACKS
# =============================================================================


@callback(
    [
        Output("price-signals-graph", "figure"),
        Output("volume-graph", "figure"),
        Output("portfolio-graph", "figure"),
        Output("no-data-alert", "style"),
        Output("metric-final-equity", "children"),
        Output("metric-max-drawdown", "children"),
        Output("metric-total-return", "children"),
        Output("metric-sharpe-ratio", "children"),
        Output("detailed-stats-table", "children"),
    ],
    [Input("backtest-data-store", "data"), Input("chart-timeframe", "value")],
    prevent_initial_call=True,
)
def update_charts(backtest_data: Optional[Dict[str, Any]], timeframe: str):
    """
    Met à jour tous les graphiques quand les données changent
    """
    if not backtest_data:
        # Aucune donnée - afficher alerte et graphiques vides
        empty_fig = charts_manager._create_empty_figure("No data available")
        return (
            empty_fig,
            empty_fig,
            empty_fig,
            {"display": "block"},  # Afficher alerte
            "$0",
            "0%",
            "0%",
            "0.00",
            html.P("No data to display", className="text-secondary"),
        )

    try:
        # Filtrer les données selon timeframe
        filtered_data = filter_data_by_timeframe(backtest_data, timeframe)

        # Générer les figures
        figures = charts_manager.get_all_figures(filtered_data)

        # Calculer les métriques
        metrics = calculate_metrics(filtered_data)

        # Créer le tableau détaillé
        stats_table = create_detailed_stats_table(metrics)

        return (
            figures["price"],
            figures["volume"],
            figures["portfolio"],
            {"display": "none"},  # Masquer alerte
            f"${metrics.get('final_equity', 0):,.2f}",
            f"{metrics.get('max_drawdown', 0):.1%}",
            f"{metrics.get('total_return', 0):+.1%}",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            stats_table,
        )

    except Exception as e:
        # En cas d'erreur
        error_fig = charts_manager._create_empty_figure(f"Error: {str(e)}")
        return (
            error_fig,
            error_fig,
            error_fig,
            {"display": "block"},
            "$0",
            "0%",
            "0%",
            "0.00",
            html.P(f"Error loading data: {str(e)}", className="text-danger"),
        )


@callback(
    Output("backtest-data-store", "data"),
    Input("submit-backtest-button", "n_clicks"),
    [
        State("asset-selector", "value"),
        State("start-date-picker", "date"),
        State("end-date-picker", "date"),
        State("initial-cash-input", "value"),
    ],
    prevent_initial_call=True,
)
def run_backtest(
    n_clicks: int, asset: str, start_date: str, end_date: str, initial_cash: float
):
    """
    Exécute le backtest et stocke les résultats
    """
    if not n_clicks:
        return None

    try:
        # Simuler des données de backtest pour la démo
        backtest_results = generate_mock_backtest_data(
            asset_name=asset or "BTC-USD",
            start_date=start_date or "2023-01-01",
            end_date=end_date or "2024-01-01",
            initial_cash=initial_cash or 10000,
        )

        return backtest_results

    except Exception as e:
        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def filter_data_by_timeframe(data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
    """Filtre les données selon la période sélectionnée"""
    if timeframe == "ALL":
        return data

    # Calculer la date de début selon le timeframe
    end_date = datetime.now()
    if timeframe == "1D":
        start_date = end_date - timedelta(days=1)
    elif timeframe == "1W":
        start_date = end_date - timedelta(weeks=1)
    elif timeframe == "1M":
        start_date = end_date - timedelta(days=30)
    elif timeframe == "3M":
        start_date = end_date - timedelta(days=90)
    else:
        return data

    # Filtrer toutes les séries temporelles
    filtered_data = data.copy()

    for key in ["price_history", "volume", "portfolio", "buy_hold"]:
        if key in filtered_data and "dates" in filtered_data[key]:
            dates = pd.to_datetime(filtered_data[key]["dates"])
            mask = (dates >= start_date) & (dates <= end_date)

            filtered_data[key] = {
                field: [val for i, val in enumerate(values) if mask.iloc[i]]
                for field, values in filtered_data[key].items()
                if isinstance(values, list)
            }

    # Filtrer les signaux
    for signal_type in ["buy_signals", "sell_signals"]:
        if signal_type in filtered_data:
            filtered_data[signal_type] = [
                signal
                for signal in filtered_data[signal_type]
                if start_date
                <= datetime.strptime(signal["date"], "%Y-%m-%d")
                <= end_date
            ]

    return filtered_data


def calculate_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Calcule les métriques de performance"""
    portfolio_data = data.get("portfolio", {})
    equity_values = portfolio_data.get("equity", [])
    initial_cash = data.get("initial_cash", 10000)

    if not equity_values:
        return {
            "final_equity": initial_cash,
            "total_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
        }

    equity_series = pd.Series(equity_values)
    final_equity = equity_series.iloc[-1]
    total_return = (final_equity - initial_cash) / initial_cash

    # Calcul du drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()

    # Calcul du Sharpe ratio (simplifié)
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = (
        returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    )

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }


def create_detailed_stats_table(metrics: Dict[str, float]) -> html.Div:
    """Crée le tableau détaillé des statistiques"""
    stats_data = [
        {"Metric": "Final Equity", "Value": f"${metrics.get('final_equity', 0):,.2f}"},
        {"Metric": "Total Return", "Value": f"{metrics.get('total_return', 0):+.2%}"},
        {"Metric": "Max Drawdown", "Value": f"{metrics.get('max_drawdown', 0):.2%}"},
        {"Metric": "Sharpe Ratio", "Value": f"{metrics.get('sharpe_ratio', 0):.3f}"},
        {"Metric": "Volatility", "Value": f"{metrics.get('volatility', 0):.2%}"},
        {"Metric": "Win Rate", "Value": f"{metrics.get('win_rate', 0):.1%}"},
    ]

    table_rows = []
    for stat in stats_data:
        color_class = ""
        if "Return" in stat["Metric"] or "Equity" in stat["Metric"]:
            color_class = (
                "text-success"
                if "+" in stat["Value"] or "$" in stat["Value"]
                else "text-danger"
            )
        elif "Drawdown" in stat["Metric"]:
            color_class = "text-danger"
        elif "Ratio" in stat["Metric"]:
            color_class = "text-info"

        table_rows.append(
            html.Tr(
                [
                    html.Td(stat["Metric"], className="text-secondary"),
                    html.Td(stat["Value"], className=f"text-end {color_class} fw-bold"),
                ]
            )
        )

    return dbc.Table(
        [html.Tbody(table_rows)],
        bordered=True,
        dark=True,
        hover=True,
        responsive=True,
        className="mb-0",
    )


def generate_mock_backtest_data(
    asset_name: str = "BTC-USD",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    initial_cash: float = 10000,
) -> Dict[str, Any]:
    """Génère des données de backtest fictives pour la démo"""

    # Générer des dates
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]

    # Générer des prix fictifs
    np.random.seed(42)  # Pour des résultats reproductibles
    base_price = 45000
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]

    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, base_price * 0.5))  # Prix minimum

    # Générer des signaux d'achat/vente
    buy_signals = []
    sell_signals = []

    for i in range(20, len(dates) - 20, 30):  # Un signal tous les 30 jours
        if i % 60 == 20:  # Signal d'achat
            buy_signals.append(
                {"date": date_strings[i], "price": prices[i], "quantity": 0.1}
            )
        elif i % 60 == 50:  # Signal de vente
            sell_signals.append(
                {"date": date_strings[i], "price": prices[i], "quantity": 0.1}
            )

    # Générer des volumes
    volumes = np.random.lognormal(10, 1, len(dates))

    # Générer l'équité du portefeuille
    equity_values = [initial_cash]
    position = 0
    cash = initial_cash

    for i in range(1, len(prices)):
        # Logique simplifiée de trading
        current_price = prices[i]

        # Vérifier s'il y a un signal à cette date
        buy_signal = next(
            (s for s in buy_signals if s["date"] == date_strings[i]), None
        )
        sell_signal = next(
            (s for s in sell_signals if s["date"] == date_strings[i]), None
        )

        if buy_signal and cash > buy_signal["price"] * buy_signal["quantity"]:
            # Acheter
            position += buy_signal["quantity"]
            cash -= buy_signal["price"] * buy_signal["quantity"]
        elif sell_signal and position >= sell_signal["quantity"]:
            # Vendre
            position -= sell_signal["quantity"]
            cash += sell_signal["price"] * sell_signal["quantity"]

        # Calculer l'équité totale
        total_equity = cash + position * current_price
        equity_values.append(total_equity)

    # Générer Buy & Hold pour comparaison
    initial_btc = initial_cash / prices[0]
    buy_hold_values = [initial_btc * price for price in prices]

    return {
        "asset_name": asset_name,
        "period_start": start_date,
        "period_end": end_date,
        "initial_cash": initial_cash,
        "final_cash": equity_values[-1],
        "total_return": (equity_values[-1] - initial_cash) / initial_cash,
        "price_history": {
            "dates": date_strings,
            "close": prices,
            "open": [p * 0.995 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
        },
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "volume": {
            "dates": date_strings,
            "total": volumes.tolist(),
            "buy": (volumes * 0.6).tolist(),
            "sell": (volumes * 0.4).tolist(),
        },
        "portfolio": {
            "dates": date_strings,
            "equity": equity_values,
            "cash": [cash] * len(date_strings),  # Simplifié
            "positions": [position * p for p in prices],
        },
        "buy_hold": {
            "dates": date_strings,
            "equity": buy_hold_values,
        },
        "metrics": {
            "total_trades": len(buy_signals) + len(sell_signals),
            "win_rate": 0.65,
            "avg_win": 500,
            "avg_loss": -300,
            "max_drawdown": -0.153,
            "sharpe_ratio": 0.89,
            "volatility": 0.35,
        },
    }


# Callback supplémentaire pour charger des données de démo automatiquement
@callback(
    Output("backtest-data-store", "data"),
    Input("no-data-alert", "is_open"),
    prevent_initial_call=False,
)
def auto_load_demo_data(alert_is_open):
    """Charge automatiquement des données de démo pour la démonstration"""
    try:
        return generate_mock_backtest_data()
    except Exception as e:
        print(f"Erreur chargement démo: {e}")
        return None
