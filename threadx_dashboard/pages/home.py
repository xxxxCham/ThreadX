"""
Page d'accueil ThreadX Dashboard
===============================

Cette page présente un aperçu des performances,
des statistiques et des liens rapides.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import THEME, BACKTEST_CONFIG
from utils.helpers import format_currency, format_percentage


def create_home_layout() -> html.Div:
    """
    Crée le layout de la page d'accueil.

    Returns:
        html.Div: Layout de la page
    """
    return html.Div(
        [
            # Hero section
            create_hero_section(),
            html.Br(),
            # Statistiques principales
            create_stats_cards(),
            html.Br(),
            # Graphiques et actions rapides
            dbc.Row(
                [
                    dbc.Col([create_performance_chart()], width=8),
                    dbc.Col([create_quick_actions()], width=4),
                ]
            ),
            html.Br(),
            # Backtests récents et documentation
            dbc.Row(
                [
                    dbc.Col([create_recent_backtests_table()], width=8),
                    dbc.Col([create_documentation_links()], width=4),
                ]
            ),
        ],
        className="p-4",
    )


def create_hero_section() -> html.Div:
    """
    Crée la section hero de la page d'accueil.

    Returns:
        html.Div: Section hero
    """
    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H1(
                                [
                                    html.I(
                                        className="fas fa-chart-line me-3",
                                        style={"color": THEME["accent_primary"]},
                                    ),
                                    "Bienvenue sur ThreadX Dashboard",
                                ],
                                className="display-4 mb-4",
                            ),
                            html.P(
                                [
                                    "Plateforme de backtesting professionnel pour le trading de cryptomonnaies. ",
                                    "Analysez vos stratégies, optimisez vos paramètres et maximisez vos performances.",
                                ],
                                className="lead mb-4",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-play me-2"),
                                            "Démarrer un Backtest",
                                        ],
                                        color="primary",
                                        size="lg",
                                        href="/backtesting",
                                        className="me-3",
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-table me-2"),
                                            "Voir les Résultats",
                                        ],
                                        color="outline-primary",
                                        size="lg",
                                        href="/results",
                                    ),
                                ]
                            ),
                        ]
                    )
                ],
                style={
                    "background": f"linear-gradient(135deg, {THEME['secondary_bg']}, {THEME['tertiary_bg']})",
                    "border": f"1px solid {THEME['border_color']}",
                },
            )
        ]
    )


def create_stats_cards() -> dbc.Row:
    """
    Crée les cartes de statistiques principales.

    Returns:
        dbc.Row: Ligne de cartes statistiques
    """
    stats = [
        {
            "title": "P&L Total",
            "value": format_currency(12450.67),
            "change": "+8.5%",
            "icon": "fas fa-dollar-sign",
            "color": THEME["success"],
            "bg_color": "rgba(0, 255, 0, 0.1)",
        },
        {
            "title": "Taux de Réussite",
            "value": "72.3%",
            "change": "+2.1%",
            "icon": "fas fa-percentage",
            "color": THEME["accent_primary"],
            "bg_color": "rgba(0, 212, 255, 0.1)",
        },
        {
            "title": "Trades Exécutés",
            "value": "1,247",
            "change": "+156",
            "icon": "fas fa-exchange-alt",
            "color": THEME["warning"],
            "bg_color": "rgba(255, 170, 0, 0.1)",
        },
        {
            "title": "Sharpe Ratio",
            "value": "1.85",
            "change": "+0.23",
            "icon": "fas fa-chart-bar",
            "color": THEME["accent_secondary"],
            "bg_color": "rgba(0, 168, 204, 0.1)",
        },
    ]

    return dbc.Row([dbc.Col([create_stat_card(stat)], width=3) for stat in stats])


def create_stat_card(stat: dict) -> dbc.Card:
    """
    Crée une carte de statistique individuelle.

    Args:
        stat: Dictionnaire contenant les données de la statistique

    Returns:
        dbc.Card: Carte de statistique
    """
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className=stat["icon"],
                                        style={
                                            "fontSize": "2rem",
                                            "color": stat["color"],
                                        },
                                    )
                                ],
                                style={
                                    "backgroundColor": stat["bg_color"],
                                    "borderRadius": "50%",
                                    "width": "60px",
                                    "height": "60px",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                },
                            ),
                            html.Div(
                                [
                                    html.H4(stat["value"], className="mb-1 fw-bold"),
                                    html.P(stat["title"], className="text-muted mb-1"),
                                    html.Small(
                                        [
                                            html.I(className="fas fa-arrow-up me-1"),
                                            stat["change"],
                                        ],
                                        className="text-success fw-bold",
                                    ),
                                ],
                                className="ms-3",
                            ),
                        ],
                        className="d-flex align-items-center",
                    )
                ]
            )
        ],
        style={
            "backgroundColor": THEME["secondary_bg"],
            "border": f"1px solid {THEME['border_color']}",
        },
    )


def create_performance_chart() -> dbc.Card:
    """
    Crée le graphique de performance.

    Returns:
        dbc.Card: Carte avec graphique
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H5(
                        [
                            html.I(className="fas fa-chart-line me-2"),
                            "Performance des 30 Derniers Jours",
                        ],
                        className="mb-0",
                    )
                ]
            ),
            dbc.CardBody(
                [dcc.Graph(id="performance-chart", style={"height": "400px"})]
            ),
        ],
        style={
            "backgroundColor": THEME["secondary_bg"],
            "border": f"1px solid {THEME['border_color']}",
        },
    )


def create_quick_actions() -> dbc.Card:
    """
    Crée la carte des actions rapides.

    Returns:
        dbc.Card: Carte d'actions
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H5(
                        [html.I(className="fas fa-bolt me-2"), "Actions Rapides"],
                        className="mb-0",
                    )
                ]
            ),
            dbc.CardBody(
                [
                    html.Div(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-plus me-2"),
                                    "Nouveau Backtest",
                                ],
                                color="primary",
                                className="w-100 mb-3",
                                href="/backtesting",
                            ),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-download me-2"),
                                    "Importer Données",
                                ],
                                color="secondary",
                                className="w-100 mb-3",
                            ),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-cogs me-2"),
                                    "Optimiser Stratégie",
                                ],
                                color="info",
                                className="w-100 mb-3",
                            ),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-file-export me-2"),
                                    "Exporter Résultats",
                                ],
                                color="success",
                                className="w-100 mb-3",
                            ),
                            html.Hr(),
                            html.H6("Stratégies Populaires", className="mb-3"),
                            html.Div(
                                [
                                    dbc.Badge(
                                        "Bollinger + ATR",
                                        color="primary",
                                        className="me-2 mb-2",
                                    ),
                                    dbc.Badge(
                                        "RSI + MA",
                                        color="secondary",
                                        className="me-2 mb-2",
                                    ),
                                    dbc.Badge(
                                        "MACD Cross",
                                        color="info",
                                        className="me-2 mb-2",
                                    ),
                                    dbc.Badge(
                                        "Mean Reversion",
                                        color="success",
                                        className="me-2 mb-2",
                                    ),
                                ]
                            ),
                        ]
                    )
                ]
            ),
        ],
        style={
            "backgroundColor": THEME["secondary_bg"],
            "border": f"1px solid {THEME['border_color']}",
        },
    )


def create_recent_backtests_table() -> dbc.Card:
    """
    Crée le tableau des backtests récents.

    Returns:
        dbc.Card: Carte avec tableau
    """
    # Données d'exemple
    backtests = [
        {
            "strategy": "BB_ATR_V2",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "return": "+15.2%",
            "trades": 78,
            "win_rate": "69.2%",
            "date": "2025-10-13",
        },
        {
            "strategy": "RSI_MA_Cross",
            "symbol": "ETHUSDT",
            "timeframe": "4h",
            "return": "-2.8%",
            "trades": 45,
            "win_rate": "42.2%",
            "date": "2025-10-12",
        },
        {
            "strategy": "MACD_Divergence",
            "symbol": "BNBUSDT",
            "timeframe": "1d",
            "return": "+8.7%",
            "trades": 23,
            "win_rate": "65.2%",
            "date": "2025-10-11",
        },
    ]

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H5(
                        [html.I(className="fas fa-history me-2"), "Backtests Récents"],
                        className="mb-0",
                    )
                ]
            ),
            dbc.CardBody(
                [
                    dbc.Table(
                        [
                            html.Thead(
                                [
                                    html.Tr(
                                        [
                                            html.Th("Stratégie"),
                                            html.Th("Symbole"),
                                            html.Th("Timeframe"),
                                            html.Th("Retour"),
                                            html.Th("Trades"),
                                            html.Th("Win Rate"),
                                            html.Th("Date"),
                                        ]
                                    )
                                ]
                            ),
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td(
                                                bt["strategy"], className="fw-bold"
                                            ),
                                            html.Td(bt["symbol"]),
                                            html.Td(bt["timeframe"]),
                                            html.Td(
                                                bt["return"],
                                                style={
                                                    "color": (
                                                        THEME["success"]
                                                        if "+" in bt["return"]
                                                        else THEME["danger"]
                                                    )
                                                },
                                            ),
                                            html.Td(bt["trades"]),
                                            html.Td(bt["win_rate"]),
                                            html.Td(bt["date"]),
                                        ]
                                    )
                                    for bt in backtests
                                ]
                            ),
                        ],
                        bordered=True,
                        hover=True,
                        striped=True,
                        style={"backgroundColor": THEME["tertiary_bg"]},
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Voir Tous les Résultats",
                                color="outline-primary",
                                href="/results",
                                className="mt-3",
                            )
                        ],
                        className="text-center",
                    ),
                ]
            ),
        ],
        style={
            "backgroundColor": THEME["secondary_bg"],
            "border": f"1px solid {THEME['border_color']}",
        },
    )


def create_documentation_links() -> dbc.Card:
    """
    Crée la carte avec les liens de documentation.

    Returns:
        dbc.Card: Carte de documentation
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H5(
                        [html.I(className="fas fa-book me-2"), "Documentation"],
                        className="mb-0",
                    )
                ]
            ),
            dbc.CardBody(
                [
                    html.Div(
                        [
                            dbc.ListGroup(
                                [
                                    dbc.ListGroupItem(
                                        [
                                            html.I(className="fas fa-rocket me-2"),
                                            "Guide de Démarrage Rapide",
                                        ],
                                        href="#",
                                        action=True,
                                        style={
                                            "backgroundColor": "transparent",
                                            "border": "none",
                                        },
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.I(className="fas fa-code me-2"),
                                            "API Documentation",
                                        ],
                                        href="#",
                                        action=True,
                                        style={
                                            "backgroundColor": "transparent",
                                            "border": "none",
                                        },
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.I(
                                                className="fas fa-question-circle me-2"
                                            ),
                                            "FAQ",
                                        ],
                                        href="#",
                                        action=True,
                                        style={
                                            "backgroundColor": "transparent",
                                            "border": "none",
                                        },
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.I(className="fas fa-video me-2"),
                                            "Tutoriels Vidéo",
                                        ],
                                        href="#",
                                        action=True,
                                        style={
                                            "backgroundColor": "transparent",
                                            "border": "none",
                                        },
                                    ),
                                    dbc.ListGroupItem(
                                        [
                                            html.I(className="fas fa-envelope me-2"),
                                            "Support",
                                        ],
                                        href="#",
                                        action=True,
                                        style={
                                            "backgroundColor": "transparent",
                                            "border": "none",
                                        },
                                    ),
                                ],
                                flush=True,
                            )
                        ]
                    )
                ]
            ),
        ],
        style={
            "backgroundColor": THEME["secondary_bg"],
            "border": f"1px solid {THEME['border_color']}",
        },
    )


# =============================================================================
# CALLBACKS
# =============================================================================


@callback(Output("performance-chart", "figure"), Input("performance-chart", "id"))
def update_performance_chart(_):
    """
    Met à jour le graphique de performance.

    Returns:
        go.Figure: Graphique Plotly
    """
    # Données d'exemple pour les 30 derniers jours
    dates = []
    values = []

    base_date = datetime.now() - timedelta(days=30)
    base_value = BACKTEST_CONFIG["initial_capital"]

    for i in range(30):
        dates.append(base_date + timedelta(days=i))
        # Simulation d'une courbe de performance
        daily_return = 0.001 * (i % 7 - 2) + 0.0005  # Variation simulée
        base_value *= 1 + daily_return
        values.append(base_value)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="Performance Cumulée",
            line=dict(color=THEME["accent_primary"], width=3),
            fill="tonexty",
            fillcolor=f"rgba(0, 212, 255, 0.1)",
        )
    )

    fig.update_layout(
        title="Évolution du Portfolio",
        xaxis_title="Date",
        yaxis_title="Valeur ($)",
        showlegend=False,
        paper_bgcolor=THEME["secondary_bg"],
        plot_bgcolor=THEME["secondary_bg"],
        font_color=THEME["text_primary"],
        xaxis=dict(gridcolor=THEME["border_color"], linecolor=THEME["border_color"]),
        yaxis=dict(gridcolor=THEME["border_color"], linecolor=THEME["border_color"]),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig
