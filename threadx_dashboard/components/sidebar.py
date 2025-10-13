"""
Barre latérale pour ThreadX Dashboard
====================================

Ce module contient la sidebar avec navigation, paramètres rapides
et statistiques en temps réel.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate

from utils.auth import auth_manager
from config import THEME


def create_sidebar() -> html.Div:
    """
    Crée la barre latérale avec navigation et informations.

    Returns:
        html.Div: Composant sidebar
    """
    return html.Div(
        [
            # Navigation principale
            create_main_navigation(),
            html.Hr(style={"borderColor": THEME["border_color"]}),
            # Paramètres rapides
            create_quick_settings(),
            html.Hr(style={"borderColor": THEME["border_color"]}),
            # Statistiques rapides
            create_quick_stats(),
            html.Hr(style={"borderColor": THEME["border_color"]}),
            # Backtests récents
            create_recent_backtests(),
        ],
        id="sidebar",
        className="sidebar",
        style={
            "backgroundColor": THEME["secondary_bg"],
            "borderRight": f"1px solid {THEME['border_color']}",
            "overflowY": "auto",
        },
    )


def create_main_navigation() -> html.Div:
    """
    Crée le menu de navigation principal.

    Returns:
        html.Div: Menu de navigation
    """
    nav_items = [
        {"label": "Accueil", "href": "/", "icon": "fas fa-home", "id": "nav-home"},
        {
            "label": "Backtesting",
            "href": "/backtesting",
            "icon": "fas fa-chart-line",
            "id": "nav-backtesting",
        },
        {
            "label": "Résultats",
            "href": "/results",
            "icon": "fas fa-table",
            "id": "nav-results",
        },
        {
            "label": "Paramètres",
            "href": "/settings",
            "icon": "fas fa-cog",
            "id": "nav-settings",
        },
    ]

    return html.Div(
        [
            html.H6(
                "Navigation", className="px-3 py-2 text-uppercase text-muted small"
            ),
            html.Div(
                [
                    dcc.Link(
                        [html.I(className=f"{item['icon']} nav-icon"), item["label"]],
                        href=item["href"],
                        className="nav-item",
                        id=item["id"],
                    )
                    for item in nav_items
                ]
            ),
        ]
    )


def create_quick_settings() -> dbc.Collapse:
    """
    Crée la section des paramètres rapides.

    Returns:
        dbc.Collapse: Section collapsible
    """
    return html.Div(
        [
            dbc.Button(
                [
                    html.I(className="fas fa-sliders-h me-2"),
                    "Paramètres Rapides",
                    html.I(
                        className="fas fa-chevron-down ms-auto",
                        id="quick-settings-icon",
                    ),
                ],
                id="quick-settings-toggle",
                color="link",
                className="text-start text-light p-3 w-100",
                style={"border": "none", "textDecoration": "none"},
            ),
            dbc.Collapse(
                [
                    html.Div(
                        [
                            # Capital initial
                            html.Div(
                                [
                                    html.Label(
                                        "Capital Initial", className="small text-muted"
                                    ),
                                    dbc.Input(
                                        id="quick-capital",
                                        type="number",
                                        value=10000,
                                        min=100,
                                        max=1000000,
                                        step=100,
                                        size="sm",
                                    ),
                                ],
                                className="mb-2 px-3",
                            ),
                            # Commission
                            html.Div(
                                [
                                    html.Label(
                                        "Commission (%)", className="small text-muted"
                                    ),
                                    dbc.Input(
                                        id="quick-commission",
                                        type="number",
                                        value=0.1,
                                        min=0,
                                        max=5,
                                        step=0.01,
                                        size="sm",
                                    ),
                                ],
                                className="mb-2 px-3",
                            ),
                            # Timeframe
                            html.Div(
                                [
                                    html.Label(
                                        "Timeframe", className="small text-muted"
                                    ),
                                    dbc.Select(
                                        id="quick-timeframe",
                                        options=[
                                            {"label": "1 minute", "value": "1m"},
                                            {"label": "5 minutes", "value": "5m"},
                                            {"label": "15 minutes", "value": "15m"},
                                            {"label": "1 heure", "value": "1h"},
                                            {"label": "4 heures", "value": "4h"},
                                            {"label": "1 jour", "value": "1d"},
                                        ],
                                        value="1h",
                                        size="sm",
                                    ),
                                ],
                                className="mb-2 px-3",
                            ),
                            # Bouton d'application
                            html.Div(
                                [
                                    dbc.Button(
                                        "Appliquer",
                                        id="apply-quick-settings",
                                        color="primary",
                                        size="sm",
                                        className="w-100",
                                    )
                                ],
                                className="px-3 pb-2",
                            ),
                        ]
                    )
                ],
                id="quick-settings-collapse",
                is_open=False,
            ),
        ]
    )


def create_quick_stats() -> html.Div:
    """
    Crée la section des statistiques rapides.

    Returns:
        html.Div: Section statistiques
    """
    return html.Div(
        [
            html.H6(
                "Statistiques", className="px-3 py-2 text-uppercase text-muted small"
            ),
            html.Div(
                [
                    # P&L Total
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Small(
                                        "P&L Total", className="text-muted d-block"
                                    ),
                                    html.Span(
                                        "+$2,450",
                                        className="fw-bold",
                                        style={"color": THEME["success"]},
                                        id="total-pnl",
                                    ),
                                ],
                                className="text-center",
                            )
                        ],
                        className="px-3 py-2",
                    ),
                    # Taux de réussite
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Small(
                                        "Taux de Réussite",
                                        className="text-muted d-block",
                                    ),
                                    html.Span(
                                        "68.5%",
                                        className="fw-bold",
                                        style={"color": THEME["accent_primary"]},
                                        id="win-rate",
                                    ),
                                ],
                                className="text-center",
                            )
                        ],
                        className="px-3 py-2",
                    ),
                    # Nombre de trades
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Small(
                                        "Trades", className="text-muted d-block"
                                    ),
                                    html.Span(
                                        "147", className="fw-bold", id="total-trades"
                                    ),
                                ],
                                className="text-center",
                            )
                        ],
                        className="px-3 py-2",
                    ),
                    # Sharpe Ratio
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Small(
                                        "Sharpe Ratio", className="text-muted d-block"
                                    ),
                                    html.Span(
                                        "1.42",
                                        className="fw-bold",
                                        style={"color": THEME["warning"]},
                                        id="sharpe-ratio",
                                    ),
                                ],
                                className="text-center",
                            )
                        ],
                        className="px-3 py-2",
                    ),
                ],
                id="quick-stats-content",
            ),
        ]
    )


def create_recent_backtests() -> html.Div:
    """
    Crée la section des backtests récents.

    Returns:
        html.Div: Section backtests récents
    """
    return html.Div(
        [
            html.H6(
                "Backtests Récents",
                className="px-3 py-2 text-uppercase text-muted small",
            ),
            html.Div(
                [
                    # Liste des backtests récents (placeholder)
                    create_backtest_item(
                        "BB_ATR_BTCUSDT", "+12.5%", "2h", THEME["success"]
                    ),
                    create_backtest_item(
                        "RSI_MA_ETHUSDT", "-3.2%", "1d", THEME["danger"]
                    ),
                    create_backtest_item(
                        "MACD_BNBUSDT", "+8.7%", "4h", THEME["success"]
                    ),
                    # Bouton voir plus
                    html.Div(
                        [
                            dbc.Button(
                                "Voir tous",
                                color="outline-primary",
                                size="sm",
                                href="/results",
                                className="w-100",
                            )
                        ],
                        className="px-3 py-2",
                    ),
                ],
                id="recent-backtests-content",
            ),
        ]
    )


def create_backtest_item(
    name: str, return_pct: str, timeframe: str, color: str
) -> html.Div:
    """
    Crée un élément de backtest récent.

    Args:
        name: Nom du backtest
        return_pct: Pourcentage de retour
        timeframe: Timeframe utilisé
        color: Couleur du retour

    Returns:
        html.Div: Élément de backtest
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Small(name, className="fw-bold d-block"),
                            html.Small(
                                f"Timeframe: {timeframe}", className="text-muted"
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(
                                return_pct, className="fw-bold", style={"color": color}
                            )
                        ],
                        className="text-end",
                    ),
                ],
                className="d-flex justify-content-between align-items-center",
            )
        ],
        className="px-3 py-2 border-bottom",
        style={"borderColor": THEME["border_color"], "cursor": "pointer"},
    )


# =============================================================================
# CALLBACKS
# =============================================================================


@callback(
    Output("quick-settings-collapse", "is_open"),
    Output("quick-settings-icon", "className"),
    Input("quick-settings-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_quick_settings(n_clicks):
    """
    Toggle la section des paramètres rapides.

    Args:
        n_clicks: Nombre de clics

    Returns:
        Tuple[bool, str]: État ouvert/fermé et classe de l'icône
    """
    if n_clicks is None:
        raise PreventUpdate

    is_open = (n_clicks % 2) == 1
    icon_class = (
        "fas fa-chevron-up ms-auto" if is_open else "fas fa-chevron-down ms-auto"
    )

    return is_open, icon_class


@callback(
    [
        Output("total-pnl", "children"),
        Output("win-rate", "children"),
        Output("total-trades", "children"),
        Output("sharpe-ratio", "children"),
    ],
    Input("stats-interval", "n_intervals"),
)
def update_quick_stats(n_intervals):
    """
    Met à jour les statistiques rapides.

    Args:
        n_intervals: Nombre d'intervalles

    Returns:
        Tuple: Nouvelles valeurs des statistiques
    """
    # Ici on récupérerait les vraies données depuis la base de données
    # Pour la démo, on utilise des valeurs statiques

    return "+$2,450", "68.5%", "147", "1.42"


# Interval pour mise à jour des stats
stats_interval = dcc.Interval(
    id="stats-interval", interval=60000, n_intervals=0  # 1 minute
)
