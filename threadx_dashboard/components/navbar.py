"""
Barre de navigation pour ThreadX Dashboard
=========================================

Ce module contient le composant de navigation principal
avec logo, titre et menu utilisateur.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
from dash.exceptions import PreventUpdate

from utils.auth import auth_manager
from config import THEME


def create_navbar() -> dbc.Navbar:
    """
    Crée la barre de navigation principale.

    Returns:
        dbc.Navbar: Composant de navigation
    """
    return dbc.Navbar(
        dbc.Container(
            [
                # Logo et titre
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        # Toggle sidebar button
                                        dbc.Button(
                                            html.I(className="fas fa-bars"),
                                            id="sidebar-toggle",
                                            color="link",
                                            className="text-light me-3",
                                            style={
                                                "border": "none",
                                                "fontSize": "1.2rem",
                                            },
                                        ),
                                        # Logo
                                        html.Img(
                                            src="/assets/logo.png",
                                            height="40px",
                                            className="me-2",
                                        ),
                                        # Titre
                                        dbc.NavbarBrand(
                                            "ThreadX Dashboard",
                                            className="fw-bold",
                                            style={"color": THEME["accent_primary"]},
                                        ),
                                    ],
                                    className="d-flex align-items-center",
                                )
                            ],
                            width="auto",
                        ),
                        dbc.Col(
                            [
                                # Status indicator
                                html.Div(
                                    id="connection-status",
                                    className="d-flex align-items-center",
                                )
                            ],
                            width=True,
                            className="d-flex justify-content-center",
                        ),
                        dbc.Col(
                            [
                                # Menu utilisateur
                                create_user_menu()
                            ],
                            width="auto",
                            className="d-flex justify-content-end",
                        ),
                    ],
                    align="center",
                    className="w-100",
                )
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        fixed="top",
        style={
            "height": "60px",
            "backgroundColor": THEME["primary_bg"],
            "borderBottom": f"1px solid {THEME['border_color']}",
        },
    )


def create_user_menu() -> dbc.DropdownMenu:
    """
    Crée le menu déroulant utilisateur.

    Returns:
        dbc.DropdownMenu: Menu utilisateur
    """
    user = auth_manager.get_current_user()
    username = user.get("username", "Guest") if user else "Guest"

    return dbc.DropdownMenu(
        children=[
            dbc.DropdownMenuItem("Profil", id="profile-menu-item", disabled=True),
            dbc.DropdownMenuItem(divider=True),
            dbc.DropdownMenuItem(
                [html.I(className="fas fa-cog me-2"), "Paramètres"],
                id="settings-menu-item",
            ),
            dbc.DropdownMenuItem(
                [html.I(className="fas fa-question-circle me-2"), "Aide"],
                id="help-menu-item",
            ),
            dbc.DropdownMenuItem(
                [html.I(className="fas fa-info-circle me-2"), "À propos"],
                id="about-menu-item",
            ),
            dbc.DropdownMenuItem(divider=True),
            dbc.DropdownMenuItem(
                [html.I(className="fas fa-sign-out-alt me-2"), "Déconnexion"],
                id="logout-menu-item",
                style={"color": THEME["danger"]},
            ),
        ],
        label=[html.I(className="fas fa-user-circle me-2"), username],
        color="link",
        className="text-light",
        toggle_style={
            "border": "none",
            "color": THEME["text_primary"],
            "backgroundColor": "transparent",
        },
    )


def create_connection_status() -> html.Div:
    """
    Crée l'indicateur de statut de connexion.

    Returns:
        html.Div: Indicateur de statut
    """
    return html.Div(
        [
            dbc.Badge(
                [html.I(className="fas fa-circle me-1"), "En ligne"],
                color="success",
                className="d-flex align-items-center",
                id="status-badge",
            )
        ]
    )


# =============================================================================
# CALLBACKS
# =============================================================================


@callback(
    Output("sidebar", "className"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar", "className"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, current_class):
    """
    Gère le toggle de la sidebar.

    Args:
        n_clicks: Nombre de clics
        current_class: Classes CSS actuelles

    Returns:
        str: Nouvelles classes CSS
    """
    if n_clicks is None:
        raise PreventUpdate

    if "collapsed" in (current_class or ""):
        return "sidebar slide-in-left"
    else:
        return "sidebar collapsed"


@callback(Output("content-area", "className"), Input("sidebar", "className"))
def update_content_margin(sidebar_class):
    """
    Ajuste la marge du contenu selon l'état de la sidebar.

    Args:
        sidebar_class: Classes CSS de la sidebar

    Returns:
        str: Classes CSS du contenu
    """
    if "collapsed" in (sidebar_class or ""):
        return "content-area sidebar-collapsed fade-in"
    else:
        return "content-area fade-in"


@callback(
    Output("connection-status", "children"), Input("status-interval", "n_intervals")
)
def update_connection_status(n_intervals):
    """
    Met à jour le statut de connexion périodiquement.

    Args:
        n_intervals: Nombre d'intervalles écoulés

    Returns:
        html.Div: Composant de statut mis à jour
    """
    # Ici on pourrait vérifier la connexion à l'API, base de données, etc.
    is_connected = True  # Placeholder

    if is_connected:
        return dbc.Badge(
            [
                html.I(
                    className="fas fa-circle me-1", style={"color": THEME["success"]}
                ),
                "Connecté",
            ],
            color="success",
            className="d-flex align-items-center",
        )
    else:
        return dbc.Badge(
            [html.I(className="fas fa-exclamation-triangle me-1"), "Déconnecté"],
            color="danger",
            className="d-flex align-items-center",
        )


@callback(
    Output("url", "pathname"),
    Input("logout-menu-item", "n_clicks"),
    prevent_initial_call=True,
)
def handle_logout(n_clicks):
    """
    Gère la déconnexion utilisateur.

    Args:
        n_clicks: Nombre de clics sur déconnexion

    Returns:
        str: URL de redirection
    """
    if n_clicks:
        auth_manager.logout()
        return "/login"

    raise PreventUpdate


# Interval pour mise à jour du statut
status_interval = dcc.Interval(
    id="status-interval", interval=30000, n_intervals=0  # 30 secondes
)
