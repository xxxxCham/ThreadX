"""
Application principale ThreadX Dashboard
=======================================

Point d'entrée de l'application Dash pour le backtesting de trading.
Gère l'authentification, le routing et le layout principal.
"""

import os
import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, ctx
from flask import session
from flask_cors import CORS
import plotly.io as pio

# Configuration et utilitaires
from config import (
    THEME,
    HOST,
    PORT,
    DEBUG,
    SECRET_KEY,
    AUTH_ENABLED,
    ASSETS_PATH,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
)
from utils.auth import auth_manager
from utils.helpers import setup_logging

# Composants
from components.navbar import create_navbar, status_interval
from components.sidebar import create_sidebar, stats_interval

# Pages
from pages.home import create_home_layout
from pages import backtesting


# =============================================================================
# CONFIGURATION DE L'APPLICATION
# =============================================================================

# Configuration du logging
logger = setup_logging()

# Styles externes
external_stylesheets = [
    dbc.themes.DARKLY,
    {
        "href": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
        "rel": "stylesheet",
    },
]

# Initialisation de l'application Dash
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    assets_folder=str(ASSETS_PATH),
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {
            "name": "description",
            "content": "ThreadX Dashboard - Plateforme de backtesting crypto",
        },
    ],
)

# Configuration du serveur Flask
server = app.server
server.secret_key = SECRET_KEY

# Configuration CORS
CORS(server, supports_credentials=True)

# Configuration Plotly
pio.templates.default = "plotly_dark"

# Titre de l'application
app.title = "ThreadX Dashboard"


# =============================================================================
# LAYOUT PRINCIPAL
# =============================================================================


def create_main_layout() -> html.Div:
    """
    Crée le layout principal de l'application.

    Returns:
        html.Div: Layout principal
    """
    return html.Div(
        [
            # URL Router
            dcc.Location(id="url", refresh=False),
            # Store pour l'état global
            dcc.Store(id="global-store"),
            # Intervals pour les mises à jour
            status_interval,
            stats_interval,
            # Contenu principal
            html.Div(id="main-content"),
        ],
        style={"backgroundColor": THEME["primary_bg"]},
    )


def create_authenticated_layout() -> html.Div:
    """
    Crée le layout pour les utilisateurs authentifiés.

    Returns:
        html.Div: Layout authentifié
    """
    return html.Div(
        [
            # Navigation
            create_navbar(),
            # Container principal
            html.Div(
                [
                    # Sidebar
                    create_sidebar(),
                    # Zone de contenu
                    html.Div(id="page-content", className="content-area fade-in"),
                ]
            ),
        ]
    )


def create_login_layout() -> html.Div:
    """
    Crée le layout de connexion.

    Returns:
        html.Div: Layout de connexion
    """
    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Img(
                                                                src="/assets/logo.png",
                                                                height="60px",
                                                                className="mb-3",
                                                            ),
                                                            html.H2(
                                                                "ThreadX Dashboard",
                                                                className="mb-4",
                                                                style={
                                                                    "color": THEME[
                                                                        "accent_primary"
                                                                    ]
                                                                },
                                                            ),
                                                        ],
                                                        className="text-center",
                                                    ),
                                                    html.Form(
                                                        [
                                                            dbc.InputGroup(
                                                                [
                                                                    dbc.InputGroupText(
                                                                        html.I(
                                                                            className="fas fa-user"
                                                                        )
                                                                    ),
                                                                    dbc.Input(
                                                                        id="login-username",
                                                                        placeholder="Nom d'utilisateur",
                                                                        type="text",
                                                                        value="admin",  # Valeur par défaut pour la démo
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.InputGroup(
                                                                [
                                                                    dbc.InputGroupText(
                                                                        html.I(
                                                                            className="fas fa-lock"
                                                                        )
                                                                    ),
                                                                    dbc.Input(
                                                                        id="login-password",
                                                                        placeholder="Mot de passe",
                                                                        type="password",
                                                                        value="admin123",  # Valeur par défaut pour la démo
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Button(
                                                                "Se connecter",
                                                                id="login-button",
                                                                color="primary",
                                                                className="w-100 mb-3",
                                                                size="lg",
                                                            ),
                                                            html.Div(id="login-alert"),
                                                        ]
                                                    ),
                                                ]
                                            )
                                        ],
                                        style={
                                            "backgroundColor": THEME["secondary_bg"],
                                            "border": f"1px solid {THEME['border_color']}",
                                        },
                                    )
                                ],
                                width=6,
                                lg=4,
                            )
                        ],
                        justify="center",
                        className="min-vh-100 align-items-center",
                    )
                ],
                fluid=True,
            )
        ],
        style={"backgroundColor": THEME["primary_bg"], "minHeight": "100vh"},
    )


# =============================================================================
# CALLBACKS PRINCIPAUX
# =============================================================================


@app.callback(Output("main-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    """
    Gère le routing principal et l'authentification.

    Args:
        pathname: Chemin de l'URL

    Returns:
        html.Div: Contenu de la page
    """
    logger.info(f"Navigation vers: {pathname}")

    # Vérifier l'authentification
    if AUTH_ENABLED and not auth_manager.is_authenticated():
        logger.info("Utilisateur non authentifié, redirection vers login")
        return create_login_layout()

    # Layout authentifié
    return create_authenticated_layout()


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def update_page_content(pathname):
    """
    Met à jour le contenu de la page selon l'URL.

    Args:
        pathname: Chemin de l'URL

    Returns:
        html.Div: Contenu de la page
    """
    if not AUTH_ENABLED or auth_manager.is_authenticated():
        if pathname == "/" or pathname == "/home":
            return create_home_layout()
        elif pathname == "/backtesting":
            return backtesting.layout
        elif pathname == "/results":
            return create_placeholder_page("Résultats", "fas fa-table")
        elif pathname == "/settings":
            return create_placeholder_page("Paramètres", "fas fa-cog")
        else:
            return create_404_page()

    return html.Div()


@app.callback(
    [
        Output("url", "pathname", allow_duplicate=True),
        Output("login-alert", "children"),
    ],
    Input("login-button", "n_clicks"),
    [State("login-username", "value"), State("login-password", "value")],
    prevent_initial_call=True,
)
def handle_login(n_clicks, username, password):
    """
    Gère la tentative de connexion.

    Args:
        n_clicks: Nombre de clics sur le bouton
        username: Nom d'utilisateur
        password: Mot de passe

    Returns:
        Tuple: (nouvelle_url, message_alerte)
    """
    if n_clicks and username and password:
        if auth_manager.login(username, password):
            logger.info(f"Connexion réussie pour: {username}")
            return "/", None
        else:
            logger.warning(f"Tentative de connexion échouée pour: {username}")
            return dash.no_update, dbc.Alert(
                "Nom d'utilisateur ou mot de passe incorrect",
                color="danger",
                className="mt-2",
            )

    return dash.no_update, dash.no_update


def create_placeholder_page(title: str, icon: str) -> html.Div:
    """
    Crée une page placeholder.

    Args:
        title: Titre de la page
        icon: Classe d'icône FontAwesome

    Returns:
        html.Div: Page placeholder
    """
    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.I(
                                                className=icon,
                                                style={
                                                    "fontSize": "4rem",
                                                    "color": THEME["accent_primary"],
                                                },
                                            ),
                                            html.H1(title, className="mt-3 mb-4"),
                                            html.P(
                                                f"La page {title} est en cours de développement.",
                                                className="lead text-muted",
                                            ),
                                            dbc.Button(
                                                "Retour à l'accueil",
                                                href="/",
                                                color="primary",
                                            ),
                                        ],
                                        className="text-center py-5",
                                    )
                                ],
                                width=12,
                            )
                        ]
                    )
                ],
                className="py-5",
            )
        ]
    )


def create_404_page() -> html.Div:
    """
    Crée une page 404.

    Returns:
        html.Div: Page 404
    """
    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.H1(
                                                "404",
                                                style={
                                                    "fontSize": "6rem",
                                                    "color": THEME["danger"],
                                                },
                                            ),
                                            html.H2(
                                                "Page non trouvée", className="mb-4"
                                            ),
                                            html.P(
                                                "La page que vous cherchez n'existe pas.",
                                                className="lead text-muted",
                                            ),
                                            dbc.Button(
                                                "Retour à l'accueil",
                                                href="/",
                                                color="primary",
                                            ),
                                        ],
                                        className="text-center py-5",
                                    )
                                ],
                                width=12,
                            )
                        ]
                    )
                ],
                className="py-5",
            )
        ]
    )


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

# Configuration du layout
app.layout = create_main_layout()

if __name__ == "__main__":
    logger.info(f"Démarrage de ThreadX Dashboard sur {HOST}:{PORT}")
    logger.info(f"Mode debug: {DEBUG}")
    logger.info(f"Authentification: {'Activée' if AUTH_ENABLED else 'Désactivée'}")

    app.run(host=HOST, port=PORT, debug=DEBUG, dev_tools_hot_reload=DEBUG)
