"""
ThreadX Dash Application - Main Entry Point
===========================================

Application Dash principale pour ThreadX.
Charge le layout modulaire, configure le serveur, et expose
l'interface web sur le port configuré (default: 8050).

Architecture:
    apps/dash_app.py (CE FICHIER)
         ↓
    src/threadx/ui/layout.py (Layout statique)
         ↓
    src/threadx/ui/components/* (P5-P6)
         ↓
    src/threadx/ui/callbacks.py (P7)

Usage:
    # Depuis racine ThreadX
    python apps/dash_app.py

    # Avec port custom
    $env:THREADX_DASH_PORT=8888
    python apps/dash_app.py

Configuration:
    - Port : Variable THREADX_DASH_PORT (fallback 8050)
    - Theme : Bootstrap DARKLY
    - Debug : False (production-ready)

Author: ThreadX Framework
Version: Prompt 4 - Layout Principal
"""

from __future__ import annotations

import os, sys, json, itertools, subprocess
from pathlib import Path
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, no_update, callback, dash_table

# Ajouter src au PYTHONPATH si nécessaire
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from threadx.ui.layout import create_layout

# Optional: Import Bridge pour signature future (P7)
# Pas d'appel métier ici, juste passage instance
try:
    from threadx.bridge import ThreadXBridge
    from threadx.ui.callbacks import register_callbacks

    bridge = ThreadXBridge(max_workers=4)
except ImportError:
    # Bridge pas encore implémenté ou tests isolés
    bridge = None
    register_callbacks = None


# Helpers d'état (inline)
def default_state():
    return {
        "tokens_blocks": [],
        "indics_blocks": [],
        "periods_blocks": [],
        "strategies_blocks": [],
        "omega_profile": {"label": "Ω", "primary": "sharpe", "constraints": {}},
        "version": 1,
    }


def _alpha_labels(blocks):
    return [b.get("label", "").strip() for b in blocks if b.get("active", True)]


def summarize_expression(state: dict) -> str:
    t = "+".join(_alpha_labels(state.get("tokens_blocks", []))) or "∅"
    i = "+".join(_alpha_labels(state.get("indics_blocks", []))) or "∅"
    p = "+".join(_alpha_labels(state.get("periods_blocks", []))) or "∅"
    s = "+".join(_alpha_labels(state.get("strategies_blocks", []))) or "∅"
    return f"({t}) × ({i}) × ({p}) → ({s})"


def _cartesian_product(*iters):
    if any(len(x) == 0 for x in iters):
        return []
    return list(itertools.product(*iters))


# Configuration application
PORT = int(os.environ.get("THREADX_DASH_PORT", 8050))
DEBUG = os.environ.get("THREADX_DASH_DEBUG", "false").lower() == "true"

# Initialiser Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="ThreadX Dashboard",
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0",
        }
    ],
)

# Serveur Flask sous-jacent (pour déploiement production)
server = app.server

############################################################
# Inline 2-column UI (opt-in via env THREADX_INLINE_UI)
############################################################

def _parse_blocks(text: str):
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\n", ",").split(",")]
    return [
        {"label": p, "active": True}
        for p in parts
        if p and not p.startswith("#")
    ]


def build_run_plan(state: dict):
    tokens = _alpha_labels(state.get("tokens_blocks", []))
    indics = _alpha_labels(state.get("indics_blocks", []))
    periods = _alpha_labels(state.get("periods_blocks", []))
    strategies = _alpha_labels(state.get("strategies_blocks", []))

    combos = _cartesian_product(tokens, indics or ["-"], periods, strategies)

    plan = []
    for idx, (tok, ind, tf, strat) in enumerate(combos, 1):
        cmd_args = [
            sys.executable,
            "-m",
            "threadx.cli",
            "backtest",
            "run",
            "--strategy",
            str(strat),
            "--symbol",
            str(tok),
            "--tf",
            str(tf),
            "--json",
        ]
        item = {
            "id": idx,
            "symbol": tok,
            "timeframe": tf,
            "strategy": strat,
            "indicator": ind,
            "cmd": " ".join(cmd_args),
            "cmd_args": cmd_args,
        }
        plan.append(item)
    return plan


def build_inline_layout():
    default_tokens = "BTCUSDT, ETHUSDT"
    default_indics = "RSI, MACD"
    default_periods = "1h, 4h"
    default_strategies = "ema_crossover, bollinger_reversion"

    # Stores
    stores = [
        dcc.Store(id="ui_state", storage_type="memory"),
        dcc.Store(id="run_plan", storage_type="memory"),
    ]

    left_controls = dbc.Card(
        [
            dbc.CardHeader("Alpha Builder", className="bg-secondary text-light"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Tokens (comma or newline)"),
                                    dcc.Textarea(
                                        id="tokens_input",
                                        value=default_tokens,
                                        style={"width": "100%", "height": "70px"},
                                    ),
                                    dbc.Label("Indicators"),
                                    dcc.Textarea(
                                        id="indics_input",
                                        value=default_indics,
                                        style={"width": "100%", "height": "70px"},
                                    ),
                                    dbc.Label("Periods / Timeframes"),
                                    dcc.Textarea(
                                        id="periods_input",
                                        value=default_periods,
                                        style={"width": "100%", "height": "70px"},
                                    ),
                                    dbc.Label("Strategies"),
                                    dcc.Textarea(
                                        id="strategies_input",
                                        value=default_strategies,
                                        style={"width": "100%", "height": "70px"},
                                    ),
                                ]
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Update State",
                                    id="btn_update_state",
                                    color="primary",
                                    className="me-2 mt-3",
                                )
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Load Demo",
                                    id="btn_load_demo",
                                    color="secondary",
                                    className="mt-3",
                                )
                            ),
                        ],
                        className="g-2",
                    ),
                    html.Hr(className="border-secondary"),
                    html.Div(
                        [
                            html.Small("Expression"),
                            html.Div(id="expr_summary", className="text-info fw-bold"),
                        ]
                    ),
                ],
                className="bg-dark",
            ),
        ],
        className="bg-dark border-secondary",
    )

    right_panel = dbc.Card(
        [
            dbc.CardHeader("Run Plan", className="bg-secondary text-light"),
            dbc.CardBody(
                [
                    dbc.Button(
                        "Generate Run Plan",
                        id="btn_generate_plan",
                        color="info",
                        className="mb-3",
                    ),
                    dash_table.DataTable(
                        id="plan_table",
                        columns=[
                            {"name": "#", "id": "id"},
                            {"name": "Symbol", "id": "symbol"},
                            {"name": "TF", "id": "timeframe"},
                            {"name": "Strategy", "id": "strategy"},
                            {"name": "Indicator", "id": "indicator"},
                            {"name": "Cmd", "id": "cmd"},
                        ],
                        data=[],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={"backgroundColor": "#1e1e1e", "color": "#ddd"},
                        style_header={"backgroundColor": "#2b2b2b", "color": "#fff"},
                    ),
                    html.Div(className="mt-3", children=[
                        dbc.Button(
                            "Run Plan (first 5)",
                            id="btn_run_plan",
                            color="success",
                            className="me-2",
                        ),
                        html.Span(id="run_status", className="text-muted"),
                    ]),
                    html.Hr(className="border-secondary"),
                    html.Pre(id="run_logs", style={"whiteSpace": "pre-wrap", "maxHeight": "300px", "overflowY": "auto"}),
                ],
                className="bg-dark",
            ),
        ],
        className="bg-dark border-secondary",
    )

    return dbc.Container(
        fluid=True,
        className="bg-dark text-light min-vh-100 p-3",
        children=[
            *stores,
            dbc.Row(
                [
                    dbc.Col(left_controls, md=4, lg=4, className="mb-3"),
                    dbc.Col(right_panel, md=8, lg=8),
                ],
                className="g-3",
            ),
        ],
    )


# Choose layout (default: existing modular layout)
USE_INLINE = os.environ.get("THREADX_INLINE_UI", "0").lower() in ("1", "true", "yes")
if USE_INLINE:
    app.layout = build_inline_layout()
else:
    # Keep existing modular layout from threadx.ui.layout
    app.layout = create_layout(bridge)

# Enregistrer callbacks (P7)
if (not USE_INLINE) and bridge and register_callbacks:
    register_callbacks(app, bridge)
    print("Callbacks: Registered (P7 active)")
else:
    print("Callbacks: Skipped (P7 not available)")


if __name__ == "__main__":
    print("=" * 60)
    print("ThreadX Dash Dashboard")
    print("=" * 60)
    print(f"Server starting on: http://127.0.0.1:{PORT}")
    print(f"Debug mode: {DEBUG}")
    print(f"Theme: Bootstrap DARKLY")
    if bridge:
        print(f"Bridge: Initialized ({bridge.config.max_workers} workers)")
    else:
        print("Bridge: Not available (will be connected in P7)")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=DEBUG, port=PORT, host="127.0.0.1")


############################################################
# Callbacks for inline UI
############################################################

@app.callback(
    Output("ui_state", "data"),
    Output("expr_summary", "children"),
    Output("tokens_input", "value"),
    Output("indics_input", "value"),
    Output("periods_input", "value"),
    Output("strategies_input", "value"),
    Input("btn_update_state", "n_clicks"),
    Input("btn_load_demo", "n_clicks"),
    State("tokens_input", "value"),
    State("indics_input", "value"),
    State("periods_input", "value"),
    State("strategies_input", "value"),
    prevent_initial_call=False,
)
def _update_state(n_update, n_demo, tokens_text, indics_text, periods_text, strategies_text):
    # Determine source
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if trigger == "btn_load_demo" or (not trigger):
        tokens_text = tokens_text or "BTCUSDT, ETHUSDT"
        indics_text = indics_text or "RSI, MACD"
        periods_text = periods_text or "1h, 4h"
        strategies_text = strategies_text or "ema_crossover, bollinger_reversion"

    state = default_state()
    state["tokens_blocks"] = _parse_blocks(tokens_text)
    state["indics_blocks"] = _parse_blocks(indics_text)
    state["periods_blocks"] = _parse_blocks(periods_text)
    state["strategies_blocks"] = _parse_blocks(strategies_text)

    return state, summarize_expression(state), tokens_text, indics_text, periods_text, strategies_text


@app.callback(
    Output("run_plan", "data"),
    Output("plan_table", "data"),
    Input("btn_generate_plan", "n_clicks"),
    State("ui_state", "data"),
    prevent_initial_call=True,
)
def _generate_plan(n_clicks, state):
    if not state:
        return no_update, no_update
    plan = build_run_plan(state)
    return plan, plan


@app.callback(
    Output("run_status", "children"),
    Output("run_logs", "children"),
    Input("btn_run_plan", "n_clicks"),
    State("run_plan", "data"),
    prevent_initial_call=True,
)
def _run_plan(n_clicks, plan):
    if not plan:
        return "No plan to run", ""

    logs = []
    max_items = min(5, len(plan))
    for item in plan[:max_items]:
        cmd_args = item.get("cmd_args")
        if not cmd_args:
            continue
        try:
            logs.append(f"$ {' '.join(cmd_args)}")
            result = subprocess.run(cmd_args, cwd=str(REPO_ROOT), capture_output=True, text=True)
            if result.returncode == 0:
                logs.append(result.stdout.strip() or "OK")
            else:
                logs.append(f"ERROR [{result.returncode}]: {result.stderr.strip()}")
        except Exception as e:
            logs.append(f"EXCEPTION: {e}")

    status = f"Executed {max_items}/{len(plan)} task(s)."
    return status, "\n".join(logs)

