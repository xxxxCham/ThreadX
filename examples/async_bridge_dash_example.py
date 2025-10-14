"""
Exemple d'intégration ThreadXBridge avec Dash UI
================================================

Démontre pattern polling non-bloquant pour UI Dash.

Usage:
    python examples/async_bridge_dash_example.py

Accès:
    http://localhost:8050

Architecture:
    Dash Callback (non-bloquant)
         ↓
    ThreadXBridge.run_backtest_async()
         ↓ (immédiat return)
    dcc.Interval polling (500ms)
         ↓
    ThreadXBridge.get_event() → update graph
"""

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from threadx.bridge import BacktestRequest, Configuration, ThreadXBridge

# Initialize Dash app
app = dash.Dash(__name__)

# Initialize ThreadXBridge (global instance)
bridge = ThreadXBridge(max_workers=4, config=Configuration(max_workers=4))

# UI Layout
app.layout = html.Div(
    [
        html.H1("ThreadX Bridge - Async Backtest Demo"),
        html.Div(
            [
                html.Label("Symbol:"),
                dcc.Input(
                    id="symbol-input",
                    type="text",
                    value="BTCUSDT",
                    placeholder="Symbol (ex: BTCUSDT)",
                ),
            ]
        ),
        html.Div(
            [
                html.Label("Timeframe:"),
                dcc.Dropdown(
                    id="timeframe-dropdown",
                    options=[
                        {"label": "1 hour", "value": "1h"},
                        {"label": "4 hours", "value": "4h"},
                        {"label": "1 day", "value": "1d"},
                    ],
                    value="1h",
                ),
            ]
        ),
        html.Div(
            [
                html.Label("Strategy:"),
                dcc.Dropdown(
                    id="strategy-dropdown",
                    options=[
                        {"label": "Bollinger Reversion", "value": "bb"},
                        {"label": "EMA Crossover", "value": "ema"},
                        {"label": "RSI Mean Reversion", "value": "rsi"},
                    ],
                    value="bb",
                ),
            ]
        ),
        html.Button("Run Backtest", id="run-button", n_clicks=0),
        html.Div(id="status-div", style={"margin-top": "20px"}),
        dcc.Graph(id="equity-graph"),
        html.Div(id="metrics-div"),
        # Polling interval pour récupérer événements (500ms)
        dcc.Interval(id="polling-interval", interval=500, n_intervals=0),  # 500ms
        # Store pour task_id actuel
        dcc.Store(id="current-task-store", data=None),
    ]
)


@app.callback(
    Output("current-task-store", "data"),
    Output("status-div", "children"),
    Input("run-button", "n_clicks"),
    State("symbol-input", "value"),
    State("timeframe-dropdown", "value"),
    State("strategy-dropdown", "value"),
    prevent_initial_call=True,
)
def submit_backtest(n_clicks, symbol, timeframe, strategy):
    """
    Callback submit backtest (non-bloquant).

    Soumet tâche async et retourne immédiatement.
    UI reste responsive pendant calculs.
    """
    if n_clicks == 0:
        return None, ""

    # Créer requête
    req = BacktestRequest(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        params=(
            {"period": 20, "std": 2.0} if strategy == "bb" else {"fast": 12, "slow": 26}
        ),
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    # Soumettre tâche async (retour IMMEDIAT, non-bloquant)
    bridge.run_backtest_async(req)
    task_id = list(bridge.active_tasks.keys())[-1]  # Récupérer dernier task_id

    # Retourner task_id pour polling + status message
    return task_id, html.Div(
        [
            html.Span("⏳ ", style={"font-size": "20px"}),
            html.Span(
                f"Backtest submitted: {symbol} {timeframe} {strategy}",
                style={"color": "blue"},
            ),
            html.Br(),
            html.Span(
                f"Task ID: {task_id}",
                style={"font-size": "12px", "color": "gray"},
            ),
        ]
    )


@app.callback(
    Output("equity-graph", "figure"),
    Output("metrics-div", "children"),
    Output("status-div", "children", allow_duplicate=True),
    Input("polling-interval", "n_intervals"),
    State("current-task-store", "data"),
    prevent_initial_call=True,
)
def poll_results(n_intervals, current_task_id):
    """
    Callback polling événements (non-bloquant).

    Appelé toutes les 500ms par dcc.Interval.
    Récupère événements de results_queue via get_event().
    """
    # Récupérer événement (non-bloquant, timeout 0.1s)
    event = bridge.get_event(timeout=0.1)

    if event is None:
        # Pas d'événement : retourner graph vide + status actuel
        state = bridge.get_state()
        status_msg = html.Div(
            [
                html.Span(
                    f"Active tasks: {state['active_tasks']} | "
                    f"Queue: {state['queue_size']}",
                    style={"color": "gray", "font-size": "12px"},
                )
            ]
        )
        return go.Figure(), "", status_msg

    # Événement reçu
    event_type = event["type"]
    task_id = event["task_id"]
    payload = event["payload"]

    if event_type == "backtest_done":
        # Backtest terminé : afficher résultats
        result = payload

        # Graph equity curve
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(result.equity_curve))),
                y=result.equity_curve,
                mode="lines",
                name="Equity",
                line={"color": "green", "width": 2},
            )
        )
        fig.update_layout(
            title=f"Equity Curve - Task {task_id}",
            xaxis_title="Days",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
        )

        # Métriques
        metrics = html.Div(
            [
                html.H3("Performance Metrics"),
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td("Sharpe Ratio:"),
                                html.Td(
                                    f"{result.sharpe_ratio:.2f}",
                                    style={"font-weight": "bold"},
                                ),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Total Return:"),
                                html.Td(
                                    f"{result.total_return:.2%}",
                                    style={"font-weight": "bold"},
                                ),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Max Drawdown:"),
                                html.Td(
                                    f"{result.max_drawdown:.2%}",
                                    style={"font-weight": "bold"},
                                ),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Win Rate:"),
                                html.Td(
                                    f"{result.win_rate:.2%}",
                                    style={"font-weight": "bold"},
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

        # Status succès
        status = html.Div(
            [
                html.Span("✅ ", style={"font-size": "20px"}),
                html.Span(
                    f"Backtest complete (Task {task_id})",
                    style={"color": "green", "font-weight": "bold"},
                ),
            ]
        )

        return fig, metrics, status

    elif event_type == "error":
        # Erreur : afficher message
        error_msg = payload

        status = html.Div(
            [
                html.Span("❌ ", style={"font-size": "20px"}),
                html.Span(
                    f"Error (Task {task_id}): {error_msg}",
                    style={"color": "red", "font-weight": "bold"},
                ),
            ]
        )

        return go.Figure(), "", status

    else:
        # Autre type événement
        return go.Figure(), "", f"Unknown event: {event_type}"


if __name__ == "__main__":
    try:
        print("Starting Dash app on http://localhost:8050")
        print("ThreadXBridge initialized with 4 workers")
        print("Polling interval: 500ms")
        print("\nPress Ctrl+C to stop\n")

        app.run_server(debug=True, port=8050)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        bridge.shutdown(wait=True, timeout=10)
        print("Goodbye!")
