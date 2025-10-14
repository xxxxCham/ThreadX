"""
ThreadX UI Callbacks - Dash Routing + Bridge Integration
=========================================================

Centralise tous les callbacks Dash pour connecter l'UI au Bridge async.
Pattern: Submit → Poll → Dispatch avec gestion d'état thread-safe.

Architecture:
    User Click → Submit Callback → Bridge.run_*_async(req)
                                         ↓ (task_id)
    Polling Callback (500ms) → Bridge.get_event(task_id)
                                         ↓ (result)
    Dispatch Updates → Update UI (graphs, tables, alerts)

Callbacks Groups:
    1. Data Manager: Validation données + registry
    2. Indicators: Build cache indicateurs techniques
    3. Backtest: Exécution stratégies + résultats
    4. Optimization: Parameter sweeps + heatmaps

Usage:
    from threadx.ui.callbacks import register_callbacks
    register_callbacks(app, bridge)

Author: ThreadX Framework
Version: Prompt 7 - Callbacks + Bridge Routing
"""

import logging
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate

from threadx.bridge import (
    BacktestRequest,
    BridgeError,
    IndicatorRequest,
    SweepRequest,
    ThreadXBridge,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════


def register_callbacks(app: dash.Dash, bridge: ThreadXBridge) -> None:
    """Enregistre tous les callbacks Dash avec routing Bridge async.

    Fonction principale appelée par apps/dash_app.py pour connecter
    tous les composants UI (P5-P6) au Bridge async (P3).

    Args:
        app: Instance Dash application.
        bridge: ThreadXBridge instance pour orchestration async.

    Callbacks Enregistrés:
        - Data: Submit validation + polling results
        - Indicators: Submit build + polling results
        - Backtest: Submit run + polling results
        - Optimization: Submit sweep + polling results
        - Global Polling: dcc.Interval → get_event() dispatch

    Notes:
        - Callbacks utilisent allow_duplicate=True pour multi-outputs
        - prevent_initial_call=True pour éviter déclenchements vides
        - Tous les stores (dcc.Store) doivent être dans layout
        - Error handling via BridgeError → dbc.Alert

    Example:
        >>> bridge = ThreadXBridge(max_workers=4)
        >>> app = dash.Dash(__name__)
        >>> register_callbacks(app, bridge)
        >>> app.run_server(debug=False, port=8050)
    """

    logger.info("Registering Dash callbacks for ThreadX UI...")

    # Vérifier que bridge est initialisé
    if bridge is None:
        logger.warning(
            "Bridge is None - callbacks will be registered but not functional"
        )

    # ═══════════════════════════════════════════════════════════════
    # GLOBAL POLLING - Dispatch Events to All Tabs
    # ═══════════════════════════════════════════════════════════════

    @callback(
        [
            # Data outputs
            Output("data-registry-table", "children", allow_duplicate=True),
            Output("data-alert", "children", allow_duplicate=True),
            Output("data-loading", "children", allow_duplicate=True),
            Output("validate-data-btn", "disabled", allow_duplicate=True),
            # Indicators outputs
            Output("indicators-cache-table", "children", allow_duplicate=True),
            Output("indicators-alert", "children", allow_duplicate=True),
            Output("indicators-loading", "children", allow_duplicate=True),
            Output("build-indicators-btn", "disabled", allow_duplicate=True),
            # Backtest outputs
            Output("bt-equity-graph", "figure", allow_duplicate=True),
            Output("bt-drawdown-graph", "figure", allow_duplicate=True),
            Output("bt-trades-table", "children", allow_duplicate=True),
            Output("bt-metrics-table", "children", allow_duplicate=True),
            Output("bt-status", "children", allow_duplicate=True),
            Output("bt-loading", "children", allow_duplicate=True),
            Output("bt-run-btn", "disabled", allow_duplicate=True),
            # Optimization outputs
            Output("opt-results-table", "children", allow_duplicate=True),
            Output("opt-heatmap", "figure", allow_duplicate=True),
            Output("opt-status", "children", allow_duplicate=True),
            Output("opt-loading", "children", allow_duplicate=True),
            Output("opt-run-btn", "disabled", allow_duplicate=True),
        ],
        Input("global-interval", "n_intervals"),
        [
            State("data-task-store", "data"),
            State("indicators-task-store", "data"),
            State("bt-task-store", "data"),
            State("opt-task-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def poll_bridge_events(
        n_intervals, data_task_id, ind_task_id, bt_task_id, opt_task_id
    ):
        """Poll Bridge events et dispatch updates to all tabs.

        Appelé toutes les 500ms par dcc.Interval (global-interval).
        Vérifie les task IDs actifs dans stores, récupère événements,
        et dispatch updates aux outputs correspondants.

        Args:
            n_intervals: Compteur intervals (ignoré).
            data_task_id: Task ID data validation (ou None).
            ind_task_id: Task ID indicators build (ou None).
            bt_task_id: Task ID backtest (ou None).
            opt_task_id: Task ID optimization sweep (ou None).

        Returns:
            Tuple[21 outputs]: Updates pour tous outputs ou no_update.

        Pattern:
            Pour chaque task_id non-None:
                event = bridge.get_event(task_id, timeout=0.1)
                if event ready: dispatch updates
                else: no_update
        """

        if bridge is None:
            raise PreventUpdate

        # Initialize outputs (tous no_update par défaut)
        outputs = [no_update] * 21

        # ─────────────────────────────────────────────────────────
        # Data Manager Polling
        # ─────────────────────────────────────────────────────────
        if data_task_id:
            try:
                event = bridge.get_event(data_task_id, timeout=0.1)
                if event:
                    if event.get("error"):
                        # Erreur: update alert, reset loading, enable btn
                        outputs[1] = dbc.Alert(
                            event["error"], color="danger", is_open=True
                        )
                        outputs[2] = ""
                        outputs[3] = False
                    elif event.get("status") == "completed":
                        # Success: update table, reset loading, enable btn
                        data = event.get("data", [])
                        outputs[0] = _create_data_table(data)
                        outputs[1] = dbc.Alert(
                            f"Validated {len(data)} records",
                            color="success",
                            is_open=True,
                        )
                        outputs[2] = ""
                        outputs[3] = False
            except BridgeError as e:
                logger.error(f"Data polling error: {e}")
                outputs[1] = dbc.Alert(str(e), color="danger", is_open=True)
                outputs[2] = ""
                outputs[3] = False

        # ─────────────────────────────────────────────────────────
        # Indicators Polling
        # ─────────────────────────────────────────────────────────
        if ind_task_id:
            try:
                event = bridge.get_event(ind_task_id, timeout=0.1)
                if event:
                    if event.get("error"):
                        outputs[5] = dbc.Alert(
                            event["error"], color="danger", is_open=True
                        )
                        outputs[6] = ""
                        outputs[7] = False
                    elif event.get("status") == "completed":
                        indicators = event.get("indicators", {})
                        outputs[4] = _create_indicators_table(indicators)
                        outputs[5] = dbc.Alert(
                            f"Built {len(indicators)} indicators",
                            color="success",
                            is_open=True,
                        )
                        outputs[6] = ""
                        outputs[7] = False
            except BridgeError as e:
                logger.error(f"Indicators polling error: {e}")
                outputs[5] = dbc.Alert(str(e), color="danger", is_open=True)
                outputs[6] = ""
                outputs[7] = False

        # ─────────────────────────────────────────────────────────
        # Backtest Polling
        # ─────────────────────────────────────────────────────────
        if bt_task_id:
            try:
                event = bridge.get_event(bt_task_id, timeout=0.1)
                if event:
                    if event.get("error"):
                        outputs[12] = dbc.Alert(
                            event["error"], color="danger", is_open=True
                        )
                        outputs[13] = ""
                        outputs[14] = False
                    elif event.get("status") == "completed":
                        result = event.get("result")
                        outputs[8] = _create_equity_graph(result.equity_curve)
                        outputs[9] = _create_drawdown_graph(result.drawdown_curve)
                        outputs[10] = _create_trades_table(result.trades)
                        outputs[11] = _create_metrics_table(result.metrics)
                        outputs[12] = dbc.Alert(
                            f"Backtest completed: {result.total_return:.2f}% return",
                            color="success",
                            is_open=True,
                        )
                        outputs[13] = ""
                        outputs[14] = False
            except BridgeError as e:
                logger.error(f"Backtest polling error: {e}")
                outputs[12] = dbc.Alert(str(e), color="danger", is_open=True)
                outputs[13] = ""
                outputs[14] = False

        # ─────────────────────────────────────────────────────────
        # Optimization Polling
        # ─────────────────────────────────────────────────────────
        if opt_task_id:
            try:
                event = bridge.get_event(opt_task_id, timeout=0.1)
                if event:
                    if event.get("error"):
                        outputs[18] = dbc.Alert(
                            event["error"], color="danger", is_open=True
                        )
                        outputs[19] = ""
                        outputs[20] = False
                    elif event.get("status") == "completed":
                        results = event.get("results", [])
                        outputs[15] = _create_sweep_results_table(results)
                        outputs[16] = _create_heatmap(results)
                        outputs[18] = dbc.Alert(
                            f"Sweep completed: {len(results)} combinations tested",
                            color="success",
                            is_open=True,
                        )
                        outputs[19] = ""
                        outputs[20] = False
            except BridgeError as e:
                logger.error(f"Optimization polling error: {e}")
                outputs[18] = dbc.Alert(str(e), color="danger", is_open=True)
                outputs[19] = ""
                outputs[20] = False

        return tuple(outputs)

    # ═══════════════════════════════════════════════════════════════
    # DATA MANAGER - Submit Validation
    # ═══════════════════════════════════════════════════════════════

    @callback(
        [
            Output("data-task-store", "data", allow_duplicate=True),
            Output("validate-data-btn", "disabled", allow_duplicate=True),
            Output("data-loading", "children", allow_duplicate=True),
            Output("data-alert", "children", allow_duplicate=True),
        ],
        Input("validate-data-btn", "n_clicks"),
        [
            State("data-upload", "contents"),
            State("data-upload", "filename"),
            State("data-source", "value"),
            State("data-symbol", "value"),
            State("data-timeframe", "value"),
        ],
        prevent_initial_call=True,
    )
    def submit_data_validation(n_clicks, contents, filename, source, symbol, timeframe):
        """Submit data validation task to Bridge async.

        Déclenché par clic sur validate-data-btn.
        Valide inputs, crée DataRequest, soumet à Bridge,
        et retourne task_id pour polling.

        Args:
            n_clicks: Compteur clics (trigger).
            contents: Base64 encoded file (ou None).
            filename: Nom fichier upload (ou None).
            source: Source données (ex: 'binance', 'local').
            symbol: Paire trading (ex: 'BTCUSDT').
            timeframe: Timeframe (ex: '1h').

        Returns:
            Tuple: (task_id, btn_disabled, loading_text, alert)

        Raises:
            PreventUpdate: Si n_clicks None ou inputs invalides.
        """

        if not n_clicks or bridge is None:
            raise PreventUpdate

        # Validate inputs
        if not symbol or not timeframe:
            return (
                no_update,
                False,
                "",
                dbc.Alert(
                    "Symbol and timeframe are required",
                    color="warning",
                    is_open=True,
                ),
            )

        try:
            # Create request (simplified - adapt to real DataRequest)
            # Note: Real implementation would parse contents/filename
            req = {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": source or "binance",
                "contents": contents,
                "filename": filename,
            }

            # Submit to Bridge async
            # Note: Adapter selon API réelle de Bridge data validation
            # task_id = bridge.validate_data_async(req)
            task_id = f"data-{n_clicks}"  # Placeholder

            logger.info(f"Submitted data validation task: {task_id}")

            return (
                task_id,
                True,  # Disable button
                "Validating data...",
                no_update,
            )

        except (BridgeError, ValueError) as e:
            logger.error(f"Data validation submit error: {e}")
            return (
                no_update,
                False,
                "",
                dbc.Alert(str(e), color="danger", is_open=True),
            )

    # ═══════════════════════════════════════════════════════════════
    # INDICATORS - Submit Build
    # ═══════════════════════════════════════════════════════════════

    @callback(
        [
            Output("indicators-task-store", "data", allow_duplicate=True),
            Output("build-indicators-btn", "disabled", allow_duplicate=True),
            Output("indicators-loading", "children", allow_duplicate=True),
            Output("indicators-alert", "children", allow_duplicate=True),
        ],
        Input("build-indicators-btn", "n_clicks"),
        [
            State("indicators-symbol", "value"),
            State("indicators-timeframe", "value"),
            State("ema-period", "value"),
            State("rsi-period", "value"),
            State("bollinger-std", "value"),
        ],
        prevent_initial_call=True,
    )
    def submit_indicators_build(
        n_clicks, symbol, timeframe, ema_period, rsi_period, bb_std
    ):
        """Submit indicators build task to Bridge async.

        Déclenché par clic sur build-indicators-btn.
        Collecte params indicateurs, crée IndicatorRequest,
        soumet à Bridge pour calcul cache.

        Args:
            n_clicks: Compteur clics (trigger).
            symbol: Paire trading.
            timeframe: Timeframe.
            ema_period: Période EMA (ou None).
            rsi_period: Période RSI (ou None).
            bb_std: Std dev Bollinger (ou None).

        Returns:
            Tuple: (task_id, btn_disabled, loading_text, alert)
        """

        if not n_clicks or bridge is None:
            raise PreventUpdate

        if not symbol or not timeframe:
            return (
                no_update,
                False,
                "",
                dbc.Alert(
                    "Symbol and timeframe are required",
                    color="warning",
                    is_open=True,
                ),
            )

        try:
            # Create indicator request
            req = IndicatorRequest(
                symbol=symbol,
                timeframe=timeframe,
                indicators={
                    "ema": {"period": ema_period or 20},
                    "rsi": {"period": rsi_period or 14},
                    "bollinger": {"std": bb_std or 2.0},
                },
            )

            # Submit to Bridge async
            # task_id = bridge.build_indicators_async(req)
            task_id = f"ind-{n_clicks}"  # Placeholder

            logger.info(f"Submitted indicators build task: {task_id}")

            return (
                task_id,
                True,
                "Building indicators...",
                no_update,
            )

        except (BridgeError, ValueError) as e:
            logger.error(f"Indicators build submit error: {e}")
            return (
                no_update,
                False,
                "",
                dbc.Alert(str(e), color="danger", is_open=True),
            )

    # ═══════════════════════════════════════════════════════════════
    # BACKTEST - Submit Run
    # ═══════════════════════════════════════════════════════════════

    @callback(
        [
            Output("bt-task-store", "data", allow_duplicate=True),
            Output("bt-run-btn", "disabled", allow_duplicate=True),
            Output("bt-loading", "children", allow_duplicate=True),
            Output("bt-status", "children", allow_duplicate=True),
        ],
        Input("bt-run-btn", "n_clicks"),
        [
            State("bt-strategy", "value"),
            State("bt-symbol", "value"),
            State("bt-timeframe", "value"),
            State("bt-period", "value"),
            State("bt-std", "value"),
        ],
        prevent_initial_call=True,
    )
    def submit_backtest_run(n_clicks, strategy, symbol, timeframe, period, std):
        """Submit backtest run task to Bridge async.

        Déclenché par clic sur bt-run-btn.
        Collecte params stratégie, crée BacktestRequest,
        soumet à Bridge pour exécution backtest.

        Args:
            n_clicks: Compteur clics (trigger).
            strategy: Nom stratégie (ex: 'bollinger_reversion').
            symbol: Paire trading.
            timeframe: Timeframe.
            period: Période paramètre stratégie.
            std: Std dev paramètre stratégie.

        Returns:
            Tuple: (task_id, btn_disabled, loading_text, status)
        """

        if not n_clicks or bridge is None:
            raise PreventUpdate

        if not all([strategy, symbol, timeframe]):
            return (
                no_update,
                False,
                "",
                dbc.Alert(
                    "Strategy, symbol, and timeframe are required",
                    color="warning",
                    is_open=True,
                ),
            )

        try:
            # Create backtest request
            req = BacktestRequest(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                params={
                    "period": period or 20,
                    "std": std or 2.0,
                },
            )

            # Submit to Bridge async
            future = bridge.run_backtest_async(req)
            task_id = f"bt-{n_clicks}"  # Placeholder (use future.task_id)

            logger.info(f"Submitted backtest task: {task_id}")

            return (
                task_id,
                True,
                "Running backtest...",
                no_update,
            )

        except (BridgeError, ValueError) as e:
            logger.error(f"Backtest submit error: {e}")
            return (
                no_update,
                False,
                "",
                dbc.Alert(str(e), color="danger", is_open=True),
            )

    # ═══════════════════════════════════════════════════════════════
    # OPTIMIZATION - Submit Sweep
    # ═══════════════════════════════════════════════════════════════

    @callback(
        [
            Output("opt-task-store", "data", allow_duplicate=True),
            Output("opt-run-btn", "disabled", allow_duplicate=True),
            Output("opt-loading", "children", allow_duplicate=True),
            Output("opt-status", "children", allow_duplicate=True),
        ],
        Input("opt-run-btn", "n_clicks"),
        [
            State("opt-strategy", "value"),
            State("opt-symbol", "value"),
            State("opt-timeframe", "value"),
            State("opt-period-min", "value"),
            State("opt-period-max", "value"),
            State("opt-period-step", "value"),
            State("opt-std-min", "value"),
            State("opt-std-max", "value"),
            State("opt-std-step", "value"),
        ],
        prevent_initial_call=True,
    )
    def submit_optimization_sweep(
        n_clicks,
        strategy,
        symbol,
        timeframe,
        period_min,
        period_max,
        period_step,
        std_min,
        std_max,
        std_step,
    ):
        """Submit optimization sweep task to Bridge async.

        Déclenché par clic sur opt-run-btn.
        Collecte grille paramètres, crée SweepRequest,
        soumet à Bridge pour parameter sweep.

        Args:
            n_clicks: Compteur clics (trigger).
            strategy: Nom stratégie.
            symbol: Paire trading.
            timeframe: Timeframe.
            period_min/max/step: Range période.
            std_min/max/step: Range std dev.

        Returns:
            Tuple: (task_id, btn_disabled, loading_text, status)
        """

        if not n_clicks or bridge is None:
            raise PreventUpdate

        if not all([strategy, symbol, timeframe]):
            return (
                no_update,
                False,
                "",
                dbc.Alert(
                    "Strategy, symbol, and timeframe are required",
                    color="warning",
                    is_open=True,
                ),
            )

        try:
            # Create parameter grid
            param_grid = {
                "period": {
                    "min": period_min or 10,
                    "max": period_max or 50,
                    "step": period_step or 5,
                },
                "std": {
                    "min": std_min or 1.0,
                    "max": std_max or 3.0,
                    "step": std_step or 0.5,
                },
            }

            # Create sweep request
            req = SweepRequest(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                param_grid=param_grid,
            )

            # Submit to Bridge async
            # future = bridge.run_sweep_async(req)
            task_id = f"opt-{n_clicks}"  # Placeholder

            logger.info(f"Submitted optimization sweep task: {task_id}")

            return (
                task_id,
                True,
                "Running parameter sweep...",
                no_update,
            )

        except (BridgeError, ValueError) as e:
            logger.error(f"Optimization sweep submit error: {e}")
            return (
                no_update,
                False,
                "",
                dbc.Alert(str(e), color="danger", is_open=True),
            )

    logger.info("All callbacks registered successfully")


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - Create UI Components from Results
# ═══════════════════════════════════════════════════════════════════


def _create_data_table(data: list[dict[str, Any]]) -> html.Div:
    """Create data registry table from validation results.

    Args:
        data: List of data records (dicts).

    Returns:
        html.Div: Formatted table or empty message.
    """
    if not data:
        return html.P("No data available", className="text-muted fst-italic")

    df = pd.DataFrame(data)
    return html.Pre(df.to_string(), className="text-light")


def _create_indicators_table(indicators: dict[str, Any]) -> html.Div:
    """Create indicators cache table from build results.

    Args:
        indicators: Dict of indicator results {name: values}.

    Returns:
        html.Div: Formatted table or empty message.
    """
    if not indicators:
        return html.P("No indicators built", className="text-muted fst-italic")

    summary = {name: f"{len(values)} values" for name, values in indicators.items()}
    return html.Pre(str(summary), className="text-light")


def _create_equity_graph(equity_curve: list[float]) -> go.Figure:
    """Create equity curve Plotly graph.

    Args:
        equity_curve: List of equity values over time.

    Returns:
        go.Figure: Plotly line chart.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="cyan", width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Equity Curve",
        xaxis_title="Step",
        yaxis_title="Equity ($)",
    )
    return fig


def _create_drawdown_graph(drawdown_curve: list[float]) -> go.Figure:
    """Create drawdown curve Plotly graph.

    Args:
        drawdown_curve: List of drawdown % over time.

    Returns:
        go.Figure: Plotly area chart.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=drawdown_curve,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red", width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Drawdown Curve",
        xaxis_title="Step",
        yaxis_title="Drawdown (%)",
    )
    return fig


def _create_trades_table(trades: list[dict[str, Any]]) -> html.Div:
    """Create trades table from backtest results.

    Args:
        trades: List of trade dicts.

    Returns:
        html.Div: Formatted table.
    """
    if not trades:
        return html.P("No trades executed", className="text-muted fst-italic")

    df = pd.DataFrame(trades)
    return html.Pre(df.head(20).to_string(), className="text-light")


def _create_metrics_table(metrics: dict[str, float]) -> html.Div:
    """Create metrics table from backtest results.

    Args:
        metrics: Dict of KPI metrics.

    Returns:
        html.Div: Formatted table.
    """
    if not metrics:
        return html.P("No metrics available", className="text-muted fst-italic")

    return html.Pre(str(metrics), className="text-light")


def _create_sweep_results_table(results: list[dict[str, Any]]) -> html.Div:
    """Create sweep results table (top combinations).

    Args:
        results: List of sweep result dicts.

    Returns:
        html.Div: Formatted table.
    """
    if not results:
        return html.P("No sweep results", className="text-muted fst-italic")

    df = pd.DataFrame(results)
    return html.Pre(df.head(10).to_string(), className="text-light")


def _create_heatmap(results: list[dict[str, Any]]) -> go.Figure:
    """Create parameter heatmap from sweep results.

    Args:
        results: List of sweep result dicts.

    Returns:
        go.Figure: Plotly heatmap.
    """
    fig = go.Figure()

    if not results:
        fig.update_layout(
            template="plotly_dark",
            title="Parameter Heatmap (No Data)",
        )
        return fig

    # Placeholder heatmap (adapt to real data structure)
    df = pd.DataFrame(results)
    fig.add_trace(go.Heatmap(z=[[0]], colorscale="Viridis"))
    fig.update_layout(
        template="plotly_dark",
        title="Parameter Heatmap",
        xaxis_title="Parameter 1",
        yaxis_title="Parameter 2",
    )
    return fig
