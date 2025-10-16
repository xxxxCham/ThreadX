"""
ThreadX Dashboard - Composants de graphiques
Gère tous les graphiques de visualisation pour le backtesting

RÈGLE ARCHITECTURE: Aucun calcul métier ici.
Tous les calculs pandas/numpy doivent passer par Bridge.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Import Bridge pour déléguer calculs métier
from threadx.bridge import MetricsController


class PriceAndSignalsChart:
    """Graphique principal affichant prix, signaux et niveaux support/résistance"""

    def __init__(self, theme_colors: Dict[str, str]):
        self.theme = theme_colors

    def create_chart(
        self,
        df_price: pd.DataFrame,
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        asset_name: str = "BTC-USD",
    ) -> go.Figure:
        """
        Crée le graphique principal prix et signaux

        Args:
            df_price: DataFrame avec colonnes [date, close, open, high, low]
            buy_signals: Liste de dict [{'date': str, 'price': float, 'quantity': float}, ...]
            sell_signals: Liste de dict [{'date': str, 'price': float, 'quantity': float}, ...]
            support_level: float ou None
            resistance_level: float ou None
            asset_name: str nom de l'asset

        Returns:
            go.Figure objet Plotly complètement configuré
        """
        fig = go.Figure()

        # Ligne de prix principale
        fig.add_trace(
            go.Scatter(
                x=df_price["date"],
                y=df_price["close"],
                mode="lines",
                name="Price",
                line=dict(color="#ffffff", width=2),
                hovertemplate="<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>",
            )
        )

        # Support level
        if support_level is not None:
            fig.add_hline(
                y=support_level,
                line=dict(color="#ffaa00", width=1.5, dash="dash"),
                annotation_text=f"Support: ${support_level:,.2f}",
                annotation_position="top right",
            )

        # Resistance level
        if resistance_level is not None:
            fig.add_hline(
                y=resistance_level,
                line=dict(color="#00ff00", width=1.5, dash="dash"),
                annotation_text=f"Resistance: ${resistance_level:,.2f}",
                annotation_position="bottom right",
            )

        # Buy signals (triangles verts pointant vers le haut)
        if buy_signals:
            buy_dates = [signal["date"] for signal in buy_signals]
            buy_prices = [signal["price"] for signal in buy_signals]
            buy_quantities = [signal.get("quantity", 1) for signal in buy_signals]

            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    name="Buy Signals",
                    marker=dict(
                        symbol="triangle-up",
                        size=15,
                        color="#00ff00",
                        line=dict(color="#ffffff", width=2),
                    ),
                    hovertemplate="<b>BUY SIGNAL</b><br>Date: %{x}<br>Price: $%{y:,.2f}<br>Quantity: %{customdata}<extra></extra>",
                    customdata=buy_quantities,
                )
            )

        # Sell signals (triangles rouges pointant vers le bas)
        if sell_signals:
            sell_dates = [signal["date"] for signal in sell_signals]
            sell_prices = [signal["price"] for signal in sell_signals]
            sell_quantities = [signal.get("quantity", 1) for signal in sell_signals]

            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    name="Sell Signals",
                    marker=dict(
                        symbol="triangle-down",
                        size=15,
                        color="#ff4444",
                        line=dict(color="#ffffff", width=2),
                    ),
                    hovertemplate="<b>SELL SIGNAL</b><br>Date: %{x}<br>Price: $%{y:,.2f}<br>Quantity: %{customdata}<extra></extra>",
                    customdata=sell_quantities,
                )
            )

        # Configuration du layout
        fig.update_layout(
            title=dict(
                text=f"{asset_name} - Price & Trading Signals",
                font=dict(color=self.theme.get("text_primary", "#ffffff"), size=16),
            ),
            xaxis=dict(
                title="Date",
                gridcolor=self.theme.get("border", "#404040"),
                zeroline=False,
                color=self.theme.get("text_primary", "#ffffff"),
                rangeslider=dict(
                    visible=True, bgcolor=self.theme.get("secondary_bg", "#242424")
                ),
            ),
            yaxis=dict(
                title="Price (USD)",
                gridcolor=self.theme.get("border", "#404040"),
                zeroline=False,
                color=self.theme.get("text_primary", "#ffffff"),
            ),
            plot_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            paper_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            font=dict(
                color=self.theme.get("text_primary", "#ffffff"),
                family="Roboto",
                size=12,
            ),
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=80),
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(36, 36, 36, 0.8)",
                bordercolor=self.theme.get("border", "#404040"),
                borderwidth=1,
            ),
            height=450,
        )

        return fig


class TradingVolumeChart:
    """Graphique de volume de trading avec moyennes mobiles"""

    def __init__(self, theme_colors: Dict[str, str]):
        self.theme = theme_colors

    def create_chart(
        self,
        df_volume: pd.DataFrame,
        buy_volume: Optional[pd.Series] = None,
        sell_volume: Optional[pd.Series] = None,
        ma_period: int = 20,
    ) -> go.Figure:
        """
        Crée le graphique de volume de trading

        Args:
            df_volume: DataFrame avec colonnes [date, volume] ou Series indexée par date
            buy_volume: Series volume d'achat par date (optionnel)
            sell_volume: Series volume de vente par date (optionnel)
            ma_period: Période moyenne mobile (default 20)

        Returns:
            go.Figure
        """
        fig = go.Figure()

        # Convertir en DataFrame si nécessaire
        if isinstance(df_volume, pd.Series):
            dates = df_volume.index
            volumes = df_volume.values
        else:
            dates = df_volume["date"]
            volumes = df_volume["volume"]

        # Volume total (barres principales)
        fig.add_trace(
            go.Bar(
                x=dates,
                y=volumes,
                name="Total Volume",
                marker_color="rgba(0, 212, 255, 0.8)",
                hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>",
            )
        )

        # Volumes buy/sell séparés si disponibles
        if buy_volume is not None and sell_volume is not None:
            fig.add_trace(
                go.Bar(
                    x=buy_volume.index,
                    y=buy_volume.values,
                    name="Buy Volume",
                    marker_color="rgba(0, 255, 0, 0.6)",
                    hovertemplate="<b>%{x}</b><br>Buy Volume: %{y:,.0f}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Bar(
                    x=sell_volume.index,
                    y=sell_volume.values,
                    name="Sell Volume",
                    marker_color="rgba(255, 68, 68, 0.6)",
                    hovertemplate="<b>%{x}</b><br>Sell Volume: %{y:,.0f}<extra></extra>",
                )
            )

        # Moyenne mobile - DÉLÈGUE À BRIDGE
        if len(volumes) >= ma_period:
            metrics_controller = MetricsController()
            ma_values_list = metrics_controller.calculate_moving_average(
                volumes, period=ma_period, ma_type="sma"
            )
            ma_values = pd.Series(ma_values_list, index=dates)

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=ma_values,
                    mode="lines",
                    name=f"MA{ma_period}",
                    line=dict(color="#ffaa00", width=2),
                    hovertemplate=f"<b>%{{x}}</b><br>MA{ma_period}: %{{y:,.0f}}<extra></extra>",
                )
            )

        # Configuration du layout
        fig.update_layout(
            title=dict(
                text="Trading Volume",
                font=dict(color=self.theme.get("text_primary", "#ffffff"), size=16),
            ),
            xaxis=dict(
                title="Date",
                gridcolor=self.theme.get("border", "#404040"),
                zeroline=False,
                color=self.theme.get("text_primary", "#ffffff"),
            ),
            yaxis=dict(
                title="Volume",
                gridcolor=self.theme.get("border", "#404040"),
                zeroline=False,
                color=self.theme.get("text_primary", "#ffffff"),
            ),
            plot_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            paper_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            font=dict(
                color=self.theme.get("text_primary", "#ffffff"),
                family="Roboto",
                size=12,
            ),
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=40),
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(36, 36, 36, 0.8)",
                bordercolor=self.theme.get("border", "#404040"),
                borderwidth=1,
            ),
            height=250,
            barmode="overlay",
        )

        return fig


class PortfolioBalanceChart:
    """Graphique d'équité du portefeuille avec drawdown et benchmarks"""

    def __init__(self, theme_colors: Dict[str, str]):
        self.theme = theme_colors

    def create_chart(
        self,
        equity_curve: pd.Series,
        buy_hold_curve: Optional[pd.Series] = None,
        initial_cash: float = 10000,
        show_drawdown: bool = True,
    ) -> go.Figure:
        """
        Crée le graphique d'équité du portefeuille

        Args:
            equity_curve: Series équité indexée par date
            buy_hold_curve: Series (optional) équité Buy & Hold
            initial_cash: float montant initial
            show_drawdown: bool afficher zones drawdown

        Returns:
            go.Figure avec metrics affichées
        """
        fig = go.Figure()

        # Calculer les statistiques
        final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_cash
        total_return = (final_equity - initial_cash) / initial_cash

        # Calcul du drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0

        # Courbe d'équité de la stratégie
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Strategy Equity",
                line=dict(color="#00d4ff", width=3),
                fill="tonexty" if show_drawdown else None,
                fillcolor="rgba(0, 212, 255, 0.1)",
                hovertemplate="<b>%{x}</b><br>Equity: $%{y:,.2f}<br>Return: %{customdata:.1%}<extra></extra>",
                customdata=((equity_curve - initial_cash) / initial_cash).values,
            )
        )

        # Buy & Hold benchmark
        if buy_hold_curve is not None:
            fig.add_trace(
                go.Scatter(
                    x=buy_hold_curve.index,
                    y=buy_hold_curve.values,
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#00a8cc", width=2, dash="dash"),
                    hovertemplate="<b>%{x}</b><br>B&H Equity: $%{y:,.2f}<extra></extra>",
                )
            )

        # Zones de drawdown
        if show_drawdown:
            drawdown_mask = drawdown < -0.01  # Seulement les drawdowns > 1%
            if drawdown_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=equity_curve.index,
                        y=peak.values,
                        mode="lines",
                        name="Peak Equity",
                        line=dict(color="rgba(255, 255, 255, 0.3)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(255, 68, 68, 0.1)",
                        hovertemplate="<b>%{x}</b><br>Peak: $%{y:,.2f}<br>Drawdown: %{customdata:.1%}<extra></extra>",
                        customdata=(drawdown * 100).values,
                    )
                )

        # Configuration du layout
        fig.update_layout(
            title=dict(
                text="Portfolio Balance Evolution",
                font=dict(color=self.theme.get("text_primary", "#ffffff"), size=16),
            ),
            xaxis=dict(
                title="Date",
                gridcolor=self.theme.get("border", "#404040"),
                zeroline=False,
                color=self.theme.get("text_primary", "#ffffff"),
            ),
            yaxis=dict(
                title="Equity ($)",
                gridcolor=self.theme.get("border", "#404040"),
                zeroline=False,
                color=self.theme.get("text_primary", "#ffffff"),
            ),
            plot_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            paper_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            font=dict(
                color=self.theme.get("text_primary", "#ffffff"),
                family="Roboto",
                size=12,
            ),
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=40),
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(36, 36, 36, 0.8)",
                bordercolor=self.theme.get("border", "#404040"),
                borderwidth=1,
            ),
            height=350,
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Max DD: {max_drawdown:.1%} | Current DD: {current_drawdown:.1%} | ROI: {total_return:.1%}",
                    showarrow=False,
                    font=dict(
                        size=10, color=self.theme.get("text_secondary", "#b0b0b0")
                    ),
                    bgcolor="rgba(36, 36, 36, 0.8)",
                    bordercolor=self.theme.get("border", "#404040"),
                    borderwidth=1,
                )
            ],
        )

        return fig


class ChartsManager:
    """Orchestrateur pour tous les graphiques de backtesting"""

    def __init__(self, theme_colors: Dict[str, str]):
        """
        Args:
            theme_colors: dictionnaire avec clés:
                - primary_bg, secondary_bg, text_primary, accent, success, danger, warning, border
        """
        self.theme = theme_colors
        self.price_chart = PriceAndSignalsChart(theme_colors)
        self.volume_chart = TradingVolumeChart(theme_colors)
        self.portfolio_chart = PortfolioBalanceChart(theme_colors)

    def create_layout_config(self) -> Dict[str, Any]:
        """Retourne configuration Plotly standard pour tous les graphiques"""
        return {
            "plot_bgcolor": self.theme.get("primary_bg", "#1a1a1a"),
            "paper_bgcolor": self.theme.get("primary_bg", "#1a1a1a"),
            "font": {
                "color": self.theme.get("text_primary", "#ffffff"),
                "family": "Roboto",
                "size": 12,
            },
            "xaxis": {
                "gridcolor": self.theme.get("border", "#404040"),
                "zeroline": False,
                "color": self.theme.get("text_primary", "#ffffff"),
            },
            "yaxis": {
                "gridcolor": self.theme.get("border", "#404040"),
                "zeroline": False,
                "color": self.theme.get("text_primary", "#ffffff"),
            },
            "hovermode": "x unified",
            "margin": {"l": 60, "r": 30, "t": 40, "b": 40},
        }

    def get_all_figures(self, backtest_data: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Génère tous les graphiques à partir des données de backtest

        Args:
            backtest_data: dict contenant:
                - 'price_history': dict avec 'dates', 'close', 'open', 'high', 'low'
                - 'buy_signals': list de dict
                - 'sell_signals': list de dict
                - 'volume': dict avec 'dates', 'total', 'buy', 'sell'
                - 'portfolio': dict avec 'dates', 'equity', 'cash', 'positions'
                - 'buy_hold': dict avec 'dates', 'equity'
                - 'asset_name': str
                - 'initial_cash': float

        Returns:
            dict avec figures: {
                'price': go.Figure,
                'volume': go.Figure,
                'portfolio': go.Figure,
            }
        """
        figures = {}

        try:
            # Préparation des données prix
            price_data = backtest_data.get("price_history", {})
            df_price = pd.DataFrame(
                {
                    "date": price_data.get("dates", []),
                    "close": price_data.get("close", []),
                    "open": price_data.get("open", []),
                    "high": price_data.get("high", []),
                    "low": price_data.get("low", []),
                }
            )

            # Calcul des niveaux support/résistance
            if len(df_price) > 0:
                support_level, resistance_level = self._calculate_support_resistance(
                    df_price["close"]
                )
            else:
                support_level = resistance_level = None

            # Graphique prix et signaux
            figures["price"] = self.price_chart.create_chart(
                df_price=df_price,
                buy_signals=backtest_data.get("buy_signals", []),
                sell_signals=backtest_data.get("sell_signals", []),
                support_level=support_level,
                resistance_level=resistance_level,
                asset_name=backtest_data.get("asset_name", "BTC-USD"),
            )

            # Préparation des données volume
            volume_data = backtest_data.get("volume", {})
            df_volume = pd.DataFrame(
                {
                    "date": volume_data.get("dates", []),
                    "volume": volume_data.get("total", []),
                }
            )

            buy_vol = pd.Series(
                volume_data.get("buy", []), index=volume_data.get("dates", [])
            )
            sell_vol = pd.Series(
                volume_data.get("sell", []), index=volume_data.get("dates", [])
            )

            # Graphique volume
            figures["volume"] = self.volume_chart.create_chart(
                df_volume=df_volume,
                buy_volume=buy_vol if len(buy_vol) > 0 else None,
                sell_volume=sell_vol if len(sell_vol) > 0 else None,
            )

            # Préparation des données portfolio
            portfolio_data = backtest_data.get("portfolio", {})
            equity_curve = pd.Series(
                portfolio_data.get("equity", []), index=portfolio_data.get("dates", [])
            )

            buy_hold_data = backtest_data.get("buy_hold", {})
            buy_hold_curve = None
            if buy_hold_data.get("equity"):
                buy_hold_curve = pd.Series(
                    buy_hold_data["equity"], index=buy_hold_data.get("dates", [])
                )

            # Graphique portfolio
            figures["portfolio"] = self.portfolio_chart.create_chart(
                equity_curve=equity_curve,
                buy_hold_curve=buy_hold_curve,
                initial_cash=backtest_data.get("initial_cash", 10000),
            )

        except Exception as e:
            # En cas d'erreur, retourner des graphiques vides
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Error loading data",
                plot_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
                paper_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            )
            figures = {"price": empty_fig, "volume": empty_fig, "portfolio": empty_fig}

        return figures

    def _calculate_support_resistance(
        self, prices: pd.Series, window: int = 20
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcule les niveaux support/résistance simples

        Args:
            prices: Series des prix
            window: Fenêtre pour le calcul

        Returns:
            Tuple (support, resistance)
        """
        if len(prices) < window:
            return None, None

        # Support = minimum des minimums locaux
        local_mins = []
        local_maxs = []

        for i in range(window, len(prices) - window):
            if prices.iloc[i] == prices.iloc[i - window : i + window + 1].min():
                local_mins.append(prices.iloc[i])
            if prices.iloc[i] == prices.iloc[i - window : i + window + 1].max():
                local_maxs.append(prices.iloc[i])

        support = np.median(local_mins) if local_mins else None
        resistance = np.median(local_maxs) if local_maxs else None

        return support, resistance

    def _create_empty_figure(self, message: str = "No data available") -> go.Figure:
        """Crée un graphique vide avec un message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.theme.get("text_secondary", "#b0b0b0")),
        )
        fig.update_layout(
            plot_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            paper_bgcolor=self.theme.get("primary_bg", "#1a1a1a"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig
