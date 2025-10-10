"""
ThreadX Charts Module - Matplotlib & Altair Wrappers
===================================================

Provides charting functionality for ThreadX UI with:
- Matplotlib backend for equity/drawdown plots
- Altair integration for interactive charts (optional)
- Export functionality with relative paths
- Dark theme optimized for TechinTerror interface

Features:
- Non-blocking chart generation
- Memory-efficient figure management
- Windows-optimized DPI handling
- Export to PNG/SVG with relative paths

Author: ThreadX Framework
Version: Phase 8 - Charts Module
"""

import logging
from pathlib import Path
from typing import Optional, Union, Any
import warnings

import pandas as pd
import numpy as np

# Matplotlib setup for non-GUI backend
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

# Optional Altair for interactive charts
try:
    import altair as alt

    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False
    alt = None

# ThreadX imports
try:
    from ..utils.log import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


def plot_equity(
    equity: pd.Series, *, save_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Plot equity curve with professional styling.

    Creates an equity curve chart showing portfolio value over time with
    proper scaling, grid, and annotations.

    Parameters
    ----------
    equity : pd.Series
        Equity curve data with datetime index and float values representing
        portfolio value over time.
    save_path : Path, optional
        If provided, saves chart to this path. If None, displays only.

    Returns
    -------
    Path or None
        Path to saved chart file if save_path provided, None otherwise.

    Raises
    ------
    ValueError
        If equity series is empty or has invalid data
    IOError
        If save_path cannot be written to

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pathlib import Path
    >>>
    >>> # Create sample equity curve
    >>> dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    >>> returns = pd.Series(np.random.randn(1000) * 0.01, index=dates)
    >>> equity = (1 + returns).cumprod() * 10000
    >>>
    >>> # Plot and save
    >>> chart_path = plot_equity(equity, save_path=Path("equity.png"))
    >>> print(f"Chart saved to: {chart_path}")
    """
    logger = get_logger(__name__)

    # Validation
    if equity.empty:
        raise ValueError("Equity series cannot be empty")

    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("Equity series must have datetime index")

    if equity.isna().any():
        logger.warning("Equity series contains NaN values, filling forward")
        equity = equity.fillna(method="ffill")

    try:
        logger.info(f"Creating equity curve chart for {len(equity)} data points")

        # Create figure with professional styling
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle("Portfolio Equity Curve", fontsize=16, fontweight="bold")

        # Plot equity curve
        ax.plot(equity.index, equity.values, linewidth=2, color="#2E86AB", alpha=0.9)

        # Styling
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Format x-axis dates
        if len(equity) > 100:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1, len(equity) // 20))
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add performance annotations
        start_value = equity.iloc[0]
        end_value = equity.iloc[-1]
        total_return = (end_value / start_value - 1) * 100
        max_value = equity.max()
        min_value = equity.min()

        # Add text box with summary stats
        stats_text = (
            f"Start: ${start_value:,.0f}\n"
            f"End: ${end_value:,.0f}\n"
            f"Return: {total_return:+.1f}%\n"
            f"Max: ${max_value:,.0f}\n"
            f"Min: ${min_value:,.0f}"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Tight layout
        plt.tight_layout()

        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Equity chart saved to: {save_path}")

            plt.close(fig)  # Free memory
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        logger.error(f"Failed to create equity chart: {e}")
        if "fig" in locals():
            plt.close(fig)
        raise


def plot_drawdown(
    equity: pd.Series, *, save_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Plot drawdown chart showing underwater periods.

    Creates a drawdown chart showing the percentage decline from peak
    equity values, highlighting maximum drawdown periods.

    Parameters
    ----------
    equity : pd.Series
        Equity curve data with datetime index and float values.
    save_path : Path, optional
        If provided, saves chart to this path.

    Returns
    -------
    Path or None
        Path to saved chart file if save_path provided, None otherwise.

    Raises
    ------
    ValueError
        If equity series is empty or invalid
    IOError
        If save_path cannot be written to

    Examples
    --------
    >>> # Using equity from backtest results
    >>> drawdown_path = plot_drawdown(equity, save_path=Path("drawdown.png"))
    >>> print(f"Drawdown chart saved to: {drawdown_path}")
    """
    logger = get_logger(__name__)

    # Validation
    if equity.empty:
        raise ValueError("Equity series cannot be empty")

    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("Equity series must have datetime index")

    try:
        logger.info(f"Creating drawdown chart for {len(equity)} data points")

        # Calculate drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity / rolling_max - 1) * 100  # Convert to percentage

        # Create figure
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.suptitle("Portfolio Equity and Drawdown", fontsize=16, fontweight="bold")

        # Top plot: Equity curve
        ax1.plot(
            equity.index,
            equity.values,
            linewidth=2,
            color="#2E86AB",
            label="Equity",
            alpha=0.9,
        )
        ax1.plot(
            rolling_max.index,
            rolling_max.values,
            linewidth=1,
            color="#A23B72",
            linestyle="--",
            label="Peak",
            alpha=0.7,
        )

        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax1.legend()

        # Bottom plot: Drawdown
        ax2.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            alpha=0.7,
            color="#F18F01",
            label="Drawdown",
        )
        ax2.plot(drawdown.index, drawdown.values, linewidth=1, color="#C73E1D")

        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.set_ylim(min(drawdown.min() * 1.1, -0.1), 1)

        # Format x-axis dates for both plots
        for ax in [ax1, ax2]:
            if len(equity) > 100:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            else:
                ax.xaxis.set_major_locator(
                    mdates.DayLocator(interval=max(1, len(equity) // 20))
                )
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add drawdown statistics
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()

        # Find recovery periods
        underwater = drawdown < -0.1  # More than 0.1% drawdown
        if underwater.any():
            # Find longest underwater period
            underwater_periods = []
            start = None
            for i, is_underwater in enumerate(underwater):
                if is_underwater and start is None:
                    start = i
                elif not is_underwater and start is not None:
                    underwater_periods.append((start, i - 1))
                    start = None

            if start is not None:  # Still underwater at end
                underwater_periods.append((start, len(underwater) - 1))

            if underwater_periods:
                longest_period = max(underwater_periods, key=lambda x: x[1] - x[0])
                longest_days = longest_period[1] - longest_period[0]
            else:
                longest_days = 0
        else:
            longest_days = 0

        # Add statistics text
        stats_text = (
            f"Max Drawdown: {max_drawdown:.2f}%\n"
            f'Max DD Date: {max_dd_date.strftime("%Y-%m-%d") if pd.notna(max_dd_date) else "N/A"}\n'
            f"Longest Recovery: {longest_days} periods"
        )

        ax2.text(
            0.02,
            0.02,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Tight layout
        plt.tight_layout()

        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Drawdown chart saved to: {save_path}")

            plt.close(fig)
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        logger.error(f"Failed to create drawdown chart: {e}")
        if "fig" in locals():
            plt.close(fig)
        raise


def altair_equity(returns_or_equity: Union[pd.Series, pd.DataFrame]):
    """
    Create interactive equity chart using Altair.

    Creates an interactive equity curve chart with zoom, pan, and tooltip
    functionality using Altair/Vega-Lite.

    Parameters
    ----------
    returns_or_equity : pd.Series or pd.DataFrame
        Either a returns series (will be converted to equity) or
        an equity series with datetime index.

    Returns
    -------
    alt.Chart
        Altair chart object with interactive features

    Raises
    ------
    ImportError
        If Altair is not available
    ValueError
        If input data is invalid

    Examples
    --------
    >>> if HAS_ALTAIR:
    ...     chart = altair_equity(returns)
    ...     chart.save('equity_interactive.html')
    """
    if not HAS_ALTAIR:
        raise ImportError("Altair not available. Install with: pip install altair")

    logger = get_logger(__name__)

    try:
        # Convert returns to equity if needed
        if returns_or_equity.name == "returns" or returns_or_equity.min() < 0:
            # Assume this is returns data
            equity = (1 + returns_or_equity).cumprod() * 10000
            logger.info("Converted returns to equity curve")
        else:
            equity = returns_or_equity

        # Prepare data for Altair
        df_chart = pd.DataFrame(
            {"date": equity.index, "equity": equity.values}
        ).reset_index(drop=True)

        # Create base chart
        base = alt.Chart(df_chart).add_selection(alt.selection_interval(bind="scales"))

        # Line chart
        line = (
            base.mark_line(
                color="#2E86AB",
                strokeWidth=2,
                point=alt.OverlayMarkDef(filled=False, fill="white", strokeWidth=2),
            )
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y(
                    "equity:Q", title="Portfolio Value ($)", scale=alt.Scale(nice=True)
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("equity:Q", title="Value", format="$,.0f"),
                ],
            )
            .resolve_scale(y="independent")
        )

        # Add rule for current value
        rule = (
            alt.Chart(df_chart)
            .mark_rule(color="#A23B72", strokeDash=[5, 5], opacity=0.7)
            .encode(y=alt.Y("mean(equity):Q"))
        )

        # Combine charts
        chart = (
            (line + rule)
            .resolve_scale(y="shared")
            .properties(
                width=800,
                height=400,
                title=alt.TitleParams(
                    text="Interactive Equity Curve", fontSize=16, fontWeight="bold"
                ),
            )
        )

        logger.info(f"Created interactive Altair chart with {len(df_chart)} points")
        return chart

    except Exception as e:
        logger.error(f"Failed to create Altair chart: {e}")
        raise


def create_summary_chart(
    equity: pd.Series, trades: pd.DataFrame, save_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Create comprehensive summary chart with multiple subplots.

    Creates a multi-panel chart showing equity curve, drawdown, trade
    distribution, and monthly returns heatmap.

    Parameters
    ----------
    equity : pd.Series
        Equity curve data
    trades : pd.DataFrame
        Trades DataFrame with columns: entry_time, exit_time, pnl
    save_path : Path, optional
        If provided, saves chart to this path

    Returns
    -------
    Path or None
        Path to saved chart file if save_path provided

    Examples
    --------
    >>> summary_path = create_summary_chart(equity, trades,
    ...                                    save_path=Path("summary.png"))
    """
    logger = get_logger(__name__)

    try:
        logger.info("Creating comprehensive summary chart")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

        # 1. Equity curve (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity.index, equity.values, linewidth=2, color="#2E86AB")
        ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # 2. Drawdown (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        rolling_max = equity.expanding().max()
        drawdown = (equity / rolling_max - 1) * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color="#F18F01")
        ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # 3. Trade PnL distribution (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        if not trades.empty and "pnl" in trades.columns:
            ax3.hist(
                trades["pnl"], bins=20, alpha=0.7, color="#A23B72", edgecolor="black"
            )
            ax3.axvline(
                trades["pnl"].mean(),
                color="red",
                linestyle="--",
                label=f'Mean: ${trades["pnl"].mean():.2f}',
            )
            ax3.legend()
        ax3.set_title("Trade PnL Distribution", fontsize=14, fontweight="bold")
        ax3.set_xlabel("PnL ($)")
        ax3.set_ylabel("Count")
        ax3.grid(True, alpha=0.3)

        # 4. Monthly returns (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        monthly_returns = equity.resample("M").last().pct_change().dropna() * 100
        colors = ["red" if x < 0 else "green" for x in monthly_returns]
        bars = ax4.bar(
            range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7
        )
        ax4.set_title("Monthly Returns", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Return (%)")
        ax4.axhline(0, color="black", linewidth=0.5)
        ax4.grid(True, alpha=0.3)

        # Format x-axis for monthly returns
        if len(monthly_returns) <= 24:  # Less than 2 years
            ax4.set_xticks(range(len(monthly_returns)))
            ax4.set_xticklabels(
                [d.strftime("%Y-%m") for d in monthly_returns.index],
                rotation=45,
                ha="right",
            )

        # 5. Performance metrics (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")  # Turn off axes for text display

        # Calculate key metrics
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        max_dd = drawdown.min()
        num_trades = len(trades) if not trades.empty else 0
        win_rate = (trades["pnl"] > 0).mean() * 100 if not trades.empty else 0

        metrics_text = f"""
Performance Summary

Total Return: {total_return:+.1f}%
Max Drawdown: {max_dd:.2f}%
Total Trades: {num_trades:,}
Win Rate: {win_rate:.1f}%

Start Value: ${equity.iloc[0]:,.0f}
End Value: ${equity.iloc[-1]:,.0f}
Period: {equity.index[0].strftime('%Y-%m-%d')} to {equity.index[-1].strftime('%Y-%m-%d')}
        """

        ax5.text(
            0.1,
            0.9,
            metrics_text,
            transform=ax5.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        # Overall title
        fig.suptitle("ThreadX Backtest Summary", fontsize=18, fontweight="bold", y=0.98)

        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Summary chart saved to: {save_path}")

            plt.close(fig)
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        logger.error(f"Failed to create summary chart: {e}")
        if "fig" in locals():
            plt.close(fig)
        raise


# Utility functions


def set_chart_style(style: str = "seaborn"):
    """
    Set global chart styling.

    Parameters
    ----------
    style : str
        Matplotlib style name ('seaborn', 'ggplot', 'classic', etc.)
    """
    try:
        if style == "seaborn":
            # Styles seaborn modernes disponibles
            try:
                plt.style.use("seaborn-v0_8")
            except OSError:
                try:
                    plt.style.use("seaborn-whitegrid")
                except OSError:
                    plt.style.use("default")
                    warnings.warn(
                        "Style seaborn non disponible, utilisation du style par défaut"
                    )
        else:
            plt.style.use(style)
        logger = get_logger(__name__)
        logger.info(f"Set chart style to: {style}")
    except Exception as e:
        warnings.warn(f"Could not set style '{style}': {e}")


def get_available_formats() -> list:
    """
    Get list of available export formats.

    Returns
    -------
    list
        List of supported file formats
    """
    return ["png", "pdf", "svg", "eps", "jpg"]


# Initialize default styling
set_chart_style("seaborn")
