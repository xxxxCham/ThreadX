"""
ThreadX Table Components - Phase 8
==================================

Table components for ThreadX UI with sorting, filtering, and export.

Provides:
- Trades table with sortable columns
- Metrics table with key performance indicators
- Export functionality (CSV, Parquet)
- Pagination for large datasets
- Search and filter capabilities

Features:
- Thread-safe operations
- Relative path handling
- Memory efficient for large datasets
- Cross-platform compatibility
- Professional styling

Author: ThreadX Framework
Version: Phase 8 - UI Components
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

import pandas as pd
import numpy as np

# ThreadX logging
try:
    from threadx.utils.log import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


def render_trades_table(trades: pd.DataFrame) -> Any:
    """
    Render trades data in a sortable table format.

    Creates a formatted table view of trades data with sortable columns,
    proper formatting for dates, currencies, and percentages.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades DataFrame with columns: entry_time, exit_time, pnl, side,
        entry_price, exit_price, and other trade-related columns.

    Returns
    -------
    Any
        Table widget or data structure (implementation dependent)

    Raises
    ------
    ValueError
        If trades DataFrame is empty or missing required columns

    Examples
    --------
    >>> import pandas as pd
    >>> trades = pd.DataFrame({
    ...     'entry_time': pd.date_range('2024-01-01', periods=10, freq='H'),
    ...     'exit_time': pd.date_range('2024-01-01 02:00', periods=10, freq='H'),
    ...     'pnl': np.random.randn(10) * 100,
    ...     'side': ['LONG'] * 5 + ['SHORT'] * 5,
    ...     'entry_price': 50000 + np.random.randn(10) * 1000,
    ...     'exit_price': 50000 + np.random.randn(10) * 1000
    ... })
    >>> table = render_trades_table(trades)
    """
    logger = get_logger(__name__)

    # Validation
    if trades.empty:
        raise ValueError("Trades DataFrame cannot be empty")

    required_columns = ["entry_time", "exit_time", "pnl", "side"]
    missing_columns = [col for col in required_columns if col not in trades.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    try:
        logger.info(f"Rendering trades table with {len(trades)} trades")

        # Create a copy for processing
        df_display = trades.copy()

        # Format datetime columns
        datetime_columns = ["entry_time", "exit_time"]
        for col in datetime_columns:
            if col in df_display.columns:
                df_display[col] = pd.to_datetime(df_display[col]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                )

        # Format currency columns
        currency_columns = ["pnl", "entry_price", "exit_price"]
        for col in currency_columns:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
                )

        # Format percentage columns
        percentage_columns = ["return_pct", "fees_pct"]
        for col in percentage_columns:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )

        # Add trade duration if both times are available
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            duration = pd.to_datetime(trades["exit_time"]) - pd.to_datetime(
                trades["entry_time"]
            )
            df_display["duration"] = duration.apply(
                lambda x: str(x).split(".")[0] if pd.notna(x) else "N/A"
            )

        # Add trade number
        df_display.insert(0, "trade_id", range(1, len(df_display) + 1))

        # Reorder columns for better display
        preferred_order = [
            "trade_id",
            "entry_time",
            "exit_time",
            "duration",
            "side",
            "entry_price",
            "exit_price",
            "pnl",
            "return_pct",
            "fees_pct",
        ]

        # Keep only columns that exist
        display_columns = [col for col in preferred_order if col in df_display.columns]
        remaining_columns = [
            col for col in df_display.columns if col not in display_columns
        ]
        final_columns = display_columns + remaining_columns

        df_display = df_display[final_columns]

        # Create summary statistics
        summary_stats = _calculate_trades_summary(trades)

        logger.info(f"Trades table rendered successfully with {len(df_display)} rows")

        # Return formatted data (in real implementation, this would be a widget)
        return {
            "data": df_display,
            "summary": summary_stats,
            "total_rows": len(df_display),
            "columns": list(df_display.columns),
        }

    except Exception as e:
        logger.error(f"Failed to render trades table: {e}")
        raise


def render_metrics_table(metrics: Dict[str, Any]) -> Any:
    """
    Render performance metrics in a formatted table.

    Creates a professional table view of performance metrics with proper
    formatting, grouping, and highlighting of key values.

    Parameters
    ----------
    metrics : dict
        Dictionary of performance metrics from Phase 6 performance.summarize()

    Returns
    -------
    Any
        Table widget or data structure (implementation dependent)

    Raises
    ------
    ValueError
        If metrics dictionary is empty

    Examples
    --------
    >>> metrics = {
    ...     'final_equity': 11000,
    ...     'total_return': 0.10,
    ...     'sharpe': 1.5,
    ...     'max_drawdown': -0.05,
    ...     'total_trades': 100,
    ...     'win_rate': 0.6
    ... }
    >>> table = render_metrics_table(metrics)
    """
    logger = get_logger(__name__)

    if not metrics:
        raise ValueError("Metrics dictionary cannot be empty")

    try:
        logger.info(f"Rendering metrics table with {len(metrics)} metrics")

        # Define metric categories and formatting
        metric_categories = {
            "Portfolio Performance": {
                "final_equity": ("Final Equity", "${:,.2f}"),
                "pnl": ("Profit/Loss", "${:,.2f}"),
                "total_return": ("Total Return", "{:.2%}"),
                "cagr": ("CAGR", "{:.2%}"),
                "annual_volatility": ("Annual Volatility", "{:.2%}"),
            },
            "Risk Metrics": {
                "sharpe": ("Sharpe Ratio", "{:.3f}"),
                "sortino": ("Sortino Ratio", "{:.3f}"),
                "max_drawdown": ("Max Drawdown", "{:.2%}"),
                "calmar": ("Calmar Ratio", "{:.3f}"),
                "var_95": ("VaR (95%)", "{:.2%}"),
            },
            "Trade Statistics": {
                "total_trades": ("Total Trades", "{:,}"),
                "win_trades": ("Winning Trades", "{:,}"),
                "loss_trades": ("Losing Trades", "{:,}"),
                "win_rate": ("Win Rate", "{:.2%}"),
                "profit_factor": ("Profit Factor", "{:.3f}"),
                "expectancy": ("Expectancy", "${:.2f}"),
            },
            "Trade Details": {
                "avg_win": ("Average Win", "${:.2f}"),
                "avg_loss": ("Average Loss", "${:.2f}"),
                "largest_win": ("Largest Win", "${:.2f}"),
                "largest_loss": ("Largest Loss", "${:.2f}"),
                "avg_trade_duration": ("Avg Trade Duration", "{}"),
            },
            "Other Metrics": {
                "duration_days": ("Backtest Period (Days)", "{:.0f}"),
                "trades_per_day": ("Trades per Day", "{:.2f}"),
                "kelly_criterion": ("Kelly %", "{:.2%}"),
            },
        }

        # Build formatted table data
        table_data = []

        for category, category_metrics in metric_categories.items():
            # Add category header
            table_data.append(
                {
                    "category": category,
                    "metric": "",
                    "value": "",
                    "formatted_value": "",
                    "is_header": True,
                }
            )

            # Add metrics in this category
            for key, (label, format_str) in category_metrics.items():
                if key in metrics:
                    raw_value = metrics[key]

                    try:
                        if isinstance(raw_value, (int, float)) and not pd.isna(
                            raw_value
                        ):
                            formatted_value = format_str.format(raw_value)
                        else:
                            formatted_value = (
                                str(raw_value) if raw_value is not None else "N/A"
                            )
                    except (ValueError, TypeError):
                        formatted_value = (
                            str(raw_value) if raw_value is not None else "N/A"
                        )

                    table_data.append(
                        {
                            "category": category,
                            "metric": label,
                            "value": raw_value,
                            "formatted_value": formatted_value,
                            "is_header": False,
                        }
                    )

        # Add any remaining metrics not in categories
        remaining_metrics = [
            k
            for k in metrics.keys()
            if not any(k in cat_metrics for cat_metrics in metric_categories.values())
        ]

        if remaining_metrics:
            table_data.append(
                {
                    "category": "Additional Metrics",
                    "metric": "",
                    "value": "",
                    "formatted_value": "",
                    "is_header": True,
                }
            )

            for key in remaining_metrics:
                value = metrics[key]
                formatted_value = (
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )

                table_data.append(
                    {
                        "category": "Additional Metrics",
                        "metric": key.replace("_", " ").title(),
                        "value": value,
                        "formatted_value": formatted_value,
                        "is_header": False,
                    }
                )

        # Convert to DataFrame for easier handling
        df_metrics = pd.DataFrame(table_data)

        # Calculate summary statistics
        summary_stats = _calculate_metrics_summary(metrics)

        logger.info(f"Metrics table rendered with {len(table_data)} rows")

        return {
            "data": df_metrics,
            "summary": summary_stats,
            "categories": list(metric_categories.keys()),
            "total_metrics": len([row for row in table_data if not row["is_header"]]),
        }

    except Exception as e:
        logger.error(f"Failed to render metrics table: {e}")
        raise


def export_table(df: pd.DataFrame, path: Path) -> Path:
    """
    Export DataFrame to file with format detection.

    Exports DataFrame to CSV or Parquet format based on file extension,
    with proper handling of datetime columns and relative paths.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export
    path : Path
        Output file path with extension (.csv or .parquet)

    Returns
    -------
    Path
        Path to exported file

    Raises
    ------
    ValueError
        If DataFrame is empty or path has unsupported extension
    IOError
        If file cannot be written

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>>
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> exported_path = export_table(df, Path("data.csv"))
    >>> print(f"Exported to: {exported_path}")
    """
    logger = get_logger(__name__)

    if df.empty:
        raise ValueError("Cannot export empty DataFrame")

    path = Path(path)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get file extension
    extension = path.suffix.lower()

    try:
        logger.info(f"Exporting DataFrame with {len(df)} rows to {path}")

        if extension == ".csv":
            # Export to CSV
            df.to_csv(path, index=False, encoding="utf-8")

        elif extension == ".parquet":
            # Export to Parquet
            df.to_parquet(path, index=False, engine="pyarrow")

        elif extension == ".json":
            # Export to JSON
            df.to_json(path, orient="records", date_format="iso", indent=2)

        elif extension == ".xlsx":
            # Export to Excel (if openpyxl available)
            try:
                df.to_excel(path, index=False, engine="openpyxl")
            except ImportError:
                raise ValueError("Excel export requires openpyxl: pip install openpyxl")

        else:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: .csv, .parquet, .json, .xlsx"
            )

        # Verify file was created and has content
        if not path.exists():
            raise IOError(f"Export failed: file not created at {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise IOError(f"Export failed: empty file created at {path}")

        logger.info(
            f"Successfully exported {len(df)} rows to {path} ({file_size:,} bytes)"
        )
        return path

    except Exception as e:
        logger.error(f"Failed to export table to {path}: {e}")
        raise


def filter_trades_table(
    trades: pd.DataFrame,
    side: Optional[str] = None,
    min_pnl: Optional[float] = None,
    max_pnl: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter trades table based on criteria.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades DataFrame to filter
    side : str, optional
        Filter by trade side ('LONG' or 'SHORT')
    min_pnl : float, optional
        Minimum PnL threshold
    max_pnl : float, optional
        Maximum PnL threshold
    start_date : str, optional
        Start date filter (YYYY-MM-DD format)
    end_date : str, optional
        End date filter (YYYY-MM-DD format)

    Returns
    -------
    pd.DataFrame
        Filtered trades DataFrame
    """
    logger = get_logger(__name__)

    try:
        filtered_trades = trades.copy()
        original_count = len(filtered_trades)

        # Filter by side
        if side:
            filtered_trades = filtered_trades[filtered_trades["side"] == side.upper()]
            logger.info(f"Filtered by side '{side}': {len(filtered_trades)} trades")

        # Filter by PnL range
        if min_pnl is not None:
            filtered_trades = filtered_trades[filtered_trades["pnl"] >= min_pnl]
            logger.info(f"Filtered by min PnL {min_pnl}: {len(filtered_trades)} trades")

        if max_pnl is not None:
            filtered_trades = filtered_trades[filtered_trades["pnl"] <= max_pnl]
            logger.info(f"Filtered by max PnL {max_pnl}: {len(filtered_trades)} trades")

        # Filter by date range
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_trades = filtered_trades[
                pd.to_datetime(filtered_trades["entry_time"]) >= start_dt
            ]
            logger.info(
                f"Filtered by start date {start_date}: {len(filtered_trades)} trades"
            )

        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_trades = filtered_trades[
                pd.to_datetime(filtered_trades["entry_time"]) <= end_dt
            ]
            logger.info(
                f"Filtered by end date {end_date}: {len(filtered_trades)} trades"
            )

        logger.info(
            f"Filter complete: {original_count} -> {len(filtered_trades)} trades"
        )
        return filtered_trades

    except Exception as e:
        logger.error(f"Failed to filter trades: {e}")
        raise


def paginate_data(
    df: pd.DataFrame, page: int = 1, page_size: int = 50
) -> Dict[str, Any]:
    """
    Paginate DataFrame for display.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to paginate
    page : int
        Page number (1-based)
    page_size : int
        Number of rows per page

    Returns
    -------
    dict
        Dictionary with paginated data and metadata
    """
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    # Validate page number
    page = max(1, min(page, total_pages))

    # Calculate slice indices
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    # Get page data
    page_data = df.iloc[start_idx:end_idx]

    return {
        "data": page_data,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "start_row": start_idx + 1,
        "end_row": end_idx,
        "has_previous": page > 1,
        "has_next": page < total_pages,
    }


# Private helper functions


def _calculate_trades_summary(trades: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for trades."""
    if trades.empty:
        return {}

    summary = {
        "total_trades": len(trades),
        "winning_trades": (
            len(trades[trades["pnl"] > 0]) if "pnl" in trades.columns else 0
        ),
        "losing_trades": (
            len(trades[trades["pnl"] < 0]) if "pnl" in trades.columns else 0
        ),
        "total_pnl": trades["pnl"].sum() if "pnl" in trades.columns else 0,
        "avg_pnl": trades["pnl"].mean() if "pnl" in trades.columns else 0,
        "max_win": trades["pnl"].max() if "pnl" in trades.columns else 0,
        "max_loss": trades["pnl"].min() if "pnl" in trades.columns else 0,
    }

    if "pnl" in trades.columns:
        summary["win_rate"] = (
            summary["winning_trades"] / summary["total_trades"]
            if summary["total_trades"] > 0
            else 0
        )

    return summary


def _calculate_metrics_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics for metrics."""
    summary = {
        "key_metrics_count": len(
            [
                k
                for k in metrics.keys()
                if k in ["sharpe", "total_return", "max_drawdown", "win_rate"]
            ]
        ),
        "risk_metrics_count": len(
            [
                k
                for k in metrics.keys()
                if k in ["sharpe", "sortino", "max_drawdown", "var_95"]
            ]
        ),
        "trade_metrics_count": len(
            [
                k
                for k in metrics.keys()
                if k in ["total_trades", "win_rate", "profit_factor"]
            ]
        ),
        "total_metrics": len(metrics),
    }

    return summary


def create_trades_excel_report(
    trades: pd.DataFrame, metrics: Dict[str, Any], output_path: Path
) -> Path:
    """
    Create comprehensive Excel report with multiple sheets.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades data
    metrics : dict
        Performance metrics
    output_path : Path
        Output Excel file path

    Returns
    -------
    Path
        Path to created Excel file
    """
    logger = get_logger(__name__)

    try:
        logger.info(f"Creating Excel report: {output_path}")

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Sheet 1: Summary metrics
            metrics_df = pd.DataFrame(
                [
                    {"Metric": k.replace("_", " ").title(), "Value": v}
                    for k, v in metrics.items()
                ]
            )
            metrics_df.to_excel(writer, sheet_name="Summary", index=False)

            # Sheet 2: All trades
            trades.to_excel(writer, sheet_name="All Trades", index=False)

            # Sheet 3: Winning trades
            winning_trades = (
                trades[trades["pnl"] > 0] if "pnl" in trades.columns else pd.DataFrame()
            )
            if not winning_trades.empty:
                winning_trades.to_excel(
                    writer, sheet_name="Winning Trades", index=False
                )

            # Sheet 4: Losing trades
            losing_trades = (
                trades[trades["pnl"] < 0] if "pnl" in trades.columns else pd.DataFrame()
            )
            if not losing_trades.empty:
                losing_trades.to_excel(writer, sheet_name="Losing Trades", index=False)

        logger.info(f"Excel report created successfully: {output_path}")
        return output_path

    except ImportError:
        raise ValueError("Excel export requires openpyxl: pip install openpyxl")
    except Exception as e:
        logger.error(f"Failed to create Excel report: {e}")
        raise
