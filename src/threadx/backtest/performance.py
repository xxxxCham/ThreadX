"""Calculs de performance déterministes avec gestion de timeout."""

from __future__ import annotations

import math
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from threadx.utils.determinism import set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def equity_curve(returns: pd.Series, initial_capital: float) -> pd.Series:
    if initial_capital <= 0:
        raise ValueError("initial_capital doit être positif")
    if returns.empty:
        return pd.Series([], dtype=float)

    cleaned = returns.fillna(0.0)
    equity = (1.0 + cleaned).cumprod() * float(initial_capital)
    return pd.Series(equity, index=returns.index, name="equity")


def drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series([], dtype=float)
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1.0
    return pd.Series(drawdown, index=equity.index, name="drawdown")


def _annualized_factor(returns: pd.Series) -> float:
    if returns.index.freq is not None:
        freq = returns.index.freq
        if freq == pd.tseries.offsets.Day():
            return 252
        if freq == pd.tseries.offsets.Hour():
            return 24 * 252
    return 252


def _sharpe_ratio(returns: pd.Series) -> float:
    scale = math.sqrt(_annualized_factor(returns))
    excess = returns.mean()
    std = returns.std(ddof=0)
    return float(excess / std * scale) if std > 0 else 0.0


def _sortino_ratio(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    if downside.empty:
        return float("inf")
    scale = math.sqrt(_annualized_factor(returns))
    downside_std = downside.std(ddof=0)
    return float(returns.mean() / downside_std * scale) if downside_std > 0 else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min()) if not equity.empty else 0.0


def _cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return 0.0
    years = max(1.0, len(equity) / 252)
    return (end / start) ** (1.0 / years) - 1.0


_METRIC_FNS: Dict[str, Any] = {
    "sharpe": _sharpe_ratio,
    "sortino": _sortino_ratio,
    "max_drawdown": _max_drawdown,
    "cagr": _cagr,
}


@dataclass(frozen=True)
class PerformanceReport:
    metrics: Dict[str, float]
    equity: pd.Series
    drawdown: pd.Series


def summarize(
    returns: pd.Series,
    trades: pd.DataFrame | None,
    *,
    initial_capital: float = 10000.0,
    seed: int = 42,
    timeout_s: float = 30.0,
) -> PerformanceReport:
    set_global_seed(seed)

    equity = equity_curve(returns, initial_capital)
    dd = drawdown_series(equity)

    with ThreadPoolExecutor(max_workers=len(_METRIC_FNS)) as executor:
        futures = {
            name: executor.submit(fn, equity if name in {"max_drawdown", "cagr"} else returns)
            for name, fn in _METRIC_FNS.items()
        }
        done, not_done = wait(futures.values(), timeout=timeout_s, return_when=FIRST_EXCEPTION)
        if not_done:
            for future in not_done:
                future.cancel()
            raise TimeoutError(f"Calcul de performance > {timeout_s}s")

    metrics = {name: futures[name].result() for name in futures}

    if trades is not None and not trades.empty:
        metrics["trades"] = len(trades)
        metrics["pnl_total"] = float(trades["pnl"].sum())

    return PerformanceReport(metrics=metrics, equity=equity, drawdown=dd)
