"""Moteur de backtesting déterministe avec timeouts sur tâches parallèles."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from threadx.config import load_settings
from threadx.utils.determinism import get_rng, set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RunResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    meta: Dict[str, Any]


class BacktestEngine:
    """Exécuteur séquentiel avec contrôle de temps d'exécution."""

    def __init__(self, *, max_workers: int | None = None, timeout_s: float | None = None) -> None:
        settings = load_settings()
        self.max_workers = max(1, max_workers or settings.MAX_WORKERS)
        self.timeout_s = float(timeout_s or 30.0)
        self.logger = logger

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        indicators: Mapping[str, Any],
        *,
        params: Mapping[str, Any],
        symbol: str,
        timeframe: str,
        seed: int = 42,
    ) -> RunResult:
        start_ts = time.perf_counter()
        self._validate_inputs(df, indicators, params)
        set_global_seed(seed)
        rng = get_rng(seed)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            signals = self._run_with_timeout(
                executor,
                self._generate_signals,
                df,
                indicators,
                params,
                rng,
                task_name="generate_signals",
            )

            trades = self._run_with_timeout(
                executor,
                self._simulate_trades,
                df,
                signals,
                params,
                rng,
                task_name="simulate_trades",
            )

            equity, returns = self._run_with_timeout(
                executor,
                self._compute_equity_and_returns,
                df,
                signals,
                params,
                task_name="compute_equity",
            )

        meta = self._build_meta(
            df=df,
            trades=trades,
            symbol=symbol,
            timeframe=timeframe,
            params=params,
            seed=seed,
            duration=time.perf_counter() - start_ts,
        )

        return RunResult(equity=equity, returns=returns, trades=trades, meta=meta)

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------
    def _run_with_timeout(
        self,
        executor: ThreadPoolExecutor,
        func: Any,
        *args: Any,
        task_name: str,
        **kwargs: Any,
    ) -> Any:
        future = executor.submit(func, *args, **kwargs)
        done, not_done = wait({future}, timeout=self.timeout_s, return_when=FIRST_EXCEPTION)
        if not_done:
            for pending in not_done:
                pending.cancel()
            raise TimeoutError(f"La tâche '{task_name}' a dépassé {self.timeout_s}s")

        future_result = next(iter(done))
        return future_result.result()

    # ------------------------------------------------------------------
    def _validate_inputs(
        self,
        df: pd.DataFrame,
        indicators: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df doit être un DataFrame")
        required_columns = {"open", "high", "low", "close", "volume"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {', '.join(sorted(missing))}")
        if df.empty:
            raise ValueError("df ne doit pas être vide")
        if not isinstance(indicators, Mapping):
            raise TypeError("indicators doit être un mapping")
        if not isinstance(params, Mapping):
            raise TypeError("params doit être un mapping")

    # ------------------------------------------------------------------
    def _generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Mapping[str, Any],
        params: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> pd.Series:
        close = df["close"].astype(float)
        window = int(params.get("signal_window", 20))
        z_entry = float(params.get("entry_z", 1.5))

        price_mean = close.rolling(window=window, min_periods=window).mean()
        price_std = close.rolling(window=window, min_periods=window).std(ddof=0)

        upper = price_mean + z_entry * price_std
        lower = price_mean - z_entry * price_std

        signals = pd.Series(0.0, index=df.index, name="signal")
        signals[close > upper] = -1.0
        signals[close < lower] = 1.0

        atr = indicators.get("atr")
        if isinstance(atr, (pd.Series, np.ndarray)) and len(atr) == len(df):
            atr_series = pd.Series(atr, index=df.index)
            threshold = atr_series.quantile(0.3)
            signals[atr_series < threshold] = 0.0

        if signals.abs().sum() == 0:
            fallback = rng.choice([0.0, 1.0, -1.0], size=len(df), p=[0.9, 0.05, 0.05])
            signals = pd.Series(fallback, index=df.index, name="signal")

        return signals

    # ------------------------------------------------------------------
    def _simulate_trades(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        params: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        leverage = float(params.get("leverage", 3.0))
        fees_bps = float(params.get("fees_bps", 10.0))
        initial_capital = float(params.get("initial_capital", 10000.0))

        trades: list[dict[str, Any]] = []
        position = 0.0
        entry_price = 0.0
        entry_ts = None

        for ts, price, signal in zip(df.index, df["close"], signals):
            if position == 0.0 and signal != 0.0:
                position = signal
                entry_price = float(price)
                entry_ts = ts
                continue

            if position != 0.0 and (signal == -position or signal == 0.0):
                exit_price = float(price)
                raw_return = (exit_price - entry_price) / entry_price * position
                fees = fees_bps * 0.0001 * 2
                net_return = raw_return - fees
                pnl = net_return * leverage * initial_capital

                trades.append(
                    {
                        "entry_ts": entry_ts,
                        "exit_ts": ts,
                        "pnl": pnl,
                        "size": leverage * initial_capital / entry_price,
                        "price_entry": entry_price,
                        "price_exit": exit_price,
                        "side": "LONG" if position > 0 else "SHORT",
                    }
                )

                position = 0.0
                entry_price = 0.0
                entry_ts = None

        if position != 0.0 and entry_ts is not None:
            exit_price = float(df.iloc[-1]["close"])
            raw_return = (exit_price - entry_price) / entry_price * position
            pnl = (raw_return - fees_bps * 0.0001 * 2) * leverage * initial_capital
            trades.append(
                {
                    "entry_ts": entry_ts,
                    "exit_ts": df.index[-1],
                    "pnl": pnl,
                    "size": leverage * initial_capital / entry_price,
                    "price_entry": entry_price,
                    "price_exit": exit_price,
                    "side": "LONG" if position > 0 else "SHORT",
                }
            )

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(
                columns=[
                    "entry_ts",
                    "exit_ts",
                    "pnl",
                    "size",
                    "price_entry",
                    "price_exit",
                    "side",
                ]
            )

        return trades_df

    # ------------------------------------------------------------------
    def _compute_equity_and_returns(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        params: Mapping[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        initial_capital = float(params.get("initial_capital", 10000.0))
        leverage = float(params.get("leverage", 3.0))
        fees_bps = float(params.get("fees_bps", 10.0))

        price_returns = df["close"].pct_change().fillna(0.0)
        position = signals.shift(1).fillna(0.0)
        gross = position * price_returns * leverage
        net = gross - fees_bps * 0.0001

        equity = (1.0 + net).cumprod() * initial_capital
        equity_series = pd.Series(equity, index=df.index, name="equity")
        returns_series = pd.Series(net, index=df.index, name="returns")

        return equity_series, returns_series

    # ------------------------------------------------------------------
    def _build_meta(
        self,
        *,
        df: pd.DataFrame,
        trades: pd.DataFrame,
        symbol: str,
        timeframe: str,
        params: Mapping[str, Any],
        seed: int,
        duration: float,
    ) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": len(df),
            "trades": len(trades),
            "seed": seed,
            "duration_s": duration,
            "params": dict(params),
        }


def create_engine(**kwargs: Any) -> BacktestEngine:
    return BacktestEngine(**kwargs)


def run(
    df: pd.DataFrame,
    indicators: Mapping[str, Any],
    *,
    params: Mapping[str, Any],
    symbol: str,
    timeframe: str,
    seed: int = 42,
    **engine_kwargs: Any,
) -> RunResult:
    engine = BacktestEngine(**engine_kwargs)
    return engine.run(
        df=df,
        indicators=indicators,
        params=params,
        symbol=symbol,
        timeframe=timeframe,
        seed=seed,
    )
