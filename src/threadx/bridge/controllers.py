"""
ThreadX Bridge Controllers - Orchestration Layer
================================================

Controllers synchrones qui orchestrent les appels vers l'Engine.
Aucune logique métier, juste wrappers fins autour des modules Engine.

Usage:
    >>> from threadx.bridge.controllers import BacktestController
    >>> from threadx.bridge.models import BacktestRequest
    >>> controller = BacktestController()
    >>> req = BacktestRequest(
    ...     symbol='BTCUSDT',
    ...     timeframe='1h',
    ...     strategy='bollinger_reversion',
    ...     params={'period': 20, 'std': 2.0}
    ... )
    >>> result = controller.run_backtest(req)
    >>> print(result.sharpe_ratio)

Author: ThreadX Framework
Version: Prompt 2 - Bridge Foundation
"""

import time
from pathlib import Path
from typing import Any

from threadx.bridge.exceptions import (
    BacktestError,
    DataError,
    IndicatorError,
    SweepError,
)
from threadx.bridge.models import (
    BacktestRequest,
    BacktestResult,
    Configuration,
    DataRequest,
    DataValidationResult,
    IndicatorRequest,
    IndicatorResult,
    SweepRequest,
    SweepResult,
)


class BacktestController:
    """Controller pour exécution de backtests.

    Wrapper synchrone autour de threadx.backtest.engine.
    Gère validation requête, appel Engine, et mapping résultat.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        """Exécute un backtest complet.

        Orchestre:
        1. Validation requête
        2. Lazy import BacktestEngine
        3. Création engine avec paramètres
        4. Exécution backtest
        5. Mapping résultat vers BacktestResult

        Args:
            request: Requête backtest avec tous paramètres.

        Returns:
            BacktestResult avec KPIs, trades, courbes.

        Raises:
            BacktestError: Si validation échoue ou exécution erreur.

        Example:
            >>> req = BacktestRequest(
            ...     symbol='BTCUSDT', timeframe='1h',
            ...     strategy='bollinger_reversion',
            ...     params={'period': 20, 'std': 2.0}
            ... )
            >>> controller = BacktestController()
            >>> result = controller.run_backtest(req)
            >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
        """
        # Validation requête
        if self.config.validate_requests and not request.validate():
            raise BacktestError("Invalid BacktestRequest: missing required fields")

        start_time = time.perf_counter()

        try:
            # Lazy import Engine (évite import lourd au démarrage)
            from threadx.backtest.engine import BacktestEngine, create_engine

            # Création engine avec configuration
            engine: BacktestEngine = create_engine(
                strategy_name=request.strategy,
                params=request.params,
                initial_cash=request.initial_cash,
                use_gpu=request.use_gpu or self.config.gpu_enabled,
            )

            # Exécution backtest (Engine gère data loading, calculs)
            raw_result = engine.run(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
            )

            # Mapping résultat Engine → BacktestResult
            execution_time = time.perf_counter() - start_time

            return BacktestResult(
                total_profit=raw_result.get("total_profit", 0.0),
                total_return=raw_result.get("total_return", 0.0),
                sharpe_ratio=raw_result.get("sharpe_ratio", 0.0),
                max_drawdown=raw_result.get("max_drawdown", 0.0),
                win_rate=raw_result.get("win_rate", 0.0),
                trades=raw_result.get("trades", []),
                equity_curve=raw_result.get("equity_curve", []),
                drawdown_curve=raw_result.get("drawdown_curve", []),
                metrics=raw_result.get("metrics", {}),
                execution_time=execution_time,
                metadata={
                    "engine": "BacktestEngine",
                    "gpu_used": request.use_gpu or self.config.gpu_enabled,
                    "cache_path": self.config.cache_path,
                },
            )

        except Exception as e:
            raise BacktestError(f"Backtest execution failed: {e}") from e


class IndicatorController:
    """Controller pour construction d'indicateurs techniques.

    Wrapper synchrone autour de threadx.indicators.bank.
    Gère cache automatique et calcul batch d'indicateurs.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def build_indicators(self, request: IndicatorRequest) -> IndicatorResult:
        """Construit ensemble d'indicateurs techniques.

        Orchestre:
        1. Validation requête
        2. Lazy import IndicatorBank
        3. Chargement données OHLCV
        4. Calcul indicateurs avec cache
        5. Retour valeurs + stats cache

        Args:
            request: Requête indicateurs avec params.

        Returns:
            IndicatorResult avec valeurs calculées et cache stats.

        Raises:
            IndicatorError: Si validation échoue ou calcul erreur.

        Example:
            >>> req = IndicatorRequest(
            ...     symbol='BTCUSDT', timeframe='1h',
            ...     indicators={'ema': {'period': 50}, 'rsi': {'period': 14}}
            ... )
            >>> controller = IndicatorController()
            >>> result = controller.build_indicators(req)
            >>> print(result.indicator_values['ema'][:5])
        """
        # Validation requête
        if self.config.validate_requests and not request.validate():
            raise IndicatorError("Invalid IndicatorRequest: missing required fields")

        start_time = time.perf_counter()
        cache_hits = 0
        cache_misses = 0

        try:
            # Lazy import IndicatorBank
            from threadx.indicators.bank import (
                IndicatorBank,
                ensure_indicator,
            )

            # Chargement données (via DataController ou direct)
            if request.data_path:
                data_path = Path(request.data_path)
            else:
                # Auto-detect path depuis registry
                from threadx.data.registry import get_data_path

                data_path = get_data_path(request.symbol, request.timeframe)

            # Création IndicatorBank avec cache
            bank = IndicatorBank(
                data_path=str(data_path),
                cache_path=self.config.cache_path,
                use_gpu=request.use_gpu or self.config.gpu_enabled,
            )

            # Calcul batch indicateurs
            indicator_values: dict[str, Any] = {}

            for indicator_name, params in request.indicators.items():
                # ensure_indicator gère cache automatiquement
                values = ensure_indicator(
                    bank=bank,
                    name=indicator_name,
                    params=params,
                    force_recompute=request.force_recompute,
                )

                indicator_values[indicator_name] = values

                # Stats cache (IndicatorBank expose cache_info)
                if bank.was_cached(indicator_name):
                    cache_hits += 1
                else:
                    cache_misses += 1

            build_time = time.perf_counter() - start_time

            return IndicatorResult(
                indicator_values=indicator_values,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                build_time=build_time,
                metadata={
                    "data_path": str(data_path),
                    "cache_path": self.config.cache_path,
                    "gpu_used": request.use_gpu or self.config.gpu_enabled,
                },
            )

        except Exception as e:
            raise IndicatorError(f"Indicator build failed: {e}") from e


class SweepController:
    """Controller pour parameter sweeps / optimisation.

    Wrapper synchrone autour de threadx.optimization.engine.
    Gère exploration grille paramètres et tri résultats.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def run_sweep(self, request: SweepRequest) -> SweepResult:
        """Exécute parameter sweep / optimisation.

        Orchestre:
        1. Validation requête
        2. Lazy import UnifiedOptimizationEngine
        3. Génération grille combinaisons
        4. Exécution backtests parallèles
        5. Tri résultats selon critères
        6. Retour top N

        Args:
            request: Requête sweep avec param_grid et critères.

        Returns:
            SweepResult avec meilleurs params et résultats top N.

        Raises:
            SweepError: Si validation échoue ou exécution erreur.

        Example:
            >>> req = SweepRequest(
            ...     symbol='BTCUSDT', timeframe='1h',
            ...     strategy='bollinger_reversion',
            ...     param_grid={'period': [10, 20, 30], 'std': [1.5, 2.0]},
            ...     optimization_criteria=['sharpe_ratio'],
            ...     top_n=5
            ... )
            >>> controller = SweepController()
            >>> result = controller.run_sweep(req)
            >>> print(result.best_params)
        """
        # Validation requête
        if self.config.validate_requests and not request.validate():
            raise SweepError("Invalid SweepRequest: missing required fields")

        start_time = time.perf_counter()

        try:
            # Lazy import OptimizationEngine
            from threadx.optimization.engine import (
                UnifiedOptimizationEngine,
            )

            # Création engine avec configuration
            engine = UnifiedOptimizationEngine(
                symbol=request.symbol,
                timeframe=request.timeframe,
                strategy=request.strategy,
                param_grid=request.param_grid,
                max_workers=request.max_workers or self.config.max_workers,
                use_gpu=request.use_gpu or self.config.gpu_enabled,
            )

            # Exécution sweep (Engine gère parallélisation, cache, pruning)
            raw_results = engine.run_sweep(
                optimization_criteria=request.optimization_criteria,
                top_n=request.top_n,
            )

            # Mapping résultat Engine → SweepResult
            execution_time = time.perf_counter() - start_time

            best_result = raw_results[0]  # Top 1 (déjà trié par Engine)

            return SweepResult(
                best_params=best_result.get("params", {}),
                best_sharpe=best_result.get("sharpe_ratio", 0.0),
                best_return=best_result.get("total_return", 0.0),
                top_results=raw_results[: request.top_n],
                total_combinations=engine.total_combinations,
                pruned_combinations=engine.pruned_combinations,
                execution_time=execution_time,
                metadata={
                    "engine": "UnifiedOptimizationEngine",
                    "max_workers": request.max_workers or self.config.max_workers,
                    "gpu_used": request.use_gpu or self.config.gpu_enabled,
                },
            )

        except Exception as e:
            raise SweepError(f"Sweep execution failed: {e}") from e


class DataController:
    """Controller pour chargement et validation de données.

    Wrapper synchrone autour de threadx.data.io et threadx.data.registry.
    Gère chargement, validation qualité, et exports.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def validate_data(self, request: DataRequest) -> DataValidationResult:
        """Valide qualité des données OHLCV.

        Orchestre:
        1. Validation requête
        2. Lazy import data modules
        3. Chargement données Parquet
        4. Validation colonnes, types, valeurs
        5. Détection missing values, duplicates, gaps, outliers
        6. Calcul quality score

        Args:
            request: Requête validation avec symbol/timeframe.

        Returns:
            DataValidationResult avec quality score et détails.

        Raises:
            DataError: Si requête invalide ou chargement échoue.

        Example:
            >>> req = DataRequest(
            ...     symbol='BTCUSDT', timeframe='1h',
            ...     validate=True
            ... )
            >>> controller = DataController()
            >>> result = controller.validate_data(req)
            >>> print(f"Quality Score: {result.quality_score}/10")
        """
        # Validation requête
        if self.config.validate_requests and not request.validate_request():
            raise DataError("Invalid DataRequest: missing required fields")

        try:
            # Lazy import data modules
            from threadx.data.io import load_parquet
            from threadx.data.registry import get_data_path

            # Résolution path
            if request.data_path:
                data_path = Path(request.data_path)
            else:
                data_path = get_data_path(request.symbol, request.timeframe)

            # Chargement données
            df = load_parquet(
                str(data_path),
                start_date=request.start_date,
                end_date=request.end_date,
            )

            # Validation si activée
            if not request.validate:
                return DataValidationResult(
                    valid=True,
                    row_count=len(df),
                    quality_score=10.0,
                    metadata={"path": str(data_path)},
                )

            # Validation complète
            errors: list[str] = []
            warnings: list[str] = []

            # Colonnes requises
            missing_cols = set(request.required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")

            # Missing values
            missing_values = int(df.isnull().sum().sum())
            if missing_values > 0:
                warnings.append(f"{missing_values} missing values detected")

            # Duplicates
            duplicate_rows = int(df.duplicated().sum())
            if duplicate_rows > 0:
                warnings.append(f"{duplicate_rows} duplicate rows detected")

            # Date gaps (si colonne timestamp existe)
            date_gaps = 0
            if "timestamp" in df.columns:
                df_sorted = df.sort_values("timestamp")
                time_diffs = df_sorted["timestamp"].diff()
                # Détection gaps > 2x timeframe normal
                expected_interval = time_diffs.median()
                date_gaps = int((time_diffs > 2 * expected_interval).sum())
                if date_gaps > 0:
                    warnings.append(f"{date_gaps} date gaps detected")

            # Outliers (OHLCV hors bornes normales)
            outliers_count = 0
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    q1 = df[col].quantile(0.01)
                    q99 = df[col].quantile(0.99)
                    outliers = ((df[col] < q1) | (df[col] > q99)).sum()
                    outliers_count += int(outliers)

            # Quality score (10 - pénalités)
            quality_score = 10.0
            quality_score -= min(len(errors) * 2.0, 5.0)
            quality_score -= min(missing_values / 100, 2.0)
            quality_score -= min(duplicate_rows / 100, 1.0)
            quality_score -= min(date_gaps / 50, 1.0)
            quality_score -= min(outliers_count / 100, 1.0)
            quality_score = max(quality_score, 0.0)

            return DataValidationResult(
                valid=len(errors) == 0,
                row_count=len(df),
                missing_values=missing_values,
                duplicate_rows=duplicate_rows,
                date_gaps=date_gaps,
                outliers_count=outliers_count,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                metadata={
                    "path": str(data_path),
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                },
            )

        except Exception as e:
            raise DataError(f"Data validation failed: {e}") from e
