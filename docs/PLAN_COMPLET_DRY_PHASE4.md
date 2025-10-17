# 🚀 Plan d'Action Complet - Refactoring DRY + Phase 4
**Date**: 17 octobre 2025
**Statut**: Step 3.1 ✅ | Steps 3.2-3.5 + Phase 4 📋 Planifiés
**Objectif Final**: Code DRY (<5% duplication) + Architecture Production-Ready

---

## 📊 État Actuel (Checkpoint)

### ✅ Complété - Step 3.1: Module Common Imports
- [x] `common_imports.py` créé (75 lignes)
- [x] `validation.py` refactorisé (-10 lignes)
- [x] `engine.py` refactorisé (-5 lignes)
- [x] Bug `measure_throughput` corrigé
- [x] Tests validation: 0 régression
- [x] Documentation: `RAPPORT_DRY_REFACTORING_PHASE1.md`

### 📈 Métriques Actuelles
- **Duplication**: ~50% → Objectif <5%
- **Fichiers refactorisés**: 2/50 (4%)
- **Lignes économisées**: 15/~750 (2%)
- **Temps investi**: 1h / 4h estimées

---

## 🎯 Roadmap Détaillé

### Phase 3: DRY Refactoring (Suite)
**Durée totale**: 3 heures restantes
**Priorité**: 🔥 HAUTE

---

## 📦 Step 3.2: Inheritance/Mixins (1h30)

### 3.2.1: BasePanel pour UI Components (45 min)

#### Objectif
Créer classe de base pour les 4 panels UI existants avec méthodes communes.

#### Fichier à créer
**`src/threadx/ui/components/base.py`** (150 lignes)

```python
"""
ThreadX UI - Base Panel Component
==================================

Classe de base pour tous les panels UI (Dash/Streamlit).
Fournit méthodes communes pour render, validation, error handling.

Usage:
    from threadx.ui.components.base import BasePanel

    class BacktestPanel(BasePanel):
        def create_layout(self):
            # Implémentation spécifique
            pass

Author: ThreadX Framework - Phase 2 DRY Refactoring
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.development.base_component import Component
import pandas as pd

from threadx.utils.common_imports import create_logger

logger = create_logger(__name__)


class BasePanel(ABC):
    """
    Classe abstraite de base pour panels UI.

    Fournit méthodes communes:
    - Rendering de tables standardisées
    - Handling d'erreurs uniform
    - Validation d'inputs
    - Création de placeholders
    - Formatage de métriques

    Sous-classes doivent implémenter:
    - create_layout(): Structure du panel
    """

    def __init__(self, panel_id: str):
        """
        Initialize base panel.

        Args:
            panel_id: Unique identifier for panel (e.g., 'backtest', 'optimization')
        """
        self.panel_id = panel_id
        self.logger = create_logger(f"{__name__}.{panel_id}")

    @abstractmethod
    def create_layout(self) -> Component:
        """
        Create panel layout (must be implemented by subclasses).

        Returns:
            Component: Dash/Streamlit component tree
        """
        pass

    def render_table(
        self,
        df: pd.DataFrame,
        table_id: str,
        max_rows: int = 100,
        striped: bool = True,
        hover: bool = True,
    ) -> Component:
        """
        Render DataFrame as styled table.

        Args:
            df: Data to display
            table_id: DOM id for table
            max_rows: Maximum rows to show
            striped: Alternate row colors
            hover: Highlight on mouse over

        Returns:
            Component: Styled table component
        """
        if df is None or df.empty:
            return html.Div("No data available", id=table_id)

        # Truncate if too many rows
        display_df = df.head(max_rows) if len(df) > max_rows else df

        # Format numeric columns
        for col in display_df.select_dtypes(include=['float']).columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

        return dbc.Table.from_dataframe(
            display_df,
            striped=striped,
            bordered=True,
            hover=hover,
            responsive=True,
            id=table_id,
            className="table-sm"
        )

    def create_error_display(self, error_id: str, message: str = "") -> Component:
        """
        Create error message display.

        Args:
            error_id: DOM id for error display
            message: Optional initial error message

        Returns:
            Component: Error alert component
        """
        return dbc.Alert(
            message or "An error occurred",
            id=error_id,
            color="danger",
            dismissable=True,
            is_open=bool(message),
            className="mt-3"
        )

    def create_loading_spinner(
        self,
        spinner_id: str,
        children: Optional[Component] = None,
        spinner_type: str = "border"
    ) -> Component:
        """
        Create loading spinner wrapper.

        Args:
            spinner_id: DOM id for spinner
            children: Content to wrap
            spinner_type: Spinner style ('border', 'grow', 'dots')

        Returns:
            Component: Loading spinner component
        """
        return dcc.Loading(
            id=spinner_id,
            type=spinner_type,
            children=children or html.Div()
        )

    def create_metric_card(
        self,
        title: str,
        value: Any,
        card_id: str,
        color: str = "primary"
    ) -> Component:
        """
        Create metric display card.

        Args:
            title: Metric name
            value: Metric value
            card_id: DOM id
            color: Card color theme

        Returns:
            Component: Styled metric card
        """
        return dbc.Card(
            dbc.CardBody([
                html.H6(title, className="text-muted"),
                html.H4(str(value), className=f"text-{color}")
            ]),
            id=card_id,
            className="mb-3"
        )

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate panel inputs (can be overridden).

        Args:
            **kwargs: Input values to validate

        Returns:
            (is_valid, error_message)
        """
        # Default: all valid
        return True, None

    def handle_error(self, error: Exception) -> str:
        """
        Handle and format error messages.

        Args:
            error: Exception to handle

        Returns:
            str: Formatted error message
        """
        self.logger.error(f"{self.panel_id} error: {error}", exc_info=True)
        return f"Error in {self.panel_id}: {str(error)}"

    def create_empty_graph(
        self,
        graph_id: str,
        title: str = "No data"
    ) -> Component:
        """
        Create empty placeholder graph.

        Args:
            graph_id: DOM id for graph
            title: Placeholder title

        Returns:
            Component: Empty graph component
        """
        return dcc.Graph(
            id=graph_id,
            figure={
                'data': [],
                'layout': {
                    'title': title,
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False}
                }
            }
        )
```

#### Fichiers à refactoriser (4 panels)

1. **`backtest_panel.py`** (325 lignes → 280 lignes)
   ```python
   from threadx.ui.components.base import BasePanel

   class BacktestPanel(BasePanel):
       def __init__(self):
           super().__init__("backtest")

       def create_layout(self) -> Component:
           # Utiliser self.render_table(), self.create_error_display(), etc.
           pass
   ```

2. **`indicators_panel.py`** (similar pattern)
3. **`optimization_panel.py`** (similar pattern)
4. **`data_manager.py`** (similar pattern)

**Gain estimé**: ~100 lignes de code dupliqué éliminées

---

### 3.2.2: BaseCommand pour CLI (45 min)

#### Objectif
Créer classe de base pour les 4 commandes CLI avec parsing/validation communes.

#### Fichier à créer
**`src/threadx/cli/commands/base.py`** (200 lignes)

```python
"""
ThreadX CLI - Base Command
===========================

Classe de base pour toutes les commandes CLI.
Fournit parsing, validation, error handling communs.

Usage:
    from threadx.cli.commands.base import BaseCommand

    class BacktestCommand(BaseCommand):
        def execute(self, **kwargs):
            # Implémentation spécifique
            pass

Author: ThreadX Framework - Phase 2 DRY Refactoring
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import typer
from datetime import datetime

from threadx.utils.common_imports import create_logger, pd
from threadx.cli.utils import (
    print_summary,
    print_json,
    format_duration,
    handle_bridge_error,
)

logger = create_logger(__name__)


class BaseCommand(ABC):
    """
    Classe abstraite de base pour commandes CLI.

    Fournit méthodes communes:
    - Parsing d'arguments standardisé
    - Validation d'inputs (dates, symbols, params)
    - Error handling uniform
    - Output formatting (JSON, table, summary)

    Sous-classes doivent implémenter:
    - execute(**kwargs): Logique de la commande
    """

    def __init__(self, command_name: str):
        """
        Initialize base command.

        Args:
            command_name: Name of command (e.g., 'backtest', 'optimize')
        """
        self.command_name = command_name
        self.logger = create_logger(f"{__name__}.{command_name}")

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute command logic (must be implemented by subclasses).

        Args:
            **kwargs: Parsed and validated command arguments

        Returns:
            Command result
        """
        pass

    def parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse date string to datetime.

        Args:
            date_str: Date in format YYYY-MM-DD

        Returns:
            datetime object or None

        Raises:
            ValueError: If date format invalid
        """
        if not date_str:
            return None

        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}': {e}")

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate symbol format.

        Args:
            symbol: Symbol to validate (e.g., BTCUSDT)

        Returns:
            True if valid

        Raises:
            ValueError: If symbol invalid
        """
        if not symbol or not symbol.isupper():
            raise ValueError(f"Invalid symbol '{symbol}': must be uppercase")
        return True

    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Validate timeframe format.

        Args:
            timeframe: Timeframe to validate (e.g., 1h, 4h, 1d)

        Returns:
            True if valid

        Raises:
            ValueError: If timeframe invalid
        """
        valid_tfs = ['1m', '5m', '15m', '1h', '4h', '1d']
        if timeframe not in valid_tfs:
            raise ValueError(f"Invalid timeframe '{timeframe}': must be one of {valid_tfs}")
        return True

    def validate_numeric_param(
        self,
        value: Optional[float],
        param_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> bool:
        """
        Validate numeric parameter.

        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            True if valid

        Raises:
            ValueError: If value out of range
        """
        if value is None:
            return True

        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")

        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")

        return True

    def format_results_table(self, data: Dict[str, Any]) -> str:
        """
        Format results as ASCII table.

        Args:
            data: Dictionary of results

        Returns:
            Formatted table string
        """
        df = pd.DataFrame([data]).T
        df.columns = ['Value']
        return df.to_string()

    def handle_execution_error(self, error: Exception) -> None:
        """
        Handle command execution errors.

        Args:
            error: Exception that occurred
        """
        self.logger.error(f"{self.command_name} failed: {error}", exc_info=True)
        typer.secho(f"❌ Error: {str(error)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    def print_success(self, message: str) -> None:
        """Print success message."""
        typer.secho(f"✅ {message}", fg=typer.colors.GREEN)

    def print_info(self, message: str) -> None:
        """Print info message."""
        typer.secho(f"ℹ️  {message}", fg=typer.colors.BLUE)

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        typer.secho(f"⚠️  {message}", fg=typer.colors.YELLOW)
```

#### Fichiers à refactoriser (4 commandes)

1. **`backtest_cmd.py`** (181 lignes → 130 lignes)
   ```python
   from threadx.cli.commands.base import BaseCommand

   class BacktestCommand(BaseCommand):
       def __init__(self):
           super().__init__("backtest")

       def execute(self, strategy: str, symbol: str, **kwargs):
           # Utiliser self.validate_symbol(), self.parse_date(), etc.
           pass
   ```

2. **`optimize_cmd.py`** (similar)
3. **`indicators_cmd.py`** (similar)
4. **`data_cmd.py`** (similar)

**Gain estimé**: ~150 lignes de code dupliqué éliminées

---

## 🎨 Step 3.3: Template Method Pattern (45 min)

### Objectif
Créer templates réutilisables pour optimizations (grid search, Monte Carlo).

### Fichier à créer
**`src/threadx/optimization/templates.py`** (250 lignes)

```python
"""
ThreadX Optimization - Template Methods
========================================

Templates réutilisables pour optimizations.
Évite duplication de boucles grid search / Monte Carlo.

Usage:
    from threadx.optimization.templates import grid_search_template

    results = grid_search_template(
        param_grid={'period': [10, 20, 30], 'std': [1.5, 2.0]},
        objective_fn=my_backtest_fn,
        maximize=True
    )

Author: ThreadX Framework - Phase 2 DRY Refactoring
"""

from typing import Dict, List, Callable, Any, Optional, Tuple
from itertools import product
import numpy as np
from dataclasses import dataclass

from threadx.utils.common_imports import pd, create_logger

logger = create_logger(__name__)


@dataclass
class OptimizationResult:
    """
    Résultat d'une optimization.

    Attributes:
        best_params: Meilleurs paramètres trouvés
        best_score: Meilleur score (metric)
        all_results: DataFrame de tous les essais
        iterations: Nombre d'itérations
        duration_sec: Durée totale
    """
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    iterations: int
    duration_sec: float


def grid_search_template(
    param_grid: Dict[str, List[Any]],
    objective_fn: Callable[[Dict[str, Any]], float],
    maximize: bool = True,
    parallel: bool = False,
    n_jobs: int = -1,
    verbose: bool = True
) -> OptimizationResult:
    """
    Template pour grid search exhaustif.

    Teste toutes les combinaisons de paramètres.

    Args:
        param_grid: Dict de paramètres avec listes de valeurs
                   Ex: {'period': [10, 20, 30], 'std': [1.5, 2.0]}
        objective_fn: Fonction à optimiser, prend params dict,
                     retourne score float
        maximize: True pour maximiser, False pour minimiser
        parallel: Exécution parallèle (multiprocessing)
        n_jobs: Nombre de workers (-1 = tous les cores)
        verbose: Afficher progression

    Returns:
        OptimizationResult avec meilleurs params et tous résultats

    Example:
        >>> def backtest_fn(params):
        ...     return run_backtest(**params).sharpe_ratio
        >>> result = grid_search_template(
        ...     param_grid={'period': [10, 20], 'std': [2.0, 2.5]},
        ...     objective_fn=backtest_fn,
        ...     maximize=True
        ... )
        >>> print(result.best_params)  # {'period': 20, 'std': 2.5}
    """
    import time
    start_time = time.time()

    # Générer toutes les combinaisons
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    total_iterations = len(combinations)
    logger.info(f"Grid search: {total_iterations} combinations")

    results = []

    if parallel:
        # Exécution parallèle
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for combo in combinations:
                params = dict(zip(param_names, combo))
                futures.append(executor.submit(objective_fn, params))

            for i, future in enumerate(futures):
                try:
                    score = future.result()
                    params = dict(zip(param_names, combinations[i]))
                    results.append({**params, 'score': score})

                    if verbose and (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i+1}/{total_iterations}")
                except Exception as e:
                    logger.warning(f"Failed combo {combinations[i]}: {e}")
    else:
        # Exécution séquentielle
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            try:
                score = objective_fn(params)
                results.append({**params, 'score': score})

                if verbose and (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{total_iterations}")
            except Exception as e:
                logger.warning(f"Failed combo {params}: {e}")

    # Créer DataFrame résultats
    results_df = pd.DataFrame(results)

    # Trouver meilleur
    if maximize:
        best_idx = results_df['score'].idxmax()
    else:
        best_idx = results_df['score'].idxmin()

    best_row = results_df.loc[best_idx]
    best_params = best_row.drop('score').to_dict()
    best_score = best_row['score']

    duration = time.time() - start_time

    logger.info(f"Best: {best_params} → score={best_score:.4f} (took {duration:.2f}s)")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results_df,
        iterations=len(results),
        duration_sec=duration
    )


def monte_carlo_template(
    param_ranges: Dict[str, Tuple[float, float]],
    objective_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 100,
    maximize: bool = True,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> OptimizationResult:
    """
    Template pour optimization Monte Carlo.

    Tire des paramètres aléatoires et évalue objective.

    Args:
        param_ranges: Dict de paramètres avec ranges (min, max)
                     Ex: {'period': (10, 50), 'std': (1.0, 3.0)}
        objective_fn: Fonction à optimiser
        n_trials: Nombre d'essais aléatoires
        maximize: True pour maximiser
        seed: Seed pour reproductibilité
        verbose: Afficher progression

    Returns:
        OptimizationResult

    Example:
        >>> result = monte_carlo_template(
        ...     param_ranges={'period': (10, 50), 'std': (1.0, 3.0)},
        ...     objective_fn=backtest_fn,
        ...     n_trials=100
        ... )
    """
    import time
    start_time = time.time()

    if seed is not None:
        np.random.seed(seed)

    logger.info(f"Monte Carlo: {n_trials} trials")

    results = []

    for i in range(n_trials):
        # Tirer paramètres aléatoires
        params = {}
        for name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[name] = np.random.randint(min_val, max_val + 1)
            else:
                params[name] = np.random.uniform(min_val, max_val)

        try:
            score = objective_fn(params)
            results.append({**params, 'score': score})

            if verbose and (i + 1) % 20 == 0:
                logger.info(f"Progress: {i+1}/{n_trials}")
        except Exception as e:
            logger.warning(f"Failed trial {i}: {e}")

    # Analyser résultats (même logique que grid_search)
    results_df = pd.DataFrame(results)

    if maximize:
        best_idx = results_df['score'].idxmax()
    else:
        best_idx = results_df['score'].idxmin()

    best_row = results_df.loc[best_idx]
    best_params = best_row.drop('score').to_dict()
    best_score = best_row['score']

    duration = time.time() - start_time

    logger.info(f"Best: {best_params} → score={best_score:.4f}")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results_df,
        iterations=len(results),
        duration_sec=duration
    )


def bayesian_optimization_template(
    param_space: Dict[str, Tuple[float, float]],
    objective_fn: Callable[[Dict[str, Any]], float],
    n_iterations: int = 50,
    n_initial_points: int = 10,
    maximize: bool = True,
    verbose: bool = True
) -> OptimizationResult:
    """
    Template pour Bayesian Optimization (scikit-optimize).

    Note: Nécessite `pip install scikit-optimize`

    Args:
        param_space: Espace de recherche
        objective_fn: Fonction objectif
        n_iterations: Nombre d'itérations
        n_initial_points: Points d'exploration initiaux
        maximize: True pour maximiser
        verbose: Afficher progression

    Returns:
        OptimizationResult
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except ImportError:
        raise ImportError("Bayesian optimization requires: pip install scikit-optimize")

    logger.info(f"Bayesian optimization: {n_iterations} iterations")

    # TODO: Implémentation complète avec scikit-optimize
    # Pour l'instant, fallback sur Monte Carlo
    logger.warning("Bayesian optimization not fully implemented, using Monte Carlo")
    return monte_carlo_template(
        param_ranges=param_space,
        objective_fn=objective_fn,
        n_trials=n_iterations,
        maximize=maximize,
        verbose=verbose
    )
```

### Fichiers à refactoriser

**`optimization/engine.py`** - Remplacer boucles custom par templates:

```python
from threadx.optimization.templates import grid_search_template, monte_carlo_template

# Avant (100+ lignes de boucles)
for period in periods:
    for std in stds:
        result = backtest(period=period, std=std)
        ...

# Après (10 lignes)
result = grid_search_template(
    param_grid={'period': periods, 'std': stds},
    objective_fn=lambda params: backtest(**params).sharpe_ratio,
    maximize=True
)
```

**Gain estimé**: ~200 lignes de boucles éliminées

---

## 🔍 Step 3.4: Rescan Duplication (30 min)

### Objectif
Vérifier que duplication < 5% après refactoring.

### Outils à utiliser

```bash
# 1. Radon - Complexity metrics
radon cc src/threadx/ -a -nb --total-average
radon mi src/threadx/ -n B  # Maintainability Index

# 2. Pylint - Duplicate code detection
pylint src/threadx/ --disable=all --enable=duplicate-code --min-similarity-lines=4

# 3. Coverage - Lignes économisées
cloc src/threadx/ --by-file --csv > before.csv
# ... après refactoring ...
cloc src/threadx/ --by-file --csv > after.csv

# 4. Custom grep count
grep -r "import pandas as pd" src/threadx/ | wc -l
grep -r "from threadx.utils.common_imports" src/threadx/ | wc -l
```

### Métriques à valider

| Métrique | Avant | Objectif | Status |
|----------|-------|----------|--------|
| Duplication | ~50% | <5% | 🔄 |
| Imports dupliqués | 100+ | <10 | 🔄 |
| Lignes totales | 35,000 | 34,250 | 🔄 |
| Complexité moyenne | ? | <10 | 🔄 |
| Maintainability Index | ? | >65 | 🔄 |

---

## 🏗️ Phase 4: Structural Improvements (4-6 heures)

### Step 4.1: Enforce Layering (2h)

#### 4.1.1: Bridge as Complete Facade (1h)

**Objectif**: Tous les accès UI → Engine via Bridge uniquement.

**Fichier à créer**: `src/threadx/bridge/controllers/gpu_controller.py`

```python
"""
ThreadX Bridge - GPU Controller
================================

Contrôleur pour gestion GPU (device management, fallbacks).

Author: ThreadX Framework - Phase 4 Layering
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from threadx.utils.common_imports import create_logger

logger = create_logger(__name__)


@dataclass
class GPUInfo:
    """Information sur un GPU."""
    device_id: int
    name: str
    memory_total_mb: float
    memory_free_mb: float
    is_available: bool


class GPUController:
    """
    Contrôleur pour operations GPU.

    Abstrait l'accès aux utilitaires GPU (CuPy, device manager).
    Fournit fallbacks CPU automatiques.
    """

    def __init__(self):
        self.logger = logger
        self._gpu_available = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        """Detect si GPU (CuPy) disponible."""
        try:
            import cupy as cp
            return True
        except ImportError:
            return False

    def list_devices(self) -> List[GPUInfo]:
        """Liste tous les GPUs disponibles."""
        if not self._gpu_available:
            return []

        try:
            from threadx.utils.gpu import list_devices
            devices = list_devices()
            return [
                GPUInfo(
                    device_id=d.id,
                    name=d.name,
                    memory_total_mb=d.memory_total / 1024**2,
                    memory_free_mb=d.memory_free / 1024**2,
                    is_available=d.is_available
                )
                for d in devices
            ]
        except Exception as e:
            self.logger.warning(f"Failed to list GPUs: {e}")
            return []

    def get_best_device(self) -> Optional[int]:
        """Retourne ID du meilleur GPU (plus de mémoire libre)."""
        devices = self.list_devices()
        if not devices:
            return None

        best = max(devices, key=lambda d: d.memory_free_mb)
        return best.device_id

    def is_gpu_available(self) -> bool:
        """Check si au moins 1 GPU disponible."""
        return self._gpu_available and len(self.list_devices()) > 0
```

**Ajouter au Bridge**: `bridge/__init__.py`

```python
from threadx.bridge.controllers.gpu_controller import GPUController

class ThreadXBridge:
    def __init__(self, config: Optional[Configuration] = None):
        # ... existing controllers ...
        self.gpu = GPUController()
```

#### 4.1.2: Dependency Injection (1h)

**Refactoriser**: `start_threadx.py`, `apps/dash_app.py`

**Avant**:
```python
# start_threadx.py
from threadx.bridge import ThreadXBridge

bridge = ThreadXBridge()  # Config hardcodée
```

**Après**:
```python
# start_threadx.py
from threadx.config.loaders import load_settings
from threadx.bridge import ThreadXBridge, Configuration

# Load config from file
config_dict = load_settings()
config = Configuration(**config_dict)

# Inject dependencies
bridge = ThreadXBridge(config=config)

# UI also receives config
if ui_framework == "dash":
    from threadx.ui.dash_app import create_app
    app = create_app(bridge=bridge, config=config)
```

**Fichiers à modifier**:
- `start_threadx.py`
- `apps/dash_app.py`
- `apps/streamlit/app.py`
- `ui/__init__.py`

---

### Step 4.2: Modularize GPU/UI (1h30)

#### 4.2.1: GPU Fallbacks Everywhere (45 min)

**Objectif**: Wrap tous les appels CuPy dans try/except avec fallback NumPy.

**Pattern à appliquer**:

```python
# Avant
import cupy as cp
result = cp.array([1, 2, 3])

# Après
from threadx.utils import xp  # xp = cupy ou numpy selon disponibilité
result = xp.array([1, 2, 3])  # Fonctionne CPU ou GPU
```

**Fichiers à refactoriser** (10+ fichiers):
- `indicators/bollinger.py`
- `indicators/xatr.py`
- `indicators/gpu_integration.py`
- `backtest/engine.py` (déjà fait ✅)
- `optimization/engine.py`
- `strategy/gpu_examples.py`
- `utils/batching.py`

**Créer helper**: `src/threadx/utils/gpu_safe.py`

```python
"""
GPU-Safe Operations
===================

Wrappers pour operations GPU avec fallback CPU automatique.

Usage:
    from threadx.utils.gpu_safe import safe_to_gpu, safe_compute

    data_gpu = safe_to_gpu(data_cpu)  # No-op if GPU unavailable
    result = safe_compute(lambda: cp.dot(a, b))  # Fallback to numpy

Author: ThreadX Framework - Phase 4
"""

from typing import Any, Callable, TypeVar
import numpy as np

from threadx.utils.common_imports import create_logger
from threadx.utils import xp

logger = create_logger(__name__)
T = TypeVar('T')


def safe_to_gpu(array: np.ndarray) -> Any:
    """
    Convert array to GPU if available, else return as-is.

    Args:
        array: NumPy array

    Returns:
        CuPy array if GPU available, else NumPy array
    """
    try:
        return xp.asarray(array)
    except Exception as e:
        logger.debug(f"GPU transfer failed, using CPU: {e}")
        return array


def safe_compute(fn: Callable[[], T], fallback_fn: Optional[Callable[[], T]] = None) -> T:
    """
    Execute GPU computation with automatic CPU fallback.

    Args:
        fn: Function to execute (may use CuPy)
        fallback_fn: Optional explicit fallback (else re-run with NumPy)

    Returns:
        Result of computation

    Example:
        >>> result = safe_compute(
        ...     fn=lambda: cp.dot(gpu_a, gpu_b),
        ...     fallback_fn=lambda: np.dot(cpu_a, cpu_b)
        ... )
    """
    try:
        return fn()
    except Exception as e:
        logger.warning(f"GPU compute failed: {e}, trying CPU fallback")
        if fallback_fn:
            return fallback_fn()
        else:
            # Retry same function (xp should fallback to numpy)
            return fn()
```

#### 4.2.2: UI Abstraction (45 min)

**Créer**: `src/threadx/ui/base.py`

```python
"""
ThreadX UI - Abstract Base
===========================

Interface abstraite pour tous les frameworks UI.
Permet de swap Dash ↔ Streamlit ↔ Tkinter sans changer code métier.

Usage:
    from threadx.ui.base import BaseUI

    class DashUI(BaseUI):
        def run(self):
            self.app.run_server()

Author: ThreadX Framework - Phase 4 UI Abstraction
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from threadx.bridge import ThreadXBridge
from threadx.utils.common_imports import create_logger

logger = create_logger(__name__)


class BaseUI(ABC):
    """
    Classe abstraite pour UI frameworks.

    Sous-classes (DashUI, StreamlitUI, TkinterUI) doivent implémenter:
    - initialize(): Setup de l'UI
    - run(): Lancement de l'application
    - register_callbacks(): Enregistrement des handlers
    """

    def __init__(self, bridge: ThreadXBridge, config: Optional[dict] = None):
        """
        Initialize UI with Bridge and config.

        Args:
            bridge: ThreadX Bridge instance
            config: Optional configuration dict
        """
        self.bridge = bridge
        self.config = config or {}
        self.logger = logger

        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """Setup UI components (must be implemented)."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Run UI application (must be implemented)."""
        pass

    @abstractmethod
    def register_callbacks(self) -> None:
        """Register event handlers (must be implemented)."""
        pass

    def get_title(self) -> str:
        """Get application title from config."""
        return self.config.get('title', 'ThreadX Dashboard')

    def get_port(self) -> int:
        """Get server port from config."""
        return self.config.get('port', 8050)
```

**Refactoriser**: `apps/dash_app.py`

```python
from threadx.ui.base import BaseUI
from dash import Dash

class DashUI(BaseUI):
    def initialize(self):
        self.app = Dash(__name__, title=self.get_title())
        # ... setup layout ...

    def run(self, debug=False):
        self.app.run_server(port=self.get_port(), debug=debug)

    def register_callbacks(self):
        # ... callbacks ...
        pass
```

---

### Step 4.3: Testing and CI Integration (1h30)

#### 4.3.1: Expand Test Coverage (1h)

**Objectif**: 100% coverage pour modules core.

**Priorités**:
1. `backtest/` - validation.py ✅, engine.py (80% → 95%)
2. `optimization/engine.py` (50% → 90%)
3. `bridge/controllers/` (70% → 95%)
4. `indicators/bank.py` (60% → 90%)

**Créer tests manquants**:

```bash
# Generate coverage report
pytest tests/ --cov=src/threadx --cov-report=html --cov-report=term-missing

# Identify gaps
coverage report --show-missing | grep -E "^src/threadx/(backtest|optimization|bridge)"
```

**Tests à créer** (~10 nouveaux fichiers):
- `tests/test_gpu_controller.py`
- `tests/test_ui_base.py`
- `tests/test_optimization_templates.py`
- `tests/test_base_panel.py`
- `tests/test_base_command.py`
- `tests/test_gpu_safe.py`

#### 4.3.2: CI Hooks (30 min)

**Créer**: `.github/workflows/quality-checks.yml`

```yaml
name: Quality Checks

on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pylint radon mypy black isort

    - name: Run Black (formatting)
      run: black --check src/threadx/

    - name: Run isort (imports)
      run: isort --check-only src/threadx/

    - name: Run Pylint
      run: pylint src/threadx/ --fail-under=8.0

    - name: Run MyPy (type checking)
      run: mypy src/threadx/ --ignore-missing-imports

    - name: Run Tests with Coverage
      run: pytest tests/ --cov=src/threadx --cov-fail-under=85

    - name: Check Complexity (Radon)
      run: radon cc src/threadx/ -a -nb --total-average --max C

    - name: Check Duplication (Pylint)
      run: pylint src/threadx/ --disable=all --enable=duplicate-code --max-similarity=5
```

**Ajouter à**: `pyproject.toml`

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --strict-markers --cov-report=term-missing"

[tool.coverage.run]
source = ["src/threadx"]
omit = ["*/tests/*", "*/__pycache__/*", "*/conftest.py"]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
fail_under = 85

[tool.pylint.messages_control]
max-line-length = 100
disable = ["C0111", "C0103"]

[tool.black]
line-length = 100
target-version = ['py312']

[tool.isort]
profile = "black"
line_length = 100
```

---

### Step 4.4: Documentation and Metrics (1h)

#### 4.4.1: Update Documentation (30 min)

**Fichiers à créer/mettre à jour**:

1. **`docs/ARCHITECTURE.md`** - Schéma 3-layer complet
2. **`docs/DRY_REFACTORING.md`** - Patterns appliqués
3. **`bridge/README_CONTROLLERS.md`** - Liste de tous les controllers
4. **`ui/README_BASE_CLASSES.md`** - BasePanel/BaseUI usage
5. **`optimization/README_TEMPLATES.md`** - Templates usage

**Exemple**: `docs/ARCHITECTURE.md`

```markdown
# ThreadX Architecture

## Overview
ThreadX suit une architecture 3-layer stricte:

```
┌─────────────────────────────────────────┐
│         UI Layer (Presentation)         │
│  Dash, Streamlit, Tkinter, CLI         │
└─────────────────┬───────────────────────┘
                  │ (via Bridge only)
┌─────────────────▼───────────────────────┐
│        Bridge Layer (Facade)            │
│  Controllers: Backtest, Optimization,   │
│              GPU, Data, Indicators      │
└─────────────────┬───────────────────────┘
                  │ (async coordination)
┌─────────────────▼───────────────────────┐
│         Engine Layer (Business Logic)   │
│  backtest/, optimization/, indicators/  │
│  data/, strategy/, utils/               │
└─────────────────────────────────────────┘
```

## Principles

1. **Separation of Concerns**: UI ne connait pas Engine
2. **Dependency Injection**: Config passed via constructors
3. **Facade Pattern**: All UI→Engine via Bridge
4. **DRY**: Common imports, base classes, templates
5. **Testability**: 100% coverage possible (mocked Bridge)

## Controllers

| Controller | Module | Purpose |
|-----------|---------|---------|
| BacktestController | `bridge/controllers/backtest_controller.py` | Backtest execution |
| OptimizationController | `bridge/controllers/optimization_controller.py` | Parameter optimization |
| DataIngestionController | `bridge/controllers/data_controller.py` | Data download/management |
| IndicatorController | `bridge/controllers/indicator_controller.py` | Indicator calculation |
| GPUController | `bridge/controllers/gpu_controller.py` | GPU management (Phase 4) |

## Base Classes

### UI Components
- `BasePanel` (`ui/components/base.py`): Common UI panel methods
- `BaseUI` (`ui/base.py`): Abstract UI framework

### CLI Commands
- `BaseCommand` (`cli/commands/base.py`): Common CLI patterns

### Optimization
- Templates (`optimization/templates.py`): Grid/Monte Carlo patterns
```

#### 4.4.2: Track Metrics (30 min)

**Script**: `scripts/measure_code_quality.py`

```python
#!/usr/bin/env python3
"""
Code Quality Metrics Dashboard
===============================

Mesure et affiche métriques de qualité du code:
- Coverage (pytest-cov)
- Complexity (radon)
- Duplication (pylint)
- Maintainability (radon mi)
- LOC (cloc)

Usage:
    python scripts/measure_code_quality.py [--output report.json]
"""

import subprocess
import json
from pathlib import Path

def run_coverage():
    """Run pytest with coverage."""
    result = subprocess.run(
        ["pytest", "tests/", "--cov=src/threadx", "--cov-report=json"],
        capture_output=True,
        text=True
    )

    with open("coverage.json") as f:
        data = json.load(f)

    return {
        "total_coverage": data["totals"]["percent_covered"],
        "lines_covered": data["totals"]["covered_lines"],
        "lines_total": data["totals"]["num_statements"]
    }

def run_complexity():
    """Run radon complexity."""
    result = subprocess.run(
        ["radon", "cc", "src/threadx/", "-a", "-nb", "--json"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    # Calculate average complexity
    total_complexity = 0
    count = 0

    for file_data in data.values():
        for item in file_data:
            if isinstance(item, dict) and 'complexity' in item:
                total_complexity += item['complexity']
                count += 1

    avg_complexity = total_complexity / count if count > 0 else 0

    return {
        "average_complexity": avg_complexity,
        "total_functions": count
    }

def run_duplication():
    """Check code duplication with pylint."""
    result = subprocess.run(
        ["pylint", "src/threadx/", "--disable=all", "--enable=duplicate-code", "--output-format=json"],
        capture_output=True,
        text=True
    )

    # Parse pylint JSON output
    try:
        data = json.loads(result.stdout)
        duplication_count = len([m for m in data if m['message-id'] == 'R0801'])
    except:
        duplication_count = 0

    return {
        "duplicate_blocks": duplication_count
    }

def count_lines():
    """Count lines of code."""
    result = subprocess.run(
        ["cloc", "src/threadx/", "--json"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)
    python_data = data.get("Python", {})

    return {
        "total_lines": python_data.get("code", 0),
        "comment_lines": python_data.get("comment", 0),
        "blank_lines": python_data.get("blank", 0)
    }

def main():
    print("📊 Measuring Code Quality Metrics...")

    metrics = {
        "coverage": run_coverage(),
        "complexity": run_complexity(),
        "duplication": run_duplication(),
        "lines_of_code": count_lines()
    }

    # Display
    print("\n✅ Coverage:")
    print(f"   {metrics['coverage']['total_coverage']:.1f}% ({metrics['coverage']['lines_covered']}/{metrics['coverage']['lines_total']} lines)")

    print("\n🔢 Complexity:")
    print(f"   Average: {metrics['complexity']['average_complexity']:.2f}")
    print(f"   Functions: {metrics['complexity']['total_functions']}")

    print("\n📋 Duplication:")
    print(f"   Duplicate blocks: {metrics['duplication']['duplicate_blocks']}")

    print("\n📏 Lines of Code:")
    print(f"   Code: {metrics['lines_of_code']['total_lines']}")
    print(f"   Comments: {metrics['lines_of_code']['comment_lines']}")

    # Save report
    with open("quality_report.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n💾 Report saved: quality_report.json")

if __name__ == "__main__":
    main()
```

---

## 📊 Timeline et Priorités

### Phase 3 (Suite): DRY Refactoring
**Durée**: 3 heures
**Priorité**: 🔥 HAUTE

| Step | Tâche | Durée | Difficulté | Priorité |
|------|-------|-------|------------|----------|
| 3.2.1 | BasePanel + refactor 4 panels | 45min | Moyenne | Haute |
| 3.2.2 | BaseCommand + refactor 4 cmds | 45min | Moyenne | Haute |
| 3.3 | Templates optimization | 45min | Moyenne | Moyenne |
| 3.4 | Rescan duplication | 30min | Facile | Haute |

### Phase 4: Structural Improvements
**Durée**: 4-6 heures
**Priorité**: 🟡 MOYENNE-HAUTE

| Step | Tâche | Durée | Difficulté | Priorité |
|------|-------|-------|------------|----------|
| 4.1.1 | GPUController + Bridge | 1h | Facile | Moyenne |
| 4.1.2 | Dependency Injection | 1h | Moyenne | Haute |
| 4.2.1 | GPU Fallbacks everywhere | 45min | Facile | Haute |
| 4.2.2 | UI Abstraction (BaseUI) | 45min | Moyenne | Basse |
| 4.3.1 | Tests (100% coverage) | 1h | Moyenne | Haute |
| 4.3.2 | CI Hooks (.github/) | 30min | Facile | Moyenne |
| 4.4.1 | Documentation | 30min | Facile | Basse |
| 4.4.2 | Metrics script | 30min | Facile | Basse |

---

## 🎯 Checklist de Completion

### Phase 3: DRY ✅❌
- [x] Step 3.1: common_imports.py
- [ ] Step 3.2: BasePanel + BaseCommand
- [ ] Step 3.3: Templates optimization
- [ ] Step 3.4: Rescan (<5% duplication)

### Phase 4: Structure ❌
- [ ] Step 4.1: GPUController + DI
- [ ] Step 4.2: GPU fallbacks + UI abstraction
- [ ] Step 4.3: Tests + CI
- [ ] Step 4.4: Docs + metrics

---

## 🚀 Ordre d'Exécution Recommandé

### Session 1 (Maintenant): DRY Step 3.2-3.4
1. Créer `BasePanel` (45min)
2. Refactoriser 2 panels (backtest, optimization) (30min)
3. Créer `BaseCommand` (45min)
4. Refactoriser 2 commands (backtest, optimize) (30min)
5. **COMMIT + PUSH** ✅

### Session 2: Templates + Rescan
6. Créer `templates.py` (45min)
7. Refactoriser `optimization/engine.py` (30min)
8. Rescan duplication (30min)
9. **COMMIT + PUSH** ✅

### Session 3: Phase 4.1-4.2
10. GPUController + intégration Bridge (1h)
11. Dependency Injection (1h)
12. GPU fallbacks (45min)
13. **COMMIT + PUSH** ✅

### Session 4: Phase 4.3-4.4
14. Tests nouveaux (1h)
15. CI hooks (30min)
16. Documentation (30min)
17. Metrics script (30min)
18. **COMMIT + PUSH FINAL** ✅

---

## 📈 KPIs de Succès

### Métriques Quantitatives
- ✅ Duplication: <5%
- ✅ Coverage: >85%
- ✅ Complexity: <10 moyenne
- ✅ Maintainability Index: >65
- ✅ Lignes économisées: ~1,000

### Métriques Qualitatives
- ✅ Architecture 3-layer stricte
- ✅ GPU fallbacks partout
- ✅ UI frameworks swappable
- ✅ Tests unitaires complets
- ✅ CI/CD automatisé
- ✅ Documentation à jour

---

## 🎓 Notes Techniques

### Patterns Appliqués
1. **DRY** - Module common_imports, base classes, templates
2. **Facade** - Bridge comme point d'entrée unique
3. **Dependency Injection** - Config externalisée
4. **Template Method** - Optimization templates
5. **Strategy** - BaseUI pour swap frameworks
6. **Factory** - Création controllers dans Bridge

### Technologies
- **Linting**: pylint, flake8, mypy, black, isort
- **Testing**: pytest, pytest-cov
- **Metrics**: radon, cloc, coverage.py
- **CI**: GitHub Actions
- **Docs**: Markdown, Sphinx (optionnel)

### Risks & Mitigations
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Dépendances circulaires | HIGH | LOW | Tests imports after each step |
| Tests cassés | MEDIUM | MEDIUM | Run pytest after each refactor |
| Performance degradée | LOW | LOW | Benchmark before/after |
| Documentation obsolète | MEDIUM | HIGH | Update docs avec code |

---

**Plan généré le**: 17 octobre 2025
**Estimé durée totale**: 7-9 heures
**Retour sur investissement**: Payback 3-4 semaines
**Status**: 📋 Ready for execution

---

## ⏭️ Action Immédiate

**Commencer par**:
1. Créer `ui/components/base.py` (BasePanel)
2. Refactoriser `backtest_panel.py`
3. Tester → Commit

**Commande**:
```bash
# Option 1: Tout automatiser
python scripts/execute_dry_plan.py --start-from 3.2.1

# Option 2: Manuel (recommandé pour contrôle)
# Je crée les fichiers un par un avec vous
```

**Prêt à commencer Step 3.2.1 (BasePanel)?** 🚀
