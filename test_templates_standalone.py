"""
Test standalone des templates d'optimisation
Sans dépendances sur le reste du framework
"""

import sys
from pathlib import Path

# Ajouter le src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time


# ===== Copie des classes pour test standalone =====


@dataclass
class OptimizationResult:
    """Résultat d'une optimisation"""

    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    iterations: int
    duration_sec: float
    convergence_history: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """
    Template Method Pattern pour optimizers
    """

    def __init__(
        self,
        objective_fn: callable,
        maximize: bool = True,
        verbose: bool = True,
        early_stopping: int = None,
        tolerance: float = 1e-6,
    ):
        self.objective_fn = objective_fn
        self.maximize = maximize
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.tolerance = tolerance

        self.best_params = None
        self.best_score = float("-inf") if maximize else float("inf")
        self.all_results = []
        self.convergence_history = []
        self.iterations_without_improvement = 0

    def prepare_data(self) -> None:
        """Hook pour préparation des données (optionnel)"""
        pass

    @abstractmethod
    def run_iteration(self, iteration: int) -> Tuple[Dict[str, Any], float]:
        """Doit retourner (params, score) pour cette itération"""
        pass

    def finalize(self) -> OptimizationResult:
        """Hook pour finalisation (optionnel)"""
        df = pd.DataFrame(self.all_results)
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_results=df,
            iterations=len(self.all_results),
            duration_sec=0,
            convergence_history=self.convergence_history,
        )

    def optimize(self, max_iterations: int) -> OptimizationResult:
        """Template method (ne pas override)"""
        start_time = time.time()

        self.prepare_data()

        for iteration in range(max_iterations):
            try:
                params, score = self.run_iteration(iteration)

                # Enregistrer
                result_row = params.copy()
                result_row["score"] = score
                result_row["iteration"] = iteration
                self.all_results.append(result_row)

                # Mettre à jour best
                improved = self._update_best(params, score)
                self.convergence_history.append(self.best_score)

                # Early stopping
                if self.early_stopping and not improved:
                    self.iterations_without_improvement += 1
                    if self.iterations_without_improvement >= self.early_stopping:
                        break
                else:
                    self.iterations_without_improvement = 0

            except Exception as e:
                if self.verbose:
                    print(f"Iteration {iteration} failed: {e}")
                continue

        result = self.finalize()
        result.duration_sec = time.time() - start_time
        return result

    def _update_best(self, params: Dict, score: float) -> bool:
        """Retourne True si amélioration"""
        is_better = (self.maximize and score > self.best_score + self.tolerance) or (
            not self.maximize and score < self.best_score - self.tolerance
        )

        if is_better:
            self.best_score = score
            self.best_params = params.copy()
            return True
        return False


class GridOptimizer(BaseOptimizer):
    """Grid Search exhaustif"""

    def __init__(self, param_grid: Dict, **kwargs):
        super().__init__(**kwargs)
        if not param_grid:
            raise ValueError("param_grid cannot be empty")
        self.param_grid = param_grid
        self.combinations = []

    def prepare_data(self):
        """Génère toutes les combinaisons"""
        import itertools

        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]

        for vals in values:
            if not vals:
                raise ValueError(f"Param must have at least one value")

        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            self.combinations.append(params)

    def run_iteration(self, iteration: int) -> Tuple[Dict, float]:
        if iteration >= len(self.combinations):
            raise StopIteration()

        params = self.combinations[iteration]
        score = self.objective_fn(params)
        return params, score

    def optimize(self, max_iterations: int = None):
        """Override pour utiliser toutes les combinaisons"""
        # Préparer d'abord pour avoir combinations
        if not self.combinations:
            self.prepare_data()
        if max_iterations is None:
            max_iterations = len(self.combinations)
        return super().optimize(max_iterations)


class MonteCarloOptimizer(BaseOptimizer):
    """Random Search (Monte Carlo)"""

    def __init__(
        self, param_ranges: Dict, n_trials: int = 100, seed: int = None, **kwargs
    ):
        super().__init__(**kwargs)
        if not param_ranges:
            raise ValueError("param_ranges cannot be empty")
        self.param_ranges = param_ranges
        self.n_trials = n_trials
        self.rng = np.random.RandomState(seed)

    def prepare_data(self):
        """Validation des ranges"""
        for param, (min_val, max_val) in self.param_ranges.items():
            if min_val >= max_val:
                raise ValueError(f"Invalid range for {param}: {min_val} >= {max_val}")

    def run_iteration(self, iteration: int) -> Tuple[Dict, float]:
        params = self._sample_params()
        score = self.objective_fn(params)
        return params, score

    def _sample_params(self) -> Dict:
        """Sample aléatoire"""
        params = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            # Détection int vs float
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param] = self.rng.randint(min_val, max_val + 1)
            else:
                params[param] = self.rng.uniform(min_val, max_val)
        return params

    def optimize(self, max_iterations: int = None):
        """Override pour utiliser n_trials"""
        if max_iterations is None:
            max_iterations = self.n_trials
        return super().optimize(max_iterations)


# ===== Tests =====


def test_grid_optimizer():
    """Test Grid Search"""

    def objective(params):
        return -((params["x"] - 5) ** 2) - (params["y"] - 3) ** 2

    optimizer = GridOptimizer(
        param_grid={"x": [1, 3, 5, 7, 9], "y": [1, 2, 3, 4, 5]},
        objective_fn=objective,
        maximize=True,
        verbose=False,
    )

    result = optimizer.optimize()

    print(f"\n✅ Grid Search: {result.iterations} iterations")
    print(f"   Best params: {result.best_params}")
    print(f"   Best score: {result.best_score:.4f}")

    assert result.iterations == 25  # 5 * 5
    assert result.best_params["x"] == 5
    assert result.best_params["y"] == 3


def test_monte_carlo_optimizer():
    """Test Monte Carlo"""

    def objective(params):
        return -((params["x"] - 5) ** 2) - (params["y"] - 3) ** 2

    optimizer = MonteCarloOptimizer(
        param_ranges={"x": (0, 10), "y": (0, 10)},
        objective_fn=objective,
        n_trials=50,
        maximize=True,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    print(f"\n✅ Monte Carlo: {result.iterations} iterations")
    print(f"   Best params: {result.best_params}")
    print(f"   Best score: {result.best_score:.4f}")

    assert result.iterations == 50
    assert abs(result.best_params["x"] - 5) < 2
    assert abs(result.best_params["y"] - 3) < 2


def test_early_stopping():
    """Test Early Stopping"""

    def objective(params):
        return 1.0  # Constant, pas d'amélioration

    optimizer = GridOptimizer(
        param_grid={"x": list(range(20))},
        objective_fn=objective,
        maximize=True,
        verbose=False,
        early_stopping=3,
    )

    result = optimizer.optimize()

    print(f"\n✅ Early Stopping: {result.iterations} iterations (stopped early)")
    assert result.iterations <= 5  # 1 best + 3 sans amélioration


def test_error_handling():
    """Test gestion erreurs"""

    def failing_objective(params):
        if params["x"] < 0:
            raise ValueError("x must be positive")
        return params["x"]

    optimizer = GridOptimizer(
        param_grid={"x": [-2, -1, 0, 1, 2]},
        objective_fn=failing_objective,
        maximize=True,
        verbose=False,
    )

    result = optimizer.optimize()

    print(f"\n✅ Error Handling: {result.iterations} iterations completed")
    print(f"   Best params: {result.best_params}")

    assert result.best_params["x"] >= 0  # Seuls les positifs ont réussi


if __name__ == "__main__":
    print("=" * 60)
    print("Tests Standalone des Templates d'Optimisation")
    print("=" * 60)

    test_grid_optimizer()
    test_monte_carlo_optimizer()
    test_early_stopping()
    test_error_handling()

    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS PASSED!")
    print("=" * 60)
