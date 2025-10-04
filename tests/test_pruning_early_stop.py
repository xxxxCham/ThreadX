#!/usr/bin/env python3
"""
ThreadX Test - Pruning Pareto
=============================

Validation du système de pruning Pareto soft:
- Efficacité du pruning
- Préservation des solutions dominantes
- Performance du pruning adaptatif
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.optimization.pruning import pareto_soft_prune
from threadx.utils.determinism import set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)

SEED_GLOBAL = 42


@pytest.fixture
def synthetic_results():
    """Génère des résultats synthétiques pour tests de pruning."""
    set_global_seed(SEED_GLOBAL)
    
    n_scenarios = 1000
    
    # Simulation résultats backtests avec trade-offs réalistes
    sharpe_ratios = np.random.gamma(2, 0.5, n_scenarios)  # Positifs, asymétriques
    max_drawdowns = np.random.gamma(3, 0.1, n_scenarios)  # Positifs (à minimiser)
    returns = np.random.normal(0.08, 0.15, n_scenarios)  # Rendements annualisés
    
    # Corrélation réaliste: drawdown vs sharpe négatif
    for i in range(n_scenarios):
        if np.random.random() < 0.3:  # 30% corrélation négative
            max_drawdowns[i] = max_drawdowns[i] * (2 - sharpe_ratios[i])
    
    results = []
    for i in range(n_scenarios):
        results.append({
            'scenario_id': i,
            'params': {'param_a': np.random.uniform(0.1, 5.0),
                      'param_b': np.random.uniform(10, 100)},
            'metrics': {
                'sharpe_ratio': sharpe_ratios[i],
                'max_drawdown': max_drawdowns[i],  # À minimiser
                'annual_return': returns[i],
                'win_rate': np.random.uniform(0.3, 0.8)
            }
        })
    
    return results


class TestParetoPruning:
    """Tests du système de pruning Pareto."""
    
    def test_pareto_efficiency_basic(self, synthetic_results):
        """Test efficacité basique du pruning Pareto."""
        logger.info("Test efficacité pruning Pareto")
        
        # Configuration pruning
        metrics_config = [
            {'name': 'sharpe_ratio', 'direction': 'maximize'},
            {'name': 'max_drawdown', 'direction': 'minimize'},
            {'name': 'annual_return', 'direction': 'maximize'}
        ]
        
        initial_count = len(synthetic_results)
        
        # Application pruning
        start = time.perf_counter()
        pruned_results, metadata = pareto_soft_prune(
            synthetic_results,
            metrics_config,
            patience=50,
            quantile_threshold=0.9
        )
        pruning_duration = time.perf_counter() - start
        
        pruned_count = len(pruned_results)
        pruning_ratio = 1 - (pruned_count / initial_count)
        
        # Vérifications
        assert pruned_count > 0, "Pruning trop agressif - aucun résultat conservé"
        assert pruned_count <= initial_count, "Pruning a ajouté des résultats"
        assert pruning_ratio > 0.1, f"Pruning insuffisant: {pruning_ratio:.2%}"
        assert pruning_ratio < 0.9, f"Pruning trop agressif: {pruning_ratio:.2%}"
        
        logger.info(f"Pruning: {initial_count} → {pruned_count} "
                   f"({pruning_ratio:.1%} éliminés en {pruning_duration:.4f}s)")
        
        # Validation qualité des solutions conservées
        pruned_sharpe = [r['metrics']['sharpe_ratio'] for r in pruned_results]
        pruned_drawdown = [r['metrics']['max_drawdown'] for r in pruned_results]
        pruned_return = [r['metrics']['annual_return'] for r in pruned_results]
        
        # Les solutions conservées doivent être dans les meilleurs quantiles
        all_sharpe = [r['metrics']['sharpe_ratio'] for r in synthetic_results]
        all_drawdown = [r['metrics']['max_drawdown'] for r in synthetic_results]
        all_return = [r['metrics']['annual_return'] for r in synthetic_results]
        
        avg_pruned_sharpe = np.mean(pruned_sharpe)
        avg_all_sharpe = np.mean(all_sharpe)
        
        assert avg_pruned_sharpe >= avg_all_sharpe, \
            "Solutions conservées pas meilleures que moyenne"
        
        results_summary = {
            'test': 'pareto_efficiency_basic',
            'initial_count': initial_count,
            'pruned_count': pruned_count,
            'pruning_ratio': pruning_ratio,
            'pruning_duration': pruning_duration,
            'avg_sharpe_before': avg_all_sharpe,
            'avg_sharpe_after': avg_pruned_sharpe,
            'sharpe_improvement': avg_pruned_sharpe / avg_all_sharpe,
            'metadata': str(metadata)
        }
        
        # Sauvegarde
        output_path = Path("artifacts/reports/test_pareto_pruning.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([results_summary]).to_csv(output_path, index=False)
        
        logger.info(f"Amélioration Sharpe: {results_summary['sharpe_improvement']:.2f}x")
    
    def test_pareto_preserves_non_dominated(self, synthetic_results):
        """Test que le pruning préserve les solutions non-dominées."""
        logger.info("Test préservation solutions non-dominées")
        
        # Ajout de quelques solutions clairement non-dominées
        elite_solutions = [
            {
                'scenario_id': 9999,
                'params': {'param_a': 1.0, 'param_b': 50},
                'metrics': {
                    'sharpe_ratio': 3.0,  # Très bon
                    'max_drawdown': 0.05,  # Très faible
                    'annual_return': 0.25,  # Très bon
                    'win_rate': 0.75
                }
            },
            {
                'scenario_id': 9998,
                'params': {'param_a': 2.0, 'param_b': 75},
                'metrics': {
                    'sharpe_ratio': 2.8,  # Excellent sharpe
                    'max_drawdown': 0.08,  # Drawdown un peu plus élevé
                    'annual_return': 0.30,  # Rendement supérieur
                    'win_rate': 0.70
                }
            }
        ]
        
        enhanced_results = synthetic_results + elite_solutions
        
        metrics_config = [
            {'name': 'sharpe_ratio', 'direction': 'maximize'},
            {'name': 'max_drawdown', 'direction': 'minimize'},
            {'name': 'annual_return', 'direction': 'maximize'}
        ]
        
        # Pruning
        pruned_results, metadata = pareto_soft_prune(
            enhanced_results,
            metrics_config,
            patience=100,
            quantile_threshold=0.85
        )
        
        # Vérification que les solutions élites sont conservées
        pruned_ids = {r['scenario_id'] for r in pruned_results}
        elite_ids = {r['scenario_id'] for r in elite_solutions}
        
        preserved_elite = elite_ids.intersection(pruned_ids)
        preservation_rate = len(preserved_elite) / len(elite_ids)
        
        logger.info(f"Solutions élites conservées: {len(preserved_elite)}/{len(elite_ids)} "
                   f"({preservation_rate:.1%})")
        
        # Au moins 50% des solutions élites doivent être conservées
        assert preservation_rate >= 0.5, \
            f"Trop de solutions élites éliminées: {preservation_rate:.1%}"
    
    def test_pruning_performance_scaling(self):
        """Test performance du pruning avec différentes tailles."""
        logger.info("Test performance scaling pruning")
        
        sizes = [100, 500, 1000, 2000]
        performance_data = []
        
        for size in sizes:
            set_global_seed(SEED_GLOBAL)
            
            # Génération dataset de taille variable
            results = []
            for i in range(size):
                results.append({
                    'scenario_id': i,
                    'params': {'param': np.random.uniform(0, 10)},
                    'metrics': {
                        'metric1': np.random.gamma(2, 1),
                        'metric2': np.random.gamma(1, 2),
                        'metric3': np.random.normal(0, 1)
                    }
                })
            
            metrics_config = [
                {'name': 'metric1', 'direction': 'maximize'},
                {'name': 'metric2', 'direction': 'minimize'},
                {'name': 'metric3', 'direction': 'maximize'}
            ]
            
            # Mesure performance
            start = time.perf_counter()
            pruned_results, metadata = pareto_soft_prune(
                results, metrics_config, patience=50
            )
            duration = time.perf_counter() - start
            
            pruning_ratio = 1 - (len(pruned_results) / len(results))
            
            performance_data.append({
                'input_size': size,
                'output_size': len(pruned_results),
                'pruning_ratio': pruning_ratio,
                'duration': duration,
                'throughput': size / duration if duration > 0 else 0
            })
            
            logger.info(f"Size {size}: {duration:.4f}s, "
                       f"throughput: {size/duration:.0f} items/s")
        
        # Analyse scaling
        perf_df = pd.DataFrame(performance_data)
        
        # Sauvegarde
        output_path = Path("artifacts/reports/test_pruning_performance.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        perf_df.to_csv(output_path, index=False)
        
        # Vérification scaling acceptable (pas pire que O(n²))
        if len(performance_data) >= 2:
            max_size = max(sizes)
            min_size = min(sizes)
            
            max_duration = perf_df[perf_df['input_size'] == max_size]['duration'].iloc[0]
            min_duration = perf_df[perf_df['input_size'] == min_size]['duration'].iloc[0]
            
            scaling_factor = max_duration / min_duration
            size_factor = max_size / min_size
            
            # Le temps ne doit pas croître plus vite que O(n²)
            worst_case_scaling = size_factor ** 2
            
            assert scaling_factor <= worst_case_scaling * 2, \
                f"Scaling trop mauvais: {scaling_factor:.2f} vs {worst_case_scaling:.2f}"
            
            logger.info(f"Scaling factor: {scaling_factor:.2f} "
                       f"(acceptable: <{worst_case_scaling:.2f})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
