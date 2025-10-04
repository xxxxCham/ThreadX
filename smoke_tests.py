#!/usr/bin/env python3
"""
Test Smoke ThreadX Optimization - Validation Rapide
==================================================

Tests de validation rapide des fonctionnalités d'optimisation ThreadX.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration du chemin Python
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_scenario_generation():
    """Test génération de scénarios."""
    print("🔍 Test génération de scénarios...")
    
    from threadx.optimization.scenarios import generate_param_grid, generate_monte_carlo, ScenarioSpec
    
    # Test grille
    grid_spec = ScenarioSpec(
        type="grid",
        params={
            "bb_period": [10, 20],
            "bb_std": [1.5, 2.0]
        },
        seed=42,
        n_scenarios=100,
        sampler="grid",
        constraints=[]
    )
    
    grid_combos = generate_param_grid(grid_spec)
    assert len(grid_combos) == 4, f"Expected 4 combinations, got {len(grid_combos)}"
    
    # Test Monte Carlo
    mc_spec = ScenarioSpec(
        type="monte_carlo",
        params={
            "bb_period": {"min": 10, "max": 30, "type": "uniform"},
            "bb_std": {"min": 1.0, "max": 3.0, "type": "uniform"}
        },
        seed=42,
        n_scenarios=50,
        sampler="sobol",
        constraints=[]
    )
    
    mc_combos = generate_monte_carlo(mc_spec)
    assert len(mc_combos) <= 50, f"Too many MC combinations: {len(mc_combos)}"
    
    print("✅ Génération de scénarios OK")


def test_pareto_pruning():
    """Test pruning Pareto."""
    print("🔍 Test pruning Pareto...")
    
    from threadx.optimization.pruning import pareto_soft_prune
    
    # Données de test
    test_df = pd.DataFrame({
        'pnl': [100, 150, 80, 200, 120, 90, 160],
        'max_drawdown': [0.1, 0.15, 0.08, 0.12, 0.09, 0.20, 0.11],
        'sharpe': [1.2, 1.8, 0.9, 2.1, 1.4, 0.8, 1.6]
    })
    
    pruned_df, metadata = pareto_soft_prune(test_df, patience=3, quantile=0.7)
    
    assert len(pruned_df) <= len(test_df), "Pruning should reduce or maintain size"
    assert 'pruned_count' in metadata, "Metadata should contain pruned_count"
    
    print("✅ Pruning Pareto OK")


def test_reporting():
    """Test fonctions de reporting."""
    print("🔍 Test reporting...")
    
    from threadx.optimization.reporting import summarize_distribution, build_heatmaps
    
    # Données de test
    test_df = pd.DataFrame({
        'bb_period': [10, 20, 30, 10, 20],
        'bb_std': [1.5, 1.5, 1.5, 2.0, 2.0],
        'pnl': [100, 150, 120, 80, 200],
        'sharpe': [1.2, 1.8, 1.4, 0.9, 2.1]
    })
    
    # Test distribution summary
    stats = summarize_distribution(test_df)
    assert 'pnl' in stats, "PnL stats should be present"
    assert 'mean' in stats['pnl'], "Mean should be calculated"
    
    # Test heatmaps
    heatmaps = build_heatmaps(test_df)
    # Au moins une heatmap devrait être générée
    
    print("✅ Reporting OK")


def test_determinism():
    """Test déterminisme."""
    print("🔍 Test déterminisme...")
    
    from threadx.utils.determinism import set_global_seed, stable_hash, enforce_deterministic_merges
    
    # Test seed global
    set_global_seed(42)
    sample1 = np.random.rand(5)
    
    set_global_seed(42)
    sample2 = np.random.rand(5)
    
    assert np.allclose(sample1, sample2), "Seeds should produce identical samples"
    
    # Test hash stable
    hash1 = stable_hash({'b': 2, 'a': 1})
    hash2 = stable_hash({'a': 1, 'b': 2})
    assert hash1 == hash2, "Stable hash should ignore key order"
    
    # Test merge déterministe
    df1 = pd.DataFrame({'a': [3, 1], 'b': ['c', 'a']})
    df2 = pd.DataFrame({'a': [2, 4], 'b': ['b', 'd']})
    
    merged = enforce_deterministic_merges([df1, df2])
    assert len(merged) == 4, "Merge should combine all rows"
    
    print("✅ Déterminisme OK")


def test_integration():
    """Test d'intégration simple."""
    print("🔍 Test d'intégration...")
    
    from threadx.optimization.scenarios import generate_param_grid, ScenarioSpec
    from threadx.optimization.pruning import pareto_soft_prune
    from threadx.utils.determinism import set_global_seed
    
    # Configuration globale
    set_global_seed(42)
    
    # Génération de scénarios
    spec = ScenarioSpec(
        type="grid",
        params={
            "param1": [1, 2, 3],
            "param2": [0.1, 0.2]
        },
        seed=42,
        n_scenarios=100,
        sampler="grid",
        constraints=[]
    )
    
    scenarios = generate_param_grid(spec)
    
    # Simulation de résultats
    results = []
    for scenario in scenarios:
        result = scenario.copy()
        result.update({
            'pnl': np.random.randn() * 100 + 50,
            'sharpe': np.random.randn() * 0.5 + 1.0,
            'max_drawdown': abs(np.random.randn() * 0.1 + 0.05)
        })
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Pruning
    pruned_df, metadata = pareto_soft_prune(results_df, patience=10)
    
    assert len(pruned_df) <= len(results_df), "Integration should work end-to-end"
    
    print("✅ Intégration OK")


def main():
    """Exécution des tests smoke."""
    print("🚀 ThreadX Pro V2 - Tests Smoke Optimization")
    print("=" * 50)
    
    try:
        test_scenario_generation()
        test_pareto_pruning()
        test_reporting()
        test_determinism()
        test_integration()
        
        print("\n" + "=" * 50)
        print("🎉 Tous les tests smoke réussis!")
        print("\n📋 Résumé des validations:")
        print("  ✅ Génération de scénarios (grille + Monte Carlo)")
        print("  ✅ Pruning Pareto soft avec early stopping")
        print("  ✅ Rapports quantitatifs et heatmaps")
        print("  ✅ Déterminisme et seeds globaux")
        print("  ✅ Intégration bout-en-bout")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur dans les tests: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())