#!/usr/bin/env python3
"""
Test Script ThreadX Optimization
===============================

Script de test simple pour valider l'intégration des modules d'optimisation.
"""

import sys
from pathlib import Path

# Ajout du chemin ThreadX
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from threadx.optimization.scenarios import generate_param_grid, ScenarioSpec
    from threadx.optimization.pruning import pareto_soft_prune
    from threadx.optimization.reporting import summarize_distribution
    from threadx.utils.determinism import set_global_seed
    
    print("✅ Tous les modules d'optimisation importés avec succès")
    
    # Test scenario generation
    test_spec = ScenarioSpec(
        type="grid",
        params={
            "bb_period": [10, 20, 30],
            "bb_std": [1.5, 2.0, 2.5]
        },
        seed=42,
        n_scenarios=100,
        sampler="grid",
        constraints=[]
    )
    
    combinations = generate_param_grid(test_spec)
    print(f"✅ Génération de grille: {len(combinations)} combinaisons")
    
    # Test determinism
    set_global_seed(42)
    print("✅ Seed global configuré")
    
    print("\n🎉 ThreadX Pro V2 Étapes 4→5 - Validation réussie!")
    print("📦 Modules disponibles:")
    print("  - scenarios.py: Génération Monte Carlo & grilles")
    print("  - pruning.py: Pruning Pareto soft")  
    print("  - reporting.py: Analyses et rapports")
    print("  - determinism.py: Seeds globaux & merges stables")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur de test: {e}")
    sys.exit(1)