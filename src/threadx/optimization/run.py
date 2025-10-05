#!/usr/bin/env python3
"""
ThreadX Optimization Runner - CLI Interface
==========================================

Interface en ligne de commande pour l'exécution de sweeps paramétriques.
Support des configurations TOML et mode dry-run pour validation.

Usage:
    python -m threadx.optimization.run --config configs/sweeps/bb_atr_grid.toml
    python -m threadx.optimization.run --config configs/sweeps/bb_atr_montecarlo.toml --dry-run
"""

import argparse
import sys
import time
from typing import Dict, Any

from threadx.config import ConfigurationError, load_config_dict
from threadx.indicators.bank import IndicatorBank
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec, validate_scenario_spec
from threadx.optimization.reporting import write_reports
from threadx.utils.log import get_logger
from threadx.utils.determinism import set_global_seed

logger = get_logger(__name__)

def validate_config(config: Dict[str, Any]) -> ScenarioSpec:
    """Valide et convertit la configuration en ScenarioSpec."""
    # Extraction des sections
    dataset = config.get('dataset', {})
    scenario = config.get('scenario', {})
    params = config.get('params', {})
    constraints = config.get('constraints', {})
    
    # Construction de la spec
    spec_dict = {
        'type': scenario.get('type', 'grid'),
        'params': params,
        'seed': scenario.get('seed', 42),
        'n_scenarios': scenario.get('n_scenarios', 100),
        'sampler': scenario.get('sampler', 'sobol' if scenario.get('type') == 'monte_carlo' else 'grid'),
        'constraints': constraints.get('rules', [])
    }
    
    # Validation
    try:
        spec = validate_scenario_spec(spec_dict)
        logger.info(f"Configuration validée: {spec['type']} avec {len(spec['params'])} paramètres")
        return spec
    except Exception as e:
        logger.error(f"Configuration invalide: {e}")
        raise


def run_sweep(config: Dict[str, Any], dry_run: bool = False) -> None:
    """Exécute le sweep selon la configuration."""
    # Validation de la configuration
    scenario_spec = validate_config(config)
    
    # Configuration globale
    execution = config.get('execution', {})
    output = config.get('output', {})
    
    # Seed global pour déterminisme
    set_global_seed(scenario_spec['seed'])
    
    if dry_run:
        logger.info("=== MODE DRY RUN ===")
        logger.info(f"Type de sweep: {scenario_spec['type']}")
        logger.info(f"Paramètres: {list(scenario_spec['params'].keys())}")
        logger.info(f"Seed: {scenario_spec['seed']}")
        logger.info(f"GPU activé: {execution.get('use_gpu', False)}")
        logger.info(f"Cache réutilisé: {execution.get('reuse_cache', True)}")
        
        if scenario_spec['type'] == 'monte_carlo':
            logger.info(f"Scénarios Monte Carlo: {scenario_spec['n_scenarios']}")
            logger.info(f"Sampler: {scenario_spec['sampler']}")
        
        logger.info("Configuration valide - prêt pour exécution")
        return
    
    # Initialisation des composants
    logger.info("Initialisation du runner de sweep...")
    
    indicator_bank = IndicatorBank()
    sweep_runner = SweepRunner(
        indicator_bank=indicator_bank,
        max_workers=execution.get('max_workers', 4)
    )
    
    # Exécution du sweep
    start_time = time.time()
    
    try:
        if scenario_spec['type'] == 'grid':
            logger.info("Exécution du sweep de grille...")
            results_df = sweep_runner.run_grid(
                scenario_spec,
                reuse_cache=execution.get('reuse_cache', True)
            )
        
        elif scenario_spec['type'] == 'monte_carlo':
            logger.info("Exécution du sweep Monte Carlo...")
            results_df = sweep_runner.run_monte_carlo(
                scenario_spec,
                reuse_cache=execution.get('reuse_cache', True)
            )
        
        else:
            raise ValueError(f"Type de sweep non supporté: {scenario_spec['type']}")
        
        # Génération des rapports
        if not results_df.empty:
            reports_dir = output.get('reports_dir', 'artifacts/reports')
            
            logger.info(f"Génération des rapports: {len(results_df)} résultats → {reports_dir}")
            
            created_files = write_reports(
                results_df,
                reports_dir,
                seeds=[scenario_spec['seed']],
                devices=['CPU', 'GPU'],  # TODO: Récupérer les vrais devices
                gpu_ratios={'5090': 0.75, '2060': 0.25},  # TODO: Récupérer les vrais ratios
                min_samples=execution.get('min_samples', 1000)
            )
            
            logger.info("Rapports générés:")
            for file_type, file_path in created_files.items():
                if isinstance(file_path, str):
                    logger.info(f"  {file_type}: {file_path}")
                else:
                    logger.info(f"  {file_type}: {file_path}")
        
        else:
            logger.warning("Aucun résultat généré")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise
    
    finally:
        execution_time = time.time() - start_time
        logger.info(f"Sweep terminé en {execution_time:.1f}s")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="ThreadX Optimization Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m threadx.optimization.run --config configs/sweeps/bb_atr_grid.toml
  python -m threadx.optimization.run --config configs/sweeps/bb_atr_montecarlo.toml --dry-run
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Chemin vers le fichier de configuration TOML'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mode dry run - valide la configuration sans exécuter'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbose'
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    if args.verbose:
        import logging
        logging.getLogger('threadx').setLevel(logging.DEBUG)
    
    try:
        # Chargement et exécution
        config = load_config_dict(args.config)
        logger.info(f"Configuration chargée: {args.config}")
        run_sweep(config, dry_run=args.dry_run)

        if not args.dry_run:
            logger.info("✅ Sweep terminé avec succès")

    except ConfigurationError as e:
        logger.error(f"❌ Erreur configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()