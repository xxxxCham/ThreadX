"""
ThreadX Scenario Generation - Monte Carlo & Grid
===============================================

Générateur de scénarios pour sweeps paramétriques avec support :
- Grilles déterministes avec dé-duplication
- Monte Carlo seedé (Latin Hypercube, Sobol, Random)
- Sérialisation JSON canonique et ordre stable
- Contraintes paramétriques simples

Author: ThreadX Framework
Version: Phase 10 - Scenario Generation
"""

import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Union, Optional
from typing_extensions import TypedDict
import itertools
from dataclasses import dataclass

from threadx.utils.log import get_logger

logger = get_logger(__name__)


class ScenarioSpec(TypedDict):
    """
    Spécification d'un scénario de génération paramétrique.
    
    Attributes:
        type: Type de génération ("grid" ou "monte_carlo")
        params: Paramètres et leurs plages de valeurs
        seed: Seed pour reproductibilité
        n_scenarios: Nombre de scénarios (pour Monte Carlo)
        sampler: Méthode d'échantillonnage ("latin_hypercube", "sobol", "random")
        constraints: Liste de contraintes sous forme de strings Python
    """
    type: str  # "grid" | "monte_carlo"
    params: Dict[str, Any]
    seed: int
    n_scenarios: int
    sampler: str  # "latin_hypercube", "sobol", "random"
    constraints: List[str]


def generate_param_grid(spec: ScenarioSpec) -> List[Dict[str, Any]]:
    """
    Génère une grille de paramètres déterministe avec ordre stable.
    
    Args:
        spec: Spécification du scénario
        
    Returns:
        Liste de combinaisons paramétriques ordonnées et dé-dupliquées
        
    Example:
        >>> spec = {
        ...     "type": "grid",
        ...     "params": {
        ...         "bb_period": [10, 20, 30],
        ...         "bb_std": [1.5, 2.0, 2.5],
        ...         "atr_period": [14, 21]
        ...     },
        ...     "seed": 42,
        ...     "constraints": ["bb_period <= atr_period * 2"]
        ... }
        >>> combinations = generate_param_grid(spec)
        >>> len(combinations)  # 3 * 3 * 2 = 18 avant contraintes
    """
    logger.info(f"Génération grille paramétrique, seed={spec['seed']}")
    
    # Extraction des paramètres et valeurs
    param_names = sorted(spec['params'].keys())  # Ordre stable
    param_values = []
    
    for param_name in param_names:
        values = spec['params'][param_name]
        if isinstance(values, list):
            param_values.append(sorted(values))  # Tri pour stabilité
        elif isinstance(values, dict) and 'start' in values and 'stop' in values:
            # Range spécification
            start = values['start']
            stop = values['stop']
            step = values.get('step', 1.0)
            range_values = []
            current = start
            while current <= stop + 1e-10:
                range_values.append(current)
                current += step
            param_values.append(range_values)
        else:
            raise ValueError(f"Format invalide pour paramètre {param_name}: {values}")
    
    # Génération du produit cartésien
    combinations = []
    for combo_values in itertools.product(*param_values):
        combo_dict = {name: value for name, value in zip(param_names, combo_values)}
        combinations.append(combo_dict)
    
    logger.info(f"Grille initiale: {len(combinations)} combinaisons")
    
    # Application des contraintes
    if spec.get('constraints'):
        filtered_combinations = []
        for combo in combinations:
            if _evaluate_constraints(combo, spec['constraints']):
                filtered_combinations.append(combo)
        combinations = filtered_combinations
        logger.info(f"Après contraintes: {len(combinations)} combinaisons")
    
    # Dé-duplication par checksum
    unique_combinations = []
    seen_checksums = set()
    
    for combo in combinations:
        checksum = _combo_checksum(combo)
        if checksum not in seen_checksums:
            unique_combinations.append(combo)
            seen_checksums.add(checksum)
    
    logger.info(f"Après dé-duplication: {len(unique_combinations)} combinaisons finales")
    
    # Tri final pour ordre déterministe
    unique_combinations.sort(key=lambda x: _combo_sort_key(x))
    
    return unique_combinations


def generate_monte_carlo(spec: ScenarioSpec) -> List[Dict[str, Any]]:
    """
    Génère des scénarios Monte Carlo avec tirages seedés et streams indépendants.
    
    Args:
        spec: Spécification du scénario Monte Carlo
        
    Returns:
        Liste de scénarios avec paramètres échantillonnés
        
    Example:
        >>> spec = {
        ...     "type": "monte_carlo",
        ...     "params": {
        ...         "bb_period": {"min": 10, "max": 50, "type": "uniform"},
        ...         "bb_std": {"min": 1.0, "max": 3.0, "type": "normal", "mean": 2.0, "std": 0.3},
        ...         "atr_period": {"min": 7, "max": 28, "type": "uniform"}
        ...     },
        ...     "n_scenarios": 1000,
        ...     "sampler": "sobol",
        ...     "seed": 42
        ... }
        >>> scenarios = generate_monte_carlo(spec)
        >>> len(scenarios) == 1000
    """
    logger.info(f"Génération Monte Carlo: {spec['n_scenarios']} scénarios, "
               f"sampler={spec['sampler']}, seed={spec['seed']}")
    
    np.random.seed(spec['seed'])
    
    # Extraction des paramètres
    param_names = sorted(spec['params'].keys())
    n_params = len(param_names)
    n_scenarios = spec['n_scenarios']
    
    # Génération des échantillons selon la méthode
    sampler = spec.get('sampler', 'sobol')
    
    if sampler == 'latin_hypercube':
        samples = _latin_hypercube_sample(n_scenarios, n_params, spec['seed'])
    elif sampler == 'sobol':
        samples = _sobol_sample(n_scenarios, n_params, spec['seed'])
    elif sampler == 'random':
        samples = _random_sample(n_scenarios, n_params, spec['seed'])
    else:
        raise ValueError(f"Sampler non supporté: {sampler}")
    
    # Transformation des échantillons [0,1] vers les plages paramétriques
    scenarios = []
    for i, sample in enumerate(samples):
        scenario = {}
        
        for j, param_name in enumerate(param_names):
            param_spec = spec['params'][param_name]
            value = _transform_sample_value(sample[j], param_spec)
            scenario[param_name] = value
        
        # Application des contraintes
        if not spec.get('constraints') or _evaluate_constraints(scenario, spec['constraints']):
            scenarios.append(scenario)
    
    logger.info(f"Monte Carlo généré: {len(scenarios)} scénarios valides "
               f"(après contraintes sur {n_scenarios} tentatives)")
    
    return scenarios


def _latin_hypercube_sample(n_scenarios: int, n_params: int, seed: int) -> np.ndarray:
    """Échantillonnage Latin Hypercube."""
    np.random.seed(seed)
    
    samples = np.zeros((n_scenarios, n_params))
    
    for j in range(n_params):
        # Stratification uniforme
        intervals = np.arange(n_scenarios, dtype=float) / n_scenarios
        intervals += np.random.rand(n_scenarios) / n_scenarios
        
        # Permutation aléatoire
        np.random.shuffle(intervals)
        samples[:, j] = intervals
    
    return samples


def _sobol_sample(n_scenarios: int, n_params: int, seed: int) -> np.ndarray:
    """Échantillonnage Sobol (quasi-aléatoire)."""
    # Implémentation Sobol simplifiée pour éviter dépendances externes
    np.random.seed(seed)
    
    # Pour cette version, utilise stratified random comme approximation
    # TODO: Implémenter vraie séquence Sobol si besoin de précision
    samples = np.zeros((n_scenarios, n_params))
    
    for j in range(n_params):
        # Approximation par échantillonnage stratifié avec perturbation
        base_points = np.linspace(0, 1, n_scenarios + 1)[:-1]
        perturbations = np.random.rand(n_scenarios) / n_scenarios
        samples[:, j] = base_points + perturbations
    
    return samples


def _random_sample(n_scenarios: int, n_params: int, seed: int) -> np.ndarray:
    """Échantillonnage aléatoire uniforme."""
    np.random.seed(seed)
    return np.random.rand(n_scenarios, n_params)


def _transform_sample_value(sample: float, param_spec: Dict[str, Any]) -> Union[int, float]:
    """Transforme un échantillon [0,1] selon la spécification du paramètre."""
    param_type = param_spec.get('type', 'uniform')
    
    if param_type == 'uniform':
        min_val = param_spec['min']
        max_val = param_spec['max']
        value = min_val + sample * (max_val - min_val)
        
        # Conversion entier si nécessaire
        if isinstance(min_val, int) and isinstance(max_val, int):
            return int(round(value))
        return value
        
    elif param_type == 'normal':
        # Transformation inverse normale
        mean = param_spec.get('mean', 0.0)
        std = param_spec.get('std', 1.0)
        
        # Approximation inverse normale via Box-Muller
        if sample < 1e-10:
            sample = 1e-10
        elif sample > 1 - 1e-10:
            sample = 1 - 1e-10
            
        # Transformation inverse normale simplifiée
        z_score = np.sqrt(-2 * np.log(sample)) * np.cos(2 * np.pi * sample)
        value = mean + std * z_score
        
        # Clipping si min/max spécifiés
        if 'min' in param_spec:
            value = max(value, param_spec['min'])
        if 'max' in param_spec:
            value = min(value, param_spec['max'])
            
        return value
        
    elif param_type == 'lognormal':
        # Log-normale
        mu = param_spec.get('mu', 0.0)
        sigma = param_spec.get('sigma', 1.0)
        
        # Via normale puis exp
        normal_sample = np.sqrt(-2 * np.log(sample)) * np.cos(2 * np.pi * sample)
        value = np.exp(mu + sigma * normal_sample)
        
        if 'min' in param_spec:
            value = max(value, param_spec['min'])
        if 'max' in param_spec:
            value = min(value, param_spec['max'])
            
        return value
    
    else:
        raise ValueError(f"Type de paramètre non supporté: {param_type}")


def _evaluate_constraints(combo: Dict[str, Any], constraints: List[str]) -> bool:
    """Évalue les contraintes sur une combinaison de paramètres."""
    if not constraints:
        return True
    
    try:
        # Création du namespace d'évaluation
        namespace = combo.copy()
        namespace.update({
            'abs': abs,
            'min': min,
            'max': max,
            'round': round,
            'int': int,
            'float': float
        })
        
        # Évaluation de chaque contrainte
        for constraint in constraints:
            if not eval(constraint, {"__builtins__": {}}, namespace):
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Erreur évaluation contrainte '{constraint}': {e}")
        return False


def _combo_checksum(combo: Dict[str, Any]) -> str:
    """Calcule un checksum stable pour une combinaison."""
    # Sérialisation JSON canonique
    combo_json = json.dumps(combo, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(combo_json.encode()).hexdigest()


def _combo_sort_key(combo: Dict[str, Any]) -> tuple:
    """Génère une clé de tri stable pour une combinaison."""
    # Tri par noms de paramètres puis valeurs
    items = sorted(combo.items())
    return tuple(str(value) for key, value in items)


# === Utilitaires de validation ===

def validate_scenario_spec(spec: Dict[str, Any]) -> ScenarioSpec:
    """
    Valide et normalise une spécification de scénario.
    
    Args:
        spec: Spécification brute
        
    Returns:
        ScenarioSpec validée
        
    Raises:
        ValueError: Si spécification invalide
    """
    # Valeurs par défaut
    validated = {
        'type': spec.get('type', 'grid'),
        'params': spec.get('params', {}),
        'seed': spec.get('seed', 42),
        'n_scenarios': spec.get('n_scenarios', 100),
        'sampler': spec.get('sampler', 'sobol' if spec.get('type') == 'monte_carlo' else 'grid'),
        'constraints': spec.get('constraints', [])
    }
    
    # Validations
    if validated['type'] not in ['grid', 'monte_carlo']:
        raise ValueError(f"Type invalide: {validated['type']}")
    
    if not validated['params']:
        raise ValueError("Paramètres manquants")
    
    if validated['type'] == 'monte_carlo' and validated['n_scenarios'] <= 0:
        raise ValueError("n_scenarios doit être > 0 pour Monte Carlo")
    
    if validated['sampler'] not in ['latin_hypercube', 'sobol', 'random', 'grid']:
        raise ValueError(f"Sampler non supporté: {validated['sampler']}")
    
    return ScenarioSpec(**validated)


if __name__ == "__main__":
    # Test rapide
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
    print(f"Test grille: {len(combinations)} combinaisons générées")
