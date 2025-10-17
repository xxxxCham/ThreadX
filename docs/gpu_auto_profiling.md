# Système d'Auto-Profilage GPU

Ce document décrit le système d'auto-profilage GPU et multi-GPU mis en œuvre dans ThreadX. Ce système améliore l'utilisation des ressources GPU en automatisant les décisions concernant quand et comment utiliser les GPUs disponibles.

## 1. Architecture Globale

Le système d'auto-profilage est composé de trois sous-systèmes principaux :

1. **Système de décision dynamique CPU/GPU** : Détermine automatiquement s'il est préférable d'utiliser le CPU ou le GPU pour une opération donnée en fonction de profils de performance antérieurs.

2. **Système d'auto-balance multi-GPU** : Optimise la distribution des charges de travail entre plusieurs GPUs en fonction de leurs capacités relatives.

3. **Système de fusion déterministe** : Garantit que les calculs distribués sur plusieurs GPUs produisent des résultats identiques entre chaque exécution.

## 2. Système de Décision Dynamique CPU/GPU

### Fonctionnement

Le système détermine automatiquement s'il faut utiliser le CPU ou le GPU pour une opération spécifique en se basant sur les performances historiques. Pour chaque opération, le système :

1. Enregistre le temps d'exécution sur CPU et GPU
2. Compare ces temps pour des opérations similaires
3. Choisit automatiquement le dispositif le plus rapide pour les futures exécutions

### Implémentation

Le cœur de l'implémentation réside dans le module `threadx.utils.gpu.profile_persistence` qui gère la persistance des profils de performance :

```python
# Exemple d'utilisation
from threadx.utils.gpu.profile_persistence import get_gpu_thresholds, update_gpu_threshold_entry

# Mettre à jour le profil avec une nouvelle mesure
update_gpu_threshold_entry(
    function_name="bollinger_bands", 
    params={"period": 20, "size": 10000},
    cpu_ms=150.0,
    gpu_ms=50.0
)

# Récupérer le profil actuel
profile = get_gpu_thresholds()
```

La décision d'utilisation est implémentée dans la fonction `_should_use_gpu_dynamic` qui analyse le profil pour déterminer le dispositif optimal.

### Format du Profil

Le profil de seuil GPU est stocké au format JSON avec la structure suivante :

```json
{
  "updated_at": "2023-07-15T14:30:22",
  "entries": {
    "bollinger_bands:{'period': 20, 'size': 10000}": {
      "cpu_ms_avg": 150.0,
      "gpu_ms_avg": 50.0,
      "samples": 3
    },
    "macd:{'slow': 26, 'fast': 12, 'size': 5000}": {
      "cpu_ms_avg": 20.0,
      "gpu_ms_avg": 35.0,
      "samples": 2
    }
  }
}
```

## 3. Système d'Auto-Balance Multi-GPU

### Fonctionnement

Le système d'auto-balance optimise la distribution des charges de travail entre plusieurs GPUs disponibles en :

1. Profilant les performances relatives des GPUs pour une charge de travail typique
2. Déterminant les ratios optimaux pour diviser les données
3. Distribuant les futurs calculs selon ces ratios optimisés

### Implémentation

L'implémentation se trouve dans le module `threadx.utils.gpu.multi_gpu` avec la méthode `profile_auto_balance` :

```python
# Exemple d'utilisation
from threadx.utils.gpu import get_default_manager

# Obtenir le manager multi-GPU
manager = get_default_manager()

# Profiler pour déterminer la balance optimale
optimized_balance = manager.profile_auto_balance(
    sample_size=100000,
    runs=3,
    workload_tag="bollinger_computation"
)

# Les futurs calculs utiliseront automatiquement cette balance optimisée
result = manager.distribute_workload(data, process_function)
```

### Format du Profil

Le profil multi-GPU est stocké au format JSON avec la structure suivante :

```json
{
  "updated_at": "2023-07-15T14:35:42",
  "workload_tag": "bollinger_computation",
  "ratios": {
    "RTX3090": 0.75,
    "GTX1660": 0.25
  }
}
```

## 4. Système de Fusion Déterministe

### Fonctionnement

Le système garantit que les résultats de calculs distribués sur plusieurs GPUs soient identiques entre les exécutions, indépendamment des variations de timing ou d'ordre d'exécution :

1. Utilise un seed aléatoire fixe pour garantir la reproductibilité
2. Applique une fusion déterministe des résultats partiels
3. Maintient l'ordre exact des opérations

### Implémentation

La fusion déterministe est implémentée dans la méthode `_merge_results_deterministic` de la classe `MultiGPUManager` :

```python
# Pour garantir des résultats déterministes, spécifier un seed
result = manager.distribute_workload(data, process_function, seed=42)
```

## 5. Configuration

### Variables d'Environnement

Les chemins des profils de performance peuvent être configurés via des variables d'environnement :

- `THREADX_GPU_THRESHOLD_PATH` : Chemin vers le fichier de profil de seuils GPU/CPU
- `THREADX_MULTIGPU_PROFILE_PATH` : Chemin vers le fichier de profil multi-GPU

### Valeurs par Défaut

Par défaut, les profils sont stockés dans :

- `artifacts/profiles/thresholds.json`
- `artifacts/profiles/multigpu.json`

## 6. Outils de Démonstration et Test

### Script de Démonstration

Un script de démonstration est disponible pour visualiser le fonctionnement du système :

```bash
python -m threadx.demo_gpu_auto
```

### Tests Automatisés

Des tests automatisés sont disponibles dans :

```bash
python -m pytest tests/test_gpu_auto_profiling.py -v
```

### Benchmarks

Des benchmarks comparatifs sont disponibles dans :

```bash
python -m threadx.benchmarks.run_auto_profile --sizes 1000 10000 100000
```

## 7. Bonnes Pratiques

### Utilisation Optimale

- **Décision Dynamique** : Utilisez l'option `use_dynamic=True` pour les fonctions d'indicateur pour bénéficier de la sélection automatique CPU/GPU.
- **Multi-GPU** : Pour les charges importantes, utilisez `MultiGPUManager` qui bénéficie automatiquement de l'auto-balance.
- **Déterminisme** : Pour les backtests et les résultats reproductibles, toujours spécifier un seed dans `distribute_workload`.

### Personnalisation

Pour les charges de travail très spécifiques, vous pouvez forcer la mise à jour des profils :

```python
from threadx.utils.gpu.profile_persistence import update_multigpu_ratio_profile

# Forcer des ratios spécifiques
update_multigpu_ratio_profile(
    ratios={"GPU0": 0.6, "GPU1": 0.4},
    workload_tag="custom_workload"
)
```

## 8. Limitations Connues

- L'auto-balance peut nécessiter plusieurs exécutions pour converger vers un ratio optimal.
- Les profils sont spécifiques à certains types de charges de travail et peuvent ne pas être optimaux pour tous les cas.
- Sur des systèmes hétérogènes (GPUs de générations différentes), des ajustements manuels peuvent parfois être nécessaires.

## 9. Perspectives Futures

- Intégration d'un système d'auto-profilage continu qui affine les profils au fil du temps.
- Support pour des décisions plus granulaires basées sur des caractéristiques supplémentaires des données.
- Optimisation dynamique de la mémoire GPU pour éviter les débordements.