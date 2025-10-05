# ThreadX · Phase C · Benchmarks & KPI Gates

## 1. Objectifs et portée

La Phase C du projet ThreadX est axée sur l'outillage de la preuve de performance et de stabilité via des scripts de benchmark et des rapports, ainsi que sur la mise en place de seuils KPI dans les tests.

### Objectifs principaux
- Mesurer systématiquement les performances CPU vs GPU pour les indicateurs techniques
- Vérifier le taux de cache hit sur les sweeps de paramètres
- Garantir le déterminisme des résultats
- S'assurer de la non-régression de l'algorithme Pareto early-stop
- Générer des rapports complets et détaillés

### Portée
- Outils de benchmark dans `tools/benchmarks_cpu_gpu.py` et `benchmarks/utils.py`
- Tests KPI automatisés dans `tests/test_kpi_gates.py`
- Génération de rapports CSV et Markdown
- Vérification de KPI gates avec badges de statut

## 2. Architecture technique

### Structure des benchmarks

```
ThreadX/
├── benchmarks/
│   ├── results/          # Résultats CSV des benchmarks
│   ├── reports/          # Rapports Markdown détaillés
│   ├── baselines/        # Références pour tests non-régressifs
│   ├── README.md         # Documentation benchmarks
│   └── utils.py          # Utilitaires communs
├── tools/
│   └── benchmarks_cpu_gpu.py  # Script principal de benchmark
├── tests/
│   └── test_kpi_gates.py      # Tests automatisés des KPI
└── run_benchmark_demo.py      # Script de démonstration
```

### Composants techniques

1. **Chronomètres spécialisés**
   - CPU: `time.perf_counter_ns()` pour précision nanosecondes
   - GPU: `cupy.cuda.Event()` pour mesurer temps kernels
   - Context manager `gpu_timer()` pour faciliter l'utilisation

2. **Générateurs de rapports**
   - Export CSV: métriques brutes avec horodatage
   - Export Markdown: rapport détaillé avec analyses et badges

3. **Outils de hashing pour déterminisme**
   - `stable_hash()`: hash stable indépendant de l'ordre des clés
   - `hash_series()`: hash SHA256 des arrays numpy

4. **Tests KPI automatisés**
   - Tests pytest pour vérifier les seuils KPI
   - Skip automatique si GPU non disponible

## 3. KPI Gates

Les KPI Gates sont des seuils bloquants qui doivent être respectés pour valider la qualité du code.

| KPI               | Description             | Seuil     | Méthode de vérification              |
| ----------------- | ----------------------- | --------- | ------------------------------------ |
| `KPI_SPEEDUP_GPU` | Accélération GPU vs CPU | ≥ 3×      | Mesure directe des temps d'exécution |
| `KPI_CACHE_HIT`   | Taux de hit du cache    | ≥ 80%     | Comptage hits/misses sur sweep       |
| `KPI_DETERMINISM` | Stabilité des résultats | Hash égal | Comparison des hashes sur 3 runs     |
| `KPI_PARETO`      | Non-régression Pareto   | ±5%       | Comparaison vs baseline              |

## 4. Méthodologie de benchmark

### 4.1 CPU vs GPU
- Test sur multiples tailles: 10K, 100K, 1M points
- 5 répétitions par point (1 warmup GPU ignoré)
- Seed fixe pour reproductibilité
- Métriques: temps moyen, écart-type, gain vs CPU, ratio kernel GPU

### 4.2 Cache hit rate
- 200+ invocations de la banque d'indicateurs
- Paramètres variés pour forcer des miss initiaux
- Comptage explicite des hits et misses

### 4.3 Déterminisme
- Multiples runs avec seed identique
- Validation par hash stable des résultats
- Vérification sur différents indicateurs

### 4.4 Pareto non-régressif
- Jeu de test synthétique reproductible
- Comparaison des métriques clés vs baseline
- Tolérance de ±5% pour variations acceptables

## 5. Formats de sortie

### 5.1 CSV
```csv
date,indicator,N,device,repeats,mean_ms,std_ms,gain_vs_cpu,gpu_kernel_ratio
2025-10-05 12:30,bollinger,10000,cpu,5,15.2,0.8,nan,nan
2025-10-05 12:30,bollinger,10000,gpu,5,4.3,0.3,3.5,0.85
```

### 5.2 Markdown
Le rapport Markdown comprend:
- Badges KPI (OK/KO)
- Méthodologie et configuration
- Tableaux de résultats par taille
- Diagnostics en cas de KO
- Informations sur l'environnement d'exécution

### 5.3 Badges
- `KPI_SPEEDUP_GPU: OK/KO`
- `KPI_CACHE_HIT: OK/KO`
- `KPI_DETERMINISM: OK/KO`
- `KPI_PARETO: OK/KO`

## 6. Utilisation

### Lancer le benchmark complet
```powershell
python -m tools.benchmarks_cpu_gpu --indicators bollinger,atr --sizes 10000,100000,1000000 --repeats 5
```

### Lancer les tests KPI
```powershell
python -m pytest tests/test_kpi_gates.py -v
```

### Lancer la démo rapide
```powershell
python run_benchmark_demo.py
```

## 7. Diagnostics et résolution des problèmes

En cas d'échec des KPI, le rapport fournit des diagnostics classés par probabilité:

### KPI_SPEEDUP_GPU: KO
1. **Taille insuffisante** - Le speedup GPU est plus important sur des grandes tailles
2. **Overhead de transfert** - Transfers CPU↔GPU dominants par rapport au calcul
3. **Indicateur non vectorisé** - Implémentation non optimisée pour GPU

### KPI_CACHE_HIT: KO
1. **Cache invalidé trop fréquemment** - TTL trop court
2. **Clés de cache instables** - Génération non déterministe des clés
3. **Cache désactivé** - Problème de configuration

### KPI_DETERMINISM: KO
1. **Seed non propagé** - Certains générateurs aléatoires non initialisés
2. **Opérations non déterministes** - Algorithmes avec composants aléatoires
3. **Problème de synchronisation GPU** - Ordre d'exécution variable

### KPI_PARETO: KO
1. **Changement d'algorithme** - Modification non intentionnelle
2. **Changement de paramètres** - Configuration modifiée
3. **Bug introduit** - Régression dans l'implémentation

## 8. Conclusion

La Phase C apporte à ThreadX une infrastructure complète pour mesurer, valider et documenter les performances du système. Les KPI gates garantissent le maintien des standards de qualité et de performance au fil des évolutions du code.