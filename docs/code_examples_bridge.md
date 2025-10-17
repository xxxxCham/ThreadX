# 📝 CODE EXAMPLES - État actuel vs Futur Bridge

**Objectif** : Montrer comment le code actuel appelle directement le moteur et comment cela changera avec Bridge

---

## 🔍 EXEMPLE 1 : Backtest avec Bollinger Bands

### ❌ Code actuel (appels directs Engine)

**Fichier** : `src/threadx/benchmarks/run_backtests.py` (lignes 122-145)

```python
from threadx.backtest import create_engine
from threadx.indicators import get_gpu_accelerated_bank

# Initialisation directe du moteur
engine = create_engine()
bank = get_gpu_accelerated_bank()

# Génération données synthétiques
df = _generate_synthetic_data(size=1000)

# Configuration stratégie
strategy_name = "bollinger_mean_reversion"
params = {
    'entry_z': -2.0,
    'exit_z': 0.0,
    'atr_filter': True
}

# Exécution directe (bloquante)
result = engine.run(
    data=df,
    strategy=strategy_name,
    params=params,
    use_gpu=False
)

# Accès direct aux résultats
cpu_equity_hash = hash_series(result.equity)
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

**Problèmes identifiés** :
- ❌ Import direct `threadx.backtest` dans code client
- ❌ Gestion manuelle initialisation Engine
- ❌ Exécution bloquante (synchrone)
- ❌ Pas de validation params centralisée
- ❌ Testabilité difficile (mock Engine complexe)

---

### ✅ Code futur (via Bridge)

**Fichier** : `apps/cli.py` ou `apps/benchmark.py` (après Prompt 2)

```python
from threadx.bridge import BacktestController, BacktestRequest

# Initialisation controller (une seule fois, config auto)
controller = BacktestController()

# Requête déclarative avec validation intégrée
request = BacktestRequest(
    symbol='SYNTHETIC',
    timeframe='1h',
    data=df,  # DataFrame déjà chargé
    strategy='bollinger_mean_reversion',
    strategy_params={
        'entry_z': -2.0,
        'exit_z': 0.0,
        'atr_filter': True
    },
    use_gpu=False,
    initial_capital=10000.0
)

# Exécution via Bridge (validation + orchestration)
result = controller.run_backtest(request)

# Résultat typé avec metadata
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Execution Time: {result.metadata['execution_time_ms']}ms")
print(f"Cache Hits: {result.metadata['indicator_cache_hits']}")
```

**Avantages** :
- ✅ Zéro import Engine (découplage total)
- ✅ Validation automatique via DataClass
- ✅ Controller gère initialisation
- ✅ Metadata enrichies (timing, cache, etc.)
- ✅ Tests faciles (mock BacktestController)

---

## 🔍 EXEMPLE 2 : Construction indicateurs avec cache

### ❌ Code actuel (appels directs IndicatorBank)

**Fichier** : Hypothétique dans UI/CLI

```python
from threadx.indicators.bank import IndicatorBank, ensure_indicator
import pandas as pd

# Initialisation manuelle
bank = IndicatorBank(cache_dir="indicators_cache")

# Chargement données
df = pd.read_parquet("data/BTCUSDC_15m.parquet")
close_prices = df['close']

# Ensure indicateur (cache ou calcul)
bb_result = ensure_indicator(
    indicator_name='bollinger',
    params={'period': 20, 'std': 2.0},
    data=close_prices,
    symbol='BTCUSDC',
    timeframe='15m'
)

# Accès valeurs
upper_band = bb_result['upper']
middle_band = bb_result['middle']
lower_band = bb_result['lower']

print(f"Bollinger Bands computed: {len(upper_band)} values")
```

**Problèmes identifiés** :
- ❌ Import direct `IndicatorBank`
- ❌ Gestion manuelle cache_dir
- ❌ Pas de validation params
- ❌ Pas de metadata sur cache hit/miss
- ❌ Code répétitif pour batch indicators

---

### ✅ Code futur (via Bridge)

```python
from threadx.bridge import IndicatorController, IndicatorRequest

# Initialisation (config auto depuis settings)
controller = IndicatorController()

# Requête déclarative
request = IndicatorRequest(
    indicator_type='bollinger',
    params={'period': 20, 'std': 2.0},
    data=close_prices,
    symbol='BTCUSDC',
    timeframe='15m',
    force_recompute=False  # Utilise cache si disponible
)

# Exécution via Bridge
result = controller.build_indicators(request)

# Résultat typé avec metadata cache
upper_band = result.values['upper']
middle_band = result.values['middle']
lower_band = result.values['lower']

print(f"Bollinger Bands computed: {len(upper_band)} values")
print(f"Cache hit: {result.metadata['from_cache']}")
print(f"Cache key: {result.metadata['cache_key']}")
print(f"Computation time: {result.metadata['compute_time_ms']}ms")
```

**Avantages** :
- ✅ Zéro import Bank (abstraction complète)
- ✅ Config cache automatique
- ✅ Validation params intégrée
- ✅ Metadata cache enrichies
- ✅ API uniforme pour tous indicateurs

---

## 🔍 EXEMPLE 3 : Sweep paramétrique

### ❌ Code actuel (appels directs UnifiedOptimizationEngine)

**Fichier** : `src/threadx/optimization/run.py` ou `src/threadx/ui/sweep.py`

```python
from threadx.optimization.engine import UnifiedOptimizationEngine, DEFAULT_SWEEP_CONFIG
from threadx.indicators.bank import IndicatorBank
import pandas as pd

# Initialisation manuelle
bank = IndicatorBank()
engine = UnifiedOptimizationEngine(
    indicator_bank=bank,
    max_workers=4
)

# Chargement données
df = pd.read_parquet("data/BTCUSDC_15m.parquet")

# Configuration sweep
sweep_config = {
    'bollinger': {
        'period': [10, 20, 30, 50],
        'std': [1.5, 2.0, 2.5]
    }
}

strategy_config = {
    'name': 'bollinger_mean_reversion',
    'params': {'entry_z': -2.0, 'exit_z': 0.0}
}

# Exécution sweep (bloquant, long)
results = engine.run_sweep(
    data=df,
    sweep_config=sweep_config,
    strategy_config=strategy_config,
    max_workers=4
)

# Tri résultats
best_params = results.sort_values('sharpe_ratio', ascending=False).iloc[0]
print(f"Best params: {best_params['params']}")
print(f"Best Sharpe: {best_params['sharpe_ratio']:.2f}")
```

**Problèmes identifiés** :
- ❌ Import direct `UnifiedOptimizationEngine`
- ❌ Gestion manuelle workers
- ❌ Exécution bloquante (pas de progress)
- ❌ Pas de pruning automatique
- ❌ Export manuel résultats

---

### ✅ Code futur (via Bridge)

```python
from threadx.bridge import SweepController, SweepRequest

# Initialisation
controller = SweepController()

# Requête déclarative
request = SweepRequest(
    symbol='BTCUSDC',
    timeframe='15m',
    data=df,
    strategy='bollinger_mean_reversion',
    strategy_params={'entry_z': -2.0, 'exit_z': 0.0},
    param_grid={
        'bollinger': {
            'period': [10, 20, 30, 50],
            'std': [1.5, 2.0, 2.5]
        }
    },
    optimization_criteria=['sharpe_ratio', 'total_return', 'max_drawdown'],
    pruning_enabled=True,
    max_workers=4
)

# Exécution avec progress callback (optionnel)
def on_progress(completed, total, current_result):
    print(f"Progress: {completed}/{total} - Current best Sharpe: {current_result:.2f}")

result = controller.run_sweep(
    request,
    progress_callback=on_progress
)

# Résultats pré-triés et pré-prunés
print(f"Best params: {result.best_params}")
print(f"Best Sharpe: {result.best_sharpe:.2f}")
print(f"Total combinations tested: {result.total_combinations}")
print(f"Pruned combinations: {result.pruned_count}")

# Export automatique
result.export_to_csv("sweep_results.csv")
result.export_report("sweep_report.html")
```

**Avantages** :
- ✅ Zéro import Engine
- ✅ Config workers automatique
- ✅ Progress tracking intégré
- ✅ Pruning automatique
- ✅ Export simplifié

---

## 🔍 EXEMPLE 4 : Chargement et validation données

### ❌ Code actuel (appels directs data.io)

```python
from threadx.data.io import read_frame
from threadx.data.registry import quick_inventory
import pandas as pd

# Listing manuel
inventory = quick_inventory()
print(f"Available files: {len(inventory)}")

# Chargement manuel
df = read_frame("data/crypto_data_parquet/BTCUSDC_15m.parquet")

# Validation manuelle basique
assert not df.empty, "DataFrame empty"
assert 'close' in df.columns, "Missing close column"
assert df['close'].notna().all(), "Null values in close"

print(f"Loaded {len(df)} rows")
```

**Problèmes identifiés** :
- ❌ Import direct data.io
- ❌ Validation manuelle (incomplet)
- ❌ Pas de metadata (source, quality, etc.)
- ❌ Gestion erreurs basique

---

### ✅ Code futur (via Bridge)

```python
from threadx.bridge import DataController, DataRequest

# Initialisation
controller = DataController()

# Listing simplifié
available_data = controller.list_available_data()
for item in available_data:
    print(f"{item.symbol} - {item.timeframe}: {item.row_count} rows")

# Requête avec validation automatique
request = DataRequest(
    symbol='BTCUSDC',
    timeframe='15m',
    validate=True,  # Active validation automatique
    required_columns=['open', 'high', 'low', 'close', 'volume']
)

# Chargement + validation
result = controller.load_data(request)

# Résultat avec validation complète
print(f"Loaded {result.row_count} rows")
print(f"Data quality: {result.validation.quality_score}/10")
print(f"Missing values: {result.validation.missing_count}")
print(f"Outliers detected: {result.validation.outlier_count}")
print(f"Date gaps: {result.validation.date_gaps}")

# Accès DataFrame validé
df = result.dataframe
```

**Avantages** :
- ✅ Zéro import data.io
- ✅ Validation automatique complète
- ✅ Metadata enrichies (quality, gaps, etc.)
- ✅ Gestion erreurs centralisée
- ✅ Inventory simplifié

---

## 📊 COMPARAISON RÉCAPITULATIVE

| Feature | Code actuel (Direct Engine) | Code futur (via Bridge) |
|---------|------------------------------|-------------------------|
| **Imports** | Engine direct | Bridge uniquement |
| **Type safety** | ❌ Partial | ✅ Full (DataClasses) |
| **Validation** | ❌ Manuelle | ✅ Automatique |
| **Testabilité** | 🟡 Difficile | ✅ Facile (mock Bridge) |
| **Metadata** | ❌ Minimale | ✅ Enrichie |
| **Progress tracking** | ❌ Non | ✅ Callbacks |
| **Error handling** | 🟡 Basique | ✅ Custom exceptions |
| **Config management** | ❌ Manuelle | ✅ Centralisée |
| **Cache visibility** | ❌ Opaque | ✅ Transparente |
| **Export results** | ❌ Manuel | ✅ Méthodes intégrées |

---

## 🎯 IMPACT MIGRATION

### Effort par fichier

| Fichier | Lignes impactées | Effort | Priorité |
|---------|------------------|--------|----------|
| `benchmarks/run_backtests.py` | ~20 lignes | 30 min | 🟡 Moyen |
| `optimization/run.py` | ~30 lignes | 45 min | 🟡 Moyen |
| `ui/sweep.py` | ~100 lignes | 3h | 🔴 Haute |
| `apps/streamlit/app.py` | ~50 lignes | 1.5h | 🔴 Haute |

### Bénéfices attendus

- ⚡ **Performance** : Inchangée (Bridge = thin wrapper)
- 🧪 **Tests** : +50% couverture (mock Bridge facile)
- 🔒 **Type safety** : +100% (mypy strict compatible)
- 📝 **Lisibilité** : +80% (requêtes déclaratives)
- 🐛 **Debugging** : +60% (exceptions claires)

---

**✅ READY TO IMPLEMENT BRIDGE (Prompt 2)**

*Examples créés le 2025-10-14*
