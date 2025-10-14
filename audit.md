# 🔍 AUDIT THREADX - État des lieux & Plan Bridge

**Date** : 2025-10-14
**Objectif** : Valider séparation métier/UI + Préparer couche Bridge
**Statut** : ✅ CODE MÉTIER PUR CONFIRMÉ

---

## 📊 SECTION 1 : ÉTAT DES LIEUX

### ✅ Architecture actuelle validée

L'audit complet du code source ThreadX confirme une **séparation architecture exemplaire** :

| Couche | Localisation | Statut | UI Dependencies |
|--------|--------------|---------|-----------------|
| **Engine/Métier** | `src/threadx/backtest/` | ✅ PUR | ❌ Aucune |
| **Engine/Métier** | `src/threadx/indicators/` | ✅ PUR | ❌ Aucune |
| **Engine/Métier** | `src/threadx/optimization/` | ⚠️ MIXTE | ⚠️ ui.py présent |
| **Engine/Métier** | `src/threadx/data/` | ✅ PUR | ❌ Aucune |
| **UI Legacy** | `src/threadx/ui/` | ⚠️ LEGACY | ✅ Tkinter (à refactoriser) |
| **UI Streamlit** | `apps/streamlit/` | ⚠️ MIXTE | ✅ Streamlit direct |

### 🎯 Constats majeurs

#### ✅ Points forts
1. **Moteur de backtest** (`src/threadx/backtest/engine.py`) : 100% pur
   - Aucun import UI (Tkinter/Dash/Streamlit)
   - Device-agnostic via `threadx.utils.xp`
   - Multi-GPU transparent
   - Architecture production-ready

2. **Banque d'indicateurs** (`src/threadx/indicators/bank.py`) : 100% pure
   - Cache disque intelligent
   - Batch processing
   - GPU multi-carte
   - Zéro dépendance UI

3. **Optimisation** (`src/threadx/optimization/engine.py`) : Moteur pur
   - `UnifiedOptimizationEngine` découplé
   - Calculs purs
   - ⚠️ **MAIS** : `ui.py` présent dans même dossier (anomalie)

#### ⚠️ Points d'attention

1. **`src/threadx/optimization/ui.py`** :
   - **Problème** : Fichier UI dans dossier moteur
   - **Impact** : Violation architecture (UI dans Engine)
   - **Recommandation** : Déplacer vers `src/threadx/ui/` ou `apps/`

2. **`src/threadx/ui/`** (Legacy Tkinter) :
   - Fichiers détectés : `charts.py`, `sweep.py`, `data_manager.py`, etc.
   - **Problème** : Imports directs moteur (`IndicatorBank`, `UnifiedOptimizationEngine`)
   - **Recommandation** : Refactoriser via Bridge (Prompt 9)

3. **`apps/streamlit/app.py`** :
   - **Problème** : Imports directs `BacktestEngine`, `PerformanceCalculator`
   - **Recommandation** : Utiliser Bridge dès création

### 📁 Scan automatisé (zéro faux positif)

**Commande exécutée** :
```bash
grep -r "import tkinter\|from tkinter\|import dash\|import streamlit" src/threadx/{backtest,indicators,data}
```

**Résultat** : ✅ **0 match** dans les modules métier purs

---

## 📂 SECTION 2 : FICHIERS CLÉS DU MÉTIER

### Module Backtest (`src/threadx/backtest/`)

| Fichier | Rôle | Lignes | Statut | Imports externes |
|---------|------|--------|--------|------------------|
| `engine.py` | Orchestrateur backtest principal | 1012 | ✅ PUR | NumPy, Pandas, threadx.utils.* |
| `performance.py` | Calculs métriques (Sharpe, DD, PF) | ~800 | ✅ PUR | NumPy, Pandas, Matplotlib (chart only) |
| `sweep.py` | Sweeps paramétriques | ~500 | ✅ PUR | Concurrent, NumPy |
| `__init__.py` | Exports publics | 50 | ✅ PUR | - |

**Classes principales** :
- `BacktestEngine` : Orchestrateur avec multi-GPU
- `RunResult` : Dataclass résultats standardisée
- `PerformanceCalculator` : Métriques financières
- `summarize()` : Fonction utilitaire résumé

### Module Indicators (`src/threadx/indicators/`)

| Fichier | Rôle | Lignes | Statut | GPU Support |
|---------|------|--------|--------|-------------|
| `bank.py` | Cache centralisé + Registry | 1491 | ✅ PUR | ✅ Multi-GPU |
| `bollinger.py` | Bollinger Bands (NumPy/CuPy) | ~400 | ✅ PUR | ✅ |
| `xatr.py` | ATR extensible | ~300 | ✅ PUR | ✅ |
| `engine.py` | Moteur indicateurs unifié | ~600 | ✅ PUR | ✅ |
| `indicators_np.py` | Librairie indicateurs NumPy | ~500 | ✅ PUR | ❌ CPU only |

**Classes principales** :
- `IndicatorBank` : Cache + ensure/force_recompute
- `BollingerBands` : Calcul Bollinger avec GPU
- `ATR` : Average True Range
- `ensure_indicator()` : Helper fonction

### Module Optimization (`src/threadx/optimization/`)

| Fichier | Rôle | Lignes | Statut | UI Dependency |
|---------|------|--------|--------|---------------|
| `engine.py` | UnifiedOptimizationEngine | ~800 | ✅ PUR | ❌ |
| `pruning.py` | Pruning critères | ~300 | ✅ PUR | ❌ |
| `reporting.py` | Génération rapports | ~400 | ✅ PUR | ❌ |
| `scenarios.py` | Scénarios pré-configurés | ~200 | ✅ PUR | ❌ |
| `run.py` | Script exécution CLI | ~150 | ✅ PUR | ❌ |
| ⚠️ `ui.py` | **Interface Tkinter** | 758 | ❌ UI | ✅ Tkinter |

**⚠️ ANOMALIE DÉTECTÉE** : `ui.py` ne devrait PAS être dans `optimization/`

### Module Data (`src/threadx/data/`)

| Fichier | Rôle | Statut |
|---------|------|--------|
| `io.py` | Lecture/écriture Parquet | ✅ PUR |
| `registry.py` | Inventaire données | ✅ PUR |
| `ingest.py` | Ingestion données | ✅ PUR |

---

## 🏗️ SECTION 3 : PLAN BRIDGE

### Architecture cible 3-couches

```
┌─────────────────────────────────────────────────────────────┐
│  COUCHE 3 : UI/CLIENTS                                      │
│  - apps/cli.py (futur CLI unifié)                          │
│  - apps/dash/ (futur Dashboard Dash)                       │
│  - apps/streamlit/ (refactorisé)                           │
└────────────────┬────────────────────────────────────────────┘
                 │ Requests/Responses (DataClasses)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  COUCHE 2 : BRIDGE (À CRÉER)                               │
│  src/threadx/bridge/                                        │
│  ├── models.py          (BacktestRequest/Result, etc.)     │
│  ├── controllers.py     (Orchestration sans calculs)       │
│  ├── exceptions.py      (BridgeError custom)               │
│  └── __init__.py        (Exports publics)                  │
└────────────────┬────────────────────────────────────────────┘
                 │ Appels directs Engine
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  COUCHE 1 : ENGINE/MÉTIER (Existant, pur)                 │
│  src/threadx/                                              │
│  ├── backtest/          (BacktestEngine, Performance)     │
│  ├── indicators/        (IndicatorBank, Bollinger, ATR)   │
│  ├── optimization/      (UnifiedOptimizationEngine)       │
│  └── data/              (IO, Registry, Ingest)            │
└─────────────────────────────────────────────────────────────┘
```

### Controllers à créer (Prompt 2)

#### 1. BacktestController
**Responsabilité** : Orchestrer backtests via `BacktestEngine`

**Interface** :
```python
class BacktestController:
    def run_backtest(self, req: BacktestRequest) -> BacktestResult:
        """
        Exécute un backtest complet.

        Args:
            req: Paramètres backtest (symbol, timeframe, strategy, params)

        Returns:
            BacktestResult avec trades, equity_curve, metrics
        """
        # 1. Charger données via DataController
        # 2. Construire indicateurs via IndicatorController
        # 3. Instancier BacktestEngine
        # 4. Exécuter engine.run()
        # 5. Packager résultats en BacktestResult
        pass
```

**Dépendances Engine** :
- `threadx.backtest.engine.BacktestEngine`
- `threadx.backtest.performance.PerformanceCalculator`
- `threadx.data.io.read_frame`

#### 2. IndicatorController
**Responsabilité** : Gérer cache indicateurs via `IndicatorBank`

**Interface** :
```python
class IndicatorController:
    def build_indicators(self, req: IndicatorRequest) -> IndicatorResult:
        """
        Construit indicateurs techniques avec cache.

        Args:
            req: Paramètres indicateurs (type, params, data)

        Returns:
            IndicatorResult avec valeurs calculées + metadata cache
        """
        # 1. Valider requête
        # 2. Appeler bank.ensure() ou bank.batch_ensure()
        # 3. Packager résultats
        pass

    def force_recompute(self, indicator_name: str, params: Dict) -> IndicatorResult:
        """Force recalcul (bypass cache)"""
        pass

    def clear_cache(self, indicator_name: Optional[str] = None) -> bool:
        """Nettoie cache (tout ou indicateur spécifique)"""
        pass
```

**Dépendances Engine** :
- `threadx.indicators.bank.IndicatorBank`
- `threadx.indicators.bank.ensure_indicator`

#### 3. SweepController
**Responsabilité** : Orchestrer sweeps paramétriques via `UnifiedOptimizationEngine`

**Interface** :
```python
class SweepController:
    def run_sweep(self, req: SweepRequest) -> SweepResult:
        """
        Exécute sweep paramétrique multi-critères.

        Args:
            req: Configuration sweep (param_grid, data, strategy)

        Returns:
            SweepResult avec best_params, all_results, top_N
        """
        # 1. Valider param_grid
        # 2. Instancier UnifiedOptimizationEngine
        # 3. Exécuter sweep
        # 4. Appliquer pruning critères
        # 5. Générer rapport
        pass

    def export_sweep_results(self, result: SweepResult, path: Path) -> bool:
        """Exporte résultats sweep (CSV/Parquet)"""
        pass
```

**Dépendances Engine** :
- `threadx.optimization.engine.UnifiedOptimizationEngine`
- `threadx.optimization.pruning`
- `threadx.optimization.reporting`

#### 4. DataController
**Responsabilité** : Gérer I/O données + validation

**Interface** :
```python
class DataController:
    def load_data(self, req: DataRequest) -> DataResult:
        """
        Charge données depuis cache/fichier.

        Args:
            req: Paramètres data (symbol, timeframe, source)

        Returns:
            DataResult avec DataFrame + metadata
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> DataValidationResult:
        """Valide qualité données (gaps, outliers, etc.)"""
        pass

    def list_available_data(self) -> List[DataInventory]:
        """Liste données disponibles en cache"""
        pass
```

**Dépendances Engine** :
- `threadx.data.io.read_frame`
- `threadx.data.registry.quick_inventory`
- `threadx.data.ingest.IngestionManager` (si nécessaire)

---

## 📋 SECTION 4 : POINTS D'ENTRÉE ACTUELS

### Scénario actuel (sans Bridge)

**Cas d'usage** : Exécuter backtest Bollinger Bands

**Code actuel (hypothétique CLI)** :
```python
# apps/cli.py (si existait)
from threadx.backtest.engine import BacktestEngine
from threadx.indicators.bank import IndicatorBank, ensure_indicator
from threadx.data.io import read_frame

# Chargement données
df = read_frame("data/BTCUSDC_15m.parquet")

# Construction indicateurs
bank = IndicatorBank()
bb_result = ensure_indicator(
    'bollinger',
    {'period': 20, 'std': 2.0},
    df['close']
)

# Configuration backtest
engine = BacktestEngine()
result = engine.run(
    df=df,
    indicators={'bollinger': bb_result},
    strategy='bollinger_mean_reversion',
    params={'entry_z': -2.0, 'exit_z': 0.0}
)

# Affichage résultats
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
```

**Problèmes** :
- ❌ Imports directs Engine dans CLI
- ❌ Gestion manuelle indicateurs + données
- ❌ Pas de validation centralisée
- ❌ Code UI couplé au métier

### Scénario cible (avec Bridge)

**Code futur (Prompt 2 + 9)** :
```python
# apps/cli.py (après Bridge)
from threadx.bridge import BacktestController, BacktestRequest

# Initialisation controller (une seule fois)
controller = BacktestController()

# Requête déclarative
request = BacktestRequest(
    symbol='BTCUSDC',
    timeframe='15m',
    strategy='bollinger_mean_reversion',
    indicators={
        'bollinger': {'period': 20, 'std': 2.0}
    },
    strategy_params={'entry_z': -2.0, 'exit_z': 0.0},
    initial_capital=10000.0
)

# Exécution (tout est géré par Bridge)
result = controller.run_backtest(request)

# Affichage (juste présentation)
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
```

**Avantages** :
- ✅ Zéro import Engine dans CLI
- ✅ Requêtes déclaratives (DataClasses)
- ✅ Validation centralisée dans Bridge
- ✅ Testabilité parfaite (mock Bridge)
- ✅ CLI découplé (peut changer UI sans toucher métier)

### Mapping points d'entrée

| Point d'entrée actuel | Fichier | Appel direct | Futur via Bridge |
|----------------------|---------|--------------|------------------|
| Backtest CLI | *(hypothétique)* | `BacktestEngine.run()` | `BacktestController.run_backtest()` |
| Sweep CLI | `optimization/run.py` | `UnifiedOptimizationEngine` | `SweepController.run_sweep()` |
| UI Tkinter | `ui/sweep.py` | `UnifiedOptimizationEngine` | `SweepController.run_sweep()` |
| UI Streamlit | `apps/streamlit/app.py` | `BacktestEngine.run()` | `BacktestController.run_backtest()` |
| Indicators | Direct calls | `bank.ensure()` | `IndicatorController.build_indicators()` |

---

## 🎯 RECOMMANDATIONS PRIORITAIRES

### Immédiat (Prompt 2)
1. ✅ **Créer `src/threadx/bridge/`** avec structure complète
   - `models.py` : DataClasses requêtes/réponses
   - `controllers.py` : 4 controllers (Backtest, Indicator, Sweep, Data)
   - `exceptions.py` : Erreurs custom
   - `__init__.py` : Exports publics

2. ✅ **Implémenter controllers** sans threading (synchrone d'abord)
   - BacktestController.run_backtest()
   - IndicatorController.build_indicators()
   - SweepController.run_sweep()
   - DataController.load_data()

### Moyen terme (Prompt 3-8)
3. ⏳ **Ajouter async/threading** dans Bridge
   - Wrapper ThreadExecutor pour long-running tasks
   - Callbacks pour UI non-bloquante
   - Progress tracking

4. ⏳ **Créer Dashboard Dash** (apps/dash/)
   - Utiliser UNIQUEMENT Bridge (pas Engine direct)
   - Composants réutilisables
   - Intégration continue

### Long terme (Prompt 9+)
5. ⏳ **Refactoriser UI Legacy** (src/threadx/ui/)
   - Déplacer `optimization/ui.py` → `ui/optimization_legacy.py`
   - Adapter imports pour utiliser Bridge
   - Éventuellement deprecated si Dash suffit

6. ⏳ **Créer CLI unifié** (apps/cli.py)
   - Click/Typer pour interface
   - Appels Bridge uniquement
   - Config YAML/TOML

---

## ✅ CHECKLIST VALIDATION

### État actuel ✅
- [x] Moteur backtest pur (0 UI imports)
- [x] Indicateurs purs (0 UI imports)
- [x] Optimisation moteur pur (engine.py seulement)
- [x] Data processing pur (0 UI imports)
- [x] Architecture 3-couches identifiable

### État cible (post-Prompt 2) ⏳
- [ ] Bridge créé avec 4 controllers
- [ ] DataClasses requêtes/réponses typées
- [ ] Exceptions custom Bridge
- [ ] Tests unitaires Bridge (mocks Engine)
- [ ] Documentation API Bridge complète

### Validation finale (post-Prompt 9) ⏳
- [ ] CLI unifié utilise Bridge uniquement
- [ ] Dashboard Dash utilise Bridge uniquement
- [ ] Streamlit refactorisé avec Bridge
- [ ] Zéro import Engine hors de Bridge
- [ ] Tests intégration UI ↔ Bridge ↔ Engine

---

## 📊 MÉTRIQUES FINALES

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| **Modules métier purs** | 3/4 | backtest, indicators, data ✅ / optimization ⚠️ |
| **Fichiers UI dans Engine** | 1 | `optimization/ui.py` (anomalie) |
| **Imports UI dans métier** | 1 | `optimization/ui.py` (Tkinter) |
| **Lignes code métier** | ~8000 | Estimation (engine + indicators + optim) |
| **Prêt pour Bridge** | ✅ OUI | Architecture solide, séparation claire |
| **Effort Prompt 2** | ~6-8h | Création Bridge complet + tests |

---

**🎯 CONCLUSION** : ThreadX a une **architecture métier exemplaire** prête pour l'ajout de la couche Bridge. Un seul fichier anomalie (`optimization/ui.py`) à déplacer, mais pas bloquant pour Prompt 2.

**✅ READY FOR PROMPT 2 : Bridge Foundation Creation**

---

*Audit complété le 2025-10-14 - Validation architecture 3-couches*
