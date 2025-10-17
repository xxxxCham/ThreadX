# TODO - Bridge Layer Corrections

## ⏸️ PROMPT 2 - Tâches Restantes

**Date :** 14 octobre 2025
**Priorité :** Moyenne (pas bloquant pour PROMPT 3)

---

## 🎯 Vue d'Ensemble

**Statut actuel :** 75% complété (3/5 fichiers production-ready)

**À faire :** Corriger controllers pour utiliser vraies APIs Engine

**Durée estimée :** 4-5 heures

**Bloque quoi ?**
- ❌ Ne bloque PAS PROMPT 3 (async wrapper)
- ❌ Ne bloque PAS PROMPT 4-7 (peut utiliser mocks)
- ✅ Bloque tests E2E complets

---

## ✅ Tâches Terminées

- [x] Créer structure Bridge (`src/threadx/bridge/`)
- [x] Créer 8 DataClasses Request/Result (`models.py`)
- [x] Créer hiérarchie 7 exceptions (`exceptions.py`)
- [x] Créer exports publics (`__init__.py`)
- [x] Documenter avec Google-style docstrings
- [x] Type hints PEP 604 partout
- [x] Analyser erreurs APIs hypothétiques
- [x] Documenter état actuel (3 fichiers .md)

---

## 📋 Tâches Restantes

### [ ] TASK 1 : Créer Data Helpers
**Fichier :** `src/threadx/data/helpers.py` (nouveau)
**Durée :** 30 minutes
**Priorité :** Haute (requis pour controllers)

**Sous-tâches :**
- [ ] Créer fonction `load_data(symbol, timeframe, path, start_date, end_date) -> DataFrame`
  - [ ] Charger depuis Parquet si path fourni
  - [ ] Auto-detect path via `get_data_path()` sinon
  - [ ] Filtrage dates si nécessaire
  - [ ] Validation colonnes OHLCV requises

- [ ] Créer fonction `get_data_path(symbol, timeframe) -> Path`
  - [ ] Résoudre path depuis `data/crypto_data_parquet/`
  - [ ] Format : `{symbol}_{timeframe}.parquet`
  - [ ] Raise FileNotFoundError si inexistant

- [ ] Créer fonction `filter_by_dates(df, start_date, end_date) -> DataFrame`
  - [ ] Filtrage pandas par index datetime
  - [ ] Support ISO 8601 dates

- [ ] Ajouter docstrings Google-style
- [ ] Ajouter type hints PEP 604
- [ ] Tests unitaires basiques

**Code exemple :**
```python
"""ThreadX Data Helpers - Fonctions utilitaires chargement données."""
from pathlib import Path
import pandas as pd

def load_data(
    symbol: str,
    timeframe: str,
    path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None
) -> pd.DataFrame:
    """Charge données OHLCV depuis Parquet."""
    # Implementation...
```

---

### [ ] TASK 2 : Corriger BacktestController
**Fichier :** `src/threadx/bridge/controllers.py`
**Durée :** 1 heure
**Priorité :** Haute

**Sous-tâches :**
- [ ] Importer `load_data` et `get_data_path` depuis helpers
- [ ] Réécrire `run_backtest()` workflow complet :
  - [ ] 1. Charger données via `load_data()`
  - [ ] 2. Créer IndicatorBank
  - [ ] 3. Calculer indicateurs via `bank.ensure()`
  - [ ] 4. Créer BacktestEngine via `create_engine()`
  - [ ] 5. Exécuter backtest via `engine.run(df_1m, indicators, ...)`
  - [ ] 6. Mapper RunResult → BacktestResult

- [ ] Implémenter helpers privés :
  - [ ] `_build_indicators(bank, df, strategy, params) -> Dict[str, Any]`
  - [ ] `_map_run_result(raw: RunResult, exec_time) -> BacktestResult`
  - [ ] `_calculate_metrics(equity, returns, trades) -> Dict[str, float]`

- [ ] Valider mypy --strict
- [ ] Test unitaire end-to-end

**Référence :**
→ `docs/CORRECTIONS_BRIDGE_API.md` section "BacktestController"

---

### [ ] TASK 3 : Corriger IndicatorController
**Fichier :** `src/threadx/bridge/controllers.py`
**Durée :** 30 minutes
**Priorité :** Moyenne

**Sous-tâches :**
- [ ] Réécrire `build_indicators()` avec vraie API :
  - [ ] Créer IndicatorSettings pour config
  - [ ] Créer IndicatorBank(settings=...)
  - [ ] Loop sur indicators : `bank.ensure(type, params, data, ...)`
  - [ ] Récupérer stats cache depuis `bank.stats`

- [ ] Valider mypy --strict
- [ ] Test unitaire avec cache

**Code exemple :**
```python
from threadx.indicators.bank import IndicatorBank, IndicatorSettings

settings = IndicatorSettings(
    cache_dir=self.config.cache_path,
    use_gpu=request.use_gpu or self.config.gpu_enabled
)
bank = IndicatorBank(settings=settings)

for indicator_name, params in request.indicators.items():
    values = bank.ensure(
        indicator_type=indicator_name,
        params=params,
        data=df,
        symbol=request.symbol,
        timeframe=request.timeframe
    )
    indicator_values[indicator_name] = values

cache_hits = bank.stats.get('cache_hits', 0)
```

---

### [ ] TASK 4 : Corriger SweepController
**Fichier :** `src/threadx/bridge/controllers.py`
**Durée :** 45 minutes
**Priorité :** Moyenne

**Sous-tâches :**
- [ ] Réécrire `run_sweep()` avec vraie API :
  - [ ] Créer IndicatorBank pour sweep
  - [ ] Créer UnifiedOptimizationEngine(indicator_bank, max_workers)
  - [ ] Construire config dict pour `run_parameter_sweep()`
  - [ ] Mapper DataFrame résultat → SweepResult

- [ ] Implémenter helper :
  - [ ] `_build_sweep_config(request) -> Dict[str, Any]`

- [ ] Valider mypy --strict
- [ ] Test unitaire avec petite grille

**Code exemple :**
```python
from threadx.optimization.engine import UnifiedOptimizationEngine

bank = IndicatorBank()
engine = UnifiedOptimizationEngine(
    indicator_bank=bank,
    max_workers=request.max_workers or self.config.max_workers
)

config = {
    "grid": request.param_grid,
    "scoring": {
        "metrics": request.optimization_criteria
    }
}

df_results = engine.run_parameter_sweep(config, df_1m)
```

---

### [ ] TASK 5 : Simplifier DataController
**Fichier :** `src/threadx/bridge/controllers.py`
**Durée :** 30 minutes
**Priorité :** Basse

**Options :**
- [ ] **Option A :** Garder validation simple (pandas checks)
  - [ ] Utiliser `load_data()` helper
  - [ ] Validation basique : colonnes, types, missing values
  - [ ] Retourner DataValidationResult simplifié

- [ ] **Option B :** Retirer DataController complètement
  - [ ] Engine fait déjà validation
  - [ ] Créer simple helper `validate_data()` dans helpers.py

- [ ] **Option C :** Transformer en property BacktestController
  - [ ] `controller.validate_data(request)` avant backtest
  - [ ] Pas besoin classe séparée

**Recommandation :** Option A (garder pour API complète)

---

### [ ] TASK 6 : Tests Unitaires
**Fichiers :** `tests/bridge/test_*.py` (nouveaux)
**Durée :** 1 heure
**Priorité :** Haute

**Sous-tâches :**
- [ ] `test_backtest_controller.py`
  - [ ] Test run_backtest avec vraies données
  - [ ] Test validation requête
  - [ ] Test error handling

- [ ] `test_indicator_controller.py`
  - [ ] Test build_indicators
  - [ ] Test cache hits/misses
  - [ ] Test GPU on/off

- [ ] `test_sweep_controller.py`
  - [ ] Test run_sweep avec petite grille
  - [ ] Test tri résultats
  - [ ] Test parallélisation

- [ ] `test_data_controller.py`
  - [ ] Test validate_data
  - [ ] Test quality_score
  - [ ] Test détection anomalies

**Coverage cible :** >80% pour controllers

---

### [ ] TASK 7 : Validation Finale
**Durée :** 30 minutes
**Priorité :** Haute

**Checklist :**
- [ ] `mypy --strict src/threadx/bridge/` sans erreurs
- [ ] `ruff check src/threadx/bridge/` sans warnings
- [ ] `black src/threadx/bridge/` formaté
- [ ] Tous tests passent
- [ ] Documentation à jour
- [ ] Examples fonctionnent

---

## 📊 Progression Estimée

```
TASK 1 : Data Helpers         [ ] 30 min  ░░░░░░░░░░░░░░░░
TASK 2 : BacktestController   [ ] 60 min  ░░░░░░░░░░░░░░░░
TASK 3 : IndicatorController  [ ] 30 min  ░░░░░░░░░░░░░░░░
TASK 4 : SweepController      [ ] 45 min  ░░░░░░░░░░░░░░░░
TASK 5 : DataController       [ ] 30 min  ░░░░░░░░░░░░░░░░
TASK 6 : Tests                [ ] 60 min  ░░░░░░░░░░░░░░░░
TASK 7 : Validation           [ ] 30 min  ░░░░░░░░░░░░░░░░

Total:                         0/285 min (0%)
```

---

## 🎯 Ordre Recommandé

1. **TASK 1** (Data Helpers) - Bloque tous controllers
2. **TASK 2** (BacktestController) - Le plus critique
3. **TASK 3** (IndicatorController) - Rapide
4. **TASK 4** (SweepController) - Optionnel pour début
5. **TASK 6** (Tests) - Valider au fur et à mesure
6. **TASK 5** (DataController) - Basse priorité
7. **TASK 7** (Validation) - Finalisation

---

## 📚 Références

**Documentation :**
- `docs/CORRECTIONS_BRIDGE_API.md` - APIs réelles détaillées
- `docs/PROMPT2_LIVRAISON_PARTIELLE.md` - Plan correction complet

**Code Source :**
- `src/threadx/backtest/engine.py` - BacktestEngine API
- `src/threadx/indicators/bank.py` - IndicatorBank API
- `src/threadx/optimization/engine.py` - UnifiedOptimizationEngine API

**Examples :**
- `src/threadx/bridge/__init__.py` - Usage examples
- `CORRECTIONS_BRIDGE_API.md` - Avant/Après code

---

## 🚀 Quand Commencer ?

### Option A : Avant PROMPT 3 (Bridge 100% fonctionnel)
**Si :**
- Vous voulez tests E2E immédiatement
- Vous préférez compléter chaque étape avant continuer
- Vous avez 4-5h disponibles

### Option B : Après PROMPT 3 (Continuer momentum)
**Si :**
- Vous voulez avancer sur async wrapper
- Controllers pas urgents (mocks suffisent)
- Correction peut attendre jusqu'à P4 (Dash UI)

**Recommandation actuelle :** Option B (continuer PROMPT 3)

---

## ✅ Critères de Complétion

Le PROMPT 2 sera **100% complété** quand :

- [x] 8 DataClasses Request/Result créées
- [x] Hiérarchie exceptions complète
- [x] Exports publics documentés
- [ ] 4 controllers utilisent vraies APIs Engine
- [ ] Data helpers créés et testés
- [ ] mypy --strict passe 100%
- [ ] Tests unitaires >80% coverage
- [ ] Documentation exemples fonctionnent

**Statut actuel :** 3/8 critères ✅ (37.5%)

---

**Créé le :** 14 octobre 2025
**Dernière mise à jour :** 14 octobre 2025
**Assigné à :** À déterminer (correction différée)
**Prochaine révision :** Avant PROMPT 4 (Dash UI)
