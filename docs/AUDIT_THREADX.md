# 🔍 AUDIT THREADX : Séparation UI / Métier

**Date** : 2025-10-14 08:30:23
**Auditeur** : Script automatisé
**Statut** : Audit complet

---

## 📊 Résumé

| Métrique | Valeur |
|----------|--------|
| Fichiers Python analysés | 34 |
| Fichiers problématiques | 8 |
| Issues totales | 15 |
| Imports métier trouvés | 7 |
| Calculs en UI détectés | 8 |

**Priorité globale** : 🔴 CRITIQUE

---

## 📁 Détail par fichier


### src\threadx\ui\charts.py
**Sévérité** : 🔴 CRITIQUE

**Calculs détectés** :
- L110: Imputation données
  ```python
  equity = equity.fillna(method="ffill")
  ```
- L556: Transformation données
  ```python
  monthly_returns = equity.resample("M").last().pct_change().dropna() * 100
  ```
- L556: Nettoyage données
  ```python
  monthly_returns = equity.resample("M").last().pct_change().dropna() * 100
  ```

**Actions requises** :
  3. [ ] Extraire la logique de calcul vers le moteur
  4. [ ] Remplacer par des appels asynchrones via bridge
  5. [ ] Créer les dataclasses de requête appropriées

**Complexité** : 🔴 HAUTE


### threadx_dashboard\engine\data_processor.py
**Sévérité** : 🔴 CRITIQUE

**Calculs détectés** :
- L221: Nettoyage données
  ```python
  cleaned = cleaned.dropna(thresh=threshold)
  ```
- L276: Transformation données
  ```python
  resampled = resampled_data.resample(timeframe).agg(available_columns)
  ```
- L290: Nettoyage données
  ```python
  values = data[col].dropna()
  ```

**Actions requises** :
  3. [ ] Extraire la logique de calcul vers le moteur
  4. [ ] Remplacer par des appels asynchrones via bridge
  5. [ ] Créer les dataclasses de requête appropriées

**Complexité** : 🔴 HAUTE


### src\threadx\ui\data_manager.py
**Sévérité** : 🟡 MOYEN

**Imports métier** :
- L23: from ..data.ingest import IngestionManager
  → *Ingestion données*

**Actions requises** :
  1. [ ] Supprimer les imports métier
  2. [ ] Utiliser des appels bridge au lieu des imports directs
  5. [ ] Créer les dataclasses de requête appropriées

**Complexité** : 🟢 FAIBLE


### src\threadx\ui\downloads.py
**Sévérité** : 🟡 MOYEN

**Imports métier** :
- L26: from ..data.ingest import IngestionManager
  → *Ingestion données*

**Actions requises** :
  1. [ ] Supprimer les imports métier
  2. [ ] Utiliser des appels bridge au lieu des imports directs
  5. [ ] Créer les dataclasses de requête appropriées

**Complexité** : 🟢 FAIBLE


### src\threadx\ui\sweep.py
**Sévérité** : 🟡 MOYEN

**Imports métier** :
- L32: from ..optimization.engine import UnifiedOptimizationEngine, DEFAULT_SWEEP_CONFIG
  → *Engine optimisation*
- L33: from ..indicators.bank import IndicatorBank
  → *Banque indicateurs*

**Actions requises** :
  1. [ ] Supprimer les imports métier
  2. [ ] Utiliser des appels bridge au lieu des imports directs
  5. [ ] Créer les dataclasses de requête appropriées

**Complexité** : 🟢 FAIBLE


### apps\streamlit\app.py
**Sévérité** : 🟡 MOYEN

**Imports métier** :
- L48: from threadx.engine.backtest import BacktestEngine
  → *Engine backtest*
- L49: from threadx.performance.metrics import PerformanceCalculator
  → *Calcul perf*

**Calculs détectés** :
- L415: Exécution backtest
  ```python
  returns, trades = st.session_state.engine.run(
  ```

**Actions requises** :
  1. [ ] Supprimer les imports métier
  2. [ ] Utiliser des appels bridge au lieu des imports directs
  3. [ ] Extraire la logique de calcul vers le moteur
  4. [ ] Remplacer par des appels asynchrones via bridge
  5. [ ] Créer les dataclasses de requête appropriées

**Complexité** : 🟡 MOYENNE


### threadx_dashboard\engine\__init__.py
**Sévérité** : 🟡 MOYEN

**Imports métier** :
- L15: from .backtest_engine import BacktestEngine
  → *Engine backtest*

**Actions requises** :
  1. [ ] Supprimer les imports métier
  2. [ ] Utiliser des appels bridge au lieu des imports directs
  5. [ ] Créer les dataclasses de requête appropriées

**Complexité** : 🟢 FAIBLE


### threadx_dashboard\utils\helpers.py
**Sévérité** : 🟡 MOYEN

**Calculs détectés** :
- L90: Nettoyage données
  ```python
  return prices.pct_change().dropna()
  ```

**Actions requises** :
  3. [ ] Extraire la logique de calcul vers le moteur
  4. [ ] Remplacer par des appels asynchrones via bridge
  5. [ ] Créer les dataclasses de requête appropriées

**Complexité** : 🟢 FAIBLE


---

## 🔧 Extractions recommandées

### Extraction #1 : Imputation données

**Fichier source** : `src\threadx\ui\charts.py:110`

**Code problématique** :
```python
equity = equity.fillna(method="ffill")
```

**Action** : Extraire vers moteur de calcul + appel bridge

**Priorité** : 🔴 HAUTE

### Extraction #2 : Transformation données

**Fichier source** : `src\threadx\ui\charts.py:556`

**Code problématique** :
```python
monthly_returns = equity.resample("M").last().pct_change().dropna() * 100
```

**Action** : Extraire vers moteur de calcul + appel bridge

**Priorité** : 🔴 HAUTE

### Extraction #3 : Nettoyage données

**Fichier source** : `src\threadx\ui\charts.py:556`

**Code problématique** :
```python
monthly_returns = equity.resample("M").last().pct_change().dropna() * 100
```

**Action** : Extraire vers moteur de calcul + appel bridge

**Priorité** : 🔴 HAUTE

### Extraction #4 : Nettoyage données

**Fichier source** : `threadx_dashboard\engine\data_processor.py:221`

**Code problématique** :
```python
cleaned = cleaned.dropna(thresh=threshold)
```

**Action** : Extraire vers moteur de calcul + appel bridge

**Priorité** : 🔴 HAUTE

### Extraction #5 : Transformation données

**Fichier source** : `threadx_dashboard\engine\data_processor.py:276`

**Code problématique** :
```python
resampled = resampled_data.resample(timeframe).agg(available_columns)
```

**Action** : Extraire vers moteur de calcul + appel bridge

**Priorité** : 🔴 HAUTE

### Extraction #6 : Nettoyage données

**Fichier source** : `threadx_dashboard\engine\data_processor.py:290`

**Code problématique** :
```python
values = data[col].dropna()
```

**Action** : Extraire vers moteur de calcul + appel bridge

**Priorité** : 🔴 HAUTE

### Extraction #7 : Exécution backtest

**Fichier source** : `apps\streamlit\app.py:415`

**Code problématique** :
```python
returns, trades = st.session_state.engine.run(
```

**Action** : Extraire vers moteur de calcul + appel bridge

**Priorité** : 🟡 MOYENNE

### Extraction #8 : Nettoyage données

**Fichier source** : `threadx_dashboard\utils\helpers.py:90`

**Code problématique** :
```python
return prices.pct_change().dropna()
```

**Action** : Extraire vers moteur de calcul + appel bridge

**Priorité** : 🟡 MOYENNE


---

## ✅ Checklist de validation

Après refactorisation, vérifier :

- [ ] Aucun `import create_engine` dans les fichiers UI
- [ ] Aucun `import IndicatorBank` dans les fichiers UI  
- [ ] Aucun `import PerformanceCalculator` dans les fichiers UI
- [ ] Tous les appels métier passent par self.bridge
- [ ] Tous les widgets UI restent dans les fichiers UI
- [ ] Tests unitaires UI passent (avec mocks bridge)
- [ ] Tests intégration Bridge ↔ Engine passent

---

## 🚀 Prochaines étapes

1. ✅ **Audit complet** (TERMINÉ - ce prompt)
2. ⏳ **Créer src/threadx/bridge/** (Prompt 2)
   - BacktestController
   - IndicatorController  
   - DataController
   - ThreadXBridge (orchestrateur principal)
3. ⏳ **Refactoriser les fichiers UI** (Prompts 3-N)
   - Supprimer imports métier
   - Ajouter appels bridge
   - Créer dataclasses de requête
4. ⏳ **Tests et validation**
   - Tests unitaires bridge
   - Tests intégration UI-Bridge-Engine
   - Validation fonctionnelle complète

---

*Rapport généré automatiquement le 2025-10-14 à 08:30:23*
