# PROMPT 5 - Composants Dash : Data + Indicators
## 📦 Livraison des Premiers Composants UI ThreadX

**Date**: 14 octobre 2025
**Statut**: ✅ COMPLET
**Phase**: P5 / 10 (Prompts Dash UI)

---

## 🎯 Objectif de la Phase

Créer les deux premiers composants Dash modulaires pour les onglets "Data Management" et "Indicators Configuration". Ces composants sont des **placeholders UI purs** (formulaires, tables vides, loading spinners) qui seront connectés au `ThreadXBridge` en P7 via des callbacks asynchrones.

---

## 📂 Fichiers Créés

### 1. `src/threadx/ui/components/data_manager.py` (253 lignes)

**Fonction Principale**: `create_data_manager_panel() -> html.Div`

**Structure du Composant**:
```
Panel Principal
├── Titre + Sous-titre
└── Row Responsive (g-3)
    ├── Col (md=6) : Configuration Card
    │   ├── Upload File (dcc.Upload)
    │   ├── Data Source Dropdown (Yahoo, Local, Binance, Custom)
    │   ├── Symbol Input (text)
    │   ├── Timeframe Dropdown (1m, 5m, 15m, 1h, 4h, 1d)
    │   └── Validate Button (primary)
    └── Col (md=6) : Results Card
        ├── Alert (data-alert)
        ├── Loading (dcc.Loading)
        │   └── DataTable (data-registry-table)
        │       └── Columns: Symbol, Timeframe, Rows, Status, Quality
        └── Empty State (placeholder)
```

**IDs Exposés** (pour callbacks P7):
- **Inputs**:
  - `data-upload`: Upload CSV/Parquet files
  - `data-source`: Dropdown source sélection
  - `data-symbol`: Input symbole (e.g., BTCUSDT)
  - `data-timeframe`: Dropdown timeframe
  - `validate-data-btn`: Button validation (n_clicks trigger)

- **Outputs**:
  - `data-registry-table`: DataTable pour registry datasets
  - `data-alert`: Alert pour messages success/error
  - `data-loading`: Loading spinner

**Style & Design**:
- Thème: Bootstrap DARKLY (`bg-dark`, `border-secondary`, `text-light`)
- Layout: Cards avec CardHeader/CardBody
- Responsive: `dbc.Col(md=6)` pour split 50/50 desktop
- Table: Dark mode (`backgroundColor: #212529`, striped rows)
- Empty State: Icon `bi-database`, texte explicatif

---

### 2. `src/threadx/ui/components/indicators_panel.py` (275 lignes)

**Fonction Principale**: `create_indicators_panel() -> html.Div`

**Structure du Composant**:
```
Panel Principal
├── Titre + Sous-titre
└── Row Responsive (g-3)
    ├── Col (md=6) : Configuration Card
    │   ├── Symbol Dropdown (dynamic from registry)
    │   ├── Timeframe Dropdown
    │   ├── EMA Parameters
    │   │   └── Period Input (number, default=20)
    │   ├── RSI Parameters
    │   │   └── Period Input (number, default=14)
    │   ├── Bollinger Bands Parameters
    │   │   ├── Period Input (number, default=20)
    │   │   └── Std Dev Input (number, default=2.0)
    │   └── Build Button (success)
    └── Col (md=6) : Results Card
        ├── Alert (indicators-alert)
        ├── Loading (dcc.Loading)
        │   └── Table (indicators-cache-table)
        │       ├── Thead: Indicator, Parameters, Status, Size
        │       └── Tbody (indicators-cache-body)
        └── Empty State (placeholder)
```

**IDs Exposés** (pour callbacks P7):
- **Inputs**:
  - `indicators-symbol`: Dropdown symbole (options dynamiques)
  - `indicators-timeframe`: Dropdown timeframe
  - `ema-period`: Input période EMA
  - `rsi-period`: Input période RSI
  - `bollinger-period`: Input période Bollinger
  - `bollinger-std`: Input standard deviation Bollinger
  - `build-indicators-btn`: Button build cache (n_clicks trigger)

- **Outputs**:
  - `indicators-cache-body`: Tbody pour status indicateurs
  - `indicators-alert`: Alert pour messages
  - `indicators-loading`: Loading spinner

**Style & Design**:
- Thème: Cohérent avec data_manager (dark theme)
- Layout: Cards symétriques
- Forms: Labels clairs, inputs groupés par indicateur
- Table: `dbc.Table` (striped, bordered, hover, dark)
- Empty State: Icon `bi-graph-up`, texte minimal

---

### 3. `src/threadx/ui/components/__init__.py` (Mis à Jour)

Exports ajoutés:
```python
from threadx.ui.components.data_manager import (
    create_data_manager_panel
)
from threadx.ui.components.indicators_panel import (
    create_indicators_panel
)

__all__ = [
    "create_data_manager_panel",
    "create_indicators_panel",
]
```

---

## 🎨 Design & Architecture

### Choix Techniques

1. **Pattern Card-Based**: Utilisation systématique de `dbc.Card` avec `CardHeader` et `CardBody` pour structuration visuelle claire.

2. **Responsive Design**:
   - `dbc.Col(md=6)` pour split 50/50 sur desktop (≥768px)
   - Auto-stacking sur mobile (< 768px)
   - `className="g-3"` pour gutters cohérents

3. **Dark Theme Cohérent**:
   - Background: `#212529` (tables), `bg-dark` (cards)
   - Borders: `border-secondary` (#6c757d)
   - Text: `text-light` (headers), `text-muted` (hints)

4. **IDs Déterministes**: Préfixes par composant (`data-*`, `indicators-*`) pour éviter conflits dans l'app globale.

5. **Placeholders Statiques**:
   - Tables vides (`data=[]`, `children=[]`)
   - Options dropdowns partiellement statiques (timeframes) ou vides (symbols dynamiques en P7)
   - Empty states visibles par défaut

### Conformité aux Contraintes

✅ **ZÉRO Logique Métier**: Pas d'imports de `threadx.backtest`, `threadx.indicators`, ou `threadx.optimization`.
✅ **Pas de Calculs**: Aucun appel à pandas, numpy, ou fonctions de calcul.
✅ **Imports Triés**: Alphabétiquement (`dash_bootstrap_components`, puis `dash`).
✅ **PEP8**: Indentation 4 espaces, line length ≤ 79 chars (reformulation profondeur).
✅ **Docstrings Google-Style**: Concises, focus sur usage.
✅ **Pas de Code Exécutable**: Pas de `if __name__ == '__main__'`.

---

## 🔗 Intégration avec P4 (Layout Principal)

Les composants sont **prêts pour intégration** dans `src/threadx/ui/layout.py` via:

```python
from threadx.ui.components import (
    create_data_manager_panel,
    create_indicators_panel,
)

# Dans create_layout()
dcc.Tab(
    label="Data",
    value="data",
    children=create_data_manager_panel(),
),
dcc.Tab(
    label="Indicators",
    value="indicators",
    children=create_indicators_panel(),
),
```

**Note**: Intégration manuelle car P4 `layout.py` utilise déjà un pattern statique. Modification mineure nécessaire pour remplacer placeholders "Coming Soon" par appels fonctions.

---

## 🚀 Callbacks Futurs (P7)

### Data Manager

**Callback 1: Upload File**
```python
@callback(
    Output("data-registry-table", "data"),
    Output("data-alert", "children"),
    Output("data-alert", "is_open"),
    Input("validate-data-btn", "n_clicks"),
    State("data-upload", "contents"),
    State("data-source", "value"),
    State("data-symbol", "value"),
    State("data-timeframe", "value"),
)
def validate_data(n_clicks, contents, source, symbol, timeframe):
    if n_clicks == 0:
        raise PreventUpdate

    # Build DataRequest
    request = DataRequest(
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        # ...
    )

    # Async call to Bridge
    task_id = bridge.validate_data_async(request)

    # Poll via dcc.Interval (separate callback)
    # Update table via bridge.get_event() result
    pass
```

**Callback 2: Polling Updates**
```python
@callback(
    Output("data-registry-table", "data"),
    Input("polling-interval", "n_intervals"),
)
def poll_data_updates(n_intervals):
    event = bridge.get_event()
    if event and event[0] == "data_validated":
        return event[2]["registry"]  # payload
    raise PreventUpdate
```

### Indicators Panel

**Callback 1: Build Cache**
```python
@callback(
    Output("indicators-cache-body", "children"),
    Output("indicators-alert", "children"),
    Output("indicators-alert", "is_open"),
    Input("build-indicators-btn", "n_clicks"),
    State("indicators-symbol", "value"),
    State("indicators-timeframe", "value"),
    State("ema-period", "value"),
    State("rsi-period", "value"),
    State("bollinger-period", "value"),
    State("bollinger-std", "value"),
)
def build_indicators(n_clicks, symbol, tf, ema, rsi, bb_p, bb_std):
    if n_clicks == 0:
        raise PreventUpdate

    # Build IndicatorRequest
    request = IndicatorRequest(
        symbol=symbol,
        timeframe=tf,
        indicators=["ema", "rsi", "bollinger"],
        params={"ema": {"period": ema}, ...},
    )

    # Async call to Bridge
    task_id = bridge.build_indicators_async(request)

    # Poll via dcc.Interval (separate callback)
    pass
```

---

## ✅ Checklist de Validation

### Tests Manuels (Préparation P7)

- [ ] **Test 1**: Lancer `python apps/dash_app.py` → App démarre sans erreur
- [ ] **Test 2**: Naviguer vers onglet "Data" → Formulaire affiché, table vide visible
- [ ] **Test 3**: Naviguer vers onglet "Indicators" → Formulaire affiché, table vide visible
- [ ] **Test 4**: Inspecter DOM → Vérifier IDs (`data-upload`, `indicators-symbol`, etc.)
- [ ] **Test 5**: Responsive → Tester sur mobile/tablet (DevTools)
- [ ] **Test 6**: Thème sombre → Vérifier cohérence visuelle (fond noir, texte clair)

**Statut**: ⏳ Tests dépendent de l'installation de `dash` et `dash-bootstrap-components` (voir section suivante).

### Tests Code

- [x] **Imports**: Triés alphabétiquement, pas d'imports inutilisés
- [x] **PEP8**: Line length ≤ 79 chars (restructuration avec variables intermédiaires)
- [x] **Docstrings**: Google-style, concises
- [x] **IDs**: Uniques, déterministes, cohérents avec nomenclature P4
- [x] **Zéro Logique Métier**: Aucun import de backtest/indicators/optimization
- [x] **Placeholders**: Tables vides, alerts fermées, loading wraps présents

---

## 📋 Dépendances & Installation

### Packages Requis (requirements.txt)

```txt
# Dash Framework
dash>=2.14.0
dash-bootstrap-components>=1.5.0
dash-table>=5.0.0

# Core ThreadX (déjà installés)
pandas>=2.0.0
numpy>=1.24.0
```

### Installation PowerShell

```powershell
# Activer environnement Python
python -m venv venv
.\venv\Scripts\Activate.ps1

# Installer dépendances Dash
pip install dash dash-bootstrap-components

# Vérifier installation
python -c "import dash; import dash_bootstrap_components; print('OK')"
```

**Note**: Les erreurs de lint "Impossible de résoudre l'importation" sont **normales** tant que les packages ne sont pas installés. Elles disparaîtront après `pip install dash dash-bootstrap-components`.

---

## 🔄 Prochaines Étapes (P6 - P10)

### Immédiat (P6): Backtest + Optimization Panels

**Fichiers à Créer**:
- `src/threadx/ui/components/backtest_panel.py`
  - Inputs: Strategy selector, params (initial_capital, fees)
  - Outputs: dcc.Graph (equity curve, drawdown), metrics table

- `src/threadx/ui/components/optimization_panel.py`
  - Inputs: Param grid (multi-range sliders)
  - Outputs: dcc.Graph (heatmap), best params table

**Structure Similaire**:
- Pattern Card 50/50
- IDs préfixés (`bt-*`, `opt-*`)
- Empty states + loading

### Moyen Terme (P7): Callbacks & Bridge Integration

**Fichier à Créer**:
- `src/threadx/ui/callbacks.py`
  - Fonction: `register_callbacks(app, bridge)`
  - Patterns: `@callback` pour chaque interaction
  - Polling: `dcc.Interval` (500ms) → `bridge.get_event()`
  - Error Handling: `dbc.Alert` pour `BridgeError`

**Dépendances**:
- `ThreadXBridge` (P3 déjà complet)
- `DataRequest`, `IndicatorRequest`, `BacktestRequest`, `SweepRequest` (P2 déjà complet)

### Long Terme (P8 - P10)

- **P8**: Tests unitaires (`test_dash_callbacks.py` avec mocks)
- **P9**: CLI refactoring pour utiliser Bridge (éliminer duplication)
- **P10**: Documentation architecture (`ARCHITECTURE.md`)

---

## 📊 Métriques de la Livraison

| Métrique | Valeur | Status |
|----------|--------|--------|
| Fichiers Créés | 2 | ✅ |
| Fichiers Modifiés | 1 (`__init__.py`) | ✅ |
| Lignes de Code | ~530 | ✅ |
| IDs Exposés | 15 | ✅ |
| Imports Métier | 0 | ✅ |
| Erreurs Lint (code) | 0 | ✅ |
| Erreurs Lint (imports) | 4 (attendues) | ⏳ |
| Tests Manuels | 0/6 | ⏳ |

**Légende**: ✅ Complet | ⏳ Dépend installation Dash

---

## 🎓 Leçons & Décisions

### Décisions Techniques

1. **Pattern Card vs Div**: Choix de `dbc.Card` pour:
   - Séparation visuelle claire (headers colorés)
   - Padding/margins cohérents
   - Responsive out-of-the-box

2. **DataTable vs Table**:
   - **Data Manager**: `dash_table.DataTable` pour features avancées (sorting, filtering futurs)
   - **Indicators Panel**: `dbc.Table` pour simplicité (données statiques, peu de colonnes)

3. **Empty States**: Inclusion systématique pour UX (éviter confusion "bug ou pas de données?").

4. **Options Dynamiques**: Dropdowns avec `options=[]` (symbols) pour remplissage dynamique en P7 via callbacks.

### Challenges & Solutions

**Challenge 1**: Indentation excessive (lignes > 79 chars)
**Solution**: Extraction variables intermédiaires (`config_card`, `results_card`) pour réduire profondeur.

**Challenge 2**: Balance entre modularité et lisibilité
**Solution**: Fonctions uniques par composant (`create_*_panel()`) plutôt que classes ou multi-fonctions.

**Challenge 3**: Compatibilité futures callbacks
**Solution**: IDs systématiques, structure dict cohérente (facilite State/Input mapping).

---

## 📝 Recommandations pour Développement Futur

1. **Validation des Inputs**: En P7, ajouter callbacks pour validation côté client (e.g., symbol format, period ranges).

2. **Tooltips**: Intégrer `dbc.Tooltip` pour expliquer params complexes (e.g., Bollinger std dev).

3. **Themes Switching**: Préparer infrastructure pour switch dark/light mode (CSS variables).

4. **Accessibility**: Ajouter `aria-label` sur inputs interactifs pour screen readers.

5. **Performance**: Utiliser `memoization` sur fonctions de layout si instanciation répétée.

---

## 📞 Support & Contact

**Questions P5**:
- Architecture composants: Voir docstrings dans `data_manager.py` et `indicators_panel.py`
- IDs callbacks: Référencer section "IDs Exposés" ci-dessus
- Intégration layout: Voir section "Intégration avec P4"

**Prochaines Questions (P6)**:
- Backtest panel: dcc.Graph usage, Plotly figure structure
- Optimization panel: Heatmap generation, param grid UI

---

**Status Final**: ✅ **PROMPT 5 COMPLET**
**Prêt pour**: P6 (Backtest + Optimization Panels)
**Bloquant**: Installation Dash (pour tests manuels)

---

*Rapport Généré: 14 octobre 2025*
*ThreadX Framework - Phase Dash UI (P5/10)*
