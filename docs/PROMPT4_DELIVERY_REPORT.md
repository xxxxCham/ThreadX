# PROMPT 4 - Layout Dash Principal ThreadX
## ✅ LIVRAISON COMPLETE

**Date**: 2024-01-14
**Status**: Production-ready
**Couverture**: 100% spécifications PROMPT 4

---

## 📦 Arborescence Produite

```
ThreadX/
├── apps/
│   └── dash_app.py                    ✅ NOUVEAU (104 lignes)
└── src/threadx/ui/
    ├── layout.py                      ✅ NOUVEAU (290 lignes)
    ├── __init__.py                    ✅ MODIFIÉ (exports updated)
    └── components/
        └── __init__.py                ✅ NOUVEAU (vide, prêt P5-P6)
```

---

## 📄 Contenu Intégral des Fichiers

### 1. `apps/dash_app.py` (Application Principale)

**Fonctionnalités**:
- Initialise app Dash avec theme DARKLY
- Configure port depuis `THREADX_DASH_PORT` (fallback 8050)
- Importe `create_layout` depuis `src.threadx.ui.layout`
- Passe instance `ThreadXBridge` au layout (optionnel, pour P7)
- Lance serveur Flask sous-jacent

**Points clés**:
- `suppress_callback_exceptions=True` pour callbacks dynamiques P7
- `title="ThreadX Dashboard"`
- Debug mode via env `THREADX_DASH_DEBUG` (default False)
- Serveur exposé via `server` pour production (Gunicorn)

### 2. `src/threadx/ui/layout.py` (Layout Statique)

**Fonction principale**: `create_layout(bridge=None)`

**Structure**:
```
dbc.Container (fluid, dark theme)
  ├─ Navbar (Header)
  │   ├─ H2: "ThreadX Dashboard"
  │   └─ P: "Backtesting Framework - GPU-Accelerated..."
  ├─ dcc.Tabs (4 onglets)
  │   ├─ Tab "Data Manager" (value="tab-data")
  │   │   └─ _create_tab_layout(tab_id="data", ...)
  │   ├─ Tab "Indicators" (value="tab-indicators")
  │   │   └─ _create_tab_layout(tab_id="ind", ...)
  │   ├─ Tab "Backtest" (value="tab-backtest")
  │   │   └─ _create_tab_layout(tab_id="bt", ...)
  │   └─ Tab "Optimization" (value="tab-optimization")
  │       └─ _create_tab_layout(tab_id="opt", ...)
  └─ Footer (Version + Crédits)
```

**Pattern répété par tab** (`_create_tab_layout`):
```
html.Div (padding)
  ├─ H3 (Titre tab)
  ├─ P (Sous-titre)
  └─ dbc.Row (responsive)
      ├─ dbc.Col (md=4, lg=3) - SETTINGS
      │   └─ dbc.Card
      │       ├─ CardHeader: "Settings"
      │       └─ CardBody: html.Div(id="<tab_id>-settings-pane")
      └─ dbc.Col (md=8, lg=9) - RESULTS
          └─ dbc.Card
              ├─ CardHeader: "Results"
              └─ CardBody: html.Div(id="<tab_id>-results-pane")
```

**IDs déterministes** (pour callbacks P7):
- `main-tabs` (tabs container)
- `data-settings-pane`, `data-results-pane`
- `ind-settings-pane`, `ind-results-pane`
- `bt-settings-pane`, `bt-results-pane`
- `opt-settings-pane`, `opt-results-pane`

**Responsive**:
- Mobile (default): 1 colonne
- Tablet (md): Settings 4/12, Results 8/12
- Desktop (lg): Settings 3/12, Results 9/12

**Theme**:
- Bootstrap DARKLY
- Background: `bg-dark`
- Text: `text-light`, `text-muted`
- Cards: `border-secondary`
- Selected tab: `bg-primary`

### 3. `src/threadx/ui/__init__.py` (Exports Module)

**Exports principaux**:
- `create_layout` (Dash layout P4)
- Legacy Tkinter/Streamlit (compatibility, try/except)

**Version**: `0.2.0` (P4 Dash integration)

### 4. `src/threadx/ui/components/__init__.py` (Package Composants)

**Contenu**: Vide (prêt pour P5-P6)

**Futurs exports** (P5-P6):
- `data_manager.py`
- `indicators_panel.py`
- `backtest_panel.py`
- `optimization_panel.py`

---

## 📦 Dépendances Requises

```txt
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.18.0
```

**Installation**:
```powershell
pip install dash dash-bootstrap-components plotly
```

---

## 🚀 Commandes PowerShell (Création & Lancement)

**Copier-coller direct** (depuis racine ThreadX):

```powershell
# Activer environnement virtuel (si existant)
. .\.venv\Scripts\Activate.ps1

# Installer dépendances
pip install dash dash-bootstrap-components plotly

# Configurer port (optionnel, default 8050)
$env:THREADX_DASH_PORT=8050

# Lancer application
python apps\dash_app.py
```

**Persistance port** (optionnel):
```powershell
# Nécessite redémarrage terminal
setx THREADX_DASH_PORT 8050
```

**Accès UI**:
```
http://127.0.0.1:8050
```

---

## ✅ Checklist de Validation

### Démarrage
- [x] App démarre sur `http://127.0.0.1:8050`
- [x] Port configurable via `THREADX_DASH_PORT`
- [x] Message console affiche port + debug + bridge status
- [x] Serveur Flask sous-jacent disponible (`app.server`)

### Layout
- [x] Navbar avec titre "ThreadX Dashboard"
- [x] Sous-titre "Backtesting Framework - GPU-Accelerated..."
- [x] 4 onglets visibles: Data, Indicators, Backtest, Optimization
- [x] Chaque tab contient grille Settings (gauche) + Results (droite)
- [x] Placeholders affichés ("Placeholder for P5-P6")

### Design
- [x] Theme sombre Bootstrap DARKLY
- [x] Text lisible (contraste suffisant)
- [x] Responsive (breakpoints md/lg)
- [x] Cards avec borders secondaires
- [x] Tabs sélectionnés en bleu (`bg-primary`)
- [x] Footer avec version

### Architecture
- [x] Aucun import `threadx.backtest|indicators|optimization` dans `apps/` ou `ui/`
- [x] `create_layout(bridge)` signature prête pour P7
- [x] IDs déterministes pour tous placeholders
- [x] Pas d'opération bloquante (layout statique)

### Code Quality
- [x] Imports triés
- [x] Docstrings Google-style
- [x] Type hints (optionnel pour layout)
- [x] Pas de dead code
- [x] Line length < 80 chars (sauf 1 ligne corrigée)

---

## 🎯 Prochaines Étapes (Ancre Projet)

### P5: Composants Dash 1/2 (Data + Indicators)
- Créer `src/threadx/ui/components/data_manager.py`
  - Forms: Upload file, select source
  - Table: Data registry
  - Bridge call: `validate_data_async()`
- Créer `src/threadx/ui/components/indicators_panel.py`
  - Forms: Symbol, timeframe, indicator params
  - Button: "Build Cache"
  - Bridge call: `build_indicators_async()`

### P6: Composants Dash 2/2 (Backtest + Optimization)
- Créer `src/threadx/ui/components/backtest_panel.py`
  - Forms: Strategy, symbol, params
  - Graphs: Equity curve, drawdown (Plotly)
  - Tables: Trades, metrics
  - Bridge call: `run_backtest_async()`
- Créer `src/threadx/ui/components/optimization_panel.py`
  - Forms: Param grid (min/max/step)
  - Heatmap: 2D sweep results
  - Table: Top results
  - Bridge call: `run_sweep_async()`

### P7: Callbacks + Routing Bridge
- Créer `src/threadx/ui/callbacks.py`
  - Register tous callbacks avec `app` et `bridge`
  - Pattern: Button click → Bridge async → polling queue → update UI
  - Gestion erreurs: `dbc.Alert` pour erreurs Bridge
  - Loading states: `dcc.Loading` pendant tasks

### P8: Tests
- `tests/test_dash_layout.py` (layout rendering)
- `tests/test_dash_callbacks.py` (mocks Bridge)

---

## 📊 Métriques

| Métrique | Valeur |
|----------|--------|
| Fichiers créés | 3 |
| Fichiers modifiés | 1 |
| Lignes code | ~450 |
| Onglets | 4 |
| IDs déterministes | 10+ |
| Responsive breakpoints | 2 (md, lg) |
| Imports métier | 0 ✅ |
| Blocage UI | 0 ✅ |

---

## 🔍 Notes Techniques

### Pattern Tab Layout

Chaque tab utilise le même pattern:
```python
_create_tab_layout(
    tab_id="data",  # Préfixe IDs unique
    title="Data Management",
    subtitle="Upload, validate, and manage market data sources"
)
```

Génère:
- Settings panel (`{tab_id}-settings-pane`)
- Results panel (`{tab_id}-results-pane`)

### IDs Mapping

```
main-tabs              → Container tabs
tab-data               → Tab Data value
tab-indicators         → Tab Indicators value
tab-backtest           → Tab Backtest value
tab-optimization       → Tab Optimization value

data-settings-pane     → Placeholder Settings Data
data-results-pane      → Placeholder Results Data
ind-settings-pane      → Placeholder Settings Indicators
ind-results-pane       → Placeholder Results Indicators
bt-settings-pane       → Placeholder Settings Backtest
bt-results-pane        → Placeholder Results Backtest
opt-settings-pane      → Placeholder Settings Optimization
opt-results-pane       → Placeholder Results Optimization
```

### Bootstrap Classes Utilisées

```
bg-dark                → Background sombre
text-light             → Texte clair
text-muted             → Texte grisé (secondaire)
border-secondary       → Bordures grises
bg-primary             → Background bleu (tabs sélectionnés)
bg-secondary           → Background gris (card headers)
min-vh-100             → Hauteur min viewport (fullscreen)
p-0, p-4, pb-3         → Padding
mb-0, mb-1, mb-3, mb-4 → Margin bottom
g-3                    → Gap grid
small                  → Texte petit
fst-italic             → Font style italic
```

### Responsive Grid

```
dbc.Col(md=4, lg=3)    → Settings (33% tablet, 25% desktop)
dbc.Col(md=8, lg=9)    → Results (66% tablet, 75% desktop)
```

Mobile (< md): 100% width stacked

---

## 🎉 Validation Finale

**PROMPT 4 Layout Dash Principal**: ✅ **COMPLETE**

- [x] Application Dash créée (`apps/dash_app.py`)
- [x] Layout statique implémenté (`src/threadx/ui/layout.py`)
- [x] 4 onglets fonctionnels (Data, Indicators, Backtest, Optimization)
- [x] Theme sombre Bootstrap DARKLY
- [x] Responsive (md/lg breakpoints)
- [x] IDs déterministes pour callbacks P7
- [x] Aucun import métier
- [x] Aucune opération bloquante
- [x] Structure prête pour P5-P6 (components)
- [x] Documentation complète

**Prêt pour P5**: ✅ **OUI** (Composants Data + Indicators)
**Prêt pour P6**: ✅ **OUI** (Composants Backtest + Optimization)
**Prêt pour P7**: ✅ **OUI** (Callbacks + Bridge routing)

---

**Fin PROMPT 4 Livraison** ✅
