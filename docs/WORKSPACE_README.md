# 🏗️ Workspace VS Code ThreadX - Guide Complet

## 📋 Vue d'Ensemble

Le workspace **ThreadX.code-workspace** englobe **TOUT le dossier ThreadX** :
- ✅ Tous les sous-dossiers (`src/`, `scripts/`, `tests/`, `apps/`, `docs/`, etc.)
- ✅ Tous les fichiers Python du projet
- ✅ Configuration complète et unifiée
- ✅ **8 configurations debug** (F5)
- ✅ **9 tâches** pré-configurées (Ctrl+Shift+P)

---

## 🚀 Ouverture du Workspace

### Méthode 1: Fichier Workspace (Recommandé)
```bash
# Depuis terminal/PowerShell
code ThreadX.code-workspace

# Ou depuis VS Code
File > Open Workspace from File > ThreadX.code-workspace
```

### Méthode 2: Dossier (Auto-détecte workspace)
```bash
code d:\ThreadX
```
*VS Code détectera automatiquement `ThreadX.code-workspace` et proposera de l'ouvrir*

---

## 🎯 Configurations Debug (F5)

Le workspace inclut **8 configurations** prêtes à l'emploi :

| # | Nom | Description | Fichier Cible |
|---|-----|-------------|---------------|
| 1 | 🐍 **Python: Fichier Actuel** | Debug fichier Python ouvert | `${file}` |
| 2 | 🔄 **ThreadX: Update Daily Tokens** | Mise à jour quotidienne top 100 | `scripts/update_daily_tokens.py` |
| 3 | 📊 **ThreadX: Analyze Token** | Analyse technique BTCUSDC | `scripts/analyze_token.py` |
| 4 | 🔍 **ThreadX: Scan All Tokens** | Scan multi-tokens | `scripts/scan_all_tokens.py` |
| 5 | ✅ **ThreadX: Tests (pytest)** | Tous les tests unitaires | `tests/` |
| 6 | 🎯 **ThreadX: Test End-to-End** | Test workflow complet | `test_end_to_end_token.py` |
| 7 | 🎨 **ThreadX: Streamlit App** | Application Streamlit | `apps/streamlit/app.py` |
| 8 | 🗂️ **ThreadX: Data Manager** | Gestionnaire données | `launch_data_manager.py` |

### Utilisation
```
1. Appuyer F5
2. Sélectionner configuration
3. Debugger démarre automatiquement
```

---

## 📋 Tâches VS Code (Ctrl+Shift+P > Tasks)

Le workspace inclut **9 tâches** pré-configurées :

| # | Tâche | Arguments | Panel |
|---|-------|-----------|-------|
| 1 | 🔄 **Update Daily Tokens** | --tokens 100 --timeframes 1h,4h --days 365 | Nouveau + Focus |
| 2 | 📊 **Analyze Token (BTCUSDC)** | BTCUSDC --timeframe 1h --days 30 | Nouveau + Focus |
| 3 | 🔍 **Scan All Tokens** | --tokens 100 --timeframe 1h --days 7 | Nouveau + Focus |
| 4 | ✅ **Run All Tests** | tests -v --tb=short | Partagé |
| 5 | 🎯 **Test End-to-End** | test_end_to_end_token.py | Nouveau + Focus |
| 6 | 📦 **Install Requirements** | pip install -r requirements.txt | Nouveau + Focus |
| 7 | 🎨 **Run Streamlit App** | streamlit run apps/streamlit/app.py | Nouveau + Background |
| 8 | 🗂️ **Launch Data Manager** | launch_data_manager.py | Nouveau + Focus |
| 9 | 🧹 **Clean Cache** | Remove cache files | Partagé |

### Utilisation
```
Ctrl+Shift+P > Tasks: Run Task > Sélectionner tâche
```

---

## ⚙️ Settings Consolidés

### 🐍 Python
```json
{
    "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
    "python.analysis.extraPaths": ["src/", "./"],
    "python.analysis.include": ["**"],  // ← TOUT le workspace
    "python.analysis.typeCheckingMode": "off",
    "python.testing.pytestEnabled": true
}
```

### 📝 Formatage
```json
{
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    }
}
```

### 🔍 Exclusions (Performance)
```json
{
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true
    },
    "search.exclude": {
        "**/.venv": true,
        "**/__pycache__": true
    }
}
```

---

## 🌳 Structure Complète Workspace

Le workspace **englobe TOUS ces dossiers** :

```
ThreadX/  ← Workspace Root (path: ".")
├── .venv/                    ✅ Environnement virtuel
├── src/                      ✅ Code source principal
│   └── threadx/
│       ├── data/             ✅ Modules data (tokens, loader)
│       ├── indicators/       ✅ Modules indicateurs
│       ├── ui/               ✅ Interfaces utilisateur
│       ├── utils/            ✅ Utilitaires
│       ├── backtest/         ✅ Backtesting
│       ├── strategy/         ✅ Stratégies
│       └── optimization/     ✅ Optimisation
├── scripts/                  ✅ Scripts production
│   ├── update_daily_tokens.py
│   ├── analyze_token.py
│   └── scan_all_tokens.py
├── tests/                    ✅ Tests unitaires
├── apps/                     ✅ Applications
│   ├── streamlit/            ✅ App Streamlit
│   └── data_manager/         ✅ Data Manager
├── docs/                     ✅ Documentation
├── data/                     ✅ Données (cache, exports)
│   ├── crypto_data_json/
│   ├── crypto_data_parquet/
│   └── exports/
├── benchmarks/               ✅ Benchmarks
├── examples/                 ✅ Exemples
├── token_diversity_manager/  ✅ Token Diversity
├── configs/                  ✅ Configurations
├── cache/                    ✅ Cache
├── indicators_cache/         ✅ Cache indicateurs
├── logs/                     ✅ Logs
└── ThreadX.code-workspace    ⭐ WORKSPACE UNIQUE
```

**Tous ces dossiers sont inclus dans le workspace !**

---

## 🔧 Personnalisation

### Modifier Arguments d'une Tâche

Ouvrir `ThreadX.code-workspace`, chercher section `"tasks"`:

```json
{
    "label": "🔄 ThreadX: Update Daily Tokens",
    "args": [
        "scripts/update_daily_tokens.py",
        "--tokens", "150",              // ← Changer 100 → 150
        "--timeframes", "15m,1h,4h",    // ← Ajouter 15m
        "--days", "180"                 // ← Changer 365 → 180
    ]
}
```

### Ajouter une Nouvelle Tâche

```json
{
    "label": "🎯 Ma Tâche Custom",
    "type": "shell",
    "command": "${config:python.defaultInterpreterPath}",
    "args": ["mon_script.py", "--arg", "value"],
    "presentation": {
        "reveal": "always",
        "panel": "new",
        "focus": true
    },
    "problemMatcher": []
}
```

### Ajouter une Config Debug

```json
{
    "name": "🚀 Mon Debug Custom",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/mon_script.py",
    "args": ["--arg", "value"],
    "cwd": "${workspaceFolder}",
    "env": {
        "PYTHONPATH": "${workspaceFolder}/src:${workspaceFolder}"
    }
}
```

---

## 💼 Workflow Quotidien

### 🌅 Matin - Ouverture Projet
```bash
# 1. Ouvrir workspace
code ThreadX.code-workspace

# 2. Vérifier environnement virtuel activé
# Terminal devrait afficher: (.venv) PS D:\ThreadX>

# 3. Lancer mise à jour quotidienne
Ctrl+Shift+P > Tasks: Run Task > 🔄 ThreadX: Update Daily Tokens
```

### ☀️ Journée - Développement
```bash
# Ouvrir fichier Python (ex: src/threadx/data/tokens.py)
# Modifier code
# Sauvegarder (auto-format Black)
# F5 pour débugger
```

### 🔬 Tests - Validation
```bash
# Méthode 1: Task
Ctrl+Shift+P > Tasks > ✅ ThreadX: Run All Tests

# Méthode 2: Debug
F5 > ✅ ThreadX: Tests (pytest)
```

### 📊 Analyse - Opportunités
```bash
# Scan tokens
Ctrl+Shift+P > Tasks > 🔍 ThreadX: Scan All Tokens

# Analyse token spécifique
Ctrl+Shift+P > Tasks > 📊 ThreadX: Analyze Token (BTCUSDC)
```

### 🌙 Soir - Streamlit App
```bash
# Lancer application web
Ctrl+Shift+P > Tasks > 🎨 ThreadX: Run Streamlit App
# Ouvre automatiquement http://localhost:8501
```

---

## 🎓 Bonnes Pratiques

### ✅ À FAIRE
- **Toujours ouvrir via workspace**: `code ThreadX.code-workspace`
- **Utiliser F5 pour débugger**: 8 configs prêtes
- **Utiliser Tasks pour scripts**: Évite commandes manuelles
- **Laisser auto-format**: Black formatte au save
- **Utiliser PYTHONPATH workspace**: `src/` et `./` inclus
- **Explorer tout le projet**: Workspace couvre 100% du dossier

### ❌ À ÉVITER
- **NE PAS créer `.vscode/settings.json`**: Redondant
- **NE PAS modifier PYTHONPATH manuellement**: Workspace le gère
- **NE PAS ignorer extensions recommandées**: Installées auto
- **NE PAS ouvrir sous-dossiers séparément**: Utiliser workspace unique

---

## 🔍 Découvrir le Workspace

### Explorer Tous les Fichiers Python
```
Ctrl+P (Quick Open)
> Taper *.py
> Tous les fichiers Python du workspace s'affichent
```

### Rechercher dans Tout le Workspace
```
Ctrl+Shift+F (Search)
> Recherche dans TOUS les fichiers (sauf exclusions)
```

### Naviguer Symboles Workspace
```
Ctrl+T (Go to Symbol in Workspace)
> Liste TOUTES les classes/fonctions du workspace
```

---

## 📦 Extensions Recommandées

Le workspace recommande **9 extensions** :

| Extension | ID | Fonction |
|-----------|-----|----------|
| Python | `ms-python.python` | Support Python |
| Pylance | `ms-python.vscode-pylance` | Language server |
| Black Formatter | `ms-python.black-formatter` | Formatage |
| Jupyter | `ms-toolsai.jupyter` | Notebooks |
| Code Spell Checker | `streetsidesoftware.code-spell-checker` | Orthographe EN |
| French Spell Checker | `streetsidesoftware.code-spell-checker-french` | Orthographe FR |
| Git Graph | `mhutchie.git-graph` | Visualisation Git |
| GitLens | `eamodio.gitlens` | Git avancé |
| PowerShell | `ms-vscode.powershell` | Support PowerShell |

VS Code propose automatiquement de les installer à l'ouverture du workspace.

---

## 🧹 Nettoyage Cache

### Via Tâche (Recommandé)
```
Ctrl+Shift+P > Tasks > 🧹 ThreadX: Clean Cache
```

### Manuel
```powershell
Remove-Item -Path cache/*,indicators_cache/*,__pycache__,.pytest_cache -Recurse -Force
```

---

## ✅ Validation Workspace

### Vérifier Workspace Actif
```
Barre inférieure VS Code devrait afficher:
📁 ThreadX (Root)
```

### Vérifier Environnement Python
```
Terminal devrait afficher:
(.venv) PS D:\ThreadX>
```

### Vérifier PYTHONPATH
```powershell
# Dans terminal VS Code
$env:PYTHONPATH
# Devrait afficher: D:\ThreadX\src;D:\ThreadX
```

---

## 🎯 Cas d'Usage Avancés

### Debug Multi-Fichiers
```bash
# 1. Mettre breakpoints dans plusieurs fichiers
#    Ex: src/threadx/data/tokens.py (ligne 50)
#        src/threadx/data/loader.py (ligne 100)
# 2. F5 > Sélectionner config
# 3. Debugger s'arrête sur tous les breakpoints
```

### Test Fichier Spécifique
```bash
# 1. Ouvrir fichier test (ex: tests/test_tokens.py)
# 2. Clic droit dans fichier
# 3. "Run Current Test File"
# OU
# 4. F5 > ✅ ThreadX: Tests (pytest)
```

### Lancer Streamlit en Background
```bash
Ctrl+Shift+P > Tasks > 🎨 ThreadX: Run Streamlit App
# Panel terminal reste ouvert
# Continuer à coder pendant que app tourne
```

---

## 📖 Documentation Associée

### Guides
- **WORKSPACE_README.md** - Guide rapide (ce fichier)
- **docs/WORKSPACE_CONFIGURATION.md** - Configuration détaillée
- **WORKSPACE_FINAL_REPORT.md** - Rapport création workspace

### Scripts
- **cleanup_workspace.ps1** - Nettoyage configs redondantes
- **scripts/update_daily_tokens.py** - Mise à jour quotidienne
- **scripts/analyze_token.py** - Analyse technique
- **scripts/scan_all_tokens.py** - Scan multi-tokens

---

## 🆘 Troubleshooting

### "Python interpreter not found"
```bash
# Vérifier .venv existe
ls .venv

# Si absent, créer:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### "Module not found" errors
```bash
# Vérifier PYTHONPATH
$env:PYTHONPATH
# Devrait inclure: src/ et ./

# Recharger window
Ctrl+Shift+P > Developer: Reload Window
```

### Tasks ne s'exécutent pas
```bash
# Vérifier workspace ouvert
# Barre inférieure: 📁 ThreadX (Root)

# Si absent, rouvrir workspace:
File > Open Workspace from File > ThreadX.code-workspace
```

---

## 🌟 Avantages Workspace Unique

### ✅ Avant (Sans Workspace)
```
- Configurations dispersées
- PYTHONPATH manuel
- Pas de configs debug
- Pas de tâches
- Settings incohérents
```

### ✅ Après (Avec Workspace)
```
✅ Configuration unique centralisée
✅ PYTHONPATH automatique (src/ + ./)
✅ 8 configs debug prêtes (F5)
✅ 9 tâches pré-configurées (Ctrl+Shift+P)
✅ Settings cohérents workspace entier
✅ Extensions recommandées automatiques
✅ Exclusions optimisées (performance)
✅ Versionnable Git (équipe entière)
✅ Portable (chemins relatifs)
```

---

## 📊 Statistiques Workspace

| Métrique | Valeur |
|----------|--------|
| **Dossiers couverts** | 100% (TOUT ThreadX) |
| **Configs debug** | 8 |
| **Tâches** | 9 |
| **Settings** | 50+ |
| **Extensions recommandées** | 9 |
| **Exclusions** | 15+ patterns |
| **PYTHONPATH** | `src/` + `./` (auto) |

---

## 🎉 Résumé

### Ce que fait le workspace
1. ✅ **Englobe TOUT le dossier ThreadX** (100% des fichiers/dossiers)
2. ✅ Configure Python automatiquement (interpreter, paths)
3. ✅ Fournit 8 configs debug (F5)
4. ✅ Fournit 9 tâches (Ctrl+Shift+P)
5. ✅ Formatte automatiquement (Black on save)
6. ✅ Active tests pytest
7. ✅ Recommande extensions
8. ✅ Optimise performance (exclusions)

### Comment l'utiliser
```bash
# 1. Ouvrir
code ThreadX.code-workspace

# 2. Débugger
F5 > Sélectionner config

# 3. Exécuter tâches
Ctrl+Shift+P > Tasks

# 4. Coder
# Auto-format, IntelliSense, imports automatiques
```

---

**✨ Workspace ThreadX : Tout le projet, une seule configuration ! ✨**

---

*Guide créé le 16 octobre 2025*
*Workspace: ThreadX.code-workspace*
*Coverage: 100% du dossier ThreadX*
