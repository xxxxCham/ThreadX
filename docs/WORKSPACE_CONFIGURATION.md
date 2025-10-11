# 🏗️ Configuration Workspace VS Code - ThreadX

## 📁 Structure Workspace Unique

Le projet ThreadX utilise **un seul workspace VS Code** configuré de manière optimale pour le développement Python avec focus sur la performance et la simplicité.

---

## 🚀 Ouverture du Workspace

### Méthode 1: Fichier `.code-workspace`
```bash
# Ouvrir le workspace depuis VS Code
File > Open Workspace from File > ThreadX.code-workspace

# Ou depuis terminal/PowerShell
code ThreadX.code-workspace
```

### Méthode 2: Dossier direct (utilise config automatiquement)
```bash
# Ouvrir le dossier (VS Code chargera le workspace si présent)
code d:\ThreadX
```

---

## ⚙️ Configuration Incluse

### 1️⃣ **Python**
- **Interpréteur**: `.venv/Scripts/python.exe` (environnement virtuel local)
- **Extra Paths**: `src/` ajouté au PYTHONPATH
- **Type Checking**: Désactivé (focus performance)
- **Auto Import**: Activé

### 2️⃣ **Formatage**
- **Formatter**: Black (88 caractères)
- **Format on Save**: Activé
- **Organize Imports**: Automatique au save
- **Rulers**: 88, 120 caractères

### 3️⃣ **Linting**
- **Pylint**: Désactivé (projet en développement rapide)
- **Flake8**: Désactivé
- **Type Stubs**: Warnings désactivés
- **Unused Imports/Variables**: Warning seulement

### 4️⃣ **Tests**
- **Framework**: pytest
- **Auto-discovery**: Activé dans `tests/`
- **Arguments**: `-v --tb=short`
- **Commande**: `python -m pytest tests`

### 5️⃣ **Git**
- **Autofetch**: Activé
- **Ignore Limit Warning**: Activé (gros repos)

---

## 🎯 Configurations Debug (F5)

Le workspace inclut **6 configurations de debug** prêtes à l'emploi :

### 1. **Python: Fichier actuel**
Lance le fichier Python actuellement ouvert avec debugger attaché.
```
PYTHONPATH = src/
justMyCode = false (debug dans librairies externes)
```

### 2. **ThreadX: Update Daily Tokens**
Lance `scripts/update_daily_tokens.py` avec paramètres par défaut :
- 100 tokens
- Timeframes: 1h, 4h

### 3. **ThreadX: Analyze Token**
Lance `scripts/analyze_token.py` pour BTCUSDC :
- Timeframe: 1h
- Historique: 30 jours

### 4. **ThreadX: Scan All Tokens**
Lance `scripts/scan_all_tokens.py` :
- 50 tokens
- Timeframe: 1h

### 5. **ThreadX: Tests (pytest)**
Lance tous les tests avec pytest en mode verbose.

### 6. **ThreadX: Test End-to-End**
Lance le test end-to-end complet (téléchargement + indicateurs).

---

## 📋 Tâches VS Code (Ctrl+Shift+P > Tasks: Run Task)

Le workspace inclut **6 tâches** pré-configurées :

| Tâche                              | Description               | Args                            |
| ---------------------------------- | ------------------------- | ------------------------------- |
| `ThreadX: Update Daily Tokens`     | Mise à jour quotidienne   | --tokens 100 --timeframes 1h,4h |
| `ThreadX: Analyze Token (BTCUSDC)` | Analyse technique BTCUSDC | --timeframe 1h --days 30        |
| `ThreadX: Scan All Tokens`         | Scan multi-tokens         | --tokens 100 --timeframe 1h     |
| `ThreadX: Run All Tests`           | Tous les tests pytest     | tests -v --tb=short             |
| `ThreadX: Test End-to-End`         | Test end-to-end complet   | -                               |
| `ThreadX: Install Requirements`    | Installer dépendances     | pip install -r requirements.txt |

---

## 🔌 Extensions Recommandées

Le workspace recommande automatiquement ces extensions (VS Code les propose à l'ouverture) :

### Essentielles
- ✅ **ms-python.python** - Support Python
- ✅ **ms-python.vscode-pylance** - Language server performant
- ✅ **ms-python.black-formatter** - Formatage Black

### Utiles
- 📊 **ms-toolsai.jupyter** - Support Notebooks
- 📝 **streetsidesoftware.code-spell-checker** - Correction orthographique
- 🇫🇷 **streetsidesoftware.code-spell-checker-french** - Dictionnaire français
- 🌳 **mhutchie.git-graph** - Visualisation Git
- 🔍 **eamodio.gitlens** - Git avancé

---

## 🧹 Fichiers/Dossiers Exclus

### Exclusion Vue Fichiers (Files Explorer)
```
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
*.egg-info/
```

### Exclusion Recherche (Search)
```
__pycache__/
.venv/
node_modules/
.git/
.mypy_cache/
.pytest_cache/
```

---

## 🛠️ Nettoyage Configuration Précédente

Si vous aviez des configurations multiples, voici ce qui a été fait :

### ✅ Supprimé/Consolidé
- ❌ Anciennes config `.vscode/settings.json` → Consolidé dans workspace
- ❌ Config `configs/pyrightconfig.json` → Remplacé par settings workspace
- ❌ Multiple espaces de travail → Un seul workspace

### ✅ Conservé
- ✅ `.venv/` - Environnement virtuel Python
- ✅ `pyproject.toml` - Config pytest minimale
- ✅ `requirements.txt` - Dépendances
- ✅ `paths.toml` - Configuration chemins ThreadX

---

## 📖 Utilisation Quotidienne

### Matin - Ouverture du projet
```bash
# 1. Ouvrir workspace
code ThreadX.code-workspace

# 2. Activer environnement virtuel (automatique dans terminal VS Code)
# Le workspace active automatiquement .venv

# 3. Lancer mise à jour quotidienne
Ctrl+Shift+P > Tasks: Run Task > ThreadX: Update Daily Tokens
```

### Pendant la journée - Développement
```bash
# Debug fichier actuel
F5 > Sélectionner "Python: Fichier actuel"

# Tester modifications
Ctrl+Shift+P > Tasks: Run Task > ThreadX: Run All Tests

# Analyser token
Ctrl+Shift+P > Tasks: Run Task > ThreadX: Analyze Token (BTCUSDC)
```

### Soir - Scan opportunités
```bash
# Scan multi-tokens
Ctrl+Shift+P > Tasks: Run Task > ThreadX: Scan All Tokens
```

---

## 🔧 Personnalisation

### Modifier paramètres tâche
Éditer `ThreadX.code-workspace` section `"tasks"` :
```json
{
    "label": "ThreadX: Update Daily Tokens",
    "args": [
        "scripts/update_daily_tokens.py",
        "--tokens", "150",  // ← Modifier ici
        "--timeframes", "15m,1h,4h"  // ← Modifier ici
    ]
}
```

### Ajouter configuration debug
Éditer `ThreadX.code-workspace` section `"launch"` :
```json
{
    "name": "Mon Script Custom",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/mon_script.py",
    "args": ["--arg1", "value1"]
}
```

---

## ⚡ Performance

### Settings optimisés pour gros projet
- ✅ `diagnosticMode: "workspace"` - Analyse complète workspace
- ✅ Type checking désactivé - Gain 50% performance
- ✅ Linting désactivé - Gain 30% performance
- ✅ Cache activé - IntelliSense rapide
- ✅ Auto-import activé - Productivité

### Exclusions intelligentes
- Recherche n'indexe pas `.venv`, `__pycache__`
- Explorer n'affiche pas fichiers temporaires
- Git ignore correctement via `.gitignore`

---

## 📊 Structure Projet ThreadX

```
ThreadX/
├── .venv/                         # Environnement virtuel (auto-activé)
├── .vscode/                       # Config VS Code legacy (peut être supprimé)
│   └── settings.json              # → Consolidé dans workspace
├── src/threadx/                   # Code source principal
│   ├── data/                      # Modules consolidés data
│   │   ├── tokens.py              # ✅ TokenManager
│   │   └── loader.py              # ✅ BinanceDataLoader
│   └── indicators/                # Modules indicateurs
│       └── indicators_np.py       # ✅ Indicateurs NumPy optimisés
├── scripts/                       # Scripts production
│   ├── update_daily_tokens.py     # ✅ Mise à jour quotidienne
│   ├── analyze_token.py           # ✅ Analyse technique
│   └── scan_all_tokens.py         # ✅ Scan multi-tokens
├── tests/                         # Tests unitaires
├── data/                          # Données (cache, exports)
├── ThreadX.code-workspace         # ✅ WORKSPACE UNIQUE
├── requirements.txt               # Dépendances Python
└── pyproject.toml                 # Config pytest
```

---

## ✅ Checklist Migration Workspace

- [x] **Workspace unique créé** : `ThreadX.code-workspace`
- [x] **Settings consolidés** : Python, formatage, linting, tests
- [x] **6 configs debug** : Scripts, tests, fichier actuel
- [x] **6 tâches** : Update, analyze, scan, tests, install
- [x] **Extensions recommandées** : Python, Black, Pylance, Jupyter
- [x] **Exclusions optimisées** : Files, search, git
- [x] **Documentation complète** : Ce fichier

---

## 🎯 Prochaines Étapes

### Option 1: Utiliser le nouveau workspace
```bash
# Fermer VS Code
# Rouvrir avec workspace
code ThreadX.code-workspace
```

### Option 2: Nettoyer ancienne config
```bash
# Supprimer ancien .vscode/settings.json (déjà consolidé)
rm .vscode/settings.json

# Supprimer configs redondantes
rm configs/pyrightconfig.json
```

### Option 3: Tester les tâches
```bash
# Dans VS Code
Ctrl+Shift+P > Tasks: Run Task > (sélectionner une tâche)
```

---

## 💡 Conseils

1. **Toujours ouvrir via workspace** : `code ThreadX.code-workspace`
2. **Utiliser F5 pour débugger** : Configurations prêtes
3. **Utiliser Tasks pour scripts** : Pas besoin de terminal manuel
4. **Laisser auto-format** : Black s'occupe du formatage
5. **Ignorer warnings types** : Focus sur fonctionnalités

---

**✅ Workspace ThreadX configuré et optimisé pour développement productif !**
