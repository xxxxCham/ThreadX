# 🚀 ThreadX - Démarrage Rapide

## Lancement Ultra-Simple

### Option 1 : Double-clic (Windows)
```
📁 Double-cliquez sur: START_THREADX.bat
```

### Option 2 : Ligne de commande
```bash
python start_threadx.py
```

### Option 3 : Dash directement
```bash
python apps/dash_app.py
```

---

## 🎯 Interfaces Disponibles

### 🌐 Interface Web (Dash)
- **URL**: http://127.0.0.1:8050
- **Fonctionnalités**:
  - 📊 Visualisation des données
  - 📈 Indicateurs techniques
  - 🎯 Backtesting
  - ⚙️ Optimisation de paramètres

### 💻 Interface CLI
```bash
# Afficher l'aide
python -m threadx.cli --help

# Gestion des données
python -m threadx.cli data validate --path data/crypto_data_parquet

# Construire des indicateurs
python -m threadx.cli indicators build --symbol BTCUSDC --timeframe 1h

# Lancer un backtest
python -m threadx.cli backtest run --strategy ma_cross --symbol BTCUSDC

# Optimiser des paramètres
python -m threadx.cli optimize sweep --strategy ma_cross --trials 100

# Afficher la version
python -m threadx.cli version
```

---

## 📋 Prérequis

- **Python**: 3.10 ou supérieur
- **Dépendances**: Installées automatiquement par `start_threadx.py`
  - dash
  - dash-bootstrap-components
  - pandas
  - plotly
  - typer
  - rich

---

## 🛠️ Dépannage

### Le script ne démarre pas ?
```bash
# Vérifier Python
python --version

# Installer les dépendances manuellement
pip install -r requirements.txt

# Lancer directement Dash
python apps/dash_app.py
```

### Port 8050 déjà utilisé ?
```bash
# Modifier le port dans apps/dash_app.py (ligne 100)
PORT = 8051  # Au lieu de 8050
```

### Erreur d'import ?
```bash
# Réinstaller les dépendances
pip install --upgrade -r requirements.txt
```

---

## 📚 Documentation

- **Architecture**: `docs/ANALYSE_COMPLETE_THREADX.md`
- **PROMPT 9 CLI**: `src/threadx/cli/README_PROMPT9.md`
- **Guides**: Dossier `docs/`

---

## 🎉 C'est Parti !

Lancez simplement `START_THREADX.bat` et accédez à http://127.0.0.1:8050

Bon trading ! 📈
