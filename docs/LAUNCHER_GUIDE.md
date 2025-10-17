# Scripts de Lancement ThreadX

## 🚀 `run_streamlit.bat` - Interface Web

Lance l'interface Streamlit de ThreadX avec configuration optimisée.

### Usage
```powershell
# Lancement direct
.\run_streamlit.bat

# Ou via PowerShell
cd D:\ThreadX
.\run_streamlit.bat
```

### Fonctionnalités
- ✅ **Activation automatique** environnement virtuel
- ✅ **Configuration optimisée** pour Windows
- ✅ **Port personnalisé** 8504 (évite conflits)
- ✅ **Cache Streamlit** organisé
- ✅ **Messages informatifs** avec emojis

### Interface Web Disponible
- **URL:** http://localhost:8504
- **Fonctionnalités:** 
  - 🏠 Accueil avec métriques projet
  - 🔧 Outils (migration, diagnostic)
  - 📊 Monitoring système temps réel
  - 📚 Documentation interactive

---

## 🖥️ `run_tkinter.py` - Interface Desktop

Application desktop native avec interface Tkinter complète.

### Usage
```powershell
# Lancement normal
python run_tkinter.py

# Mode debug
python run_tkinter.py --debug

# Thème clair  
python run_tkinter.py --theme light

# Mode développement
python run_tkinter.py --dev --debug
```

### Options CLI
- `--debug` : Logs détaillés
- `--dev` : Features expérimentales 
- `--theme {dark,light,auto}` : Thème interface
- `--config PATH` : Config personnalisée

### Fonctionnalités Interface
- 🎛️ **Onglets multiples** (Backtest, Config, Résultats, Logs)
- ⚡ **Actions rapides** avec boutons toolbar
- 📊 **Monitoring temps réel** avec barres progression
- 🔧 **Menu complet** (Fichier, Outils, Aide)
- 🎨 **Thèmes** dark/light avec couleurs adaptées
- 📝 **Logs intégrés** avec ScrolledText coloré

---

## 🔧 Prérequis

### Environnement
```powershell
# Environnement virtuel actif
.venv\Scripts\activate

# Dépendances installées
pip install -r requirements.txt
```

### Structure Attendue
```
ThreadX/
├── .venv/                    # Environnement virtuel
├── apps/
│   └── streamlit/
│       └── app_minimal.py    # App Streamlit minimale
├── tools/
│   ├── migrate_from_tradxpro.py
│   └── check_env.py
├── logs/                     # Logs Tkinter
├── cache/
│   └── streamlit/           # Cache Streamlit
├── run_streamlit.bat        # Script batch
└── run_tkinter.py          # Script Python
```

---

## 🚦 Statut et Tests

### Tests de Validation
```powershell
# Test Tkinter (dépendances)
python -c "import tkinter; print('✅ Tkinter OK')"

# Test Streamlit (lancement)
.\run_streamlit.bat  # Devrait ouvrir http://localhost:8504

# Test app Tkinter
python run_tkinter.py --help  # Affiche aide CLI
```

### Dépannage Courant

**Problème : Tkinter non disponible**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Windows
# Tkinter inclus avec Python standard
```

**Problème : Port 8504 occupé**
```powershell
# Trouver processus
netstat -ano | findstr :8504

# Tuer processus
taskkill /PID <PID> /F
```

**Problème : Environnement virtuel**
```powershell
# Recréer si corrompu
Remove-Item .venv -Recurse -Force
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🎯 Intégration Outils Phase 10

### Migration TradXPro
Les deux interfaces permettent d'accéder à l'outil de migration :

```powershell
# Directement via CLI
python tools/migrate_from_tradxpro.py --root "D:\TradXPro\crypto_data_json" --symbols BTCUSDC --dry-run

# Via interface Streamlit (page Outils)
# Via interface Tkinter (Menu Outils > Migration)
```

### Diagnostic Environnement
```powershell
# CLI direct
python tools/check_env.py --json-output env_report.json

# Interfaces graphiques proposent boutons intégrés
```

---

## 📈 Roadmap Interface

### À Venir
- 🔄 **Intégration complète** outils Phase 10
- 📊 **Visualisation** résultats backtests  
- ⚙️ **Configuration avancée** stratégies
- 🎮 **Support joystick/gamepad** (mode fun)
- 🌐 **Mode multi-utilisateur** Streamlit

### Architecture
- **Tkinter** → Interface principale desktop
- **Streamlit** → Interface web/remote + démos
- **CLI tools** → Automatisation et scripts