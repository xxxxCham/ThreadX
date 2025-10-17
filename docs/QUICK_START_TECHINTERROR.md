# 🚀 Guide de Démarrage - Interface TechinTerror ThreadX

## Lancement Immédiat

```bash
# Option 1 : Lancement direct
python launch_techinterror.py

# Option 2 : Import Python
python -c "from threadx.ui.app import run_app; run_app()"
```

## Interface TechinTerror

### 🏠 **Home Tab** 
- **Auto-load BTC/USDC 1h** au démarrage
- Métriques temps réel (PnL, Sharpe, Max DD)
- Graphiques Equity + Drawdown intégrés

### 📥 **Downloads Tab**
- Téléchargements manuels avec vérification **1m + 3h**
- Progress bars en temps réel
- Logs de validation automatique

### 📊 **Charts Tab**
- Graphiques matplotlib avec **thème Nord**
- Export PNG/SVG haute résolution
- Equity, Drawdown, Candlesticks, Volume

### 📋 **Tables Tab** 
- Tables Trades, Metrics, Positions
- **Tri par colonnes** + filtres
- Export CSV, Parquet, Excel

### 📝 **Logs Tab**
- Logs temps réel avec auto-scroll
- Filtrage par niveau (DEBUG, INFO, etc.)
- Export vers fichiers

## Fonctionnalités Clés

- ✅ **Threading non-bloquant** - Interface toujours réactive
- ✅ **Thème Nord** - Design professionnel (#2E3440, #3B4252)
- ✅ **Auto-scan** - Détection automatique des fichiers de données
- ✅ **BTC Homepage** - Chargement automatique au démarrage
- ✅ **Tests validés** - 15 tests unitaires passés

## Support

- **Tests** : `python -m pytest tests/test_ui_basic.py -v`
- **Debug** : Logs disponibles dans le Tab Logs
- **Docs** : Voir `TECHINTERROR_IMPLEMENTATION_REPORT.md`

**Interface TechinTerror prête ! 🎯**