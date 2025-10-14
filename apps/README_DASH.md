# ThreadX Dash Dashboard - Quick Start Guide

## 🚀 Démarrage Rapide (5 minutes)

### 1. Installation Dépendances

```powershell
# Depuis racine ThreadX
pip install dash dash-bootstrap-components plotly
```

### 2. Configuration Port (Optionnel)

```powershell
# Port par défaut: 8050
$env:THREADX_DASH_PORT=8050

# Persistant (redémarrer terminal après)
setx THREADX_DASH_PORT 8050
```

### 3. Lancement Application

```powershell
python apps\dash_app.py
```

### 4. Accès UI

Ouvrir navigateur: **http://127.0.0.1:8050**

---

## 📖 Structure Dashboard

### Onglets Disponibles

1. **Data Manager**
   - Upload fichiers market data
   - Validation données
   - Registry sources

2. **Indicators**
   - Build cache indicateurs techniques
   - Configuration paramètres
   - Visualisation cache

3. **Backtest**
   - Exécution stratégies
   - Graphiques equity/drawdown
   - Métriques performance

4. **Optimization**
   - Parameter sweeps
   - Heatmaps 2D
   - Top results ranking

---

## 🎨 Design

- **Theme**: Bootstrap DARKLY
- **Layout**: Responsive (mobile/tablet/desktop)
- **Pattern**: Settings (gauche) + Results (droite)

---

## 🔧 Configuration Avancée

### Variables Environnement

```powershell
# Port serveur
$env:THREADX_DASH_PORT=8888

# Debug mode (development)
$env:THREADX_DASH_DEBUG=true
```

### Production Deployment

```python
# Utiliser serveur Flask exposé
from apps.dash_app import server

# Avec Gunicorn (Linux)
gunicorn apps.dash_app:server -b 0.0.0.0:8050
```

---

## 📝 État Actuel (PROMPT 4)

**Implémenté**:
- ✅ Layout statique 4 onglets
- ✅ Theme sombre responsive
- ✅ IDs déterministes pour callbacks

**À venir** (P5-P7):
- ⏳ Composants forms/tables/graphs
- ⏳ Callbacks Bridge integration
- ⏳ Async tasks + polling

---

## 🐛 Troubleshooting

### Port Already in Use

```powershell
# Changer port
$env:THREADX_DASH_PORT=8888
python apps\dash_app.py
```

### Import Error (Dash)

```powershell
pip install --upgrade dash dash-bootstrap-components
```

### Layout Not Updating

```powershell
# Clear cache browser (Ctrl+F5)
# Restart app
```

---

## 📚 Documentation

- **PROMPT4_DELIVERY_REPORT.md**: Documentation complète
- **PROMPT4_SUMMARY.md**: Résumé exécutif
- **threadx_10_prompts_esquisse.md**: Plan global 10 prompts

---

**Version**: 0.1.0 (PROMPT 4)
**Status**: Layout statique complet
**Next**: P5 Composants Data + Indicators
