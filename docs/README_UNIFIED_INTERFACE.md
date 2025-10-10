# 🚀 ThreadX Unified Interface

## Interface unifiée inspirée de TradXPro, adaptée à ThreadX

### 📋 Description

Cette interface unifiée combine toutes les fonctionnalités ThreadX dans une interface graphique moderne inspirée de l'architecture TradXPro du fichier `unified_data_historique_with_indicators.py`.

### ✨ Fonctionnalités

- **🏠 Dashboard** : Vue d'ensemble du système et statut
- **🎯 Backtesting** : Interface de test de stratégies de trading
- **📊 Data Management** : Gestion et visualisation des données
- **📋 Logging** : Journalisation en temps réel
- **⚙️ Configuration** : Paramètres et configuration système

### 🎮 Utilisation

#### Mode GUI (Recommandé)
```bash
cd D:\ThreadX
python apps/threadx_unified_simple.py
```

#### Mode CLI (si GUI indisponible)
L'interface bascule automatiquement en mode CLI si Tkinter n'est pas disponible.

### 🔧 Modes de Fonctionnement

#### Mode ThreadX Complet
- Toutes les fonctionnalités ThreadX disponibles
- Backtesting réel avec les moteurs ThreadX
- Gestion complète des données

#### Mode Demo
- Interface complète disponible
- Données de démonstration
- Résultats simulés pour les tests

### 📊 Interface Utilisateur

#### 🏠 Dashboard
- Statut système en temps réel
- Actions rapides
- Monitoring des performances

#### 🎯 Backtesting
- Sélection de stratégies
- Configuration des symboles
- Timeframes personnalisables
- Résultats détaillés en JSON

#### 📊 Data Management
- Liste des symboles disponibles
- Statut des données
- Outils de validation

#### 📋 Logs
- Journalisation en temps réel
- Sauvegarde des logs
- Nettoyage automatique

### 🔨 Développement

#### Structure du Code
```
ThreadX/
├── apps/
│   ├── threadx_unified_simple.py     # Interface principale
│   └── threadx_unified_interface.py  # Version complète
├── launch_simple.py                  # Lanceur simple
└── README_UNIFIED_INTERFACE.md       # Cette documentation
```

#### Intégration ThreadX
- Import conditionnel des modules ThreadX
- Fallback gracieux en mode démo
- Logging unifié avec le système ThreadX

### 📈 Exemple d'Utilisation

1. **Lancement de l'interface**
   ```bash
   python apps/threadx_unified_simple.py
   ```

2. **Test de backtest**
   - Sélectionner l'onglet "🎯 Backtest"
   - Choisir une stratégie (ex: bb_atr)
   - Entrer les symboles (ex: BTCUSDC,ETHUSDC)
   - Cliquer "🚀 Run Backtest"

3. **Visualisation des résultats**
   - Résultats affichés en temps réel
   - Export possible en JSON
   - Logs détaillés disponibles

### 🎨 Inspiration TradXPro

Cette interface s'inspire du système TradXPro original :

- **Architecture modulaire** : Composants séparés et réutilisables
- **Logging centralisé** : Queue de logs pour l'affichage temps réel
- **Graceful fallbacks** : Dégradation gracieuse si modules indisponibles
- **Interface intuitive** : Onglets organisés par fonctionnalité
- **Configuration flexible** : Variables d'environnement et paths adaptatifs

### 🔮 Fonctionnalités Futures

- [ ] Optimisation de stratégies
- [ ] Calcul d'indicateurs techniques
- [ ] Graphiques interactifs
- [ ] Export de rapports avancés
- [ ] API REST pour intégration externe
- [ ] Notifications en temps réel

### 🐛 Dépannage

#### Interface ne se lance pas
```bash
# Vérifier les dépendances
pip install tkinter pandas numpy

# Lancer en mode debug
python -c "import tkinter; print('Tkinter OK')"
```

#### ThreadX non disponible
- L'interface fonctionne en mode démo
- Toutes les fonctionnalités UI sont disponibles
- Les résultats sont simulés mais réalistes

### 📞 Support

Pour tout problème ou suggestion :
1. Vérifier les logs dans l'onglet "📋 Logs"
2. Consulter le fichier `logs/threadx_unified.log`
3. Utiliser le menu "Help > About" pour les informations système

---

## 🎉 Interface Prête !

L'interface ThreadX unifiée est maintenant opérationnelle et prête à être utilisée pour tous vos besoins de trading et d'analyse quantitative !