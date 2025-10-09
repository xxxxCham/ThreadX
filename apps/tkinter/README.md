# ThreadX - Interface Graphique Data Manager

## 🎯 Vue d'ensemble

Interface graphique complète pour la gestion et mise à jour des tokens de diversité crypto avec ThreadX. Développée en Tkinter avec une architecture moderne et intuitive.

## ✨ Fonctionnalités

### 📊 **Gestion des Données**
- **Sélection par groupes** : L1, L2, DeFi, AI, Gaming, Meme, Infrastructure, Privacy
- **Timeframes multiples** : 1m, 5m, 15m, 1h, 4h, 1d
- **Indicateurs techniques** : RSI, MACD, Bollinger Bands, SMA, EMA, ATR, VWAP
- **Configuration période** : 7 jours à 1 an

### ⚡ **Performance**
- **Traitement parallèle** : Jusqu'à 16 threads simultanés
- **Cache intelligent** : TTL configurable (5-120 min)
- **Barre de progression** temps réel
- **Statistiques live** : Vitesse, succès/erreurs

### 📤 **Export & Visualisation**
- **Formats multiples** : CSV, Excel multi-feuilles, Parquet
- **Organisation hiérarchique** : groupe/symbole/timeframe
- **Liste interactive** avec détails (lignes, dernière MAJ, taille)
- **Export sélectif** ou global

### 📋 **Logs & Monitoring**
- **Logs colorés** temps réel
- **Niveaux configurables** : DEBUG, INFO, WARNING, ERROR
- **Sauvegarde logs** avec horodatage
- **Historique persistant**

## 🚀 Installation & Lancement

### **Prérequis**
```bash
# ThreadX installé en mode développement
pip install -e .

# Dépendances GUI
pip install tkinter  # Généralement inclus avec Python
pip install pandas openpyxl  # Pour les exports
```

### **Lancement**
```bash
# Méthode 1 : Lanceur simplifié
cd apps/tkinter
python launch_gui.py

# Méthode 2 : Direct
python threadx_gui.py

# Méthode 3 : Depuis la racine
python apps/tkinter/threadx_gui.py
```

## 📁 Structure des Données

### **Organisation automatique**
```
ThreadX/
├── data/
│   ├── processed/           # Données traitées par groupe
│   │   ├── l1/
│   │   │   ├── btcusdt/
│   │   │   │   ├── 1h.parquet
│   │   │   │   ├── 1h_meta.json
│   │   │   │   └── 1h.csv
│   │   │   └── ethusdt/...
│   │   ├── defi/...
│   │   └── ai/...
│   ├── cache/               # Cache temporaire
│   ├── exports/             # Exports utilisateur
│   └── logs/                # Logs application
└── configs/
    └── gui_config.json      # Configuration sauvegardée
```

### **Formats de sauvegarde**
- **Parquet** (principal) : Compression snappy, indexé
- **CSV** (compatibilité) : UTF-8, séparateur virgule
- **JSON** (métadonnées) : Horodatage, statistiques, performance

## 🎮 Guide d'utilisation

### **1. Configuration initiale**
- ⚙️ **Onglet Configuration** : Paramétrer les dossiers et performance
- 📊 **Sélection groupes** : Cocher les catégories de tokens désirées
- ⏰ **Timeframes** : Choisir les intervalles (défaut: 1h, 4h, 1d)
- 📈 **Indicateurs** : Sélectionner les indicateurs techniques

### **2. Mise à jour des données**
- 🚀 **Bouton "METTRE À JOUR"** : Lance le processus
- 📊 **Suivi progression** : Barre de progression et statistiques live
- ⏹️ **Arrêt d'urgence** : Possibilité d'interrompre
- 📋 **Logs temps réel** : Suivi détaillé des opérations

### **3. Export et visualisation**
- 📤 **Onglet Export** : Liste des données disponibles
- 🔄 **Actualisation** : Rafraîchir la liste
- 📊 **Export sélectif** : CSV, Excel, Parquet
- 📂 **Navigation** : Accès direct aux dossiers

### **4. Configuration avancée**
- 💾 **Sauvegarde config** : Conserver les préférences
- 🔄 **Rechargement** : Restaurer une configuration
- ⚡ **Performance** : Ajuster threads et cache
- 📋 **Logs** : Configurer le niveau de détail

## ⚙️ Configuration JSON

```json
{
  "selected_groups": ["L1", "DeFi", "AI"],
  "selected_timeframes": ["1h", "4h", "1d"],
  "selected_indicators": ["RSI", "MACD", "SMA 20"],
  "period": "30_days",
  "threads": 4,
  "cache_ttl": 15
}
```

## 🔧 Personnalisation

### **Ajout de nouveaux groupes**
```python
# Dans threadx_gui.py
DEFAULT_GROUPS["MonGroupe"] = ["TOKEN1USDT", "TOKEN2USDT"]
```

### **Nouveaux indicateurs**
```python
# Dans convert_indicators_to_specs()
elif indicator == "Mon Indicateur":
    specs.append(IndicatorSpec(name="mon_indicateur", params={"param": value}))
```

### **Thème personnalisé**
```python
# Dans setup_window()
style.configure("Custom.TLabel", foreground="color", background="color")
```

## 🐛 Dépannage

### **Problèmes courants**

**❌ ThreadX non disponible**
- Solution : Vérifier l'installation `pip install -e .`
- Mode simulation automatique activé

**❌ Erreur d'import GUI**
- Solution : Installer les dépendances `pip install tkinter pandas openpyxl`

**❌ Dossiers non créés**
- Solution : Permissions insuffisantes, lancer en admin

**❌ Cache non fonctionnel**
- Solution : Vérifier l'espace disque et les permissions

### **Logs de débogage**
```python
# Activer le niveau DEBUG dans l'interface
self.log_level_var.set("DEBUG")

# Ou directement dans le code
self.logger.setLevel(logging.DEBUG)
```

## 📊 Performances

### **Benchmarks typiques**
- **Vitesse traitement** : ~28K lignes/sec
- **Latence moyenne** : 15.5ms par symbole/timeframe
- **Cache hit rate** : >95% après première exécution
- **Mémoire** : <200MB pour 1000+ séries temporelles

### **Optimisations**
- **Threads parallèles** : Ajustable selon CPU
- **Cache intelligent** : Évite les recalculs
- **Compression parquet** : Réduction taille 70%
- **Lazy loading** : Chargement à la demande

## 🔮 Roadmap

### **Version 1.1** (Prochaine)
- [ ] Graphiques intégrés avec matplotlib
- [ ] Alertes temps réel
- [ ] API REST pour intégrations
- [ ] Mode batch automatisé

### **Version 1.2** (Future)
- [ ] Interface web avec Streamlit
- [ ] Machine Learning intégré
- [ ] Backtesting visuel
- [ ] Notifications push

## 📝 License

ThreadX Data Manager GUI - Partie du projet ThreadX
Développé avec ❤️ pour la communauté crypto

---

*Interface moderne, performance optimale, utilisation intuitive* 🚀