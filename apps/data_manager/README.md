# ThreadX Data Manager - README

## 🎯 Objectif
Interface de gestion pour découvrir, valider et intégrer les données d'indicateurs existantes dans ThreadX.

## 📁 Structure
```
apps/data_manager/
├── run_data_manager.py     # Point d'entrée
├── main_window.py          # Interface principale
├── models.py               # Modèles de données
├── discovery/
│   └── local_scanner.py    # Scanner de données locales
├── integration/            # (À développer)
│   ├── importer.py
│   ├── merger.py
│   └── cleaner.py
└── ui/
    ├── discovery_tab.py    # Onglet découverte ✅
    └── validation_tab.py   # Onglets validation/intégration (stubs)
```

## 🚀 Lancement
```bash
cd d:\ThreadX
.venv\Scripts\python.exe apps\data_manager\run_data_manager.py
```

## 📊 Fonctionnalités - Étape 1

### ✅ Onglet Découverte
- [x] Sélection de chemins à scanner (g:\indicators_db, i:\indicators_db)
- [x] Scan récursif avec options configurables
- [x] Détection automatique structure Symbol/Timeframe/Indicateur
- [x] Parsing intelligent des noms de fichiers (atr_p14.parquet, bollinger_p20_s2.0.parquet)
- [x] Analyse rapide métadonnées (taille, colonnes, row_count)
- [x] Interface TreeView avec hiérarchie Symbol > Timeframe > Fichiers
- [x] Statistiques globales (symboles, fichiers, taille totale)
- [x] Données de démonstration pour tests

### 🚧 Onglet Validation (À développer)
- [ ] Validation intégrité fichiers Parquet
- [ ] Détection gaps temporels dans les données
- [ ] Vérification schémas OHLCV + indicateurs
- [ ] Analyse qualité des données (outliers, NaN)
- [ ] Rapport de validation avec recommandations
- [ ] Tests de cohérence inter-fichiers

### 🚧 Onglet Intégration (À développer)
- [ ] Plan d'intégration intelligent
- [ ] Import sélectif vers structure ThreadX
- [ ] Fusion datasets multiples
- [ ] Nettoyage et déduplication
- [ ] Création IndicatorBank ThreadX-compatible
- [ ] Tests de smoke automatiques

## 🎭 Données de Démonstration
Utilise `create_demo_catalog()` pour simuler:
- Symbol: ETHUSDC
- Timeframe: 5m
- Indicateurs: ATR (p14), Bollinger (p20_s2.0)
- Métadonnées réalistes

## 🔧 Intégration ThreadX
- Import structure `DataCatalog` → ThreadX `IndicatorBank`
- Configuration via `threadx.config.settings`
- Logging intégré avec interface utilisateur
- Compatible avec système de cache ThreadX

## 📈 Roadmap
1. **Phase 1**: Scanner + Interface découverte ✅
2. **Phase 2**: Validation complète des données
3. **Phase 3**: Intégration avec ThreadX
4. **Phase 4**: Automation et workflows
5. **Phase 5**: Support formats additionnels (CSV, Feather)

## 🎯 Usage Typique

### Workflow Option A Révisée:
1. **Découverte**: Scanner g:\indicators_db → Catalogue complet
2. **Validation**: Vérifier intégrité → Rapport de qualité  
3. **Intégration**: Import sélectif → IndicatorBank ThreadX
4. **Test**: Backtesting avec données validées
5. **Production**: Interface complète + API temps réel

### Résultat Attendu:
```python
# Structure générée pour ThreadX
catalog = DataCatalog(
    symbols={"ETHUSDC": SymbolData(...)},
    total_files=156,
    unique_indicators={"atr", "bollinger", ...},
    size_mb=245.6
)

# Intégration future
indicator_bank = create_indicator_bank_from_catalog(catalog)
```

## 📞 Points d'Extension
- `LocalDataScanner._parse_filename()`: Ajouter patterns indicateurs
- `ValidationTab`: Implémenter logique validation complète
- `IntegrationTab`: Implémenter pipeline d'import ThreadX
- Export formats: JSON, CSV, SQLite pour interopérabilité

---
*ThreadX Data Manager - Transformez vos données existantes en IndicatorBank opérationnel*