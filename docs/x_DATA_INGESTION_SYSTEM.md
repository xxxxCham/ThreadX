# ThreadX Data Ingestion System - Documentation Complète

## Vue d'ensemble

Le système d'ingestion de données ThreadX implémente le principe **"1m truth"** où toutes les fréquences temporelles (1h, 3h, 4h, 1d) sont dérivées uniquement à partir des données 1-minute. Ce système garantit la cohérence et la précision des données à travers tous les timeframes.

## Architecture

### Composants Principaux

1. **LegacyAdapter** (`src/threadx/data/legacy_adapter.py`)
   - Adaptation du code legacy pour téléchargement OHLCV
   - Gestion des erreurs réseau et retry automatique
   - Normalisation des timestamps UTC
   - Détection et comblement des gaps

2. **IngestionManager** (`src/threadx/data/ingest.py`)
   - Orchestrateur principal du téléchargement
   - Implémentation du principe "1m truth"
   - Traitement par batch avec parallélisation
   - Vérification de cohérence des resamples

3. **DataManagerPage** (`src/threadx/ui/data_manager.py`)
   - Interface Tkinter pour téléchargement manuel
   - Opérations non-bloquantes via threading
   - Barres de progression et logs temps réel
   - Mode dry-run pour validation

## Principe "1m Truth"

### Concept

Toutes les données temporelles proviennent d'une source unique : les chandeliers 1-minute téléchargés depuis l'API Binance. Les autres timeframes sont calculés par resampling.

```
API Binance (1m) → DataFrame 1m → Resample vers 1h, 3h, 4h, 1d
```

### Avantages

- **Cohérence garantie** : Pas de divergences entre timeframes
- **Précision maximale** : Conservation de tous les ticks
- **Flexibilité** : Possibilité de créer n'importe quel timeframe
- **Validation** : Vérification automatique de la cohérence

## API Principale

### LegacyAdapter

```python
from threadx.data.legacy_adapter import LegacyAdapter

adapter = LegacyAdapter(settings)

# Téléchargement OHLCV 1m
df = adapter.fetch_klines_1m(
    symbol="BTCUSDC",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# Détection des gaps
gaps = adapter.detect_gaps_1m(df)

# Comblement conservatif
df_filled = adapter.fill_gaps_conservative(df, gaps)
```

### IngestionManager

```python
from threadx.data.ingest import IngestionManager

manager = IngestionManager(settings)

# Téléchargement simple
df_1m = manager.download_ohlcv_1m("BTCUSDC", "2024-01-01", "2024-01-31")

# API resample depuis 1m
df_1h = manager.resample_from_1m_api("BTCUSDC", "1h", "2024-01-01", "2024-01-31")

# Batch update
results = manager.update_assets_batch(
    symbols=["BTCUSDC", "ETHUSDC"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    target_timeframes=["1h", "4h"]
)
```

### Interface Utilisateur

```python
from threadx.ui.data_manager import DataManagerPage

# Intégration dans app principale
class ThreadXApp(tk.Tk):
    def _create_data_manager_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Data Manager")
        
        self.data_manager_page = DataManagerPage(tab, padding="10")
        self.data_manager_page.pack(fill=tk.BOTH, expand=True)
```

## Configuration

### Fichier paths.toml

```toml
[data]
base_data_path = "data"
raw_data_path = "data/raw"
processed_data_path = "data/processed"

[data.api]
base_url = "https://api.binance.com"
klines_endpoint = "/api/v3/klines"
rate_limit_calls = 1200
rate_limit_window = 60

[data.processing]
default_batch_size = 5
max_gap_fill_hours = 4
resample_validation = true
```

### Variables d'environnement

**Aucune variable d'environnement n'est requise**. Le système utilise uniquement la configuration TOML.

## Tests

### Suite de tests complète

```bash
# Tests offline avec mocks
python -m pytest tests/test_legacy_adapter.py -v
python -m pytest tests/test_ingest_manager.py -v

# Test démonstration
python demo_data_ingestion.py
```

### Tests inclus

1. **test_legacy_adapter.py** (15 tests)
   - Configuration et initialisation
   - Construction d'URLs API
   - Transformation JSON vers DataFrame
   - Détection de gaps
   - Comblement de gaps
   - Gestion d'erreurs

2. **test_ingest_manager.py** (10 tests)
   - Téléchargement OHLCV 1m
   - API resample
   - Vérification cohérence
   - Traitement batch
   - Mode dry-run

## Utilisation Pratique

### Cas d'usage typique

```python
# 1. Configuration
from threadx.config import get_settings
settings = get_settings()

# 2. Manager principal
from threadx.data.ingest import IngestionManager
manager = IngestionManager(settings)

# 3. Téléchargement et resample
df_1m = manager.download_ohlcv_1m("BTCUSDC", "2024-01-01", "2024-01-31")
df_1h = manager.resample_from_1m_api("BTCUSDC", "1h", "2024-01-01", "2024-01-31")

# 4. Vérification cohérence
is_valid = manager.verify_resample_consistency(df_1m, df_1h, "1h")
print(f"Cohérence 1m→1h: {is_valid}")
```

### Interface utilisateur

1. Lancer l'application : `python run_tkinter.py`
2. Aller dans l'onglet "Data Manager"
3. Sélectionner symboles (Ctrl+clic pour multi-sélection)
4. Configurer période start/end
5. Cliquer "Download Selected" pour téléchargement background
6. Suivre progression via logs temps réel

### Mode dry-run

```python
# Test sans téléchargement réel
results = manager.update_assets_batch(
    symbols=["BTCUSDC"],
    start_date="2024-01-01", 
    end_date="2024-01-31",
    target_timeframes=["1h"],
    dry_run=True  # Simulation uniquement
)
```

## Gestion d'erreurs

### Stratégies de retry

```python
# Configuration dans paths.toml
[data.api]
max_retries = 3
retry_delay = 1.0
timeout_seconds = 30.0
```

### Types d'erreurs gérées

1. **Erreurs réseau** : Retry automatique avec backoff exponentiel
2. **Erreurs API** : Log détaillé et continuation du batch
3. **Gaps de données** : Détection et comblement conservatif
4. **Erreurs de cohérence** : Validation et re-téléchargement si nécessaire

## Performance

### Optimisations

- **Parallélisation** : Téléchargement concurrent par symbole
- **Batch processing** : Traitement groupé pour efficacité
- **Cache intelligent** : Évite re-téléchargement données existantes
- **Validation rapide** : Vérification cohérence sans re-calcul complet

### Métriques typiques

- **1 symbole, 1 mois (1m)** : ~30 secondes
- **5 symboles, 1 mois (1m+1h)** : ~2-3 minutes
- **Validation cohérence** : ~1-2 secondes par timeframe

## Maintenance

### Fichiers logs

```
logs/
├── data_ingestion.log      # Logs téléchargement
├── legacy_adapter.log      # Logs adapter
└── tkinter_app.log        # Logs interface
```

### Surveillance

```python
# Monitoring intégré
manager.download_ohlcv_1m("BTCUSDC", "2024-01-01", "2024-01-31")
# → Logs automatiques avec timestamps, durées, statistiques
```

## Migration depuis code legacy

### Avant (code legacy)

```python
# Code dispersé avec variables d'environnement
import os
API_KEY = os.getenv('BINANCE_API_KEY')
df = download_with_env_vars(symbol, API_KEY)
```

### Après (ThreadX)

```python
# Code centralisé avec configuration TOML
from threadx.data.ingest import IngestionManager
manager = IngestionManager(get_settings())
df = manager.download_ohlcv_1m(symbol, start, end)
```

### Points d'attention

1. **Configuration** : Migrer variables env → TOML
2. **Chemins** : Utiliser chemins relatifs ThreadX
3. **Logging** : Adopter système logging ThreadX
4. **Tests** : Ajouter tests offline appropriés

## Support et débogage

### Debug mode

```python
# Configuration debug dans paths.toml
[logging]
level = "DEBUG"
console_output = true

# Ou via code
import logging
logging.getLogger('threadx.data').setLevel(logging.DEBUG)
```

### Problèmes courants

1. **Timeouts API** : Augmenter `timeout_seconds` dans config
2. **Rate limiting** : Vérifier `rate_limit_calls` et `rate_limit_window`
3. **Gaps importants** : Ajuster `max_gap_fill_hours`
4. **Mémoire insuffisante** : Réduire `default_batch_size`

---

**Version** : Phase 8 - Iteration 3/3  
**Auteur** : ThreadX Framework  
**Dernière mise à jour** : Système d'ingestion complet avec UI intégrée