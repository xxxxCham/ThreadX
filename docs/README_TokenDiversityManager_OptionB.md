# TokenDiversityManager - Contrat Option B

Pipeline unifié ThreadX pour orchestration de données sans calcul natif d'indicateurs.

## Objectif

Le `TokenDiversityManager` implémente le contrat **Option B** strict :
- ✅ **AUCUN calcul d'indicateur natif**
- ✅ **Délégation totale à IndicatorBank** 
- ✅ **Orchestration unifiée** : ingestion → validation → fusion → publication
- ✅ **Cache TTL** avec clés stables et métriques
- ✅ **Device-agnostic** via helpers ThreadX
- ✅ **Déterminisme** avec seed global

## API Principale

### TokenDiversityManager.prepare_dataframe()

```python
def prepare_dataframe(
    self,
    market: str,                           # "BTCUSDT", "ETHUSDT"
    timeframe: str,                        # "1h", "5m", "1d"
    start: Union[str, pd.Timestamp],       # "2023-01-01" ou timestamp
    end: Union[str, pd.Timestamp],         # "2023-06-01" ou timestamp  
    indicators: List[IndicatorSpec],       # Specs pour IndicatorBank
    price_source: PriceSourceSpec,         # Source données OHLCV
    strict: bool = True,                   # Validations complètes
    cache_ttl_sec: Optional[int] = None,   # TTL cache (None = permanent)
    seed: Optional[int] = None,            # Seed déterminisme
) -> Tuple[pd.DataFrame, RunMetadata]:
```

**Retour :**
- `DataFrame` : Index UTC tz-aware, colonnes `[open, high, low, close, volume, ind_*]`
- `RunMetadata` : Métriques d'exécution, cache hit, warnings, device used

## Types de Données

### IndicatorSpec
```python
IndicatorSpec = TypedDict("IndicatorSpec", {
    "name": str,              # "rsi", "bbands", "atr", "macd"
    "params": Dict[str, Any]  # {"window": 14, "n_std": 2.0}
})
```

### PriceSourceSpec  
```python
PriceSourceSpec = TypedDict("PriceSourceSpec", {
    "name": str,              # "binance_spot", "local_cache", "external_manager"
    "params": Dict[str, Any]  # Paramètres source-spécifiques
})
```

### RunMetadata
```python
RunMetadata = TypedDict("RunMetadata", {
    "market": str,
    "timeframe": str, 
    "start": str,                    # ISO format
    "end": str,                      # ISO format
    "indicators_count": int,
    "price_source": str,
    "execution_time_ms": float,
    "cache_hit": bool,
    "rows_processed": int,
    "coverage_pct": float,           # % données sans NaN
    "device_used": str,              # "cpu" | "gpu" | "mixed"
    "seed": Optional[int],
    "warnings": List[str]
})
```

## Schéma Colonnes

### OHLCV (Standard ThreadX)
| Colonne  | Type    | Description    |
| -------- | ------- | -------------- |
| `open`   | float64 | Prix ouverture |
| `high`   | float64 | Prix maximum   |
| `low`    | float64 | Prix minimum   |
| `close`  | float64 | Prix clôture   |
| `volume` | float64 | Volume échangé |

### Indicateurs (Préfixe `ind_`)
| Pattern          | Exemple        | Description                               |
| ---------------- | -------------- | ----------------------------------------- |
| `ind_{name}`     | `ind_rsi`      | Indicateur simple (RSI, ATR)              |
| `ind_{name}_{i}` | `ind_bbands_0` | Indicateur multi-sortie (Bollinger bands) |

**Règles nommage :**
- Snake_case obligatoire
- Préfixe `ind_` pour tous les indicateurs
- Pas de collision avec colonnes OHLCV
- Dtype explicite (float32/float64 selon Settings)

## Exemples d'Usage

### 1. Données Prix Seules
```python
from threadx.data.providers.token_diversity import TokenDiversityManager, PriceSourceSpec

manager = TokenDiversityManager()

df, meta = manager.prepare_dataframe(
    market="BTCUSDT",
    timeframe="1h",
    start="2023-01-01",
    end="2023-06-01", 
    indicators=[],  # Pas d'indicateurs
    price_source=PriceSourceSpec(name="binance_spot", params={})
)

print(f"OHLCV: {len(df)} rows, columns: {list(df.columns)}")
print(f"Period: {df.index[0]} → {df.index[-1]}")
```

### 2. Prix + 1 Indicateur  
```python
from threadx.data.providers.token_diversity import IndicatorSpec

indicators = [
    IndicatorSpec(name="rsi", params={"window": 14})
]

df, meta = manager.prepare_dataframe(
    market="ETHUSDT",
    timeframe="5m",
    start="2023-03-01",
    end="2023-03-15",
    indicators=indicators,
    price_source=PriceSourceSpec(name="local_cache", params={})
)

# Colonnes disponibles: [open, high, low, close, volume, ind_rsi]
rsi_values = df["ind_rsi"]
print(f"RSI range: {rsi_values.min():.2f} - {rsi_values.max():.2f}")
```

### 3. Multiple Indicateurs + Cache + Resample
```python
indicators = [
    IndicatorSpec(name="rsi", params={"window": 14}),
    IndicatorSpec(name="bbands", params={"window": 20, "n_std": 2.0}),
    IndicatorSpec(name="atr", params={"window": 14}),
    IndicatorSpec(name="macd", params={"fast": 12, "slow": 26, "signal": 9})
]

df, meta = manager.prepare_dataframe(
    market="SOLUSDT", 
    timeframe="1h",
    start="2023-01-01",
    end="2023-12-31",  # 1 année complète
    indicators=indicators,
    price_source=PriceSourceSpec(name="external_manager", params={}),
    cache_ttl_sec=3600,  # Cache 1h
    seed=42  # Reproductibilité
)

print(f"Indicators computed: {meta['indicators_count']}")
print(f"Cache hit: {meta['cache_hit']}")
print(f"Execution time: {meta['execution_time_ms']:.1f}ms")
print(f"Coverage: {meta['coverage_pct']:.1f}%")

# Export CSV pour analyse
df.to_csv("solusdt_1h_with_indicators.csv")
```

### 4. Export Parquet Optimisé
```python
# Configuration high-performance
df, meta = manager.prepare_dataframe(
    market="ADAUSDT",
    timeframe="15m", 
    start="2023-06-01",
    end="2023-12-01",
    indicators=[
        IndicatorSpec(name="bbands", params={"window": 20, "n_std": 2.0})
    ],
    price_source=PriceSourceSpec(name="binance_spot", params={"rate_limit": 1200}),
    cache_ttl_sec=1800,  # 30min cache
)

# Export Parquet avec compression
df.to_parquet(
    "adausdt_15m_bbands.parquet", 
    compression="snappy",
    engine="pyarrow"
)

print(f"Parquet exported: {len(df)} rows, {meta['execution_time_ms']:.1f}ms")
```

## Pipeline Interne

### 1. Validation & Cache Lookup
- Normalisation timestamps (UTC tz-aware)
- Génération clé cache stable (hash paramètres)
- Lookup cache avec TTL

### 2. Ingestion OHLCV
- Délégation à TokenDiversityDataSource
- Fallback stub pour tests/développement
- Validation format (index, colonnes, dtypes)

### 3. Calcul Indicateurs (IndicatorBank)
- Appels batch/async selon disponibilité
- Adaptation format sortie (simple/multi-values)  
- Préfixage colonnes (`ind_*`)
- Retry logique sur échecs partiels

### 4. Alignement Temporel
- Détection fréquence source vs cible
- Resample avec règles OHLC :
  - `open`: first
  - `high`: max
  - `low`: min  
  - `close`: last
  - `volume`: sum
- Alignement indicateurs (moyenne)

### 5. Contrôles Qualité
- Détection trous temporels
- NaN head/tail après warmup
- Invariants OHLC (high ≥ open,close ; low ≤ open,close)
- Clipping outliers optionnel (conservateur)

### 6. Cache Update & Métriques
- Stockage cache avec timestamp
- Métriques performance (latence, throughput)
- Device tracking (CPU/GPU via IndicatorBank)
- Warnings aggregation

## Gestion d'Erreurs

### Codes d'Erreur Stables
```python
# Paramètres invalides
ValueError: "Invalid date range: start >= end"

# Données indisponibles  
DataNotFoundError: "No OHLCV data for BTCUSDT@1h"

# Timeframe non supporté
UnsupportedTimeframeError: "Unsupported timeframe: 2h (supported: 1m,5m,15m,1h,4h,1d)"

# Indicateur inconnu (warning, pas exception)
Warning: "Failed to compute unknown_indicator: not found in IndicatorBank"
```

### Messages User-Friendly
- Contexte précis (market, timeframe, range)
- Suggestions correctives
- Codes erreur stables pour intégration UI

## Performance & Benchmarks

### Budgets Cibles
- **OHLCV seul** : < 50ms pour 1000 bars
- **OHLCV + 3 indicateurs** : < 200ms pour 1000 bars  
- **Cache hit** : < 5ms (toute taille)
- **Memory usage** : < 100MB pour 10K bars × 10 colonnes

### Optimisations
- Cache TTL intelligent
- Batch IndicatorBank calls
- Vectorisation Pandas/NumPy
- Device selection automatique (CPU/GPU)
- Lazy loading modules

## Tests & Validation

### Suite de Tests
```bash
# Tests complets
python -m pytest test_token_diversity_manager_option_b.py -v

# Benchmarks  
python -m pytest test_token_diversity_manager_option_b.py::TestTokenDiversityManagerBenchmarks --benchmark-only

# Tests smoke rapides
python test_token_diversity_manager_option_b.py
```

### Fixtures & Goldens
- Données de référence déterministes (seed=42)
- Comparaison bitwise pour reproductibilité
- Fixtures multi-timeframes et multi-indicateurs

## Intégration ThreadX

### Settings & Configuration
```toml
# paths.toml
[data.token_diversity]
cache_dir = "cache/token_diversity"
default_ttl_sec = 3600
supported_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

[performance] 
gpu_threshold_rows = 5000
max_concurrent_indicators = 4
```

### UI Integration Hooks
```python
# Pour Data Preview UI
def get_dataframe_for_ui(
    self, 
    market: str, 
    timeframe: str, 
    max_rows: int = 1000
) -> Tuple[pd.DataFrame, RunMetadata]:
    """Version UI avec pagination/échantillonnage."""
    
# Pour Downloads UI
def export_data(
    self,
    df: pd.DataFrame, 
    format: str = "csv",  # "csv", "parquet", "json"
    compression: Optional[str] = None
) -> str:
    """Export avec formats multiples."""
```

## Migration & Compatibilité

### Depuis unified_data_historique_with_indicators
- Interface similaire (market, timeframe, indicators)
- Métadonnées enrichies (RunMetadata vs dict)
- Cache unifié (clés compatibles)
- Logs structurés (même format)

### Avec IndicatorBank Existant
- Pas de modification IndicatorBank requise
- Appels standard `ensure()` et `ensure_batch()`
- Device selection déléguée
- Retry logic encapsulée

---

**Contrat Option B respecté** ✅  
*Délégation totale, interface stable, performance optimisée*