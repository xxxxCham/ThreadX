# Data Creation & Management - How It Works

## Vue d'ensemble

Le module **Data Creation & Management** de ThreadX fournit une interface complète pour créer, valider et gérer la banque de données OHLCV depuis Binance, avec validation UDFI stricte et mise à jour automatique des indicateurs.

## Architecture (3 couches)

### 1. Engine Layer (Calculs purs)
- **`threadx.data.ingest`**: Pipeline d'ingestion Binance
  - `ingest_binance()`: Téléchargement single symbol
  - `ingest_batch()`: Téléchargement batch (top 100 / groups / custom list)
- **`threadx.data.loader`**: Wrapper Binance API (fetch + pagination)
- **`threadx.data.legacy_adapter`**: Retry logic + normalisation
- **`threadx.data.tokens`**: Gestion listes symboles (top 100, groups L1/DeFi/L2/Stable)
- **`threadx.data.udfi_contract`**: Validation schéma UDFI
- **`threadx.data.registry`**: Gestion checksums + idempotence
- **`threadx.data.io`**: I/O Parquet optimisé
- **`threadx.data.unified_diversity_pipeline`**: Mise à jour indicateurs batch

### 2. Bridge Layer (Orchestration async)
- **`ThreadXBridge`**: Coordinateur async (non utilisé directement ici mais prêt pour futures améliorations)
- Les callbacks appellent directement les fonctions data pour l'instant (synchrone)
- **TODO**: Migrer vers pattern async via Bridge pour UI non-bloquante

### 3. UI Layer (Dash)
- **`threadx.ui.components.data_manager`**: Composant UI
  - Mode selection: Single / Top 100 / Group
  - Date range picker
  - Preview candlestick
  - Registry table
  - Indicators batch selector
- **`threadx.ui.callbacks`**: Callbacks Dash
  - `download_and_validate_data()`: Téléchargement + validation + sauvegarde
  - `update_indicators_batch()`: MAJ indicateurs en batch
  - `toggle_source_inputs()`: Toggle UI inputs selon mode

## Flux de données

### 1. Téléchargement OHLCV

```
User Input (UI)
    ↓
[Mode Selection]
    ├─ Single: 1 symbole
    ├─ Top 100: TokenManager.get_top_tokens(limit=100)
    └─ Group: create_default_config().groups["L1"|"DeFi"|"L2"|"Stable"]
    ↓
[Date Range Selection]
    → start_iso, end_iso (ISO 8601)
    ↓
[Callback: download_and_validate_data()]
    ↓
ingest_batch(mode, symbols_or_group, interval, start_iso, end_iso)
    ↓
[Parallélisation ThreadPool]
    → ingest_binance() pour chaque symbole
        ↓
    [Fetch Binance API]
        → LegacyAdapter.fetch_klines_1m()
        → Pagination automatique (limit 1000)
        → Retry logic (3 tentatives, backoff exponentiel)
        ↓
    [Normalisation UDFI]
        → Timezone UTC
        → Colonnes: timestamp, open, high, low, close, volume
        → Types: float64
        ↓
    [Validation stricte]
        → assert_udfi(df, strict=True)
        → Vérification schéma Pandera
        ↓
    [Resample si nécessaire]
        → Si interval != "1m": resample_from_1m(df, interval)
        ↓
    [Sauvegarde]
        → Parquet: processed/{symbol}/{interval}.parquet
        → Checksum: file_checksum(path) → registry
        ↓
[Mise à jour UI]
    → Registry table: symbole, rows, dates, checksum
    → Preview: candlestick chart (premier symbole)
    → Global store: persistance sélections pour autres onglets
```

### 2. Mise à jour indicateurs (Batch)

```
User Selection
    ↓
[Indicators Dropdown]
    → RSI, MACD, Bollinger Bands, SMA, EMA, ATR
    ↓
[Callback: update_indicators_batch()]
    ↓
UnifiedDiversityPipeline.process_batch()
    ↓
[Pour chaque (symbole, indicator)]
    → IndicatorBank.calculate(symbol, indicator, params)
    → Cache: indicators_cache/{symbol}/{indicator}.parquet
    ↓
[Retour UI]
    → Alert: "✅ Updated N indicator(s) successfully"
```

## Schéma UDFI (Unified Data Format for Indicators)

### Colonnes requises
```python
REQUIRED_COLS = {"symbol", "open", "high", "low", "close", "volume"}
CRITICAL_COLS = {"open", "high", "low", "close"}
EXPECTED_DTYPES = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}
```

### Index
- **Type**: `pd.DatetimeIndex`
- **Timezone**: UTC obligatoire
- **Fréquence**: Dépend du timeframe (1m, 5m, 15m, 1h, 4h, 1d)
- **Nom**: `"timestamp"` ou index sans nom

### Validation
```python
from threadx.data.udfi_contract import assert_udfi

assert_udfi(df, strict=True)
# Vérifie:
# - Colonnes présentes
# - Types corrects
# - Index DatetimeIndex UTC
# - Pas de NaN dans colonnes critiques
# - Cohérence OHLC (high >= low, etc.)
```

## Gestion de la pagination Binance

### Limite API
- **Max klines/requête**: 1000
- **Gestion automatique**: `LegacyAdapter.fetch_klines_1m()`

### Algorithme
```python
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)
interval_ms = 60_000  # 1m = 60s = 60_000ms

while current_ms < end_ms:
    # Requête 1000 klines max
    response = requests.get(BINANCE_API, params={
        "symbol": symbol,
        "interval": "1m",
        "startTime": current_ms,
        "limit": 1000,
    })

    # Avancer curseur
    last_kline_time = response[-1][0]
    current_ms = last_kline_time + interval_ms

    # Anti-loop protection
    if len(response) < 1000:
        break  # Fin des données
```

## Idempotence & Checksums

### Registry structure
```python
{
    "symbol": "BTCUSDC",
    "timeframe": "1h",
    "path": "processed/BTCUSDC/1h.parquet",
    "checksum": "a3f5c7d9...",  # SHA-256 du fichier
    "rows": 8760,
    "start": "2024-01-01",
    "end": "2024-12-31",
}
```

### Garanties
- **Checksum SHA-256**: Détection modifications/corruptions
- **Re-run safe**: Si checksum identique, pas de re-download
- **Gap filling**: Télécharge seulement plages manquantes

### Implémentation
```python
from threadx.data.registry import file_checksum

# Après sauvegarde Parquet
checksum = file_checksum(parquet_path)

# Vérification avant re-download
if dataset_exists(symbol, timeframe):
    existing_checksum = get_checksum(symbol, timeframe)
    if existing_checksum == expected_checksum:
        logger.info("Data up-to-date, skipping download")
        return
```

## Gestion d'erreurs

### Erreurs API Binance

#### Rate limiting
```python
# LegacyAdapter implémente retry automatique
max_retries = 3
backoff_factor = 2.0

for attempt in range(max_retries):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        if e.response.status_code == 429:  # Too Many Requests
            wait = backoff_factor ** attempt
            logger.warning(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
        else:
            raise
```

#### Erreurs réseau
- **Timeout**: 10s par défaut
- **Retry**: 3 tentatives avec backoff exponentiel
- **Fallback**: Si échec total, alerte UI avec message clair

### Erreurs validation UDFI

```python
try:
    assert_udfi(df, strict=True)
except UDFIColumnError as e:
    logger.error(f"Missing columns: {e}")
    # UI: Alert danger "Validation failed: Missing columns [...]"
except UDFITypeError as e:
    logger.error(f"Invalid types: {e}")
    # UI: Alert danger "Validation failed: Invalid types [...]"
except UDFIIndexError as e:
    logger.error(f"Invalid index: {e}")
    # UI: Alert danger "Validation failed: Index must be DatetimeIndex UTC"
```

### Erreurs I/O

```python
try:
    write_frame(df, parquet_path)
except PermissionError:
    # UI: Alert danger "Cannot write to {path}: Permission denied"
except FileNotFoundError:
    # Création auto répertoire parent
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    write_frame(df, parquet_path)
```

## Persistance globale (dcc.Store)

### Structure
```python
{
    "symbols": ["BTCUSDC", "ETHUSDC", "SOLUSDC"],
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "last_downloaded": "2024-01-31T15:30:00Z",
}
```

### Réutilisation
```python
# Dans callbacks autres onglets (Backtest, Indicators)
@callback(
    Output("bt-results", "children"),
    Input("bt-run-btn", "n_clicks"),
    State("data-global-store", "data"),
)
def run_backtest(n_clicks, store_data):
    symbols = store_data["symbols"]  # Récupère sélection Data Manager
    timeframe = store_data["timeframe"]

    # Lance backtest sur ces symboles
    results = bridge.run_backtest(symbols, timeframe, ...)
    return results
```

## Modes d'ingestion

### Mode "Single"
```python
ingest_batch(
    mode="single",
    symbols_or_group=["BTCUSDC"],
    interval="1h",
    start_iso="2024-01-01T00:00:00Z",
    end_iso="2024-01-31T23:59:59Z",
)
# Télécharge 1 symbole
```

### Mode "Top 100"
```python
ingest_batch(
    mode="top",
    symbols_or_group=None,  # Ignoré, récupère auto top 100
    interval="4h",
    start_iso="2024-01-01T00:00:00Z",
    end_iso="2024-01-31T23:59:59Z",
)
# Télécharge top 100 (market cap + volume combinés)
# Filtrage: paires USDC tradables sur Binance
```

### Mode "Group"
```python
ingest_batch(
    mode="group",
    symbols_or_group="L1",  # "L1" | "DeFi" | "L2" | "Stable"
    interval="1d",
    start_iso="2024-01-01T00:00:00Z",
    end_iso="2024-01-31T23:59:59Z",
)
# Télécharge groupe prédéfini:
# - L1: BTC, ETH, SOL, ADA
# - DeFi: UNI, AAVE, LINK, DOT
# - L2: MATIC, ARB, OP
# - Stable: EUR, FDUSD, USDE
```

## Pièges courants

### 1. Timezone confusion
❌ **Mauvais**:
```python
start = datetime(2024, 1, 1)  # Timezone naive
```

✅ **Bon**:
```python
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
# Ou
start_iso = "2024-01-01T00:00:00Z"
start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
```

### 2. Pagination manuelle
❌ **Mauvais**:
```python
# Ne pas implémenter pagination manuellement
while True:
    response = requests.get(url, params={...})
    # Risque de boucle infinie
```

✅ **Bon**:
```python
# Utiliser LegacyAdapter ou BinanceDataLoader
from threadx.data.legacy_adapter import LegacyAdapter

adapter = LegacyAdapter()
klines = adapter.fetch_klines_1m(symbol, start, end)
# Pagination automatique + retry logic
```

### 3. Validation UDFI après modifications
❌ **Mauvais**:
```python
df = ingest_binance(...)
df['custom_column'] = df['close'] * 2  # Modifie sans revalidation
save(df)  # Peut casser autres modules
```

✅ **Bon**:
```python
df = ingest_binance(...)  # Déjà validé UDFI
# Si modifications nécessaires:
df_modified = df.copy()
df_modified['custom_column'] = df['close'] * 2
assert_udfi(df_modified, strict=True)  # Revalider
save(df_modified)
```

### 4. Ignorance checksums
❌ **Mauvais**:
```python
# Re-télécharger systématiquement
df = ingest_binance(symbol, interval, start, end)
```

✅ **Bon**:
```python
from threadx.data.registry import dataset_exists

if dataset_exists(symbol, interval):
    df = read_frame(f"processed/{symbol}/{interval}.parquet")
else:
    df = ingest_binance(symbol, interval, start, end)
```

## Tests manuels recommandés

### 1. Single symbol
```bash
python apps/dash_app.py
# UI:
# - Mode: Single Symbol
# - Symbol: BTCUSDC
# - Timeframe: 1h
# - Date: Last 7 days
# - Click "Download & Validate Data"
# ✓ Registry table mise à jour
# ✓ Preview candlestick affiché
# ✓ Fichier: processed/BTCUSDC/1h.parquet
```

### 2. Top 100
```bash
# UI:
# - Mode: Top 100
# - Timeframe: 4h
# - Date: Last 30 days
# - Click "Download & Validate Data"
# ✓ ~100 symboles téléchargés
# ✓ Registry table avec tous symboles
# ✓ Temps exécution < 5 min (parallélisation 4 workers)
```

### 3. Group L1
```bash
# UI:
# - Mode: Group
# - Group: L1
# - Timeframe: 1d
# - Date: Last 365 days
# - Click "Download & Validate Data"
# ✓ 4 symboles (BTC, ETH, SOL, ADA)
# ✓ Preview BTC
```

### 4. Indicateurs batch
```bash
# Après téléchargement:
# - Select: RSI, MACD, BB
# - Click "Update Indicators"
# ✓ Alert: "Updated N indicators"
# ✓ Cache: indicators_cache/{symbol}/{indicator}.parquet
```

## Dépendances requises

```bash
pip install python-binance pandas pyarrow requests plotly dash dash-bootstrap-components
```

## Prochaines améliorations

1. **Async via Bridge**: Migrer callbacks vers pattern async pour UI non-bloquante
2. **Progress bar**: dcc.Interval + polling pour afficher progression batch
3. **WebSocket Binance**: Stream real-time pour données live
4. **Cache intelligent**: Détection auto gaps + téléchargement incrémental
5. **Export CSV/Excel**: Bouton export depuis registry table
6. **Tests unitaires**: Coverage ingest.py + callbacks

## Support

Pour questions/issues: Voir `docs/` ou créer issue GitHub.
