# 📊 CONFIGURATION DES CHEMINS THREADX - VALIDATION COMPLÈTE

## ✅ RÉSUMÉ DE LA CONFIGURATION

Tous les chemins ont été mis à jour pour correspondre exactement à la structure ThreadX.

### 🎯 Chemins de base configurés

```python
# Fichier: unified_data_historique_with_indicators.py

JSON_ROOT = "D:\ThreadX\data\crypto_data_json"
PARQUET_ROOT = "D:\ThreadX\data\crypto_data_parquet"
INDICATORS_DB_ROOT = "D:\ThreadX\data\indicators"
OUTPUT_DIR = "D:\ThreadX\data\exports"
```

---

## 📁 STRUCTURE DES DOSSIERS

### 1️⃣ Données OHLCV JSON
**Répertoire:** `D:\ThreadX\data\crypto_data_json\`

**Format:** `{SYMBOL}_{TIMEFRAME}.json`

**Exemples validés:**
```
D:\ThreadX\data\crypto_data_json\ADAUSDC_30m.json
D:\ThreadX\data\crypto_data_json\ALGOUSDC_1h.json
D:\ThreadX\data\crypto_data_json\AAVEUSDC_15m.json
D:\ThreadX\data\crypto_data_json\ALTUSDC_3m.json
D:\ThreadX\data\crypto_data_json\APTUSDC_5m.json
```

### 2️⃣ Données OHLCV Parquet
**Répertoire:** `D:\ThreadX\data\crypto_data_parquet\`

**Format:** `{SYMBOL}_{TIMEFRAME}.parquet`

**Exemples validés:**
```
D:\ThreadX\data\crypto_data_parquet\ALTUSDC_1h.parquet
D:\ThreadX\data\crypto_data_parquet\ALTUSDC_3m.parquet
D:\ThreadX\data\crypto_data_parquet\ALTUSDC_5m.parquet
D:\ThreadX\data\crypto_data_parquet\ALTUSDC_15m.parquet
D:\ThreadX\data\crypto_data_parquet\ALTUSDC_30m.parquet
D:\ThreadX\data\crypto_data_parquet\APTUSDC_1h.parquet
D:\ThreadX\data\crypto_data_parquet\ARBUSDC_3m.parquet
```

### 3️⃣ Indicateurs Techniques
**Répertoire:** `D:\ThreadX\data\indicators\{BASE_SYMBOL}\{TIMEFRAME}\`

**Format:** `{INDICATOR}_{PARAMS}.parquet`

**Structure arborescente:**
```
D:\ThreadX\data\indicators\
├── ZKC\
│   ├── 3m\
│   │   ├── bollinger_period10_std1.5.parquet
│   │   ├── bollinger_period20_std2.0.parquet
│   │   ├── rsi_period14.parquet
│   │   └── ...
│   ├── 5m\
│   ├── 15m\
│   ├── 30m\
│   └── 1h\
│       ├── bollinger_period10_std1.5.parquet
│       ├── bollinger_period10_std2.0.parquet
│       ├── bollinger_period10_std2.5.parquet
│       ├── bollinger_period20_std1.5.parquet
│       ├── bollinger_period20_std2.0.parquet
│       ├── bollinger_period20_std2.5.parquet
│       ├── bollinger_period50_std1.5.parquet
│       ├── bollinger_period50_std2.0.parquet
│       ├── bollinger_period50_std2.5.parquet
│       ├── ema_period20.parquet
│       ├── ema_period50.parquet
│       ├── ema_period200.parquet
│       ├── macd_fast12_signal9_slow26.parquet
│       ├── obv.parquet
│       ├── rsi_period14.parquet
│       ├── rsi_period21.parquet
│       ├── vortex_period14.parquet
│       ├── vortex_period21.parquet
│       ├── vwap_window96.parquet
│       ├── atr_period14.parquet
│       └── atr_period21.parquet
└── ZK\
    ├── 3m\
    ├── 5m\
    ├── 15m\
    ├── 30m\
    └── 1h\
```

---

## 🔧 FONCTIONS DE CHEMINS

### `parquet_path(symbol: str, tf: str) -> str`
Génère le chemin pour un fichier Parquet OHLCV.

**Exemple:**
```python
parquet_path("ALTUSDC", "1h")
# Retourne: "D:\ThreadX\data\crypto_data_parquet\ALTUSDC_1h.parquet"
```

### `json_path_symbol(symbol: str, tf: str) -> str`
Génère le chemin pour un fichier JSON OHLCV.

**Exemple:**
```python
json_path_symbol("ALTUSDC", "30m")
# Retourne: "D:\ThreadX\data\crypto_data_json\ALTUSDC_30m.json"
```

### `indicator_path(symbol: str, tf: str, name: str, key: str) -> str`
Génère le chemin pour un fichier d'indicateur.

**Exemple:**
```python
indicator_path("ZKUSDC", "1h", "bollinger", "period20_std2.0")
# Retourne: "D:\ThreadX\data\indicators\ZK\1h\bollinger_period20_std2.0.parquet"
```

**Note importante:** Le symbole est automatiquement normalisé :
- `ZKUSDC` → `ZK`
- `ZKCUSDC` → `ZKC`
- `BTCUSDC` → `BTC`

---

## ✅ VALIDATION

### Scripts de validation créés

1. **`validate_paths.py`** - Valide que tous les chemins sont corrects
2. **`generate_example_paths.py`** - Génère des exemples de chemins

### Exécution de la validation

```bash
python validate_paths.py
```

**Résultat:**
```
🎉 VALIDATION RÉUSSIE !
✅ Tous les chemins correspondent à la structure ThreadX attendue

📊 Structure validée:
  • JSON OHLCV: D:\ThreadX\data\crypto_data_json\
  • Parquet OHLCV: D:\ThreadX\data\crypto_data_parquet\
  • Indicateurs: D:\ThreadX\data\indicators\{SYMBOL}\{tf}\
  • Exports: D:\ThreadX\data\exports\
```

---

## 📦 CRÉATION AUTOMATIQUE DES RÉPERTOIRES

Les répertoires sont créés automatiquement au démarrage de `unified_data_historique_with_indicators.py` :

```python
for _p in (JSON_ROOT, PARQUET_ROOT, INDICATORS_DB_ROOT, OUTPUT_DIR):
    os.makedirs(_p, exist_ok=True)
```

**Répertoires créés:**
- ✅ `D:\ThreadX\data\crypto_data_json\`
- ✅ `D:\ThreadX\data\crypto_data_parquet\`
- ✅ `D:\ThreadX\data\indicators\`
- ✅ `D:\ThreadX\data\exports\`
- ✅ `D:\ThreadX\logs\`

---

## 🔄 MIGRATION DEPUIS TRADXPRO

Si vous aviez des données dans TradXPro, les chemins ont été migrés :

| Ancien (TradXPro)                   | Nouveau (ThreadX)                      |
| ----------------------------------- | -------------------------------------- |
| `D:\TradXPro\crypto_data_json\`     | `D:\ThreadX\data\crypto_data_json\`    |
| `D:\TradXPro\crypto_data_parquet\`  | `D:\ThreadX\data\crypto_data_parquet\` |
| `I:\indicators_db\`                 | `D:\ThreadX\data\indicators\`          |
| `D:\TradXPro\best_token_DataFrame\` | `D:\ThreadX\data\exports\`             |

---

## 🎯 TIMEFRAMES SUPPORTÉS

```python
INTERVALS = ["3m", "5m", "15m", "30m", "1h"]
```

Tous les fichiers sont générés pour ces 5 timeframes.

---

## 📝 NOTES IMPORTANTES

1. **Symboles de base** : Les indicateurs utilisent le symbole sans "USDC"
   - `ZKUSDC` → stocké dans `ZK/`
   - `ZKCUSDC` → stocké dans `ZKC/`

2. **Format des paramètres** : Convention cohérente
   - Bollinger: `period{N}_std{X}`
   - RSI: `period{N}`
   - EMA: `period{N}`
   - MACD: `fast{N}_signal{M}_slow{P}`

3. **Extension `.parquet`** : Tous les indicateurs et données converties

---

## ✅ CONCLUSION

**Tous les chemins sont maintenant configurés selon la structure ThreadX exacte.**

Les données seront automatiquement stockées dans les bons répertoires lors de :
- Téléchargement OHLCV
- Conversion JSON → Parquet
- Calcul des indicateurs techniques

**Date de validation:** 11 octobre 2025
**Version:** ThreadX cleanup-2025-10-09
