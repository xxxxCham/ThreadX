# 📚 GUIDE DES BONNES PRATIQUES - ThreadX Data Management

## Date: 11 octobre 2025
## Objectif: Utilisation quotidienne optimale du système consolidé

---

## 🎯 WORKFLOW RECOMMANDÉ

### Scénario 1: Mise à Jour Quotidienne des Tokens

**Fréquence:** 1x par jour (matin, avant trading)

**Script recommandé:** `update_daily_tokens.py`

```python
#!/usr/bin/env python3
"""
Mise à jour quotidienne des top tokens et leurs données OHLCV.
À exécuter chaque matin avant le début du trading.
"""

import sys
from pathlib import Path
from datetime import datetime

# Import modules consolidés
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

# Import direct TokenManager (évite dépendances config)
spec = importlib.util.spec_from_file_location(
    "tokens",
    Path(__file__).parent / "src" / "threadx" / "data" / "tokens.py"
)
tokens_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokens_module)
TokenManager = tokens_module.TokenManager

# Import direct BinanceDataLoader
spec = importlib.util.spec_from_file_location(
    "loader",
    Path(__file__).parent / "src" / "threadx" / "data" / "loader.py"
)
loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader_module)
BinanceDataLoader = loader_module.BinanceDataLoader


def update_daily_tokens():
    """Mise à jour quotidienne complète des tokens."""
    
    print("=" * 60)
    print(f"📅 MISE À JOUR QUOTIDIENNE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # ÉTAPE 1: Mise à jour liste tokens top 100
    print("\n📝 ÉTAPE 1/3: Récupération top 100 tokens...")
    print("-" * 60)
    
    token_mgr = TokenManager(
        cache_path=Path("data/crypto_data_json/tokens_top100.json")
    )
    
    # Récupérer top 100 avec données fraîches
    top_tokens = token_mgr.get_top_tokens(
        limit=100,
        usdc_only=True,
        force_refresh=True  # Force rafraîchissement quotidien
    )
    
    print(f"✅ {len(top_tokens)} tokens USDC sélectionnés")
    print(f"   Top 5: {top_tokens[:5]}")
    
    # ÉTAPE 2: Téléchargement données OHLCV
    print("\n\n📥 ÉTAPE 2/3: Téléchargement OHLCV...")
    print("-" * 60)
    
    loader = BinanceDataLoader(
        json_cache_dir=Path("data/crypto_data_json"),
        parquet_cache_dir=Path("data/crypto_data_parquet")
    )
    
    # Timeframes à mettre à jour (selon stratégie)
    timeframes = ["1h", "4h"]  # Adapter selon besoins
    
    for tf in timeframes:
        print(f"\n⏱️  Timeframe: {tf}")
        
        # Téléchargement avec callback progression
        def progress_callback(pct, done, total):
            if done % 10 == 0:  # Afficher tous les 10 symboles
                print(f"   [{done}/{total}] {pct:.1f}%")
        
        results = loader.download_multiple(
            symbols=top_tokens,
            interval=tf,
            days_history=365,  # 1 an d'historique
            max_workers=4,     # Parallèle pour vitesse
            progress_callback=progress_callback
        )
        
        print(f"   ✅ {len(results)}/{len(top_tokens)} symboles téléchargés ({tf})")
    
    # ÉTAPE 3: Résumé
    print("\n\n📊 ÉTAPE 3/3: Résumé")
    print("-" * 60)
    
    # Statistiques stockage
    json_dir = Path("data/crypto_data_json")
    parquet_dir = Path("data/crypto_data_parquet")
    
    json_files = len(list(json_dir.glob("*.json")))
    parquet_files = len(list(parquet_dir.glob("*.parquet")))
    
    print(f"📁 Fichiers JSON:    {json_files}")
    print(f"📁 Fichiers Parquet: {parquet_files}")
    print(f"✅ Mise à jour terminée!")
    
    return {
        "tokens_count": len(top_tokens),
        "timeframes": timeframes,
        "json_files": json_files,
        "parquet_files": parquet_files
    }


if __name__ == "__main__":
    stats = update_daily_tokens()
    print("\n" + "=" * 60)
    print("✅ MISE À JOUR QUOTIDIENNE COMPLÈTE")
    print("=" * 60)
```

---

### Scénario 2: Analyse d'un Token Spécifique

**Fréquence:** À la demande (analyse technique)

**Script recommandé:** `analyze_token.py`

```python
#!/usr/bin/env python3
"""
Analyse technique complète d'un token spécifique.
Télécharge données + calcule tous les indicateurs.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

# Import modules
spec = importlib.util.spec_from_file_location(
    "loader",
    Path(__file__).parent / "src" / "threadx" / "data" / "loader.py"
)
loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader_module)
BinanceDataLoader = loader_module.BinanceDataLoader

spec = importlib.util.spec_from_file_location(
    "indicators_np",
    Path(__file__).parent / "src" / "threadx" / "indicators" / "indicators_np.py"
)
indicators_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indicators_module)


def analyze_token(symbol: str, timeframe: str = "1h", days: int = 30):
    """
    Analyse technique complète d'un token.
    
    Args:
        symbol: Symbole (ex: "BTCUSDC")
        timeframe: Timeframe (ex: "1h", "4h")
        days: Nombre de jours historique
        
    Returns:
        DataFrame avec OHLCV + tous indicateurs
    """
    
    print(f"🔍 ANALYSE TECHNIQUE: {symbol} ({timeframe})")
    print("-" * 60)
    
    # 1. Téléchargement données
    print(f"📥 Téléchargement {days} jours de données...")
    
    loader = BinanceDataLoader(
        json_cache_dir=Path("data/crypto_data_json"),
        parquet_cache_dir=Path("data/crypto_data_parquet")
    )
    
    df = loader.download_ohlcv(
        symbol=symbol,
        interval=timeframe,
        days_history=days,
        force_update=False  # Utiliser cache si disponible
    )
    
    if df.empty:
        print(f"❌ Impossible de télécharger {symbol}")
        return None
    
    print(f"✅ {len(df)} bougies chargées")
    
    # 2. Calcul indicateurs
    print(f"\n📊 Calcul indicateurs...")
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # RSI
    df['rsi'] = indicators_module.rsi_np(close, period=14)
    
    # EMA rapide et lente
    df['ema_12'] = indicators_module.ema_np(close, span=12)
    df['ema_26'] = indicators_module.ema_np(close, span=26)
    
    # Bollinger Bands
    lower, ma, upper, z = indicators_module.boll_np(close, period=20, std=2.0)
    df['bb_lower'] = lower
    df['bb_middle'] = ma
    df['bb_upper'] = upper
    df['bb_zscore'] = z
    
    # MACD
    macd, signal, hist = indicators_module.macd_np(close, fast=12, slow=26, signal=9)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    
    # ATR
    df['atr'] = indicators_module.atr_np(high, low, close, period=14)
    
    # VWAP
    df['vwap'] = indicators_module.vwap_np(close, high, low, volume, window=96)
    
    # OBV
    df['obv'] = indicators_module.obv_np(close, volume)
    
    # Vortex
    vortex_df = indicators_module.vortex_df(high, low, close, period=14)
    df['vi_plus'] = vortex_df['vi_plus']
    df['vi_minus'] = vortex_df['vi_minus']
    
    print(f"✅ {len(df.columns) - 5} indicateurs calculés")
    
    # 3. Sauvegarde résultats
    output_path = Path(f"data/exports/{symbol}_{timeframe}_analysis.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    
    print(f"💾 Résultats sauvegardés: {output_path}")
    
    # 4. Résumé statistiques
    print(f"\n📈 RÉSUMÉ STATISTIQUES:")
    print(f"   Prix actuel:   ${df['close'].iloc[-1]:.4f}")
    print(f"   RSI:           {df['rsi'].iloc[-1]:.2f}")
    print(f"   MACD:          {df['macd'].iloc[-1]:.4f}")
    print(f"   ATR:           {df['atr'].iloc[-1]:.4f}")
    print(f"   BB Position:   {df['bb_zscore'].iloc[-1]:.2f} σ")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse technique token")
    parser.add_argument("symbol", help="Symbole (ex: BTCUSDC)")
    parser.add_argument("--timeframe", default="1h", help="Timeframe (défaut: 1h)")
    parser.add_argument("--days", type=int, default=30, help="Jours historique (défaut: 30)")
    
    args = parser.parse_args()
    
    df = analyze_token(args.symbol, args.timeframe, args.days)
    
    if df is not None:
        print("\n" + "=" * 60)
        print("✅ ANALYSE TERMINÉE")
        print("=" * 60)
```

**Utilisation:**
```bash
# Analyser BTCUSDC sur 30 jours
python analyze_token.py BTCUSDC

# Analyser ETHUSDC en 4h sur 90 jours
python analyze_token.py ETHUSDC --timeframe 4h --days 90
```

---

### Scénario 3: Scan Complet Multi-Tokens

**Fréquence:** 2-3x par jour (recherche opportunités)

**Script recommandé:** `scan_all_tokens.py`

```python
#!/usr/bin/env python3
"""
Scan complet de tous les top tokens avec critères de filtrage.
Trouve les meilleures opportunités selon indicateurs.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

# Imports modules (comme précédemment)
# ... code import ...


def scan_all_tokens(
    tokens: List[str],
    timeframe: str = "1h",
    filters: Dict = None
):
    """
    Scan complet avec filtres.
    
    Args:
        tokens: Liste symboles à scanner
        timeframe: Timeframe analyse
        filters: Critères de filtrage (RSI, MACD, etc.)
        
    Returns:
        DataFrame résultats filtrés triés par score
    """
    
    if filters is None:
        filters = {
            "rsi_min": 30,
            "rsi_max": 70,
            "macd_positive": True,
            "bb_zscore_min": -2.0,
            "bb_zscore_max": 2.0
        }
    
    print(f"🔍 SCAN DE {len(tokens)} TOKENS")
    print(f"⏱️  Timeframe: {timeframe}")
    print(f"📊 Filtres: {filters}")
    print("-" * 60)
    
    loader = BinanceDataLoader(
        json_cache_dir=Path("data/crypto_data_json"),
        parquet_cache_dir=Path("data/crypto_data_parquet")
    )
    
    # Téléchargement parallèle
    data = loader.download_multiple(
        symbols=tokens,
        interval=timeframe,
        days_history=30,
        max_workers=4
    )
    
    results = []
    
    for symbol, df in data.items():
        if df.empty:
            continue
        
        # Calcul indicateurs (même logique que analyze_token)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        rsi = indicators_module.rsi_np(close, period=14)
        macd, signal, _ = indicators_module.macd_np(close)
        _, _, _, bb_z = indicators_module.boll_np(close, period=20)
        atr = indicators_module.atr_np(high, low, close, period=14)
        
        # Dernières valeurs
        current_rsi = rsi[-1]
        current_macd = macd[-1]
        current_bb_z = bb_z[-1]
        current_price = close[-1]
        current_volume = volume[-1]
        
        # Filtrage
        if filters.get("rsi_min") and current_rsi < filters["rsi_min"]:
            continue
        if filters.get("rsi_max") and current_rsi > filters["rsi_max"]:
            continue
        if filters.get("macd_positive") and current_macd < 0:
            continue
        if filters.get("bb_zscore_min") and current_bb_z < filters["bb_zscore_min"]:
            continue
        if filters.get("bb_zscore_max") and current_bb_z > filters["bb_zscore_max"]:
            continue
        
        # Score composite (exemple simple)
        score = 0
        if 40 <= current_rsi <= 60:  # RSI neutre
            score += 30
        if current_macd > 0:  # MACD bullish
            score += 25
        if -1 <= current_bb_z <= 1:  # BB dans limites
            score += 25
        if current_volume > df['volume'].mean():  # Volume élevé
            score += 20
        
        results.append({
            "symbol": symbol.replace("USDC", ""),
            "price": current_price,
            "rsi": current_rsi,
            "macd": current_macd,
            "bb_zscore": current_bb_z,
            "volume_24h": current_volume,
            "score": score
        })
    
    # Tri par score
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("score", ascending=False)
    
    print(f"\n✅ {len(df_results)} tokens passent les filtres")
    print(f"\n🏆 TOP 10:")
    print(df_results.head(10).to_string(index=False))
    
    # Sauvegarde
    output_path = Path(f"data/exports/scan_{timeframe}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    print(f"\n💾 Résultats: {output_path}")
    
    return df_results


if __name__ == "__main__":
    # Récupérer top 100 tokens
    token_mgr = TokenManager()
    tokens = token_mgr.get_top_tokens(limit=100)
    
    # Scan avec filtres
    results = scan_all_tokens(
        tokens=tokens,
        timeframe="1h",
        filters={
            "rsi_min": 35,
            "rsi_max": 65,
            "macd_positive": True
        }
    )
```

---

## 📋 BONNES PRATIQUES

### 1. Fréquence Mises à Jour

| Type                  | Fréquence         | Script                   | Durée      |
| --------------------- | ----------------- | ------------------------ | ---------- |
| **Top 100 tokens**    | 1x par jour       | `update_daily_tokens.py` | ~5-10 min  |
| **OHLCV 1h**          | Toutes les heures | `update_hourly.py`       | ~2-3 min   |
| **OHLCV 4h**          | Toutes les 4h     | `update_4h.py`           | ~2-3 min   |
| **Scan opportunités** | 2-3x par jour     | `scan_all_tokens.py`     | ~3-5 min   |
| **Analyse token**     | À la demande      | `analyze_token.py`       | ~10-30 sec |

### 2. Gestion du Cache

**✅ BON:**
```python
# Utiliser cache pour analyse rapide
loader = BinanceDataLoader(
    json_cache_dir=Path("data/crypto_data_json"),
    parquet_cache_dir=Path("data/crypto_data_parquet")
)

df = loader.download_ohlcv(
    symbol="BTCUSDC",
    interval="1h",
    days_history=30,
    force_update=False  # Utilise cache si récent
)
```

**❌ MAUVAIS:**
```python
# Force download à chaque fois (lent!)
df = loader.download_ohlcv(
    symbol="BTCUSDC",
    interval="1h",
    days_history=30,
    force_update=True  # ❌ Toujours télécharger
)
```

### 3. Gestion des Erreurs

**✅ BON:**
```python
try:
    df = loader.download_ohlcv(symbol="BTCUSDC", interval="1h")
    if df.empty:
        logger.warning(f"Aucune donnée pour BTCUSDC")
        return None
    
    # Traitement...
    
except Exception as e:
    logger.error(f"Erreur téléchargement BTCUSDC: {e}")
    return None
```

### 4. Téléchargement Parallèle

**✅ BON (Rapide):**
```python
# Téléchargement parallèle (4 workers)
data = loader.download_multiple(
    symbols=top_100_tokens,
    interval="1h",
    max_workers=4  # Parallèle
)
```

**❌ LENT (Séquentiel):**
```python
# Téléchargement séquentiel (1 par 1)
data = {}
for symbol in top_100_tokens:
    data[symbol] = loader.download_ohlcv(symbol, "1h")
    # ❌ Très lent!
```

### 5. Sauvegarde Résultats

**Format recommandé:** Parquet (compression ZSTD)

```python
# ✅ BON: Parquet compressé (rapide, petit)
df.to_parquet(
    "data/exports/BTCUSDC_analysis.parquet",
    engine="pyarrow",
    compression="zstd",
    index=True
)

# ⚠️ OK: CSV (humainement lisible mais gros)
df.to_csv("data/exports/BTCUSDC_analysis.csv", index=True)

# ❌ ÉVITER: JSON (très gros fichiers)
df.to_json("data/exports/BTCUSDC_analysis.json")
```

---

## 🔄 WORKFLOW QUOTIDIEN COMPLET

### Matin (avant trading)
```bash
# 1. Mettre à jour top 100 tokens + OHLCV
python update_daily_tokens.py

# 2. Scan opportunités
python scan_all_tokens.py

# 3. Analyser tokens intéressants
python analyze_token.py BTCUSDC --days 30
python analyze_token.py ETHUSDC --days 30
```

### Pendant la journée
```bash
# Mise à jour OHLCV 1h (toutes les heures via cron/scheduler)
python update_hourly.py

# Analyse token spécifique à la demande
python analyze_token.py SOLUSDC --timeframe 1h --days 7
```

### Soir (après trading)
```bash
# Scan final de la journée
python scan_all_tokens.py --timeframe 4h

# Backup données importantes
python backup_data.py
```

---

## 🛠️ MAINTENANCE

### Hebdomadaire
- Vérifier espace disque (Parquet cache)
- Nettoyer vieux fichiers exports (> 30 jours)
- Vérifier logs erreurs

### Mensuel
- Mettre à jour dépendances Python
- Vérifier nouveaux tokens top 100
- Backup complet données

---

## 📦 STRUCTURE FICHIERS RECOMMANDÉE

```
ThreadX/
├── scripts/
│   ├── update_daily_tokens.py      ← Mise à jour quotidienne
│   ├── update_hourly.py             ← Mise à jour horaire
│   ├── analyze_token.py             ← Analyse token
│   ├── scan_all_tokens.py           ← Scan multi-tokens
│   └── backup_data.py               ← Backup
│
├── data/
│   ├── crypto_data_json/            ← Cache JSON (raw)
│   ├── crypto_data_parquet/         ← Cache Parquet (optimisé)
│   └── exports/                     ← Résultats analyses
│       ├── scans/                   ← Scans quotidiens
│       └── tokens/                  ← Analyses tokens
│
└── logs/
    ├── update.log                   ← Log mises à jour
    └── errors.log                   ← Log erreurs
```

---

## ✅ CHECKLIST QUOTIDIENNE

- [ ] Mise à jour top 100 tokens (matin)
- [ ] Téléchargement OHLCV toutes les heures
- [ ] Scan opportunités 2-3x par jour
- [ ] Analyser tokens intéressants
- [ ] Vérifier logs erreurs
- [ ] Backup données importantes

---

**Auteur:** ThreadX Core Team  
**Date:** 11 octobre 2025  
**Version:** 1.0
