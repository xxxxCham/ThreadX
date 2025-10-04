# TradXPro Core Manager - Guide d'Intégration

Ce module unifie toute la logique TradXPro en un seul gestionnaire facile à incorporer dans n'importe quel programme.

## 🚀 Démarrage Rapide

### Installation
```bash
# Aucune installation requise - tout est inclus
cd D:\TradXPro\scripts\mise_a_jour_dataframe
python quick_start_tradxpro.py
```

### Usage Basique (3 lignes de code)
```python
from tradxpro_core_manager import TradXProManager

# 1. Initialisation
manager = TradXProManager()

# 2. Récupérer les top 100 tokens crypto avec diversité garantie
tokens = manager.get_top_100_tokens()  # 🔒 Assure ≥3 tokens par catégorie !

# 3. Analyser avec indicateurs techniques
df = manager.get_trading_data("BTCUSDC", "1h", indicators=["rsi", "bollinger", "atr"])
```

### 🔒 Nouvelle Fonctionnalité : Diversité Garantie

Le système garantit automatiquement qu'au moins **3 tokens de chaque catégorie importante** sont inclus dans les top 100 :

- 🌐 **Layer 1 Blockchain** : BTC, ETH, ADA, SOL, etc.
- 🏦 **DeFi Protocols** : UNI, AAVE, COMP, MKR, etc.  
- ⚡ **Layer 2 Scaling** : MATIC, ARB, OP, etc.
- 🎮 **Gaming & AI** : AXS, SAND, FET, AGIX, etc.
- 🏪 **Exchange Tokens** : BNB, CRO, FTT, etc.
- 💰 **Stablecoins** : USDT, USDC, DAI, etc.
- 🔒 **Privacy Coins** : XMR, ZEC, DASH, etc.
- 🛠️ **Infrastructure** : LINK, GRT, FIL, etc.

```python
# Vérifier la diversité
diversity_stats = manager.analyze_token_diversity(tokens)
print(f"Score de diversité: {diversity_stats['global']['diversity_score']:.1f}%")

# Afficher un rapport complet
manager.print_diversity_report(tokens)
```

## 📋 Fonctionnalités Complètes

### 🏆 Gestion des Tokens avec Diversité Garantie
- **`get_top_100_tokens()`** : Récupère les 100 meilleurs tokens (CoinGecko + Binance)
- **`load_saved_tokens()`** : Charge les tokens sauvegardés
- **Fusion intelligente** : Combine marketcap et volume pour un scoring optimal
- **🔒 Diversité garantie** : Assure au moins 3 représentants de chaque catégorie importante
- **`analyze_token_diversity()`** : Analyse la répartition par catégorie
- **`print_diversity_report()`** : Affiche un rapport détaillé de diversité

### 📥 Téléchargement des Données
- **`download_crypto_data(symbols, intervals)`** : Télécharge les données historiques
- **`download_top_100_data()`** : Télécharge automatiquement les données des top 100
- **Multi-threading** : Téléchargements parallèles pour la performance

### 📈 Indicateurs Techniques
- **RSI** (Relative Strength Index)
- **Bollinger Bands** (Bandes de Bollinger) 
- **ATR** (Average True Range)
- **EMA** (Exponential Moving Average)
- **MACD** (Moving Average Convergence Divergence)

### 💾 Gestion des Données
- **Double format** : JSON (compatibilité) + Parquet (performance)
- **Cache automatique** : Conversion JSON → Parquet transparente
- **Compression optimale** : ZSTD pour réduire l'espace disque

## 📚 Exemples d'Utilisation

### Exemple 1: Analyse Simple avec Diversité
```python
from tradxpro_core_manager import TradXProManager

manager = TradXProManager()

# Récupérer les top 100 avec diversité garantie
tokens = manager.get_top_100_tokens()

# Vérifier la diversité obtenue
diversity_stats = manager.analyze_token_diversity(tokens)
print(f"✅ Score de diversité: {diversity_stats['global']['diversity_score']:.1f}%")
print(f"✅ Catégories représentées: {len([c for c in diversity_stats if c != 'global' and diversity_stats[c]['count'] >= 3])}/10")

# Analyser Bitcoin avec indicateurs
df = manager.get_trading_data(
    symbol="BTCUSDC", 
    interval="1h", 
    indicators=["rsi", "bollinger", "atr"]
)

if df is not None:
    latest = df.iloc[-1]
    print(f"Prix BTC: ${latest['close']:.2f}")
    print(f"RSI: {latest['rsi']:.1f}")
    
    # Signal simple
    if latest['rsi'] < 30:
        print("🟢 Signal d'achat potentiel (RSI oversold)")
```

### Exemple 2: Stratégie Multi-Assets
```python
from tradxpro_core_manager import TradXProManager

manager = TradXProManager()

# Récupérer les top tokens
tokens = manager.get_top_100_tokens()
top_10_symbols = [token["symbol"] + "USDC" for token in tokens[:10]]

# Analyser en parallèle
pairs = [(symbol, "1h") for symbol in top_10_symbols]
results = manager.get_multiple_trading_data(pairs, indicators=["rsi", "macd"])

signals = []
for symbol_interval, df in results.items():
    if df is not None:
        latest = df.iloc[-1]
        if latest['rsi'] < 35 and latest['macd'] > latest['macd_signal']:
            signals.append(symbol_interval.replace("_1h", ""))

print(f"Signaux d'achat détectés: {signals}")
```

### Exemple 2b: Stratégie par Catégorie (Nouveau !)
```python
from tradxpro_core_manager import TradXProManager

manager = TradXProManager()

# Récupérer les tokens avec diversité garantie
tokens = manager.get_top_100_tokens()

# Analyser par catégorie
diversity_stats = manager.analyze_token_diversity(tokens)

# Stratégie ciblée sur les DeFi protocols
defi_tokens = diversity_stats["defi_protocols"]["tokens"]
defi_signals = []

for token_symbol in defi_tokens:
    symbol = token_symbol + "USDC"
    df = manager.get_trading_data(symbol, "1h", ["rsi", "macd"])
    
    if df is not None:
        latest = df.iloc[-1]
        
        # Stratégie spécifique DeFi (plus volatile)
        if (latest['rsi'] < 40 and  # Légèrement surassuré
            latest['macd'] > latest['macd_signal']):  # Momentum positif
            defi_signals.append({
                "symbol": symbol,
                "category": "DeFi",
                "signal": "BUY_DEFI",
                "rsi": latest['rsi']
            })

print(f"Signaux DeFi détectés: {len(defi_signals)}")
for signal in defi_signals:
    print(f"🔵 {signal['symbol']} - RSI: {signal['rsi']:.1f}")
```

### Exemple 3: Intégration dans une Classe
```python
from tradxpro_core_manager import TradXProManager

class MonRobot:
    def __init__(self):
        self.tradx = TradXProManager()
        self.watchlist = []
    
    def setup_watchlist(self):
        """Configure une watchlist avec les meilleurs tokens"""
        tokens = self.tradx.get_top_100_tokens()
        self.watchlist = [token["symbol"] + "USDC" for token in tokens[:20]]
        return len(self.watchlist)
    
    def scan_opportunities(self):
        """Scanne les opportunités de trading"""
        opportunities = []
        
        for symbol in self.watchlist:
            df = self.tradx.get_trading_data(symbol, "1h", ["rsi", "bollinger"])
            
            if df is not None:
                latest = df.iloc[-1]
                
                # Votre logique de trading ici
                if (latest['rsi'] < 30 and 
                    latest['close'] < latest['bb_lower']):
                    opportunities.append({
                        "symbol": symbol,
                        "signal": "BUY",
                        "price": latest['close'],
                        "rsi": latest['rsi']
                    })
        
        return opportunities

# Usage
robot = MonRobot()
robot.setup_watchlist()
opportunities = robot.scan_opportunities()
```

## 🛠️ Configuration Avancée

### Personnaliser les Chemins
```python
# Utiliser un dossier personnalisé
manager = TradXProManager(root_path="/mon/dossier/custom")
```

### Personnaliser les Paramètres
```python
manager = TradXProManager()

# Modifier les paramètres par défaut
manager.history_days = 180  # 6 mois au lieu de 1 an
manager.max_workers = 8     # Plus de threads pour téléchargements
manager.intervals = ["1h", "4h", "1d"]  # Timeframes personnalisés
```

### Filtrage Temporel
```python
# Analyser seulement une période spécifique
df = manager.get_trading_data(
    symbol="BTCUSDC",
    interval="1h", 
    indicators=["rsi"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

## 📊 Utilitaires

### Statistiques des Données
```python
stats = manager.get_data_statistics()
print(f"Symboles disponibles: {stats['symbols_count']}")
print(f"Taille totale: {stats['total_size_mb']} MB")
```

### Données Disponibles
```python
available = manager.get_available_data()
for symbol, intervals in available.items():
    print(f"{symbol}: {intervals}")
```

### Nettoyage
```python
# Supprimer les fichiers de plus de 7 jours
stats = manager.cleanup_old_files(days_old=7)
print(f"Fichiers supprimés: {stats['json_removed'] + stats['parquet_removed']}")
```

## 🎯 Cas d'Usage Typiques

### 1. Bot de Trading
```python
class TradingBot:
    def __init__(self):
        self.tradx = TradXProManager()
    
    def run_strategy(self):
        # Récupérer les meilleurs tokens
        tokens = self.tradx.get_top_100_tokens()
        
        # Analyser et prendre des décisions
        for token in tokens[:50]:
            symbol = token["symbol"] + "USDC"
            df = self.tradx.get_trading_data(symbol, "15m", ["rsi", "macd"])
            
            if df is not None:
                # Votre logique de trading
                self.evaluate_signal(symbol, df)
```

### 2. Analyse de Marché
```python
class MarketAnalyzer:
    def __init__(self):
        self.tradx = TradXProManager()
    
    def daily_report(self):
        # Récupérer les données du top 20
        tokens = self.tradx.get_top_100_tokens()[:20]
        symbols = [token["symbol"] + "USDC" for token in tokens]
        
        # Analyser les tendances
        pairs = [(s, "1d") for s in symbols]
        data = self.tradx.get_multiple_trading_data(pairs, ["rsi", "bollinger"])
        
        # Générer le rapport
        self.generate_report(data)
```

### 3. Screener de Cryptos
```python
class CryptoScreener:
    def __init__(self):
        self.tradx = TradXProManager()
    
    def find_oversold_assets(self):
        tokens = self.tradx.get_top_100_tokens()
        oversold = []
        
        for token in tokens:
            symbol = token["symbol"] + "USDC"
            df = self.tradx.get_trading_data(symbol, "1h", ["rsi"])
            
            if df is not None and df.iloc[-1]['rsi'] < 30:
                oversold.append(symbol)
        
        return oversold
```

## 🔧 Dépendances

Le module utilise uniquement des bibliothèques standard Python :
- `pandas` : Manipulation des données
- `numpy` : Calculs numériques  
- `requests` : APIs externes
- `pathlib` : Gestion des chemins
- `json` : Format de données
- `concurrent.futures` : Parallélisation

## 📁 Structure des Fichiers

```
TradXPro/
├── scripts/mise_a_jour_dataframe/
│   ├── tradxpro_core_manager.py      # Module principal
│   ├── exemple_integration_tradxpro.py # Exemples détaillés
│   ├── quick_start_tradxpro.py       # Démarrage rapide
│   └── resultats_choix_des_100tokens.json # Tokens sauvegardés
├── crypto_data_json/                 # Données JSON
├── crypto_data_parquet/              # Données Parquet (rapides)
└── indicators_db/                    # Cache des indicateurs
```

## 🚨 Notes Importantes

1. **Première utilisation** : Le système télécharge automatiquement les données manquantes
2. **Cache intelligent** : Les fichiers Parquet sont créés automatiquement pour accélérer les chargements futurs
3. **APIs externes** : Utilise CoinGecko et Binance (respect des limites de taux)
4. **Gestion d'erreurs** : Système robuste avec fallbacks automatiques

## 💡 Conseils d'Optimisation

1. **Utilisez Parquet** : Plus rapide que JSON (10x+)
2. **Chargement parallèle** : `get_multiple_trading_data()` pour plusieurs assets
3. **Cache des indicateurs** : Les calculs sont mis en cache automatiquement
4. **Filtrage temporel** : Limitez les données avec `start_date`/`end_date`

## 🆘 Support

Pour obtenir de l'aide :
1. Lancez `python quick_start_tradxpro.py` pour les exemples
2. Consultez `exemple_integration_tradxpro.py` pour des cas d'usage avancés
3. Vérifiez les logs pour diagnostiquer les problèmes

---

**TradXPro Core Manager v1.0** - Toute la puissance de TradXPro en un seul module ! 🚀