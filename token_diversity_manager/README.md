# 🔒 TradXPro Token Diversity Manager

**Module spécialisé pour la gestion des tokens crypto avec diversité garantie**

## 📦 Qu'est-ce que c'est ?

Ce module unifie toute la logique TradXPro en un gestionnaire qui garantit automatiquement une sélection diversifiée et équilibrée des cryptomonnaies, couvrant tous les secteurs importants du marché.

## ✨ Fonctionnalités Principales

- 🏆 **Récupération automatique** des top 100 tokens (CoinGecko + Binance)
- 🔒 **Diversité garantie** : Au moins 3 tokens de chaque catégorie importante
- 📥 **Téléchargement intelligent** des données historiques
- 📈 **Indicateurs techniques** intégrés (RSI, Bollinger, ATR, EMA, MACD)
- 💾 **Stockage optimisé** (JSON + Parquet avec compression)
- 📊 **Analyse de diversité** avec métriques et rapports
- ⚡ **Performance** : Chargement parallèle et cache automatique

## 🚀 Installation et Usage

### Installation Simple
```bash
# Aucune installation requise - tout est inclus
cd d:\TradXPro\modules\token_diversity_manager
```

### Usage de Base (3 lignes)
```python
from tradxpro_token_diversity_manager import TradXProManager

manager = TradXProManager()
tokens = manager.get_top_100_tokens()  # 🔒 Diversité automatique !
df = manager.get_trading_data("BTCUSDC", "1h", ["rsi", "bollinger"])
```

## 📊 Catégories Garanties

Le système assure une représentation équilibrée de 10 catégories :

1. 🌐 **Layer 1 Blockchain** : BTC, ETH, ADA, SOL, AVAX, DOT, NEAR, ALGO
2. 🏦 **DeFi Protocols** : UNI, AAVE, COMP, MKR, SUSHI, CRV, 1INCH, YFI
3. ⚡ **Layer 2 Scaling** : MATIC, ARB, OP, IMX, LRC, MINA
4. 🎮 **AI Gaming** : FET, AGIX, OCEAN, AXS, SAND, MANA, ENJ
5. 🏪 **Exchange Tokens** : BNB, CRO, FTT, HT, KCS, OKB
6. 💰 **Stablecoins** : USDT, USDC, BUSD, DAI, FRAX, TUSD
7. 🔒 **Privacy Coins** : XMR, ZEC, DASH, SCRT
8. 🛠️ **Infrastructure** : LINK, GRT, FIL, AR, STORJ, SIA
9. 🃏 **Meme Coins** : DOGE, SHIB, PEPE, FLOKI, BONK
10. 📈 **Smart Contracts** : ETH, ADA, SOL, AVAX, DOT, ALGO, NEAR, ATOM

## 📁 Structure du Module

```
token_diversity_manager/
├── tradxpro_core_manager.py      # 🔧 Module principal
├── __init__.py                   # 📦 Configuration du module
├── README.md                     # 📖 Ce fichier
├── tests/                        # 🧪 Tests et validations
│   ├── test_token_diversity.py   #   Test complet
│   └── test_diversite_simple.py  #   Test rapide
├── examples/                     # 💡 Exemples d'utilisation
│   ├── exemple_integration_tradxpro.py
│   └── quick_start_tradxpro.py
└── docs/                         # 📚 Documentation
    ├── README_CORE_MANAGER.md    #   Guide complet
    └── DIVERSITE_GARANTIE.md     #   Détails de la diversité
```

## 🧪 Tests et Validation

### Test Rapide
```bash
cd d:\TradXPro\modules\token_diversity_manager
python tests/test_diversite_simple.py
```

### Test Complet
```bash
python tests/test_token_diversity.py
```

### Exemples Pratiques
```bash
python examples/quick_start_tradxpro.py
python examples/exemple_integration_tradxpro.py
```

## 📈 Résultats Typiques

Avec la diversité garantie, vous obtiendrez :
- ✅ **Score de diversité** : 70%+ (Excellent)
- ✅ **Catégories représentées** : 7-10/10 avec ≥3 tokens chacune
- ✅ **Couverture équilibrée** : Tous les secteurs crypto importants
- ✅ **Qualité assurée** : Tokens ajoutés automatiquement si nécessaire

## 💡 Exemples d'Utilisation

### Analyse de Diversité
```python
# Vérifier la diversité obtenue
diversity_stats = manager.analyze_token_diversity(tokens)
manager.print_diversity_report(tokens)

print(f"Score: {diversity_stats['global']['diversity_score']:.1f}%")
```

### Stratégie par Catégorie
```python
# Cibler les tokens DeFi
defi_tokens = diversity_stats["defi_protocols"]["tokens"]
for token in defi_tokens:
    df = manager.get_trading_data(token + "USDC", "1h", ["rsi", "macd"])
    # Votre logique DeFi...
```

### Intégration dans une Classe
```python
class MonRobot:
    def __init__(self):
        self.tradx = TradXProManager()
        
    def scan_market(self):
        tokens = self.tradx.get_top_100_tokens()
        # Diversité automatiquement garantie !
        return self.analyze_all_tokens(tokens)
```

## 🔧 Configuration Avancée

### Chemins Personnalisés
```python
manager = TradXProManager(root_path="/mon/dossier/custom")
```

### Paramètres Personnalisés
```python
manager.history_days = 180        # 6 mois de données
manager.max_workers = 8           # Plus de threads
manager.intervals = ["1h", "4h"]  # Timeframes spécifiques
```

## 📊 API Principales

### Gestion des Tokens
- `get_top_100_tokens()` - Récupère avec diversité garantie
- `analyze_token_diversity()` - Analyse la répartition
- `print_diversity_report()` - Rapport détaillé

### Données et Indicateurs
- `get_trading_data()` - Données OHLCV + indicateurs
- `download_crypto_data()` - Téléchargement en masse
- `get_multiple_trading_data()` - Chargement parallèle

### Utilitaires
- `get_data_statistics()` - Stats des données
- `get_available_data()` - Inventaire des fichiers
- `cleanup_old_files()` - Nettoyage automatique

## 🎯 Cas d'Usage

- **Bots de Trading** : Diversité automatique pour réduire les biais
- **Analyse de Marché** : Vue complète de l'écosystème crypto
- **Screening** : Recherche d'opportunités dans tous les secteurs
- **Portfolios** : Construction équilibrée automatique
- **Recherche** : Données représentatives pour études

## 🆘 Support

1. **Documentation** : Consultez `docs/README_CORE_MANAGER.md`
2. **Tests** : Lancez `tests/test_diversite_simple.py`
3. **Exemples** : Explorez le dossier `examples/`
4. **Logs** : Vérifiez les messages du système

## 🎉 Avantages vs Version Standard

| Aspect | Version Standard | **Token Diversity Manager** |
|--------|-----------------|------------------------------|
| Sélection | Score marketcap/volume | ✅ **Score + Diversité garantie** |
| Représentation | Biais possibles | ✅ **10 catégories équilibrées** |
| Couverture | Partielle | ✅ **Écosystème complet** |
| Analyse | Basique | ✅ **Métriques de diversité** |
| Stratégies | Limitées | ✅ **Par catégorie + globales** |

---

**TradXPro Token Diversity Manager v1.1** - La diversité crypto garantie ! 🔒✨