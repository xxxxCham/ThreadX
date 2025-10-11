# ThreadX - Framework de Trading Crypto Haute Performance

> **Version 2.0** - Architecture refactorisée avec bonnes pratiques professionnelles

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Vue d'Ensemble

ThreadX est un framework Python haute performance pour le trading crypto, offrant:

- ⚡ **Indicateurs optimisés NumPy** (50x plus rapides que pandas)
- 📊 **Backtesting engine** robuste avec métriques détaillées
- 🔄 **Pipeline de données** complet (Binance → Parquet)
- 🎨 **Interface UI** (Tkinter + Streamlit)
- 🧪 **Architecture testable** avec dependency injection
- 📈 **Gestion diversité tokens** (Top 100 avec garanties)

---

## 🚀 Démarrage Rapide

### Installation

```bash
# Cloner le repository
git clone https://github.com/threadx/threadx.git
cd threadx

# Installer en mode développement
make install

# Ou manuellement
pip install -e ".[dev]"
pre-commit install
```

### Utilisation Basique

```python
from threadx.config import get_settings
from threadx.indicators.numpy import rsi_np, macd_np
from threadx.data.ingest import IngestionManager

# Configuration
settings = get_settings()

# Téléchargement données
manager = IngestionManager(settings)
manager.download(["BTCUSDT"], timeframes=["1h"])

# Calcul indicateurs (NumPy ultra-rapide)
import pandas as pd
df = pd.read_parquet("data/processed/parquet/BTCUSDT_1h.parquet")
df["rsi"] = rsi_np(df["close"].values, period=14)

# Backtest
from threadx.backtest.engine import BacktestEngine
engine = BacktestEngine()
results = engine.run(df, strategy_params={...})
```

---

## 📚 Documentation Complète

### 🌟 **COMMENCER ICI**: [GUIDE_MIGRATION_RAPIDE.md](GUIDE_MIGRATION_RAPIDE.md)

### Guides Principaux

| Document                                                                       | Description                 | Audience        |
| ------------------------------------------------------------------------------ | --------------------------- | --------------- |
| [GUIDE_MIGRATION_RAPIDE.md](GUIDE_MIGRATION_RAPIDE.md)                         | Guide démarrage & migration | Tous            |
| [docs/BONNES_PRATIQUES_ARCHITECTURE.md](docs/BONNES_PRATIQUES_ARCHITECTURE.md) | Architecture moderne        | Développeurs    |
| [docs/RAPPORT_REDONDANCES_PIPELINE.md](docs/RAPPORT_REDONDANCES_PIPELINE.md)   | Analyse & optimisations     | Mainteneurs     |
| [docs/GUIDE_MIGRATION_TRADXPRO_V2.md](docs/GUIDE_MIGRATION_TRADXPRO_V2.md)     | Migration v1→v2             | Utilisateurs v1 |

---

## ⚡ Performance

### Benchmarks Indicateurs

| Indicateur        | Pandas | NumPy ThreadX | Speedup   |
| ----------------- | ------ | ------------- | --------- |
| **RSI** (10k pts) | 125ms  | 2.5ms         | **50x** ⚡ |
| **Bollinger**     | 110ms  | 3.1ms         | **35x** ⚡ |
| **MACD**          | 95ms   | 2.8ms         | **34x** ⚡ |
| **ATR**           | 80ms   | 2.2ms         | **36x** ⚡ |

---

## 🛠️ Commandes Utiles

```bash
# Installation & Setup
make install        # Installe dépendances + pre-commit

# Développement
make test           # Tests avec coverage
make lint           # Vérification qualité
make format         # Formatage automatique
make clean          # Nettoie fichiers temp

# Migration
python scripts/migrate_to_best_practices.py --phase 1
```

---

## 🏗️ Architecture

```
ThreadX/
├── src/threadx/              # Code source
│   ├── config/               # Configuration Pydantic
│   ├── data/                 # Pipeline données
│   ├── indicators/           
│   │   └── numpy.py         # ✨ Optimisations (50x)
│   ├── backtest/             # Moteur backtesting
│   └── ui/                   # Interfaces
├── tests/                    # Tests pytest
├── configs/                  # TOML configs
├── docs/                     # Documentation
└── scripts/                  # Scripts utilitaires
```

---

## 🌟 Fonctionnalités Clés

### 1. Indicateurs Ultra-Rapides

```python
from threadx.indicators.numpy import add_all_indicators

df = add_all_indicators(df, rsi_period=14, bb_period=20)
# → RSI, MACD, BB, ATR, VWAP, OBV en < 50ms
```

### 2. Configuration Type-Safe

```python
from threadx.config import Settings

settings = Settings.load_from_toml("configs/production.toml")
# Validation Pydantic automatique
```

### 3. Pipeline Données Complet

```python
from threadx.data.ingest import IngestionManager

manager = IngestionManager(settings)
manager.download_top_100(timeframes=["1h", "4h"])
```

---

## 🎯 Roadmap

### v2.0 (Actuel) ✅
- Architecture refactorisée sans redondances
- Indicateurs NumPy optimisés (50x)
- Configuration Pydantic + TOML
- Tests automatisés CI/CD

### v2.1 (Prochaine)
- Support multi-exchange
- Dashboard Streamlit temps réel
- Stratégies ML
- API REST

---

## 📜 Licence

MIT License - Voir [LICENSE](LICENSE)

---

**Créé avec ❤️ par ThreadX Core Team**  
**Version**: 2.0.0 | **Date**: 11 octobre 2025
