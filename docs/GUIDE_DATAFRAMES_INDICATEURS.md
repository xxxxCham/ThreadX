# Guide des Structures de Données dans ThreadX

Ce document décrit les formats de données attendus par les composants clés de ThreadX, notamment le moteur de backtest (`BacktestEngine`) et la banque d'indicateurs (`IndicatorBank`).

## 1. DataFrame OHLCV

Le DataFrame OHLCV est la structure de base pour toutes les données de marché.

### Structure Attendue

- **Type**: `pandas.DataFrame`
- **Index**: Doit être un `pandas.DatetimeIndex`. Il est fortement recommandé d'utiliser des timestamps avec fuseau horaire (UTC) pour éviter toute ambiguïté.
- **Colonnes**: Les noms de colonnes suivants sont obligatoires, en minuscules.
  - `open` (float): Prix d'ouverture.
  - `high` (float): Prix le plus haut.
  - `low` (float): Prix le plus bas.
  - `close` (float): Prix de clôture.
  - `volume` (float ou int): Volume des transactions.

### Exemple de Création

```python
import pandas as pd

# Création d'un DataFrame OHLCV valide pour les tests
df_ohlcv = pd.DataFrame({
    'open': [50000.0, 50100.0, 50200.0],
    'high': [50200.0, 50300.0, 50400.0],
    'low': [49800.0, 49900.0, 50000.0],
    'close': [50100.0, 50200.0, 50300.0],
    'volume': [1000, 1500, 2000]
}, index=pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:01:00', '2024-01-01 00:02:00'], utc=True))

print(df_ohlcv)
```

### Producteurs et Consommateurs

- **Producteurs Principaux**:
  - Scripts de chargement de données (ex: `data.loader`, non montré ici) qui lisent des fichiers Parquet, CSV, ou interrogent des APIs.
  - Fonctions de génération de données de test (ex: dans `tests/` et `benchmarks/`).

- **Consommateurs Principaux**:
  - `threadx.backtest.engine.BacktestEngine.run(df_1m, ...)`: Le moteur de backtest prend ce DataFrame comme entrée principale.
  - `threadx.indicators.bank.IndicatorBank.ensure(...)`: La banque d'indicateurs l'utilise pour calculer les indicateurs. Par exemple, le calcul des Bandes de Bollinger extrait la colonne `close`.

## 2. Dictionnaire d'Indicateurs

Le dictionnaire d'indicateurs est la structure qui regroupe les résultats des calculs d'indicateurs pour les fournir au moteur de stratégie.

### Structure Attendue

- **Type**: `dict` (dictionnaire Python).
- **Clés**: `str`. Le nom de l'indicateur, en minuscules (ex: `"bollinger"`, `"atr"`).
- **Valeurs**: Le format de la valeur dépend de ce que retourne l'indicateur.
  - **Indicateur à sortie multiple (ex: Bandes de Bollinger)**: Un `tuple` de `numpy.ndarray` ou `pandas.Series`. Pour `"bollinger"`, c'est un tuple de 3 arrays : `(upper_band, middle_band, lower_band)`.
  - **Indicateur à sortie unique (ex: ATR)**: Un simple `numpy.ndarray` ou `pandas.Series`.

### Exemple de Création

```python
from threadx.indicators.bank import IndicatorBank
# Supposons que df_ohlcv est le DataFrame de l'étape précédente

# 1. Initialiser la banque d'indicateurs
bank = IndicatorBank()

# 2. Calculer les indicateurs nécessaires
bollinger_result = bank.ensure(
    indicator_type='bollinger',
    params={'period': 20, 'std': 2.0},
    data=df_ohlcv
)

atr_result = bank.ensure(
    indicator_type='atr',
    params={'period': 14},
    data=df_ohlcv
)

# 3. Assembler le dictionnaire pour le moteur de backtest
indicators_for_backtest = {
    "bollinger": bollinger_result,  # Ceci est un tuple (upper, middle, lower)
    "atr": atr_result               # Ceci est un array
}

print(indicators_for_backtest)
```

### Producteurs et Consommateurs

- **Producteur Principal**:
  - Le code utilisateur qui orchestre le calcul via `IndicatorBank` et assemble le dictionnaire final.

- **Consommateur Principal**:
  - `threadx.backtest.engine.BacktestEngine.run(indicators, ...)`: Le dictionnaire est passé via le paramètre `indicators`.
  - `threadx.backtest.engine._generate_trading_signals()`: Cette méthode interne au moteur accède aux clés du dictionnaire (`indicators['bollinger']`, `indicators['atr']`) pour implémenter la logique de la stratégie.