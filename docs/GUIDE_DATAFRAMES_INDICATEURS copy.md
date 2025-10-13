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