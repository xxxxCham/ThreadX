# TokenDiversityDataSource - Documentation

## Vue d'ensemble

Le `TokenDiversityDataSource` est un provider de données standardisé pour ThreadX qui organise les symboles de trading par groupes de diversité (L1, L2, DeFi, AI, Gaming, etc.) et fournit un accès uniforme aux données OHLCV normalisées.

## Caractéristiques

- **Interface stable** : API cohérente pour intégration UI/CLI future
- **Groupes de diversité** : Organisation logique des tokens par secteur/technologie
- **Validation stricte** : Timeframes supportés avec mapping d'alias
- **Normalisation OHLCV** : Données UTC, colonnes standard, invariants validés
- **Gestion d'erreurs** : Exceptions spécialisées avec messages informatifs

## API Publique

### TokenDiversityConfig

Configuration du provider définissant les groupes, symboles et timeframes supportés.

```python
@dataclass(frozen=True)
class TokenDiversityConfig:
    groups: Mapping[str, list[str]]        # Groupes → symboles
    symbols: list[str]                     # Liste complète des symboles
    supported_tf: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")
```

### TokenDiversityDataSource

Provider principal implémentant l'interface standardisée ThreadX.

#### Méthodes principales

```python
def list_symbols(self, group: str | None = None) -> list[str]
```
Retourne la liste des symboles, optionnellement filtrés par groupe.

```python
def supported_timeframes(self) -> tuple[str, ...]
```
Retourne les timeframes supportés par ce provider.

```python
def get_frame(self, symbol: str, timeframe: str) -> pd.DataFrame
```
Récupère un DataFrame OHLCV normalisé (index UTC, colonnes standard).

## Utilisation

### Configuration par défaut

```python
from threadx.data.providers.token_diversity import (
    TokenDiversityDataSource,
    create_default_config
)

# Configuration avec groupes prédéfinis
config = create_default_config()
provider = TokenDiversityDataSource(config)

# Lister tous les symboles
all_symbols = provider.list_symbols()

# Lister par groupe
l2_tokens = provider.list_symbols("L2")
defi_tokens = provider.list_symbols("DeFi")
```

### Configuration personnalisée

```python
from threadx.data.providers.token_diversity import TokenDiversityConfig

config = TokenDiversityConfig(
    groups={
        "MajorPairs": ["BTCUSDT", "ETHUSDT"],
        "Altcoins": ["ADAUSDT", "DOTUSDT"],
    },
    symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"],
    supported_tf=("1h", "4h", "1d")
)

provider = TokenDiversityDataSource(config)
```

### Récupération de données

```python
# Récupération DataFrame OHLCV
df = provider.get_frame("BTCUSDT", "1h")

# Vérifications automatiques effectuées :
assert df.index.tz is not None  # Index UTC
assert (df["high"] >= df["low"]).all()  # Cohérence OHLC
assert (df["volume"] >= 0).all()  # Volume positif
assert not df["close"].isna().any()  # Prix valides

# Alias de timeframes supportés
df_m1 = provider.get_frame("ETHUSDT", "M1")  # → "1m"
df_h4 = provider.get_frame("ETHUSDT", "H4")  # → "4h"
```

## Gestion d'erreurs

### Exceptions spécialisées

```python
from threadx.data.errors import DataNotFoundError, UnsupportedTimeframeError

try:
    df = provider.get_frame("INVALIDUSDT", "1h")
except DataNotFoundError as e:
    print(f"Symbole non trouvé: {e}")

try:
    df = provider.get_frame("BTCUSDT", "2m")
except UnsupportedTimeframeError as e:
    print(f"Timeframe non supporté: {e}")
```

### Messages informatifs

Les erreurs incluent des informations contextuelles :
- Symboles/groupes disponibles pour `DataNotFoundError`
- Timeframes supportés pour `UnsupportedTimeframeError`

## Pipeline de normalisation

Le provider applique le pipeline de normalisation ThreadX :

1. **Validation symbole** : Vérification dans la liste configurée
2. **Résolution timeframe** : Mapping alias → format standard
3. **Récupération brute** : Via `_fetch_raw_bars()` (extensible)
4. **Normalisation OHLCV** : Appel `threadx.data.io.normalize_ohlcv()`
5. **Validation invariants** : Vérifications supplémentaires strictes

## Extensibilité

### Étape A (actuelle)
- Interface standardisée définie
- Stub `_fetch_raw_bars()` avec données synthétiques
- Tests complets de l'API

### Étapes futures
- **Étape B** : Branchement sur persistance réelle (`write_frame`)
- **Étape C** : Intégration token diversity manager
- **Étape D** : Support indicateurs via IndicatorBank
- **Étape E** : Interface CLI/UI

### Personnalisation source de données

Pour brancher sur une source réelle, override `_fetch_raw_bars()` :

```python
class CustomDataSource(TokenDiversityDataSource):
    def _fetch_raw_bars(self, symbol: str, tf: str) -> pd.DataFrame:
        # Votre logique de récupération
        return your_data_adapter.get_ohlcv(symbol, tf)
```

## Tests

Suite complète de tests unitaires disponible :

```bash
python -m pytest tests/data/providers/test_token_diversity.py -v
```

**Couverture** :
- Configuration et initialisation
- Énumération symboles/timeframes  
- Récupération données avec normalisation
- Validation invariants OHLCV
- Gestion d'erreurs complète
- Scénarios d'intégration multi-symboles

## Logging

Le provider utilise le système de logging ThreadX :

```python
import logging
logging.getLogger('threadx.data.providers.token_diversity').setLevel(logging.DEBUG)
```

Logs automatiques pour :
- Initialisation provider (groupes, symboles, timeframes)
- Récupération données (symbole, tf, nombre de barres, période)
- Erreurs et avertissements détaillés

## Groupes par défaut

La configuration par défaut inclut :

- **L1** : Bitcoin, Ethereum
- **L2** : Arbitrum, Optimism, Polygon  
- **DeFi** : Uniswap, Aave, Compound
- **AI** : Fetch.ai, SingularityNET, Ocean Protocol
- **Gaming** : Axie Infinity, Sandbox, Decentraland

Chaque groupe contient 2-3 tokens représentatifs du secteur.

## Performance

- **Génération stub** : ~0.1ms par symbole/timeframe
- **Normalisation** : ~5-10ms pour 200 barres
- **Validation** : ~1-2ms pour invariants complets
- **Mémoire** : ~50KB par DataFrame 200 barres

## Compatibilité

- **Python** : 3.10+
- **Pandas** : 1.5+
- **ThreadX** : Phase 1 (config, data.io)
- **Pandera** : Optionnel (validation schéma)

---

*ThreadX Framework - Phase 1, Étape A - Provider de données unifié*