# 🎯 Rapport Intermédiaire - Débogage Token Gestion

**Date**: 10 octobre 2025  
**Durée**: 30 minutes  
**Fichiers traités**: 2 (token_diversity.py créé, diversity_pipeline.py en cours)

---

## ✅ Progrès Réalisés

### 1. Création token_diversity.py ✅

**Fichier**: `src/threadx/data/providers/token_diversity.py` (316 lignes)

**Classes implémentées**:
- ✅ `TokenDiversityConfig` - Configuration des groupes/symboles
- ✅ `TokenDiversityDataSource` - Provider OHLCV brut
- ✅ `create_default_config()` - Configuration par défaut

**État**: **COMPLET** avec implémentation stub

**Détails**:
```python
# Classes principales
class TokenDiversityConfig:
    groups: Mapping[str, List[str]]
    symbols: List[str]
    supported_tf: Tuple[str, ...]

class TokenDiversityDataSource:
    def __init__(self, config: TokenDiversityConfig)
    def list_symbols(self, group: Optional[str] = None) -> List[str]
    def list_groups(self) -> List[str]
    def fetch_ohlcv(...) -> pd.DataFrame  # STUB - À implémenter
    def validate_symbol(self, symbol: str) -> bool
    def validate_timeframe(self, timeframe: str) -> bool

def create_default_config() -> TokenDiversityConfig:
    # Groupes: L1, DeFi, L2, Stable
```

**Note**: `fetch_ohlcv()` est un **STUB** qui lève `NotImplementedError`. 
Nécessite implémentation pour:
- Lecture depuis fichiers Parquet locaux, OU
- API Binance/exchange, OU
- Intégration TradXProManager

---

### 2. Corrections diversity_pipeline.py (Partielles) 🔄

**Fichier**: `src/threadx/data/diversity_pipeline.py` (417 lignes)

**Corrections appliquées**:
- ✅ Supprimé `Tuple` inutilisé (ligne 14)
- ✅ Supprimé `normalize_ohlcv` inutilisé (ligne 19)
- ✅ Supprimé `read_frame` inutilisé (ligne 19)
- ✅ Supprimé import `RegistryManager` (ligne 25 - n'existe pas)
- ✅ Corrigé `provider.get_frame()` → `provider.fetch_ohlcv()` (ligne 137)

**Corrections restantes**:
- ❌ `bank.compute_batch()` n'existe pas (ligne 170)
- ❌ `td_config.cache_dir` n'existe pas (ligne 197)  
- ❌ `provider.list_symbols(limit=10)` - paramètre invalide (ligne 256)
- ❌ Type `List[int]` devrait être `List[float]` (ligne 329)
- ⚠️ 15 lignes >79 chars (formatage)

---

## ❌ Problèmes Bloquants Restants

### 1. API IndicatorBank Incompatible 🚨

**Ligne 170-172**:
```python
# Code actuel (INVALIDE)
indicators_result = bank.compute_batch(
    data=ohlcv_df, indicators=indicators, symbol=symbol
)
```

**Problème**: `IndicatorBank.compute_batch()` **n'existe pas**

**API réelle disponible**:
```python
# bank.py ligne 499
def batch_ensure(
    self,
    indicator_type: str,        # ❌ Pas "indicators" list
    params_list: List[Dict],    # ❌ Params structurés
    data: DataFrame,
    symbol: str = "",
    timeframe: str = ""
) -> Dict[str, result]
```

**Différence**:
- `compute_batch()` attendrait une liste d'indicateurs (ex: `["rsi_14", "bb_20"]`)
- `batch_ensure()` attend un type unique + liste de paramètres

**Solutions possibles**:

#### Option A : Adapter diversity_pipeline.py à l'API batch_ensure
```python
# Transformer indicators=["rsi_14", "bb_20", "sma_50"]
# En:
results = {}
for indicator in indicators:
    ind_type, params = parse_indicator(indicator)  # ex: "rsi_14" → ("rsi", {period: 14})
    result = bank.batch_ensure(ind_type, [params], ohlcv_df, symbol)
    results.update(result)
```

#### Option B : Créer méthode wrapper compute_batch() dans IndicatorBank
```python
# Dans bank.py
def compute_batch(
    self,
    data: DataFrame,
    indicators: List[str],  # ex: ["rsi_14", "bb_20_2.0"]
    symbol: str = ""
) -> DataFrame:
    """
    Wrapper pour calculer plusieurs indicateurs d'un coup.
    
    Args:
        indicators: Liste format "type_param1_param2" 
                    ex: "rsi_14", "bb_20_2.0", "sma_50"
    
    Returns:
        DataFrame avec colonnes d'indicateurs ajoutées
    """
    # Implémenter parsing + appels batch_ensure
    ...
```

#### Option C : Utiliser ensure() individuel
```python
# Plus simple mais moins performant
for indicator in indicators:
    ind_spec = parse_indicator_spec(indicator)
    result = bank.ensure(ind_spec, data, symbol)
    # Ajouter au DataFrame
```

**Recommandation**: **Option B** (créer `compute_batch()` wrapper)
- Conserve l'intention du code original
- API plus intuitive pour les utilisateurs
- Performance raisonnable via batch_ensure interne

---

### 2. Configuration cache_dir Manquante 🚨

**Ligne 197**:
```python
output_dir or td_config.cache_dir,  # ❌ cache_dir n'existe pas
```

**Problème**: `TokenDiversityConfig` n'a pas d'attribut `cache_dir`

**Solution simple**:
```python
# Option 1: Ajouter cache_dir à TokenDiversityConfig
@dataclass(frozen=True)
class TokenDiversityConfig:
    groups: Mapping[str, List[str]]
    symbols: List[str]
    supported_tf: Tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")
    cache_dir: str = "./data/cache"  # ✅ Ajouté

# Option 2: Utiliser valeur par défaut dans diversity_pipeline.py
output_dir or Path("./data/diversity_cache")  # ✅ Fallback
```

**Recommandation**: **Option 1** (ajouter à Config)
- Plus cohérent avec la configuration centralisée
- Permet override par utilisateur

---

### 3. Paramètre list_symbols(limit=...) Invalide 🚨

**Ligne 256**:
```python
return provider.list_symbols(limit=10)  # ❌ Paramètre invalide
```

**Solution**:
```python
# Option 1: Limiter après récupération
symbols = provider.list_symbols()
return symbols[:10]  # ✅ Simple

# Option 2: Ajouter paramètre limit à la méthode
def list_symbols(self, group: Optional[str] = None, limit: Optional[int] = None):
    symbols = self.config.groups.get(group, []) if group else self.config.symbols
    return symbols[:limit] if limit else symbols  # ✅ Flexible
```

**Recommandation**: **Option 1** (limiter après)
- Plus simple, pas de modification API
- Suffisant pour le cas d'usage

---

### 4. Type Corrélation List[int] → List[float] ⚠️

**Ligne 329**:
```python
avg_correlations: List[int] = []  # ❌ Type incorrect
```

**Solution triviale**:
```python
avg_correlations: List[float] = []  # ✅ Type correct
```

---

## 📊 État Actuel des Erreurs

### Fichier diversity_pipeline.py

| Catégorie              | Count | État                     |
| ---------------------- | ----- | ------------------------ |
| **Erreurs critiques**  | 4     | 🔄 1 résolue, 3 restantes |
| **Erreurs mineures**   | 1     | ❌ Non résolue            |
| **Warnings formatage** | 15    | ❌ Non résolus            |
| **TOTAL**              | 20    | 🔄 **En cours**           |

### Détail Erreurs Critiques Restantes

1. ❌ `bank.compute_batch()` n'existe pas (ligne 170)
2. ❌ `td_config.cache_dir` n'existe pas (ligne 197)
3. ❌ `list_symbols(limit=10)` paramètre invalide (ligne 256)
4. ⚠️ Type `List[int]` vs `List[float]` (ligne 329)

---

## 🚀 Plan d'Action Recommandé

### Prochaine Session (1-2h)

#### Étape 1 : Créer compute_batch() dans IndicatorBank (30 min)

**Fichier**: `src/threadx/indicators/bank.py`

**Code à ajouter**:
```python
def compute_batch(
    self,
    data: pd.DataFrame,
    indicators: List[str],
    symbol: str = "",
    timeframe: str = ""
) -> pd.DataFrame:
    """
    Calcule plusieurs indicateurs et retourne DataFrame enrichi.
    
    Args:
        data: DataFrame OHLCV
        indicators: Liste format "type_period" ex: ["rsi_14", "bb_20"]
        symbol, timeframe: Métadonnées pour cache
    
    Returns:
        DataFrame avec colonnes d'indicateurs ajoutées
    
    Example:
        >>> bank = IndicatorBank()
        >>> df_with_ind = bank.compute_batch(
        ...     ohlcv_df,
        ...     indicators=["rsi_14", "bb_20_2.0", "sma_50"]
        ... )
        >>> print(df_with_ind.columns)
        ['open', 'high', 'low', 'close', 'volume', 
         'rsi_14', 'bb_upper_20', 'bb_middle_20', 'bb_lower_20', 'sma_50']
    """
    result_df = data.copy()
    
    for indicator_str in indicators:
        # Parse format "type_param1_param2..."
        parts = indicator_str.split("_")
        ind_type = parts[0]
        
        # Parser selon type
        if ind_type == "rsi":
            period = int(parts[1]) if len(parts) > 1 else 14
            params = {"period": period}
            rsi_result = self.batch_ensure(ind_type, [params], data, symbol, timeframe)
            result_df[f"rsi_{period}"] = list(rsi_result.values())[0]
        
        elif ind_type == "bb":
            period = int(parts[1]) if len(parts) > 1 else 20
            std = float(parts[2]) if len(parts) > 2 else 2.0
            params = {"period": period, "std_dev": std}
            bb_result = self.batch_ensure(ind_type, [params], data, symbol, timeframe)
            upper, middle, lower = list(bb_result.values())[0]
            result_df[f"bb_upper_{period}"] = upper
            result_df[f"bb_middle_{period}"] = middle
            result_df[f"bb_lower_{period}"] = lower
        
        # Ajouter autres indicateurs (sma, ema, macd, atr...)
    
    return result_df
```

#### Étape 2 : Ajouter cache_dir à TokenDiversityConfig (5 min)

**Fichier**: `src/threadx/data/providers/token_diversity.py`

```python
@dataclass(frozen=True)
class TokenDiversityConfig:
    groups: Mapping[str, List[str]]
    symbols: List[str]
    supported_tf: Tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")
    cache_dir: str = "./data/diversity_cache"  # ✅ Ajouté
```

#### Étape 3 : Corrections finales diversity_pipeline.py (15 min)

1. Corriger `list_symbols(limit=10)` → `list_symbols()[:10]`
2. Corriger `List[int]` → `List[float]`
3. Formatter avec Black (optionnel)

#### Étape 4 : Tests de validation (30 min)

```python
# test_token_diversity.py
def test_token_provider():
    config = create_default_config()
    provider = TokenDiversityDataSource(config)
    
    # Test list_symbols
    assert len(provider.list_symbols()) > 0
    assert "BTCUSDT" in provider.list_symbols("L1")
    
    # Test fetch_ohlcv (NotImplementedError attendu)
    with pytest.raises(NotImplementedError):
        provider.fetch_ohlcv("BTCUSDT", "1h")

def test_diversity_pipeline():
    # Test avec données mockées
    ...
```

---

## 💡 Décision Requise

Pour continuer le débogage, j'ai besoin de votre choix :

### Option 1 : Implémenter compute_batch() (Recommandé) ⭐
- **Durée**: 30-45 min
- **Avantages**: API propre, réutilisable
- **Inconvénient**: Modification IndicatorBank

### Option 2 : Adapter diversity_pipeline.py à batch_ensure()
- **Durée**: 20-30 min
- **Avantages**: Pas de modification IndicatorBank
- **Inconvénient**: Code moins lisible

### Option 3 : Reporter IndicatorBank, finir autres corrections
- **Durée**: 10 min
- **Avantages**: Résolution rapide 3 autres erreurs
- **Inconvénient**: diversity_pipeline.py reste cassé

**Que préférez-vous ?**

---

**Auteur**: GitHub Copilot  
**Date**: 10 octobre 2025  
**Temps écoulé**: 30 minutes  
**Progrès**: 40% (1/2 fichiers complets)
