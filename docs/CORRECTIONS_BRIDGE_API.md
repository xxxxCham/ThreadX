# Corrections Controllers Bridge - API Réelles Engine

## Problème Identifié
Les controllers utilisent des signatures **hypothétiques** au lieu des **vraies APIs** des modules Engine existants.

## Vraies Signatures à Utiliser

### 1. BacktestEngine (src/threadx/backtest/engine.py)

**Factory Function:**
```python
def create_engine(
    gpu_balance: Optional[Dict[str, float]] = None,
    use_multi_gpu: bool = True
) -> BacktestEngine
```

**Méthode run():**
```python
def run(
    self,
    df_1m: pd.DataFrame,
    indicators: Dict[str, Any],
    *,
    params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    seed: int = 42,
    use_gpu: Optional[bool] = None,
) -> RunResult
```

**RunResult (DataClass):**
```python
@dataclass
class RunResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)
```

### 2. IndicatorBank (src/threadx/indicators/bank.py)

**Constructeur:**
```python
def __init__(self, settings: Optional[IndicatorSettings] = None)
```

**Méthode ensure():**
```python
def ensure(
    self,
    indicator_type: str,
    params: Dict[str, Any],
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    symbol: str = "",
    timeframe: str = "",
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]
```

**Fonction globale ensure_indicator():**
```python
def ensure_indicator(
    indicator_type: str,
    params: Dict[str, Any],
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    symbol: str = "",
    timeframe: str = "",
    cache_dir: str = "indicators_cache",
) -> Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]
```

### 3. Data Module (src/threadx/data/loader.py)

**Classe principale:**
```python
class BinanceDataLoader:
    def __init__(
        self,
        json_cache_dir: Optional[Path] = None,
        parquet_cache_dir: Optional[Path] = None,
    )
```

**Note:** Pas de `load_parquet()` ou `get_data_path()` dans les modules actuels.
→ Il faut soit créer ces helpers ou utiliser pandas directement.

### 4. UnifiedOptimizationEngine

**À VÉRIFIER** - ne semble pas exister dans l'état actuel.
→ Peut-être dans optimization/engine.py mais avec API différente.

## Corrections à Appliquer

### BacktestController.run_backtest()

**AVANT (hypothétique):**
```python
engine = create_engine(
    strategy_name=request.strategy,
    params=request.params,
    initial_cash=request.initial_cash,
    use_gpu=request.use_gpu
)
raw_result = engine.run(symbol, timeframe, start_date, end_date)
```

**APRÈS (réel):**
```python
# 1. Charger données OHLCV
df_1m = load_data(request.symbol, request.timeframe)

# 2. Calculer indicateurs via IndicatorBank
bank = IndicatorBank()
indicators = build_strategy_indicators(bank, df_1m, request.strategy, request.params)

# 3. Créer engine
engine = create_engine(use_gpu=request.use_gpu)

# 4. Run backtest
raw_result: RunResult = engine.run(
    df_1m=df_1m,
    indicators=indicators,
    params=request.params,
    symbol=request.symbol,
    timeframe=request.timeframe,
    use_gpu=request.use_gpu,
)

# 5. Mapper RunResult (DataClass) → BacktestResult (dict-based)
return BacktestResult(
    total_profit=calculate_profit(raw_result.equity),
    total_return=calculate_return(raw_result.returns),
    sharpe_ratio=calculate_sharpe(raw_result.returns),
    max_drawdown=calculate_drawdown(raw_result.equity),
    win_rate=calculate_win_rate(raw_result.trades),
    trades=raw_result.trades.to_dict('records'),
    equity_curve=raw_result.equity.tolist(),
    drawdown_curve=calculate_drawdown_curve(raw_result.equity).tolist(),
    ...
)
```

### IndicatorController.build_indicators()

**APRÈS (simplifié):**
```python
bank = IndicatorBank()  # Pas de params constructeur complexes
indicator_values = {}

for indicator_name, params in request.indicators.items():
    # Charger données si nécessaire
    df = load_data(request.symbol, request.timeframe, request.data_path)

    # Calculer indicateur
    values = bank.ensure(
        indicator_type=indicator_name,
        params=params,
        data=df,
        symbol=request.symbol,
        timeframe=request.timeframe,
    )
    indicator_values[indicator_name] = values

# Stats cache (utiliser bank.stats)
cache_hits = bank.stats.get('cache_hits', 0)
cache_misses = bank.stats.get('cache_misses', 0)
```

## Actions Immédiates

1. **Créer helpers data/** :
   - `load_data(symbol, timeframe, path=None) -> pd.DataFrame`
   - `get_data_path(symbol, timeframe) -> Path`

2. **Modifier BacktestController** pour suivre workflow réel :
   - Chargement données
   - Calcul indicateurs
   - Exécution backtest
   - Mapping RunResult → BacktestResult

3. **Simplifier IndicatorController** :
   - Utiliser vraie API bank.ensure()
   - Pas de cache_path/use_gpu en constructeur IndicatorBank

4. **Vérifier UnifiedOptimizationEngine** :
   - Lire src/threadx/optimization/engine.py
   - Adapter SweepController selon API réelle

5. **Créer validation helpers** :
   - Data validation peut être simple pandas checks
   - Pas besoin wrapper complexe si Engine fait déjà

## Philosophie Bridge

> **RAPPEL CRITIQUE** : Bridge ne FAIT PAS de calculs métier.
> Bridge ORCHESTRE des appels vers Engine existant.
>
> Si Engine n'a pas une fonction, Bridge ne doit PAS la créer.
> → Soit ajouter fonction à Engine, soit simplifier Bridge Request.
