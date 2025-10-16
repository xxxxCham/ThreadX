# 🔄 Migration Notice - ThreadX Engine

**Date**: 16 octobre 2025
**Status**: ⚠️ DEPRECATED - Use `src/threadx/` instead

---

## ⚠️ Important Notice

Le dossier `threadx_dashboard/engine/` contient des **implémentations legacy** qui sont progressivement dépréciées au profit de l'architecture centralisée dans `src/threadx/`.

### ✅ Current Status

| Module | Status | Action Required |
|--------|--------|-----------------|
| `indicators.py` | ❌ **REMOVED** | Use `src/threadx/indicators/` |
| `backtest_engine.py` | ⚠️ Legacy | Migrate to `src/threadx/backtest/` |
| `data_processor.py` | ⚠️ Legacy | Migrate to `src/threadx/data/` |

---

## 🎯 Migration Path

### For Indicators (COMPLETED ✅)

**OLD (Removed)**:
```python
from threadx_dashboard.engine.indicators import IndicatorCalculator
calculator = IndicatorCalculator()
result = calculator.calculate_indicator(...)
```

**NEW (Use this)**:
```python
# Option 1: NumPy optimized functions (50x faster)
from threadx.indicators.indicators_np import ema_np, rsi_np, macd_np
ema_values = ema_np(close_prices, span=20)

# Option 2: High-level engine with specs
from threadx.indicators.engine import enrich_indicators
df_with_indicators = enrich_indicators(df, specs=[
    {"name": "EMA", "params": {"span": 20}, "outputs": ["ema_20"]}
])

# Option 3: Via Bridge (recommended for UI)
from threadx.bridge import IndicatorController, IndicatorRequest
controller = IndicatorController()
result = controller.build_indicators(request)
```

### For Backtest (TODO)

**CURRENT**:
```python
from threadx_dashboard.engine.backtest_engine import BacktestEngine
```

**SHOULD MIGRATE TO**:
```python
# Via Bridge (recommended)
from threadx.bridge import BacktestController, BacktestRequest
controller = BacktestController()
result = controller.run_backtest(request)

# Or direct Engine (for tests/scripts)
from threadx.backtest.engine import BacktestEngine
engine = BacktestEngine(...)
```

### For Data Processing (TODO)

**CURRENT**:
```python
from threadx_dashboard.engine.data_processor import DataProcessor
```

**SHOULD MIGRATE TO**:
```python
# Via Bridge (recommended)
from threadx.bridge import DataController, DataRequest
controller = DataController()
result = controller.validate_data(request)

# Or direct modules
from threadx.data.io import load_parquet
from threadx.data.validation import validate_ohlcv
```

---

## 🔧 Why Migrate?

### Architecture Benefits

1. **Single Source of Truth**: One implementation, no duplication
2. **Better Performance**: NumPy optimizations (50x faster)
3. **Type Safety**: Full type hints with mypy compliance
4. **Bridge Pattern**: Clean UI/Engine separation
5. **Testing**: Isolated unit tests for each layer

### Before (threadx_dashboard/engine)
```
UI → threadx_dashboard.engine → Calculations
     (Duplicated code, pandas-based)
```

### After (src/threadx via Bridge)
```
UI → Bridge → Controllers → Engine
     (Centralized, NumPy-optimized, type-safe)
```

---

## 📋 Checklist for Full Migration

- [x] **Phase 1**: Remove `indicators.py` duplication
- [ ] **Phase 2**: Deprecate `backtest_engine.py` with wrapper
- [ ] **Phase 3**: Deprecate `data_processor.py` with wrapper
- [ ] **Phase 4**: Update all references in threadx_dashboard app
- [ ] **Phase 5**: Remove legacy engine/ directory completely

---

## 🚀 Next Steps

1. **For new code**: ALWAYS use `src/threadx/` modules via Bridge
2. **For existing code**: Gradually migrate to new architecture
3. **Testing**: Ensure all tests pass after migration

---

## 📚 References

- **Bridge Documentation**: `src/threadx/bridge/README.md`
- **Indicators Documentation**: `src/threadx/indicators/README.md`
- **Architecture Audit**: `RAPPORT_COHERENCE_ARCHITECTURE.md`

---

**Questions?** Check the main ThreadX documentation or ask the team.

**DO NOT** add new code to `threadx_dashboard/engine/` - it's deprecated!
