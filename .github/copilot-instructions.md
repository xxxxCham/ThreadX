# ThreadX AI Coding Instructions

ThreadX is a high-performance algorithmic trading framework with GPU acceleration and multi-device support. This guide helps AI agents understand the project's architecture and conventions.

## Architecture Overview

### Core Components
- **`src/threadx/backtest/engine.py`**: Production backtest orchestrator using device-agnostic computing
- **`src/threadx/indicators/`**: GPU-accelerated technical indicators with caching via `bank.py`
- **`src/threadx/strategy/bb_atr.py`**: Main Bollinger Bands + ATR strategy implementation
- **`src/threadx/utils/gpu/multi_gpu.py`**: Multi-GPU workload distribution (75%/25% split by default)
- **`src/threadx/utils/xp.py`**: Device-agnostic NumPy/CuPy abstraction layer
- **`src/threadx/ui/app.py`**: Main Tkinter application with tabbed interface

### Data Flow
```
bank.ensure(indicators) → engine.run(df, indicators, params) → RunResult → performance.summarize(result.returns, result.trades)
```

## Configuration System

- **TOML configuration + Environment Variables for API keys**
- Settings loaded via `src/threadx/config/settings.py`
- API authentication via `src/threadx/config/auth.py`
- Default paths: `./data`, `./cache`, `./logs`
- GPU devices configured as `["5090", "2060"]` with load balance `{"5090": 0.75, "2060": 0.25}`

### API Authentication
- Use environment variables for API keys: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `COINGECKO_API_KEY`
- Graceful fallback to public endpoints when credentials unavailable
- AuthManager provides centralized credential management
- DataClient handles authenticated requests with rate limiting

## GPU Architecture

### Device Management
- **Fallback-first design**: All GPU code gracefully falls back to NumPy
- Use `from threadx.utils import xp; xp_module = xp.get_xp()` for device-agnostic arrays
- Multi-GPU via `MultiGPUManager` with automatic workload splitting
- Memory management with 80% threshold and auto-fallback

### Key Patterns
```python
# Device-agnostic computing
from threadx.utils import xp
xp_module = xp.get_xp()  # Returns numpy or cupy
result = xp_module.sum(data)

# Multi-GPU distribution
manager = MultiGPUManager()
result = manager.distribute_workload(data, vectorized_func)
```

## Testing Framework

### Test Organization
- **Markers**: `@pytest.mark.gpu`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Run tests: `python -m pytest tests -v`
- GPU tests: `python -m pytest -m gpu`
- Integration tests: `python -m pytest -m integration`

### Key Test Files
- `tests/test_integration.py`: Complete pipeline testing
- `smoke_tests.py`: Quick validation of core functionality
- GPU tests require actual hardware or gracefully skip

## Backtest Engine Patterns

### RunResult Structure
```python
@dataclass
class RunResult:
    equity: pd.Series      # Datetime-indexed equity curve
    returns: pd.Series     # Daily returns
    trades: pd.DataFrame   # Trade log with entry/exit
    meta: Dict[str, Any]   # Metadata (execution time, device used)
```

### Engine Usage
```python
engine = create_engine()
result = engine.run(
    data=ohlcv_df,
    indicators={"bollinger": bb_params, "atr": atr_params},
    seed=42  # Deterministic results
)
```

## Indicator Bank System

### Caching Strategy
- TTL-based cache (3600 seconds default)
- Automatic batch processing (threshold: 100 parameters)
- Registry updates with checksums for integrity
- Path: `indicators_cache/{indicator_type}/`

### Usage Pattern
```python
from threadx.indicators.bank import ensure_indicator

# Single indicator
bb_result = ensure_indicator('bollinger', {'period': 20, 'std': 2.0}, close_data)

# Batch processing
bank = IndicatorBank()
results = bank.batch_ensure('bollinger', params_list, close_data)
```

## Development Workflows

### Essential Commands
```bash
# Setup
.venv/Scripts/python.exe -m pip install -e .

# Environment Configuration
.\setup_env.ps1 -Interactive              # Windows PowerShell setup
python test_auth.py --verbose            # Test API authentication

# Testing
.venv/Scripts/python.exe -m pytest tests -v
.venv/Scripts/python.exe smoke_tests.py

# Applications
.venv/Scripts/python.exe -m streamlit run apps/streamlit/app.py
.venv/Scripts/python.exe apps/tkinter/run_tkinter.py

# Benchmarks
.venv/Scripts/python.exe -m threadx.benchmarks.run_indicators
.venv/Scripts/python.exe -m threadx.optimization.run --config configs/sweeps/bb_atr_grid.toml
```

### VS Code Tasks
Use `run_task` tool with these predefined tasks:
- `"shell: Run Tests"`: Execute test suite
- `"shell: Start Streamlit App"`: Launch web interface
- `"shell: Run Benchmarks - Indicators"`: Performance testing
- `"shell: Optimization Grid Sweep"`: Parameter optimization

## Code Conventions

### Error Handling
- **Graceful degradation**: GPU → CPU fallback throughout
- Import-level fallbacks with `try/except ImportError`
- Use `logger = get_logger(__name__)` for consistent logging

### Strategy Implementation
- Extend `threadx.strategy.model.Strategy` base class
- Parameters as dataclasses (e.g., `BBAtrParams`)
- Vectorized operations using xp abstraction
- Deterministic with configurable seeds

### UI Integration
- Threading for operations > 100ms to prevent blocking
- Pipeline: Data → Parameters → Backtest → Results → Export
- Real-time logging via `src/threadx/utils/log.py`

## File Naming
- Test files: `test_*.py` or `*_test.py`
- Configuration: `*.toml` files in `configs/`
- GPU utilities: `src/threadx/utils/gpu/`
- Strategies: `src/threadx/strategy/*.py`
- Indicators: `src/threadx/indicators/*.py`