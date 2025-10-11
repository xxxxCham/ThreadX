# Guide des Bonnes Pratiques - Architecture ThreadX
**Date**: 11 octobre 2025  
**Version**: 1.0  
**Objectif**: Structure professionnelle et maintenable

---

## 📐 Architecture Recommandée

### Structure de Répertoires Idéale

```
ThreadX/
├── .github/                      # CI/CD & GitHub configs
│   ├── workflows/
│   │   ├── ci.yml               # Tests automatiques
│   │   ├── release.yml          # Publication releases
│   │   └── lint.yml             # Vérification code quality
│   └── ISSUE_TEMPLATE/          # Templates issues
│
├── src/threadx/                 # Code source principal (PEP 420)
│   ├── __init__.py
│   ├── config/                  # Configuration centralisée
│   │   ├── __init__.py
│   │   ├── settings.py          # Classe Settings (Pydantic)
│   │   └── paths.py             # Gestion chemins
│   │
│   ├── data/                    # Pipeline données
│   │   ├── __init__.py
│   │   ├── providers/           # Sources de données
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Interface abstraite
│   │   │   ├── binance.py       # Provider Binance
│   │   │   └── token_diversity.py
│   │   ├── ingest.py            # Ingestion manager
│   │   ├── io.py                # I/O Parquet/JSON
│   │   └── registry.py          # Métadonnées
│   │
│   ├── indicators/              # Indicateurs techniques
│   │   ├── __init__.py
│   │   ├── numpy.py             # ✅ Créé (optimisations NumPy)
│   │   ├── bank.py              # Cache & registry
│   │   └── custom/              # Indicateurs personnalisés
│   │       └── __init__.py
│   │
│   ├── backtest/                # Moteur backtesting
│   │   ├── __init__.py
│   │   ├── engine.py            # BacktestEngine
│   │   ├── strategy.py          # Stratégies base
│   │   └── results.py           # Analyse résultats
│   │
│   ├── ui/                      # Interfaces utilisateur
│   │   ├── __init__.py
│   │   ├── data_manager.py      # ✅ Existe (Tkinter)
│   │   ├── cli.py               # Interface CLI
│   │   └── dashboard.py         # Streamlit dashboard
│   │
│   ├── utils/                   # Utilitaires
│   │   ├── __init__.py
│   │   ├── logging_utils.py     # Configuration logging
│   │   ├── performance.py       # Profiling
│   │   └── validators.py        # Validation données
│   │
│   └── models/                  # Modèles de données (Pydantic)
│       ├── __init__.py
│       ├── ohlcv.py             # Modèle OHLCV
│       └── backtest.py          # Résultats backtest
│
├── tests/                       # Tests unitaires & intégration
│   ├── __init__.py
│   ├── conftest.py              # Fixtures pytest
│   ├── unit/                    # Tests unitaires
│   │   ├── test_indicators.py
│   │   ├── test_data_providers.py
│   │   └── test_backtest.py
│   ├── integration/             # Tests intégration
│   │   └── test_pipeline.py
│   └── fixtures/                # Données de test
│       └── sample_ohlcv.parquet
│
├── docs/                        # Documentation
│   ├── index.md                 # Page d'accueil
│   ├── api/                     # Documentation API
│   ├── tutorials/               # Tutoriels
│   └── architecture/            # Diagrammes
│
├── scripts/                     # Scripts utilitaires
│   ├── download_data.py         # Téléchargement données
│   ├── compute_indicators.py    # Calcul batch indicateurs
│   └── benchmark.py             # Performance tests
│
├── configs/                     # Fichiers configuration
│   ├── default.toml             # Config par défaut
│   ├── development.toml         # Dev environment
│   ├── production.toml          # Production
│   └── paths.toml               # Chemins TradXPro
│
├── data/                        # Données (gitignored)
│   ├── raw/                     # Données brutes
│   ├── processed/               # Données traitées
│   ├── indicators/              # Cache indicateurs
│   └── backtest_results/        # Résultats backtests
│
├── notebooks/                   # Jupyter notebooks
│   ├── exploratory/             # Analyse exploratoire
│   └── examples/                # Exemples d'utilisation
│
├── docker/                      # Conteneurisation
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── .venv/                       # Virtual environment (gitignored)
│
├── pyproject.toml               # ✅ Configuration projet moderne
├── setup.py                     # Installation (fallback)
├── requirements.txt             # Dépendances production
├── requirements-dev.txt         # Dépendances développement
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .gitignore                   # Git ignore
├── .env.example                 # Variables d'environnement template
├── README.md                    # Documentation principale
├── CHANGELOG.md                 # Historique changements
├── LICENSE                      # Licence
└── Makefile                     # Commandes utilitaires
```

---

## 🎯 Bonnes Pratiques Appliquées

### 1. **Configuration Centralisée (TOML + Pydantic)**

#### ❌ Mauvaise Pratique (Ancien Code)
```python
# Variables globales éparpillées
JSON_ROOT = r"D:\TradXPro\crypto_data_json"
PARQUET_ROOT = r"D:\TradXPro\crypto_data_parquet"
HISTORY_DAYS = 365
BINANCE_LIMIT = 1000
```

#### ✅ Bonne Pratique
```python
# configs/default.toml
[paths]
json_root = "data/raw/json"
parquet_root = "data/processed/parquet"
indicators_db = "data/indicators"

[data]
history_days = 365
binance_limit = 1000
supported_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]

[indicators]
default_rsi_period = 14
default_bb_period = 20
default_bb_std = 2.0

# src/threadx/config/settings.py
from pathlib import Path
from typing import List
from pydantic import BaseSettings, Field

class PathsConfig(BaseSettings):
    json_root: Path = Field(default=Path("data/raw/json"))
    parquet_root: Path = Field(default=Path("data/processed/parquet"))
    indicators_db: Path = Field(default=Path("data/indicators"))
    
    class Config:
        env_prefix = "THREADX_"  # Variables d'environnement

class DataConfig(BaseSettings):
    history_days: int = Field(default=365, ge=1, le=3650)
    binance_limit: int = Field(default=1000, ge=1, le=1000)
    supported_timeframes: List[str] = ["1m", "3m", "5m", "15m", "30m", "1h"]

class Settings(BaseSettings):
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    
    @classmethod
    def load_from_toml(cls, config_file: Path) -> "Settings":
        """Charge depuis fichier TOML."""
        import toml
        config = toml.load(config_file)
        return cls(**config)

# Usage
from threadx.config import Settings

settings = Settings.load_from_toml(Path("configs/default.toml"))
print(settings.data.history_days)  # Type-safe!
```

---

### 2. **Dependency Injection (Pas de Singletons)**

#### ❌ Mauvaise Pratique
```python
# Singleton global (couplage fort)
class IndicatorBank:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Utilisation couplée
bank = IndicatorBank()  # Toujours la même instance
```

#### ✅ Bonne Pratique
```python
# Injection de dépendances
from abc import ABC, abstractmethod
from typing import Protocol

class IndicatorProvider(Protocol):
    """Interface pour fournisseurs d'indicateurs."""
    def calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        ...

class NumPyIndicatorProvider:
    """Implémentation NumPy optimisée."""
    def calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        return rsi_np(close, period)

class BacktestEngine:
    """Moteur backtest avec injection de dépendances."""
    
    def __init__(
        self,
        indicator_provider: IndicatorProvider,
        settings: Settings
    ):
        self.indicators = indicator_provider
        self.settings = settings
    
    def run(self, df: pd.DataFrame) -> BacktestResult:
        # Utilise l'interface, pas l'implémentation
        rsi = self.indicators.calculate_rsi(
            df['close'].values,
            self.settings.indicators.default_rsi_period
        )

# Usage (testable et flexible)
provider = NumPyIndicatorProvider()
engine = BacktestEngine(
    indicator_provider=provider,
    settings=settings
)
```

---

### 3. **Type Hints & Validation (Pydantic)**

#### ❌ Mauvaise Pratique
```python
def calculate_indicators(df, indicators):
    """Calcule indicateurs (pas de type hints)."""
    results = {}
    for ind in indicators:
        if ind == "rsi":
            results[ind] = calc_rsi(df)
    return results
```

#### ✅ Bonne Pratique
```python
from typing import List, Dict, Optional
from pydantic import BaseModel, validator
import pandas as pd

class OHLCVFrame(BaseModel):
    """Modèle validé pour DataFrame OHLCV."""
    data: pd.DataFrame
    symbol: str
    timeframe: str
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('data')
    def validate_ohlcv_columns(cls, df):
        required = {'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        return df
    
    @validator('timeframe')
    def validate_timeframe(cls, tf):
        valid = {'1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'}
        if tf not in valid:
            raise ValueError(f"Timeframe invalide: {tf}")
        return tf

class IndicatorRequest(BaseModel):
    """Requête de calcul d'indicateurs."""
    ohlcv: OHLCVFrame
    indicators: List[str]
    params: Optional[Dict[str, int]] = None

class IndicatorResponse(BaseModel):
    """Réponse avec indicateurs calculés."""
    data: pd.DataFrame
    computed: List[str]
    errors: List[str] = []
    
    class Config:
        arbitrary_types_allowed = True

def calculate_indicators(
    request: IndicatorRequest
) -> IndicatorResponse:
    """
    Calcule indicateurs avec validation Pydantic.
    
    Args:
        request: Requête validée
    
    Returns:
        Réponse avec données enrichies
    
    Raises:
        ValidationError: Si données invalides
    """
    df = request.ohlcv.data.copy()
    computed = []
    errors = []
    
    for indicator in request.indicators:
        try:
            if indicator == "rsi":
                period = request.params.get("rsi_period", 14) if request.params else 14
                df["rsi"] = rsi_np(df["close"].values, period)
                computed.append("rsi")
        except Exception as e:
            errors.append(f"Erreur {indicator}: {str(e)}")
    
    return IndicatorResponse(
        data=df,
        computed=computed,
        errors=errors
    )

# Usage (type-safe et validé)
ohlcv = OHLCVFrame(data=df, symbol="BTCUSDT", timeframe="1h")
request = IndicatorRequest(
    ohlcv=ohlcv,
    indicators=["rsi", "macd"],
    params={"rsi_period": 21}
)
response = calculate_indicators(request)  # ✅ Validé automatiquement
```

---

### 4. **Logging Structuré (pas de print)**

#### ❌ Mauvaise Pratique
```python
def download_data(symbol):
    print(f"Téléchargement {symbol}...")
    try:
        data = fetch_klines(symbol)
        print(f"Succès: {len(data)} bougies")
    except Exception as e:
        print(f"ERREUR: {e}")
```

#### ✅ Bonne Pratique
```python
# src/threadx/utils/logging_utils.py
import logging
import logging.config
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_logs: bool = False
) -> logging.Logger:
    """
    Configure le système de logging ThreadX.
    
    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        log_file: Fichier de log optionnel
        json_logs: Format JSON pour parsing facile
    
    Returns:
        Logger configuré
    """
    import logging.handlers
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if json_logs else "standard",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"]
        }
    }
    
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json" if json_logs else "standard"
        }
        config["root"]["handlers"].append("file")
    
    logging.config.dictConfig(config)
    return logging.getLogger("threadx")

# Usage dans modules
import logging
from threadx.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def download_data(symbol: str) -> List[Dict]:
    """Télécharge données avec logging structuré."""
    logger.info("download_started", extra={
        "symbol": symbol,
        "source": "binance"
    })
    
    try:
        data = fetch_klines(symbol)
        logger.info("download_success", extra={
            "symbol": symbol,
            "candles_count": len(data),
            "duration_ms": 1234
        })
        return data
    
    except Exception as e:
        logger.error("download_failed", extra={
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, exc_info=True)
        raise
```

---

### 5. **Testing Complet (pytest + fixtures)**

#### Structure Tests
```
tests/
├── conftest.py              # Fixtures globales
├── unit/                    # Tests unitaires isolés
│   ├── test_indicators.py
│   └── test_validators.py
├── integration/             # Tests bout-en-bout
│   └── test_pipeline.py
└── fixtures/                # Données de test
    └── sample_data.parquet
```

#### ✅ Bonnes Pratiques Tests
```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Fixture DataFrame OHLCV réaliste."""
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_price = np.roll(close, 1)
    volume = np.random.randint(1000, 10000, n)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

@pytest.fixture
def mock_settings() -> Settings:
    """Settings de test."""
    return Settings(
        data=DataConfig(history_days=30, binance_limit=500),
        paths=PathsConfig(json_root=Path("/tmp/test"))
    )

# tests/unit/test_indicators.py
import pytest
import numpy as np
from threadx.indicators.numpy import rsi_np, boll_np

class TestRSI:
    """Tests unitaires RSI."""
    
    def test_rsi_basic(self, sample_ohlcv):
        """Test calcul RSI basique."""
        rsi = rsi_np(sample_ohlcv['close'].values, period=14)
        
        assert len(rsi) == len(sample_ohlcv)
        assert np.all((rsi >= 0) & (rsi <= 100))
        assert not np.all(np.isnan(rsi))
    
    def test_rsi_extreme_values(self):
        """Test RSI sur valeurs extrêmes."""
        # Prix constants → RSI = 50
        close_flat = np.ones(100) * 100
        rsi = rsi_np(close_flat, period=14)
        np.testing.assert_allclose(rsi[14:], 50.0, atol=1e-6)
        
        # Prix haussiers → RSI proche 100
        close_up = np.arange(100) * 10
        rsi = rsi_np(close_up, period=14)
        assert rsi[-1] > 80
    
    @pytest.mark.parametrize("period", [7, 14, 21, 50])
    def test_rsi_different_periods(self, sample_ohlcv, period):
        """Test RSI avec différentes périodes."""
        rsi = rsi_np(sample_ohlcv['close'].values, period=period)
        
        assert len(rsi) == len(sample_ohlcv)
        assert np.all((rsi >= 0) & (rsi <= 100))
    
    def test_rsi_empty_input(self):
        """Test RSI sur input vide."""
        rsi = rsi_np(np.array([]), period=14)
        assert len(rsi) == 0

class TestBollingerBands:
    """Tests unitaires Bollinger Bands."""
    
    def test_bollinger_basic(self, sample_ohlcv):
        """Test calcul Bollinger basique."""
        lower, middle, upper, z = boll_np(
            sample_ohlcv['close'].values,
            period=20,
            std=2.0
        )
        
        assert len(lower) == len(sample_ohlcv)
        assert np.all(lower <= middle)
        assert np.all(middle <= upper)
    
    def test_bollinger_zscore_range(self, sample_ohlcv):
        """Test z-score dans plage raisonnable."""
        _, _, _, z = boll_np(sample_ohlcv['close'].values, 20, 2.0)
        
        # 95% des valeurs doivent être entre -2 et +2 (2 std)
        within_range = np.sum(np.abs(z) <= 2)
        percentage = within_range / len(z)
        assert percentage > 0.90  # Au moins 90%

# tests/integration/test_pipeline.py
import pytest
from threadx.data.ingest import IngestionManager
from threadx.indicators.numpy import add_all_indicators

@pytest.mark.integration
def test_full_pipeline(mock_settings, tmp_path):
    """Test pipeline complet end-to-end."""
    # 1. Ingestion
    manager = IngestionManager(mock_settings)
    # ... (mock Binance API)
    
    # 2. Calcul indicateurs
    df = load_ohlcv("BTCUSDT", "1h")
    df_enriched = add_all_indicators(df)
    
    # 3. Backtest
    # ...
    
    assert "rsi" in df_enriched.columns
    assert "macd" in df_enriched.columns
```

---

### 6. **CI/CD avec GitHub Actions**

#### ✅ Workflow Complet
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install ruff mypy black isort
      
      - name: Ruff (linter)
        run: ruff check src/
      
      - name: Black (formatter)
        run: black --check src/
      
      - name: isort (imports)
        run: isort --check-only src/
      
      - name: mypy (type checker)
        run: mypy src/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests with coverage
        run: |
          pytest tests/ \
            --cov=src/threadx \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=80
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build package
        run: |
          pip install build
          python -m build
      
      - name: Check package
        run: |
          pip install twine
          twine check dist/*
```

---

### 7. **Pre-commit Hooks (Qualité Automatique)**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=5000']
      - id: check-json
      - id: check-toml
  
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.11
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict, --ignore-missing-imports]

# Installation
# pip install pre-commit
# pre-commit install
```

---

### 8. **pyproject.toml Moderne (PEP 621)**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "threadx"
version = "2.0.0"
description = "Framework de trading crypto haute performance"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "ThreadX Team", email = "team@threadx.dev"}
]
keywords = ["trading", "crypto", "backtest", "indicators", "binance"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Office/Business :: Financial :: Investment",
]

dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "pydantic>=2.0.0",
    "requests>=2.31.0",
    "toml>=0.10.2",
    "tqdm>=4.66.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.12.0",
    "ruff>=0.1.0",
    "black>=23.12.0",
    "isort>=5.13.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]
ui = [
    "streamlit>=1.29.0",
    "plotly>=5.18.0",
]
docs = [
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=2.0.0",
    "myst-parser>=2.0.0",
]

[project.scripts]
threadx = "threadx.cli:main"
threadx-download = "threadx.scripts.download_data:main"
threadx-backtest = "threadx.backtest.cli:main"

[project.urls]
Homepage = "https://github.com/threadx/threadx"
Documentation = "https://threadx.readthedocs.io"
Repository = "https://github.com/threadx/threadx"
Changelog = "https://github.com/threadx/threadx/blob/main/CHANGELOG.md"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
threadx = ["py.typed", "configs/*.toml"]

# Configuration Ruff (linter moderne)
[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]

ignore = [
    "E501",  # line too long (géré par black)
    "B008",  # function call in argument defaults
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # unused imports OK
"tests/*" = ["S101"]      # assert OK in tests

# Configuration Black (formatter)
[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312']
include = '\.pyi?$'
extend-exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''

# Configuration isort (tri imports)
[tool.isort]
profile = "black"
line_length = 100
skip_gitignore = true
known_first_party = ["threadx"]

# Configuration mypy (type checker)
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# Configuration pytest
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src/threadx",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html",
    "--cov-report=xml",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests (deselect with '-m \"not slow\"')",
]

# Configuration coverage
[tool.coverage.run]
source = ["src/threadx"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/site-packages/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

---

### 9. **Documentation (Sphinx + ReadTheDocs)**

```python
# docs/conf.py
project = 'ThreadX'
copyright = '2025, ThreadX Team'
author = 'ThreadX Team'
release = '2.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',  # Markdown support
]

# Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Auto-documentation
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
}

html_theme = 'sphinx_rtd_theme'
```

---

### 10. **Makefile (Commandes Utilitaires)**

```makefile
# Makefile
.PHONY: help install test lint format clean docs build

help:  ## Affiche l'aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Installe le projet en mode développement
	pip install -e ".[dev]"
	pre-commit install

test:  ## Lance les tests avec coverage
	pytest tests/ --cov=src/threadx --cov-report=html --cov-report=term

test-fast:  ## Tests rapides (skip slow)
	pytest tests/ -m "not slow" -v

lint:  ## Vérification qualité code
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/
	mypy src/

format:  ## Formate le code automatiquement
	black src/ tests/
	isort src/ tests/
	ruff check src/ tests/ --fix

clean:  ## Nettoie les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml

docs:  ## Génère la documentation
	cd docs && make html

build:  ## Build le package
	python -m build

publish-test:  ## Publie sur TestPyPI
	twine upload --repository testpypi dist/*

publish:  ## Publie sur PyPI
	twine upload dist/*

docker-build:  ## Build image Docker
	docker build -t threadx:latest -f docker/Dockerfile .

docker-run:  ## Lance container Docker
	docker-compose -f docker/docker-compose.yml up

benchmark:  ## Lance les benchmarks performance
	python scripts/benchmark.py

download-data:  ## Télécharge données crypto
	python scripts/download_data.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h

.DEFAULT_GOAL := help
```

---

## 🚀 Plan de Migration vers Bonnes Pratiques

### Phase 1: Fondations (Semaine 1-2)

- [ ] Créer `pyproject.toml` moderne
- [ ] Configurer pre-commit hooks
- [ ] Setup CI/CD GitHub Actions
- [ ] Créer structure tests/ avec fixtures
- [ ] Ajouter type hints progressivement

### Phase 2: Architecture (Semaine 3-4)

- [ ] Créer `Settings` avec Pydantic
- [ ] Refactorer chemins (configs/paths.toml)
- [ ] Implémenter dependency injection
- [ ] Migrer vers logging structuré
- [ ] Séparer concerns (data/indicators/backtest)

### Phase 3: Qualité (Semaine 5-6)

- [ ] Atteindre 80%+ coverage tests
- [ ] Documentation Sphinx complète
- [ ] Performance profiling
- [ ] Docker images
- [ ] Release automatisée

---

## 📊 Checklist Qualité

### Code Quality
- [ ] Type hints sur toutes fonctions publiques
- [ ] Docstrings Google-style
- [ ] Pas de print() (logging uniquement)
- [ ] Pas de variables globales
- [ ] Séparation config/code
- [ ] Dependency injection

### Tests
- [ ] Coverage > 80%
- [ ] Tests unitaires isolés
- [ ] Tests intégration end-to-end
- [ ] Fixtures réutilisables
- [ ] Mocks pour APIs externes

### Documentation
- [ ] README.md complet
- [ ] CHANGELOG.md à jour
- [ ] API docs générée (Sphinx)
- [ ] Exemples d'utilisation
- [ ] Architecture diagrammes

### DevOps
- [ ] CI/CD fonctionnel
- [ ] Pre-commit hooks
- [ ] Release automatique
- [ ] Docker images
- [ ] Monitoring performance

---

**Créé le**: 11 octobre 2025  
**Auteur**: GitHub Copilot (ThreadX Core Team)  
**Statut**: Guide de Référence
