# ThreadX Configuration - Compatibility & Migration Guide

## 🔄 API Publique & Compatibilité Ascendante

### ✅ **API Stable**
```python
# GARANTI - API stable pour toutes versions futures
from threadx.config.loaders import load_settings

# Usage recommandé
settings = load_settings(config_path="paths.toml", cli_args=None)
# OU
settings = load_settings()  # Utilise paths.toml par défaut
```

### 🔄 **Aliases de Compatibilité**
```python
# DEPRECATED mais maintenu pour compatibilité
from threadx.config.loaders import load_settings as load_config  # Legacy alias
```

### 📊 **DEFAULT_SETTINGS Usage**
```python
from threadx.config.settings import DEFAULT_SETTINGS

# Fallbacks homogènes dans tout le codebase
cache_ttl = config.get("cache_ttl", DEFAULT_SETTINGS.CACHE_TTL_SEC)
```

---

## ⚠️ **MIGRATION REQUIRED - Anciens Noms**

### 🔍 **Noms Changés** (À vérifier dans le repo)
```python
# ANCIENS (ne marchent plus)
Settings.LOGS_DIR         → Settings.LOGS
Settings.INDICATORS_ROOT  → Settings.INDICATORS  
Settings.DATA_PATH        → Settings.DATA_ROOT

# NOUVEAUX (actuels)  
Settings.LOGS
Settings.INDICATORS
Settings.DATA_ROOT
```

### 🛠️ **Actions Required**
```bash
# 1. Chercher les anciens noms dans le repo
grep -r "LOGS_DIR\|INDICATORS_ROOT\|DATA_PATH" src/

# 2. Remplacer par nouveaux noms
# 3. OU créer passerelle temporaire
```

### 🔄 **Passerelle Temporaire** (si nécessaire)
```python
# src/threadx/config/settings.py
@dataclass(frozen=True)  
class Settings:
    # ... nouveaux champs ...
    
    # Passerelle compatibility (deprecated)
    @property
    def LOGS_DIR(self) -> str:
        warnings.warn("LOGS_DIR deprecated, use LOGS", DeprecationWarning)
        return self.LOGS
        
    @property 
    def INDICATORS_ROOT(self) -> str:
        warnings.warn("INDICATORS_ROOT deprecated, use INDICATORS", DeprecationWarning)
        return self.INDICATORS
```

---

## 🧪 **Plan de Tests Complet**

### **A. Tests Unitaires Config**

#### **A1. Fichiers & TOML**
```python
def test_config_file_not_found():
    """Fichier introuvable → ConfigurationError"""
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config_dict("absent.toml")

def test_invalid_toml_syntax():
    """TOML invalide → ConfigurationError avec détails"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("a = [1, , 2]")  # Syntax error
        f.flush()
        
    with pytest.raises(ConfigurationError, match="Invalid TOML syntax"):
        load_config_dict(f.name)
```

#### **A2. Sections Requises**
```python
def test_missing_required_sections():
    """Sections manquantes → Erreur validation"""
    config_data = {"paths": {"data_root": "./data"}}  # Missing gpu, performance, trading
    
    loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
    loader.config_data = config_data
    
    errors = loader.validate_config()
    assert "Missing required configuration section: gpu" in errors
    assert "Missing required configuration section: performance" in errors 
    assert "Missing required configuration section: trading" in errors
```

#### **A3. Validation Paths**
```python
def test_absolute_paths_forbidden():
    """Chemins absolus interdits quand allow_absolute_paths=false"""
    config_data = {
        "paths": {"logs": "/tmp/x"},
        "security": {"allow_absolute_paths": False},
        "gpu": {}, "performance": {}, "trading": {}
    }
    
    loader = TOMLConfigLoader.__new__(TOMLConfigLoader)  
    loader.config_data = config_data
    
    errors = loader._validate_paths(check_only=True)
    assert any("Absolute path not allowed" in err for err in errors)
```

#### **A4. GPU Load Balance**
```python
def test_gpu_load_balance_invalid_sum():
    """Load balance doit sommer à 1.0"""
    config_data = {
        "paths": {}, "performance": {}, "trading": {},
        "gpu": {"load_balance": {"A": 0.6, "B": 0.5}}  # Sum = 1.1
    }
    
    loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
    loader.config_data = config_data
    
    errors = loader._validate_gpu_config(check_only=True)
    assert any("must sum to 1.0" in err for err in errors)
```

#### **A5. Performance Validation**
```python
def test_performance_negative_values():
    """Valeurs performance négatives interdites"""
    config_data = {
        "paths": {}, "gpu": {}, "trading": {},
        "performance": {"max_workers": -1, "cache_ttl_sec": 0}
    }
    
    loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
    loader.config_data = config_data  
    
    errors = loader._validate_performance_config(check_only=True)
    assert any("must be a positive number" in err for err in errors)
```

#### **A6. CLI Overrides**
```python
def test_cli_overrides():
    """Overrides CLI fonctionnent correctement"""
    # Créer config minimal valide
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(MINIMAL_VALID_CONFIG)  # Défini ailleurs
        f.flush()
        
    settings = load_settings(
        config_path=f.name,
        cli_args=["--data-root", "./custom", "--disable-gpu"]
    )
    
    assert settings.DATA_ROOT == "./custom"
    assert settings.ENABLE_GPU == False
```

#### **A7. Migration Legacy**
```python  
def test_timeframes_migration():
    """Migration timeframes.supported → trading.supported_timeframes"""
    config_data = {
        "paths": {"data_root": "./data"}, 
        "gpu": {}, "performance": {},
        "trading": {},  # Pas de supported_timeframes
        "timeframes": {"supported": ["1h", "4h"]}  # Legacy
    }
    
    loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
    loader.config_data = config_data
    
    # Trigger migration
    loader.create_settings()
    
    # Vérifier migration effectuée
    assert loader.get_section("trading")["supported_timeframes"] == ["1h", "4h"]
```

### **B. Tests Fumée**
```python
def test_print_config_no_crash():
    """print_config ne doit jamais crasher"""
    settings = DEFAULT_SETTINGS
    
    # Capture stdout 
    from io import StringIO
    import sys
    captured = StringIO()
    sys.stdout = captured
    
    print_config(settings)  # Ne doit pas lever
    
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    
    # Vérifier contenu clé
    assert "Target Tasks/Min" in output
    assert "GPU Enabled" in output
    assert "Data Root" in output
```

### **C. Tests XP Additionnels**
```python
def test_memory_pool_info():
    """memory_pool_info retourne structure correcte"""
    info = memory_pool_info()
    
    assert "backend" in info
    assert "device_id" in info
    assert isinstance(info["backend"], str)

def test_to_device_preserves_dtype():
    """to_device garde le dtype original"""
    import numpy as np
    
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = to_device(arr)
    
    # Dtype préservé même après transfert device
    assert result.dtype == np.float32
```

---

## 📝 **Exemple paths.toml Cohérent**

```toml
# ThreadX Configuration - Production Ready
# ========================================

[paths]
data_root   = "./data"
raw_json    = "{data_root}/raw/json"  
processed   = "{data_root}/processed"
indicators  = "{data_root}/indicators"
runs        = "{data_root}/runs"
logs        = "./logs"
cache       = "./cache"
config      = "./config"

[gpu]
devices          = ["5090", "2060"] 
load_balance     = { "5090" = 0.75, "2060" = 0.25 }
memory_threshold = 0.8
auto_fallback    = true
enable_gpu       = true

[performance]
target_tasks_per_min     = 2500
vectorization_batch_size = 10000
cache_ttl_sec           = 3600
max_workers             = 4
memory_limit_mb         = 8192

[trading]
supported_timeframes = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"]
default_timeframe    = "1h"
base_currency        = "USDT"
fee_rate             = 0.001
slippage_rate        = 0.0005

[backtesting]
initial_capital = 10000.0
max_positions   = 10
position_size   = 0.1
stop_loss       = 0.02
take_profit     = 0.04

[logging]
level            = "INFO"
max_file_size_mb = 100
max_files        = 10
log_rotate       = true
format           = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

[security]
read_only_data       = true
validate_paths       = true
allow_absolute_paths = false
max_file_size_mb     = 1000

[monte_carlo]
default_simulations = 10000
max_simulations     = 1000000
default_steps       = 252
seed                = 42
confidence_levels   = [0.95, 0.99]

[cache]
enable      = true
max_size_mb = 2048
ttl_seconds = 3600
compression = true
strategy    = "LRU"
```

---

## ⚙️ **Commandes CI Locale**

```bash
# === LINT & TYPE CHECKING ===
ruff check .                                    # Lint rapide
ruff check . --fix                             # Auto-fix
mypy src/threadx/config --strict               # Type checking strict

# === TESTS CIBLÉS ===
pytest tests/test_utils.py -v                  # Tests utils xp
pytest tests/test_config_loaders.py -v         # Tests config (à créer)
pytest tests/test_config_loaders.py::test_invalid_toml -v
pytest tests/test_config_loaders.py::test_cli_overrides -v

# === SANITY CHECKS ===
python -c "import threadx.config.loaders as L; L.print_config(L.load_settings(cli_args=[]))"
python -c "from threadx.config.settings import DEFAULT_SETTINGS; print(f'GPU: {DEFAULT_SETTINGS.ENABLE_GPU}')"

# === COVERAGE ===
pytest --cov=src/threadx/config tests/         # Coverage config module
pytest --cov-report=html                       # Rapport HTML

# === INTÉGRATION ===
python -m threadx.config.loaders --print-config     # Test CLI
python -m threadx.config.loaders --data-root=./test --disable-gpu --print-config
```

---

## ✅ **Résumé Exécutable**

### **MERGE MAINTENANT** :
1. ✅ `tests/test_utils.py` - xp helpers avec micro-amélioration assert backend
2. ✅ `src/threadx/config/settings.py` - dataclass + DEFAULT_SETTINGS + docstrings
3. ✅ `src/threadx/config/loaders.py` - architecture CLI + validation robuste  
4. ✅ `src/threadx/config/errors.py` - hiérarchie exceptions (déjà fait)

### **AJOUTER POUR COMPATIBILITÉ** :
```python
# src/threadx/config/__init__.py
from .loaders import load_settings
from .loaders import load_settings as load_config  # DEPRECATED alias
from .settings import DEFAULT_SETTINGS, Settings

__all__ = ["load_settings", "load_config", "DEFAULT_SETTINGS", "Settings"]
```

### **TODO APRÈS MERGE** :
1. **Créer** `tests/test_config_loaders.py` avec les 8 cas listés
2. **Vérifier** anciens noms `LOGS_DIR`, `INDICATORS_ROOT` dans le repo
3. **Ajouter** passerelle si nécessaire
4. **Documenter** migration dans CHANGELOG.md

---

## 🎉 **CONCLUSION**

Ton **Option A révisée** est **PARFAITE** ! Tu as identifié la vraie valeur : tes données existantes. L'approche "validate & integrate existing assets first" est la plus professionnelle.

**Next action** : Veux-tu que je crée l'interface Data Manager pour commencer l'étape 1 ? 🚀