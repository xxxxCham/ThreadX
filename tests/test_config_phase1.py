"""
Tests pour Phase 1: Configuration and Paths
Validation du système de configuration TOML sans variables d'environnement.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

# Ajout du path pour imports ThreadX
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from threadx.config.settings import Settings
from threadx.config.loaders import TOMLConfigLoader, load_settings, print_config, ConfigurationError


class TestTOMLConfigLoader:
    """Tests du chargeur de configuration TOML."""
    
    def test_find_config_file_provided_path(self):
        """Test: Fichier config fourni explicitement."""
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as tmp:
            tmp.write(b'[paths]\ndata_root = "./test"')
            tmp.flush()
            
            loader = TOMLConfigLoader(tmp.name)
            assert loader.config_path == Path(tmp.name)
            
            os.unlink(tmp.name)
    
    def test_find_config_file_not_found(self):
        """Test: Fichier config introuvable."""
        with pytest.raises(ConfigurationError):
            TOMLConfigLoader("/nonexistent/path.toml")
    
    def test_load_config_success(self):
        """Test: Chargement TOML réussi."""
        toml_content = """
[paths]
data_root = "./data"
indicators = "./indicators"

[gpu]
devices = ["5090", "2060"]
load_balance = {5090 = 0.75, 2060 = 0.25}

[performance]
target_tasks_per_min = 2500
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(toml_content)
            tmp.flush()
            
            loader = TOMLConfigLoader(tmp.name)
            
            # Vérification sections
            assert "paths" in loader.config_data
            assert "gpu" in loader.config_data
            assert "performance" in loader.config_data
            
            # Vérification valeurs
            assert loader.get_value("paths", "data_root") == "./data"
            assert loader.get_value("gpu", "devices") == ["5090", "2060"]
            assert loader.get_value("performance", "target_tasks_per_min") == 2500
            
            os.unlink(tmp.name)
    
    def test_load_config_invalid_toml(self):
        """Test: TOML invalide."""
        invalid_toml = """
[paths
data_root = "./data"  # Bracket manquant
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(invalid_toml)
            tmp.flush()
            
            with pytest.raises(ConfigurationError):
                TOMLConfigLoader(tmp.name)
            
            os.unlink(tmp.name)
    
    def test_expand_paths_simple(self):
        """Test: Expansion de chemins sans variables."""
        toml_content = """
[paths]
data_root = "./data"
indicators = "./data/indicators"
cache = "./cache"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(toml_content)
            tmp.flush()
            
            loader = TOMLConfigLoader(tmp.name)
            expanded = loader.expand_paths()
            
            assert expanded["data_root"] == Path("./data")
            assert expanded["indicators"] == Path("./data/indicators")
            assert expanded["cache"] == Path("./cache")
            
            os.unlink(tmp.name)
    
    def test_expand_paths_with_variables(self):
        """Test: Expansion avec substitution de variables."""
        toml_content = """
[paths]
data_root = "./data"
raw_json = "{data_root}/raw/json"
processed = "{data_root}/processed"
indicators = "{data_root}/indicators"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(toml_content)
            tmp.flush()
            
            loader = TOMLConfigLoader(tmp.name)
            expanded = loader.expand_paths()
            
            assert expanded["data_root"] == Path("./data")
            assert expanded["raw_json"] == Path("./data/raw/json")
            assert expanded["processed"] == Path("./data/processed")
            assert expanded["indicators"] == Path("./data/indicators")
            
            os.unlink(tmp.name)


class TestConfigValidation:
    """Tests de validation de configuration."""
    
    def test_validate_config_missing_sections(self):
        """Test: Sections manquantes."""
        minimal_toml = """
[paths]
data_root = "./data"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(minimal_toml)
            tmp.flush()
            
            loader = TOMLConfigLoader(tmp.name)
            errors = loader.validate_config()
            
            # Doit détecter les sections manquantes
            assert len(errors) > 0
            assert any("gpu" in error for error in errors)
            assert any("performance" in error for error in errors)
            
            os.unlink(tmp.name)
    
    def test_validate_config_valid(self):
        """Test: Configuration valide."""
        valid_toml = """
[paths]
data_root = "./data"

[gpu]
devices = ["5090"]
load_balance = {5090 = 1.0}

[performance]
target_tasks_per_min = 1000

[timeframes]
supported = ["1m", "5m", "1h"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(valid_toml)
            tmp.flush()
            
            loader = TOMLConfigLoader(tmp.name)
            errors = loader.validate_config()
            
            # Aucune erreur attendue
            assert len(errors) == 0
            
            os.unlink(tmp.name)


class TestSettingsCreation:
    """Tests de création d'instances Settings."""
    
    def test_create_settings_defaults(self):
        """Test: Création Settings avec valeurs par défaut."""
        complete_toml = """
[paths]
data_root = "./data"
raw_json = "{data_root}/raw/json"
processed = "{data_root}/processed"
indicators = "{data_root}/indicators"
runs = "{data_root}/runs"
cache = "{data_root}/cache"
logs = "./logs"

[gpu]
devices = ["5090", "2060"]
load_balance = {5090 = 0.75, 2060 = 0.25}
memory_threshold = 0.8
auto_fallback = true
enable_cupy = true
enable_numba = true

[performance]
target_tasks_per_min = 2500
vectorization_batch_size = 10000
cache_ttl_sec = 3600
max_workers_default = -1
memory_limit_gb = 8

[indicators]
enable_disk_cache = true
enable_ram_cache = true
max_ram_cache_entries = 20000
batch_compute_threshold = 100
force_recompute_on_schema_change = true

[timeframes]
supported = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
base_timeframe = "1m"

[logging]
level = "INFO"
console_output = true
file_rotation = true
max_file_size_mb = 10
backup_count = 5

[security]
allow_absolute_paths = false
readonly_data_root = true
validate_file_extensions = true
max_file_size_mb = 1000

[backtest]
default_leverage = 3.0
default_risk_per_trade = 0.01
default_fees_bps = 4.5
default_slippage_bps = 0.5
max_concurrent_positions = 1

[ui]
primary_ui = "tkinter"
theme = "dark"
auto_refresh_sec = 5
enable_drag_drop = true
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(complete_toml)
            tmp.flush()
            
            loader = TOMLConfigLoader(tmp.name)
            settings = loader.create_settings()
            
            # Vérification types et valeurs
            assert isinstance(settings, Settings)
            assert settings.DATA_ROOT == Path("./data")
            assert settings.INDICATORS_ROOT == Path("./data/indicators")
            assert settings.GPU_DEVICES == ["5090", "2060"]
            assert settings.GPU_LOAD_BALANCE == {"5090": 0.75, "2060": 0.25}
            assert settings.TARGET_TASKS_PER_MIN == 2500
            assert settings.SUPPORTED_TIMEFRAMES == ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")
            assert settings.LOG_LEVEL == "INFO"
            assert settings.ALLOW_ABSOLUTE_PATHS is False
            assert settings.PRIMARY_UI == "tkinter"
            
            os.unlink(tmp.name)
    
    def test_create_settings_with_overrides(self):
        """Test: Overrides CLI."""
        minimal_toml = """
[paths]
data_root = "./data"

[gpu]
devices = ["5090"]

[performance]
target_tasks_per_min = 1000

[timeframes]
supported = ["1m"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(minimal_toml)
            tmp.flush()
            
            loader = TOMLConfigLoader(tmp.name)
            settings = loader.create_settings(
                data_root="./custom_data",
                indicators="./custom_indicators",
                logs="./custom_logs"
            )
            
            # Vérification overrides appliqués
            assert settings.DATA_ROOT == Path("./custom_data")
            assert settings.INDICATORS_ROOT == Path("./custom_indicators")
            assert settings.LOGS_DIR == Path("./custom_logs")
            
            os.unlink(tmp.name)


class TestUtilityFunctions:
    """Tests des fonctions utilitaires."""
    
    def test_load_settings_function(self):
        """Test: Fonction load_settings()."""
        simple_toml = """
[paths]
data_root = "./test_data"

[gpu]
devices = ["5090"]

[performance]
target_tasks_per_min = 1500

[timeframes]
supported = ["1m", "5m"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(simple_toml)
            tmp.flush()
            
            settings = load_settings(tmp.name)
            
            assert isinstance(settings, Settings)
            assert settings.DATA_ROOT == Path("./test_data")
            assert settings.TARGET_TASKS_PER_MIN == 1500
            
            os.unlink(tmp.name)
    
    def test_print_config_function(self, capsys):
        """Test: Fonction print_config()."""
        toml_content = """
[paths]
data_root = "./data"

[gpu]
devices = ["5090"]

[performance]
target_tasks_per_min = 2000

[timeframes]
supported = ["1m"]
base_timeframe = "1m"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(toml_content)
            tmp.flush()
            
            settings = load_settings(tmp.name)
            print_config(settings)
            
            captured = capsys.readouterr()
            
            # Vérification sortie
            assert "CONFIGURATION THREADX" in captured.out
            assert "./data" in captured.out
            assert "5090" in captured.out
            assert "2000" in captured.out
            
            os.unlink(tmp.name)


class TestNoEnvironmentVariables:
    """Tests critiques: Aucune variable d'environnement utilisée."""
    
    def test_no_env_vars_used(self):
        """Test: Aucune variable d'environnement ne doit être utilisée."""
        # Simulation env vars TradXPro style
        env_vars = {
            "TRADX_DATA_ROOT": "/old/tradx/data",
            "TRADX_IND_DB": "/old/tradx/indicators",
            "INDICATORS_DB_ROOT": "/old/indicators",
            "TRADX_USE_GPU": "1",
            "TRADX_CACHE_SIZE": "50000"
        }
        
        toml_content = """
[paths]
data_root = "./new_data"
indicators = "./new_indicators"

[gpu]
devices = ["5090"]

[performance]
target_tasks_per_min = 3000

[timeframes]
supported = ["1m"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(toml_content)
            tmp.flush()
            
            # Simulation environnement pollué
            with patch.dict(os.environ, env_vars):
                settings = load_settings(tmp.name)
                
                # ThreadX doit ignorer complètement les env vars
                assert settings.DATA_ROOT == Path("./new_data")
                assert settings.INDICATORS_ROOT == Path("./new_indicators")
                assert settings.TARGET_TASKS_PER_MIN == 3000
                
                # Pas les valeurs des env vars
                assert str(settings.DATA_ROOT) != "/old/tradx/data"
                assert str(settings.INDICATORS_ROOT) != "/old/tradx/indicators"
            
            os.unlink(tmp.name)
    
    def test_reproducible_config(self):
        """Test: Configuration reproductible sans dépendance environnement."""
        toml_content = """
[paths]
data_root = "./reproducible"

[gpu]
devices = ["5090"]

[performance]
target_tasks_per_min = 1234

[timeframes]
supported = ["1m"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
            tmp.write(toml_content)
            tmp.flush()
            
            # Chargement 1: env propre
            with patch.dict(os.environ, {}, clear=True):
                settings1 = load_settings(tmp.name)
            
            # Chargement 2: env pollué  
            polluted_env = {
                "TRADX_DATA_ROOT": "/different/path",
                "HOME": "/home/user",
                "PATH": "/usr/bin"
            }
            with patch.dict(os.environ, polluted_env, clear=True):
                settings2 = load_settings(tmp.name)
            
            # Résultats identiques
            assert settings1.DATA_ROOT == settings2.DATA_ROOT
            assert settings1.TARGET_TASKS_PER_MIN == settings2.TARGET_TASKS_PER_MIN
            assert settings1.GPU_DEVICES == settings2.GPU_DEVICES
            
            os.unlink(tmp.name)


if __name__ == "__main__":
    # Exécution des tests
    pytest.main([__file__, "-v"])