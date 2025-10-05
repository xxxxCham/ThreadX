import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from threadx.config import ConfigurationError, load_config_dict
from threadx.optimization import run as opt_run
from threadx.benchmarks import run_backtests as bench_run


def test_load_config_alias_emits_warning(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[scenario]\ntype='grid'\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        data = opt_run.load_config(str(config_path))

    assert data["scenario"]["type"] == "grid"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_validate_cli_config_rejects_invalid_params():
    config = {"params": {"alpha": "invalid"}}
    with pytest.raises(ConfigurationError):
        opt_run.validate_cli_config(config, "config.toml")


def test_validate_cli_config_requires_rules_list():
    config = {"params": {}, "constraints": {"rules": "oops"}}
    with pytest.raises(ConfigurationError):
        opt_run.validate_cli_config(config, "config.toml")


def test_build_scenario_spec_invalid_type():
    config = {"scenario": {"type": "unknown"}, "params": {}, "constraints": {"rules": []}}
    with pytest.raises(ConfigurationError):
        opt_run.build_scenario_spec(config, "config.toml")


def test_validate_benchmark_config_sizes():
    with pytest.raises(ConfigurationError):
        bench_run.validate_benchmark_config({"sizes": ["1000"]}, "config.toml")


def test_load_config_dict_invalid_toml(tmp_path):
    config_path = tmp_path / "broken.toml"
    config_path.write_text("[invalid", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config_dict(config_path)
