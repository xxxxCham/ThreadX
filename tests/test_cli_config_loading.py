"""Tests de fumée pour le chargement de configuration des CLIs ThreadX."""

from pathlib import Path
import importlib
import sys
import textwrap
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from threadx.config import load_config_dict


def _write_toml(path: Path, content: str) -> Path:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(textwrap.dedent(content), encoding="utf-8")
    temp_path.replace(path)
    return path


def test_load_config_dict_returns_plain_dict(tmp_path):
    config_file = tmp_path / "optimization_config.toml"
    _write_toml(
        config_file,
        """
        [dataset]
        name = "demo"

        [scenario]
        type = "grid"
        seed = 123

        [params]
        alpha = [1, 2, 3]

        [constraints]
        rules = []
        """,
    )

    loaded = load_config_dict(config_file)

    assert isinstance(loaded, dict)
    assert loaded["scenario"]["type"] == "grid"
    assert loaded["params"]["alpha"] == [1, 2, 3]


def test_run_backtests_load_config_uses_shared_loader(tmp_path, monkeypatch):
    config_file = tmp_path / "benchmark_config.toml"
    _write_toml(
        config_file,
        """
        sizes = [16]

        [strategies.demo.params]
        window = 14
        threshold = 0.2
        """,
    )

    stub_backtest = types.ModuleType("threadx.backtest")
    stub_backtest.create_engine = lambda: None
    monkeypatch.setitem(sys.modules, "threadx.backtest", stub_backtest)

    stub_indicators = types.ModuleType("threadx.indicators")
    stub_indicators.get_gpu_accelerated_bank = lambda: None
    monkeypatch.setitem(sys.modules, "threadx.indicators", stub_indicators)

    run_backtests = importlib.import_module("threadx.benchmarks.run_backtests")

    loaded = run_backtests.load_config(str(config_file))

    assert isinstance(loaded, dict)
    assert loaded["sizes"] == [16]
    assert loaded["strategies"]["demo"]["params"]["window"] == 14
