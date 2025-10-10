"""Public API surface for the indicator toolkit."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BollingerBands",
    "BollingerSettings",
    "compute_bollinger_bands",
    "compute_bollinger_batch",
    "ATR",
    "ATRSettings",
    "compute_atr",
    "compute_atr_batch",
    "IndicatorBank",
    "IndicatorSettings",
    "batch_ensure_indicators",
    "ensure_indicator",
    "force_recompute_indicator",
]

_EXPORTS = {
    "BollingerBands": ("threadx.indicators.bollinger", "BollingerBands"),
    "BollingerSettings": ("threadx.indicators.bollinger", "BollingerSettings"),
    "compute_bollinger_bands": ("threadx.indicators.bollinger", "compute_bollinger_bands"),
    "compute_bollinger_batch": ("threadx.indicators.bollinger", "compute_bollinger_batch"),
    "ATR": ("threadx.indicators.xatr", "ATR"),
    "ATRSettings": ("threadx.indicators.xatr", "ATRSettings"),
    "compute_atr": ("threadx.indicators.xatr", "compute_atr"),
    "compute_atr_batch": ("threadx.indicators.xatr", "compute_atr_batch"),
    "IndicatorBank": ("threadx.indicators.bank", "IndicatorBank"),
    "IndicatorSettings": ("threadx.indicators.bank", "IndicatorSettings"),
    "batch_ensure_indicators": ("threadx.indicators.bank", "batch_ensure_indicators"),
    "ensure_indicator": ("threadx.indicators.bank", "ensure_indicator"),
    "force_recompute_indicator": ("threadx.indicators.bank", "force_recompute_indicator"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'threadx.indicators' n'a pas d'attribut {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
