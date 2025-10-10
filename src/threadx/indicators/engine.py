"""Utilities to enrich market data with technical indicators."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = ["enrich_indicators"]

logger = logging.getLogger(__name__)

_BACKEND_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "gpu": "gpu",
}


class IndicatorSpecificationError(ValueError):
    """Raised when an indicator specification is invalid."""


def _normalise_backend(global_backend: str | None, spec_backend: str | None) -> str:
    """Return the effective backend to use for a spec."""

    backend = (spec_backend or global_backend or "auto").lower()
    if backend not in _BACKEND_ALIASES:
        raise IndicatorSpecificationError(f"Backend inconnu: {backend}")
    return _BACKEND_ALIASES[backend]


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], indicator: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise IndicatorSpecificationError(
            f"Colonnes manquantes pour {indicator}: {', '.join(missing)}"
        )


def _prepare_output_names(spec: Mapping[str, Any], default: str) -> list[str]:
    outputs = spec.get("outputs")
    if outputs is None:
        return [default]
    if isinstance(outputs, str):
        return [outputs]
    if isinstance(outputs, Sequence):
        outputs_list = [str(name) for name in outputs]
        if not outputs_list:
            raise IndicatorSpecificationError("La liste outputs ne peut être vide")
        return outputs_list
    raise IndicatorSpecificationError("outputs doit être une chaîne ou une séquence")


def _apply_xatr(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    spec: Mapping[str, Any],
    backend: str,
) -> None:
    from .xatr import GPU_AVAILABLE, compute_atr

    _ensure_columns(source_df, ("high", "low", "close"), "xatr")
    params = dict(spec.get("params") or {})

    period = int(params.get("period", 14))
    method = str(params.get("method", "ema"))

    effective_backend = _normalise_backend(backend, params.get("backend"))
    use_gpu = effective_backend == "gpu" or (
        effective_backend == "auto" and bool(params.get("use_gpu", True))
    )
    if use_gpu and not GPU_AVAILABLE:
        logger.debug("GPU indisponible pour xatr, utilisation CPU")
        use_gpu = False

    atr_values = compute_atr(
        source_df["high"].to_numpy(),
        source_df["low"].to_numpy(),
        source_df["close"].to_numpy(),
        period=period,
        method=method,
        use_gpu=use_gpu,
    )

    output_names = _prepare_output_names(spec, "xatr")
    main_output = output_names[0]
    target_df[main_output] = pd.Series(atr_values, index=source_df.index)


_INDICATOR_DISPATCH: Mapping[str, Any] = {
    "xatr": _apply_xatr,
}


def enrich_indicators(
    df: pd.DataFrame,
    specs: Iterable[Mapping[str, Any]] | None,
    backend: str = "auto",
) -> pd.DataFrame:
    """Return a copy of *df* enriched with the requested indicators."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas.DataFrame")

    specs = list(specs or [])
    if not specs:
        return df.copy()

    result_df = df.copy()
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise IndicatorSpecificationError("Chaque spec doit être un mapping")

        name = spec.get("name") or spec.get("indicator")
        if not name:
            raise IndicatorSpecificationError("Spec sans nom d'indicateur")

        handler = _INDICATOR_DISPATCH.get(str(name).lower())
        if handler is None:
            logger.warning("Indicateur inconnu ignoré: %s", name)
            continue

        handler(result_df, df, spec, backend)

    return result_df
