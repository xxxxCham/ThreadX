"""Fonctions d'enrichissement des indicateurs techniques."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import logging
from typing import Any

import pandas as pd

__all__ = ["enrich_indicators", "register_indicator", "IndicatorSpecificationError"]

logger = logging.getLogger(__name__)


class IndicatorSpecificationError(ValueError):
    """Erreur levée lorsqu'une spécification d'indicateur est invalide."""


IndicatorHandler = Callable[[pd.DataFrame, Mapping[str, Any], str], Mapping[str, pd.Series]]

_BACKEND_ALIASES: dict[str, str] = {
    "auto": "auto",
    "cpu": "cpu",
    "gpu": "gpu",
}


def _normalise_backend(global_backend: str | None, spec_backend: str | None) -> str:
    backend = (spec_backend or global_backend or "auto").lower()
    try:
        return _BACKEND_ALIASES[backend]
    except KeyError as exc:
        raise IndicatorSpecificationError(f"Backend inconnu: {backend}") from exc


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], indicator: str) -> None:
    missing = [column for column in columns if column not in df.columns]
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
        names = [str(name) for name in outputs if str(name)]
        if not names:
            raise IndicatorSpecificationError("outputs ne peut pas être vide")
        if len(names) != 1:
            raise IndicatorSpecificationError("xatr ne supporte qu'une seule sortie")
        return names
    raise IndicatorSpecificationError("outputs doit être une chaîne ou une séquence")


def _apply_xatr(df: pd.DataFrame, spec: Mapping[str, Any], backend: str) -> Mapping[str, pd.Series]:
    from .xatr import GPU_AVAILABLE, compute_atr

    _ensure_columns(df, ("high", "low", "close"), "xatr")

    params = dict(spec.get("params") or {})
    period = int(params.get("period", 14))
    method = str(params.get("method", "ema"))

    effective_backend = _normalise_backend(backend, params.get("backend"))
    use_gpu = params.get("use_gpu", True)
    if effective_backend == "cpu":
        use_gpu = False
    elif effective_backend == "gpu":
        use_gpu = True
    elif effective_backend == "auto":
        use_gpu = bool(use_gpu)

    if use_gpu and not GPU_AVAILABLE:
        logger.debug("GPU indisponible pour xatr, utilisation CPU")
        use_gpu = False

    atr_values = compute_atr(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        period=period,
        method=method,
        use_gpu=use_gpu,
    )

    output_name = _prepare_output_names(spec, "xatr")[0]
    return {output_name: pd.Series(atr_values, index=df.index, name=output_name)}


_INDICATOR_REGISTRY: dict[str, IndicatorHandler] = {"xatr": _apply_xatr}


def register_indicator(name: str, handler: IndicatorHandler, *, override: bool = False) -> None:
    """Enregistre dynamiquement un nouvel indicateur."""

    key = name.lower()
    if not override and key in _INDICATOR_REGISTRY:
        raise ValueError(f"L'indicateur {name} est déjà enregistré")
    _INDICATOR_REGISTRY[key] = handler


def _normalise_specs(specs: Iterable[Mapping[str, Any] | str]) -> list[Mapping[str, Any]]:
    normalised: list[Mapping[str, Any]] = []
    for spec in specs:
        if isinstance(spec, str):
            normalised.append({"name": spec})
            continue
        if isinstance(spec, Mapping):
            normalised.append(spec)
            continue
        raise IndicatorSpecificationError("Chaque spec doit être un mapping ou une chaîne")
    return normalised


def enrich_indicators(
    df: pd.DataFrame,
    specs: Iterable[Mapping[str, Any] | str] | None,
    backend: str = "auto",
) -> pd.DataFrame:
    """Retourne une copie de *df* enrichie avec les indicateurs demandés."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas.DataFrame")

    specs = _normalise_specs(specs or [])
    if not specs:
        return df.copy()

    effective_backend = _normalise_backend(backend, None)
    result = df.copy()

    for spec in specs:
        name = spec.get("name") or spec.get("indicator")
        if not name:
            raise IndicatorSpecificationError("Spec sans nom d'indicateur")

        handler = _INDICATOR_REGISTRY.get(str(name).lower())
        if handler is None:
            logger.warning("Indicateur inconnu ignoré: %s", name)
            continue

        outputs = handler(df, spec, effective_backend)
        for column, series in outputs.items():
            if not isinstance(series, pd.Series):
                raise IndicatorSpecificationError(
                    f"L'indicateur {name} doit retourner des pandas.Series"
                )
            result[column] = series

    return result
