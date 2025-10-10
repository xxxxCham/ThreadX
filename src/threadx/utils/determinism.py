"""Utilitaires garantissant la reproductibilité de ThreadX."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable, List, Union

import numpy as np
import pandas as pd

from threadx.utils.log import get_logger

logger = get_logger(__name__)

_GLOBAL_SEED: int = 42
_GLOBAL_RNG: np.random.Generator | None = None


def set_global_seed(seed: int) -> None:
    """Initialise tous les générateurs pseudo-aléatoires avec *seed*."""

    global _GLOBAL_SEED, _GLOBAL_RNG

    _GLOBAL_SEED = int(seed)
    _GLOBAL_RNG = np.random.default_rng(_GLOBAL_SEED)

    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)

    try:
        import cupy as cp  # type: ignore

        cp.random.seed(_GLOBAL_SEED)
    except Exception:  # pragma: no cover - dépendance optionnelle
        logger.debug("CuPy non disponible pour la configuration du seed")

    try:
        import torch  # type: ignore

        torch.manual_seed(_GLOBAL_SEED)
        if torch.cuda.is_available():  # pragma: no cover - dépendance optionnelle
            torch.cuda.manual_seed_all(_GLOBAL_SEED)
    except Exception:
        logger.debug("PyTorch non disponible pour la configuration du seed")

    logger.info("Seed global configuré à %s", _GLOBAL_SEED)


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Retourne un générateur NumPy déterministe."""

    if seed is not None:
        return np.random.default_rng(int(seed))

    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        _GLOBAL_RNG = np.random.default_rng(_GLOBAL_SEED)
    return _GLOBAL_RNG


def enforce_deterministic_merges(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatène des DataFrames avec un ordre stable."""

    frames = list(frames)
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].copy()

    merged = pd.concat(frames, ignore_index=True, copy=False)
    if merged.empty or len(merged.columns) == 0:
        return merged

    return merged.sort_values(list(merged.columns), kind="mergesort").reset_index(drop=True)


def stable_hash(payload: Any) -> str:
    """Retourne un hash SHA-256 stable du contenu fourni."""

    try:
        json_str = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        json_str = json.dumps(str(payload))

    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def create_deterministic_splits(
    data: Union[pd.DataFrame, np.ndarray],
    n_splits: int,
    seed: int | None = None,
) -> List[Union[pd.DataFrame, np.ndarray]]:
    """Découpe *data* en *n_splits* segments reproductibles."""

    if n_splits <= 0:
        raise ValueError("n_splits doit être strictement positif")

    rng = get_rng(seed)

    if isinstance(data, pd.DataFrame):
        indices = rng.permutation(len(data))
        splits = np.array_split(indices, n_splits)
        return [data.iloc[chunk].copy() for chunk in splits if len(chunk) > 0]

    array = np.asarray(data)
    indices = rng.permutation(len(array))
    splits = np.array_split(indices, n_splits)
    return [array[chunk].copy() for chunk in splits if len(chunk) > 0]
