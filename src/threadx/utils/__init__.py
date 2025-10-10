"""Utilitaires centraux ThreadX."""

from __future__ import annotations

from threadx.utils.log import get_logger  # noqa: F401

try:  # Timing (optionnel)
    from .timing import Timer, measure_throughput, track_memory  # type: ignore
except Exception:  # pragma: no cover - dépendances optionnelles
    Timer = None  # type: ignore
    measure_throughput = lambda *args, **kwargs: (lambda func: func)  # type: ignore
    track_memory = lambda *args, **kwargs: (lambda func: func)  # type: ignore

try:  # Cache
    from .cache import CacheStats, LRUCache, TTLCache, TimedLRUCache, cached  # type: ignore
except Exception:  # pragma: no cover - dépendances optionnelles
    CacheStats = None  # type: ignore
    LRUCache = None  # type: ignore
    TTLCache = None  # type: ignore
    TimedLRUCache = None  # type: ignore
    cached = lambda func=None, **_: func  # type: ignore

try:  # Couche xp (optionnelle)
    from . import xp  # type: ignore
except Exception:  # pragma: no cover
    xp = None  # type: ignore

__all__ = [
    "get_logger",
    "Timer",
    "measure_throughput",
    "track_memory",
    "CacheStats",
    "LRUCache",
    "TTLCache",
    "TimedLRUCache",
    "cached",
    "xp",
]
