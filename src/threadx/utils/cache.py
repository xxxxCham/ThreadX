"""Cache simple combinant TTL et LRU pour ThreadX."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Generic, Hashable, Optional, Tuple, TypeVar

from threadx.config import load_settings
from threadx.utils.log import get_logger

logger = get_logger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[K, V] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if key not in self._items:
            self.stats.misses += 1
            return default
        self._items.move_to_end(key)
        self.stats.hits += 1
        return self._items[key]

    def set(self, key: K, value: V) -> None:
        if key in self._items:
            self._items.move_to_end(key)
            self._items[key] = value
            return
        self._items[key] = value
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
            self.stats.evictions += 1

    def clear(self) -> None:
        self._items.clear()


class TTLCache(Generic[K, V]):
    def __init__(self, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl = float(ttl_seconds)
        self._items: Dict[K, Tuple[V, float]] = {}
        self.stats = CacheStats()

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        item = self._items.get(key)
        if item is None:
            self.stats.misses += 1
            return default
        value, expiry = item
        if expiry < time.time():
            self.stats.expirations += 1
            self._items.pop(key, None)
            return default
        self.stats.hits += 1
        return value

    def set(self, key: K, value: V) -> None:
        self._items[key] = (value, time.time() + self.ttl)

    def clear(self) -> None:
        self._items.clear()


class TimedLRUCache(Generic[K, V]):
    """Cache combinant politique LRU et expiration TTL."""

    def __init__(self, capacity: int, ttl_seconds: float) -> None:
        self.lru = LRUCache[K, Tuple[V, float]](capacity)
        self.ttl = float(ttl_seconds)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        item = self.lru.get(key)
        if item is None:
            return default
        value, expiry = item
        if expiry < time.time():
            self.lru._items.pop(key, None)
            return default
        return value

    def set(self, key: K, value: V) -> None:
        expiry = time.time() + self.ttl
        self.lru.set(key, (value, expiry))

    def clear(self) -> None:
        self.lru.clear()


def _default_cache_config() -> Tuple[int, float]:
    settings = load_settings()
    return settings.MAX_WORKERS * 16, float(settings.CACHE_TTL_SECONDS)


def cached(func: Callable | None = None, *, ttl: Optional[float] = None, capacity: Optional[int] = None) -> Callable:
    if func is None:
        return lambda f: cached(f, ttl=ttl, capacity=capacity)

    default_capacity, default_ttl = _default_cache_config()
    cache = TimedLRUCache(capacity or default_capacity, ttl or default_ttl)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, frozenset(kwargs.items()))
        cached_value = cache.get(key)
        if cached_value is not None:
            logger.debug("Cache hit pour %s", func.__name__)
            return cached_value

        result = func(*args, **kwargs)
        cache.set(key, result)
        return result

    return wrapper


__all__ = ["LRUCache", "TTLCache", "TimedLRUCache", "cached", "CacheStats"]
