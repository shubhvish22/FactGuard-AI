"""
FactGuard AI — Thread-Safe In-Memory TTL & LRU Cache
Provides sub-millisecond retrieval for repeated text queries, RSS feeds,
and fact-checking lookups to dramatically reduce response time.
"""
import time
import threading
import functools
import hashlib
from typing import Any, Optional, Dict, Tuple


class TTLCache:
    """Thread-safe In-Memory LRU cache with Time-To-Live (TTL) expiration."""

    def __init__(self, maxsize: int = 2048, default_ttl: int = 900):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expire_at)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve item if present and unexpired."""
        with self._lock:
            if key in self._cache:
                val, expire_at = self._cache[key]
                if time.time() < expire_at:
                    self.hits += 1
                    return val
                else:
                    # Expired
                    del self._cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store item with expiration timestamp."""
        duration = ttl if ttl is not None else self.default_ttl
        expire_at = time.time() + duration
        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.maxsize and key not in self._cache:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (value, expire_at)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            active_entries = sum(1 for _, (_, exp) in self._cache.items() if exp > now)
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "active_entries": active_entries,
                "total_stored": len(self._cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_pct": round(hit_rate * 100, 2),
            }


# Global shared cache instances
evidence_cache = TTLCache(maxsize=4096, default_ttl=900)  # 15 min TTL for web evidence
text_cache = TTLCache(maxsize=2048, default_ttl=600)      # 10 min TTL for predictions


def cached(cache_instance: TTLCache, ttl: Optional[int] = None, prefix: str = ""):
    """Decorator to cache function results with thread safety."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Create deterministic cache key
            raw_key = f"{prefix}:{fn.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            cached_val = cache_instance.get(cache_key)
            if cached_val is not None:
                return cached_val
            result = fn(*args, **kwargs)
            if result is not None:
                cache_instance.set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator

