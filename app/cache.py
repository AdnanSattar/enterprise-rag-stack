"""
Caching layer for RAG service.
Cache high-frequency queries and reranker outputs.
"""

import hashlib
import json
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class InMemoryCache:
    """
    Simple in-memory cache with TTL.
    Use for development or single-instance deployments.
    """

    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._cache: dict = {}
        self._timestamps: dict = {}

    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired."""
        if key not in self._timestamps:
            return True
        timestamp, ttl = self._timestamps[key]
        return (datetime.now().timestamp() - timestamp) > ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache and not self._is_expired(key):
            logger.debug(f"Cache hit: {key[:50]}...")
            return self._cache[key]

        # Clean up expired entry
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]

        return None

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        self._cache[key] = value
        self._timestamps[key] = (datetime.now().timestamp(), ttl)

    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        valid_count = sum(1 for k in self._cache if not self._is_expired(k))
        return {
            "total_keys": len(self._cache),
            "valid_keys": valid_count,
            "expired_keys": len(self._cache) - valid_count,
        }


class RedisCache:
    """
    Redis-backed cache for distributed deployments.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,
        prefix: str = "rag:",
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._client = None

    @property
    def client(self):
        """Lazy load Redis client."""
        if self._client is None:
            try:
                import redis

                self._client = redis.from_url(self.redis_url)
                self._client.ping()
            except Exception as e:
                logger.warning(
                    f"Redis connection failed: {e}. Using in-memory fallback."
                )
                return None
        return self._client

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if self.client is None:
            return None

        try:
            full_key = self._make_key(key)
            value = self.client.get(full_key)
            if value:
                logger.debug(f"Cache hit: {key[:50]}...")
                return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get error: {e}")

        return None

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in Redis with TTL."""
        if self.client is None:
            return

        try:
            full_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            self.client.set(full_key, json.dumps(value), ex=ttl)
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str) -> None:
        """Delete a key from Redis."""
        if self.client is None:
            return

        try:
            self.client.delete(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    def clear_prefix(self) -> None:
        """Clear all keys with this prefix."""
        if self.client is None:
            return

        try:
            pattern = f"{self.prefix}*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


def make_cache_key(*args, **kwargs) -> str:
    """
    Create a deterministic cache key from arguments.
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(cache: Optional[Any] = None, ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results.

    Usage:
        @cached(cache=my_cache, ttl=3600, key_prefix="query:")
        def expensive_computation(query: str):
            ...
    """

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if cache is None:
                return fn(*args, **kwargs)

            # Build cache key
            cache_key = key_prefix + make_cache_key(*args, **kwargs)

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Compute and cache
            result = fn(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator


def served_answer(
    cache: Any, cache_key: str, compute_fn: Callable, ttl: int = 3600
) -> Any:
    """
    Pattern for serving cached answers.

    Args:
        cache: Cache instance
        cache_key: Key for this query
        compute_fn: Function to compute answer if not cached
        ttl: Cache TTL in seconds

    Returns:
        Cached or computed result
    """
    # Try cache first
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # Compute fresh result
    result = compute_fn()

    # Cache for future
    cache.set(cache_key, result, ttl)

    return result


# Global cache instance
_cache: Optional[Any] = None


def get_cache(use_redis: bool = False, redis_url: str = None) -> Any:
    """Get or create global cache instance."""
    global _cache

    if _cache is None:
        if use_redis:
            from .config import settings

            _cache = RedisCache(
                redis_url=redis_url or settings.REDIS_URL,
                default_ttl=settings.CACHE_TTL,
            )
        else:
            _cache = InMemoryCache()

    return _cache
