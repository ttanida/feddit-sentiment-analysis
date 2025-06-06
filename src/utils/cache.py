"""Simple in-memory cache implementation."""

import hashlib
import time
from typing import Any


class InMemoryCache:
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 3600):
        """Initialize cache with default TTL in seconds."""
        self._cache: dict[str, tuple[Any, float]] = {}  # hash of text mapping to (SentimentResult, expiry_time)
        self._default_ttl = default_ttl  # default TTL (time to live) in seconds

    def __is_expired(self, expiry_time: float) -> bool:
        """
        Check if a cache entry is expired.

        Args:
            expiry_time: The expiry timestamp

        Returns:
            True if expired, False otherwise
        """
        return time.time() > expiry_time

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        value, expiry_time = self._cache[key]

        # Check if expired
        if self.__is_expired(expiry_time):
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self._default_ttl

        expiry_time = time.time() + ttl
        self._cache[key] = (value, expiry_time)

    @staticmethod
    def create_key(text: str) -> str:
        """Create a cache key from text content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()


# Global cache instance
sentiment_cache = InMemoryCache()
