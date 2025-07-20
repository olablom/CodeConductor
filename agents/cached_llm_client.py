#!/usr/bin/env python3
"""
Cached LLM Client for CodeConductor

Implements smart caching policies (LRU, TTL) for improved performance.
"""

import time
import hashlib
import json
import pickle
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
from collections import OrderedDict
import threading
import sqlite3
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CachePolicy:
    """Base class for cache policies"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size

    def should_evict(self, cache: Dict[str, Any]) -> bool:
        """Determine if cache should evict items"""
        return len(cache) > self.max_size

    def select_eviction_candidate(self, cache: Dict[str, Any]) -> str:
        """Select which item to evict"""
        raise NotImplementedError


class LRUCachePolicy(CachePolicy):
    """Least Recently Used cache policy"""

    def __init__(self, max_size: int = 1000):
        super().__init__(max_size)
        self.access_order = OrderedDict()

    def access_item(self, key: str):
        """Mark item as recently accessed"""
        if key in self.access_order:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = time.time()

    def select_eviction_candidate(self, cache: Dict[str, Any]) -> str:
        """Select least recently used item"""
        if not self.access_order:
            return next(iter(cache))
        return next(iter(self.access_order))


class TTLCachePolicy(CachePolicy):
    """Time To Live cache policy"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        super().__init__(max_size)
        self.ttl_seconds = ttl_seconds

    def is_expired(self, timestamp: float) -> bool:
        """Check if item is expired"""
        return time.time() - timestamp > self.ttl_seconds

    def select_eviction_candidate(self, cache: Dict[str, Any]) -> str:
        """Select oldest item"""
        oldest_key = None
        oldest_time = float("inf")

        for key, value in cache.items():
            if isinstance(value, dict) and "timestamp" in value:
                if value["timestamp"] < oldest_time:
                    oldest_time = value["timestamp"]
                    oldest_key = key

        return oldest_key if oldest_key else next(iter(cache))


class HybridCachePolicy(CachePolicy):
    """Combines LRU and TTL policies"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        super().__init__(max_size)
        self.ttl_seconds = ttl_seconds
        self.access_order = OrderedDict()

    def access_item(self, key: str):
        """Mark item as recently accessed"""
        if key in self.access_order:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = time.time()

    def is_expired(self, timestamp: float) -> bool:
        """Check if item is expired"""
        return time.time() - timestamp > self.ttl_seconds

    def select_eviction_candidate(self, cache: Dict[str, Any]) -> str:
        """Select item based on both LRU and TTL"""
        # First, check for expired items
        for key, value in cache.items():
            if isinstance(value, dict) and "timestamp" in value:
                if self.is_expired(value["timestamp"]):
                    return key

        # If no expired items, use LRU
        if self.access_order:
            return next(iter(self.access_order))

        return next(iter(cache))


class CachedLLMClient:
    """LLM client with intelligent caching"""

    def __init__(
        self,
        cache_policy: str = "hybrid",
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        enable_persistence: bool = True,
    ):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0

        # Initialize cache policy
        if cache_policy == "lru":
            self.policy = LRUCachePolicy(max_size)
        elif cache_policy == "ttl":
            self.policy = TTLCachePolicy(max_size, ttl_seconds)
        elif cache_policy == "hybrid":
            self.policy = HybridCachePolicy(max_size, ttl_seconds)
        else:
            raise ValueError(f"Unknown cache policy: {cache_policy}")

        self.enable_persistence = enable_persistence
        self.cache_file = Path("data/llm_cache.db")
        self.lock = threading.Lock()

        # Load persistent cache
        if enable_persistence:
            self._load_cache()

    def _generate_cache_key(self, prompt: str, model: str = "default", temperature: float = 0.7) -> str:
        """Generate a unique cache key for the request"""

        # Create a hash of the request parameters
        request_data = {"prompt": prompt, "model": model, "temperature": temperature}

        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()

    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""

        if not isinstance(cache_entry, dict) or "timestamp" not in cache_entry:
            return True

        if hasattr(self.policy, "is_expired"):
            return self.policy.is_expired(cache_entry["timestamp"])

        return False

    def _evict_if_needed(self):
        """Evict items if cache is full"""

        if not self.policy.should_evict(self.cache):
            return

        # Remove expired items first
        expired_keys = []
        for key, value in self.cache.items():
            if self._is_expired(value):
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]
            if hasattr(self.policy, "access_order") and key in self.policy.access_order:
                del self.policy.access_order[key]

        # If still full, evict based on policy
        while self.policy.should_evict(self.cache):
            evict_key = self.policy.select_eviction_candidate(self.cache)
            if evict_key in self.cache:
                del self.cache[evict_key]
                if hasattr(self.policy, "access_order") and evict_key in self.policy.access_order:
                    del self.policy.access_order[evict_key]

    def get(self, prompt: str, model: str = "default", temperature: float = 0.7) -> Optional[str]:
        """Get response from cache if available"""

        with self.lock:
            self.total_requests += 1
            cache_key = self._generate_cache_key(prompt, model, temperature)

            # Check cache
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]

                # Check if expired
                if self._is_expired(cache_entry):
                    del self.cache[cache_key]
                    self.cache_misses += 1
                    return None

                # Mark as accessed for LRU
                if hasattr(self.policy, "access_item"):
                    self.policy.access_item(cache_key)

                self.cache_hits += 1
                logger.debug(f"Cache HIT for key: {cache_key[:8]}...")
                return cache_entry.get("response")

            self.cache_misses += 1
            logger.debug(f"Cache MISS for key: {cache_key[:8]}...")
            return None

    def set(
        self,
        prompt: str,
        response: str,
        model: str = "default",
        temperature: float = 0.7,
        metadata: Dict[str, Any] = None,
    ):
        """Store response in cache"""

        with self.lock:
            cache_key = self._generate_cache_key(prompt, model, temperature)

            # Create cache entry
            cache_entry = {
                "response": response,
                "timestamp": time.time(),
                "model": model,
                "temperature": temperature,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "metadata": metadata or {},
            }

            # Evict if needed
            self._evict_if_needed()

            # Store in cache
            self.cache[cache_key] = cache_entry

            # Mark as accessed for LRU
            if hasattr(self.policy, "access_item"):
                self.policy.access_item(cache_key)

            logger.debug(f"Cached response for key: {cache_key[:8]}...")

    def _load_cache(self):
        """Load cache from persistent storage"""

        try:
            if not self.cache_file.exists():
                return

            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp REAL,
                    model TEXT,
                    temperature REAL,
                    prompt_length INTEGER,
                    response_length INTEGER,
                    metadata TEXT
                )
            """
            )

            # Load cache entries
            cursor.execute("SELECT * FROM llm_cache")
            rows = cursor.fetchall()

            for row in rows:
                (
                    key,
                    response,
                    timestamp,
                    model,
                    temperature,
                    prompt_length,
                    response_length,
                    metadata,
                ) = row

                # Check if expired
                if hasattr(self.policy, "is_expired") and self.policy.is_expired(timestamp):
                    continue

                cache_entry = {
                    "response": response,
                    "timestamp": timestamp,
                    "model": model,
                    "temperature": temperature,
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                    "metadata": json.loads(metadata) if metadata else {},
                }

                self.cache[key] = cache_entry

                # Update access order for LRU
                if hasattr(self.policy, "access_item"):
                    self.policy.access_item(key)

            conn.close()
            logger.info(f"Loaded {len(rows)} cache entries from persistent storage")

        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    def _save_cache(self):
        """Save cache to persistent storage"""

        try:
            # Create directory if needed
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()

            # Create table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp REAL,
                    model TEXT,
                    temperature REAL,
                    prompt_length INTEGER,
                    response_length INTEGER,
                    metadata TEXT
                )
            """
            )

            # Clear existing data
            cursor.execute("DELETE FROM llm_cache")

            # Insert cache entries
            for key, entry in self.cache.items():
                cursor.execute(
                    """
                    INSERT INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        key,
                        entry["response"],
                        entry["timestamp"],
                        entry["model"],
                        entry["temperature"],
                        entry["prompt_length"],
                        entry["response_length"],
                        json.dumps(entry["metadata"]),
                    ),
                )

            conn.commit()
            conn.close()
            logger.info(f"Saved {len(self.cache)} cache entries to persistent storage")

        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""

        with self.lock:
            hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0

            # Calculate cache size metrics
            total_size = sum(len(entry.get("response", "")) for entry in self.cache.values())
            avg_response_length = total_size / len(self.cache) if self.cache else 0

            # Find most common models
            model_counts = {}
            for entry in self.cache.values():
                model = entry.get("model", "unknown")
                model_counts[model] = model_counts.get(model, 0) + 1

            return {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "cache_size": len(self.cache),
                "max_size": self.policy.max_size,
                "total_size_bytes": total_size,
                "avg_response_length": avg_response_length,
                "model_distribution": model_counts,
                "cache_policy": self.policy.__class__.__name__,
            }

    def clear_cache(self):
        """Clear all cache entries"""

        with self.lock:
            self.cache.clear()
            if hasattr(self.policy, "access_order"):
                self.policy.access_order.clear()

            if self.enable_persistence:
                self._save_cache()

            logger.info("Cache cleared")

    def cleanup_expired(self):
        """Remove expired entries from cache"""

        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]
                if hasattr(self.policy, "access_order") and key in self.policy.access_order:
                    del self.policy.access_order[key]

            if expired_keys:
                logger.info(f"Removed {len(expired_keys)} expired cache entries")

    def shutdown(self):
        """Clean shutdown - save cache if persistence is enabled"""

        if self.enable_persistence:
            self._save_cache()

        logger.info("CachedLLMClient shutdown complete")


def main():
    """Test the cached LLM client"""

    print("🧠 Testing Cached LLM Client...")
    print("=" * 50)

    # Create client with hybrid policy
    client = CachedLLMClient(cache_policy="hybrid", max_size=100, ttl_seconds=3600, enable_persistence=True)

    # Test prompts
    test_prompts = [
        "Create a simple calculator function",
        "Write a REST API endpoint",
        "Implement user authentication",
        "Create a database schema",
        "Write unit tests for a function",
    ]

    # Simulate some requests
    for i, prompt in enumerate(test_prompts):
        print(f"\nRequest {i + 1}: {prompt[:50]}...")

        # Try to get from cache first
        cached_response = client.get(prompt)

        if cached_response:
            print(f"  ✅ Cache HIT: {len(cached_response)} chars")
        else:
            # Simulate LLM response
            response = f"Generated response for: {prompt} (response #{i + 1})"
            client.set(prompt, response)
            print(f"  ❌ Cache MISS: Generated new response")

    # Test cache hit
    print(f"\nTesting cache hit...")
    cached_response = client.get(test_prompts[0])
    if cached_response:
        print(f"  ✅ Cache HIT confirmed: {len(cached_response)} chars")

    # Print statistics
    stats = client.get_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Cache size: {stats['cache_size']}/{stats['max_size']}")
    print(f"  Policy: {stats['cache_policy']}")

    # Cleanup
    client.shutdown()

    print("\n🎉 Cache test completed!")


if __name__ == "__main__":
    main()
