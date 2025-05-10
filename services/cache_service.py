import logging
import pickle
import random
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

from services.config_service import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LRUCache:
    """A size-limited Least Recently Used (LRU) cache."""

    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Any | None:
        """
        Get an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """
        Add or update an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Add/update and move to end
        self.cache[key] = value
        self.cache.move_to_end(key)
        # Remove the oldest if over the size limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

class CacheService:
    """Complete caching system with multi-level storage and metrics."""

    _instance = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the cache with configuration settings."""
        # Load configuration
        memory_size = config.get("cache", "memory_size") or 100
        disk_cache_dir = Path(config.get("cache", "disk_cache_dir") or "./cache")
        max_age_hours = config.get("cache", "max_age_hours") or 72
        enable_compression = config.get("cache", "enable_compression")
        if enable_compression is None:
            enable_compression = True

        # Initialize components
        self.memory_cache = LRUCache(memory_size)
        self.disk_cache_dir = disk_cache_dir
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        self.enable_compression = enable_compression

        # Initialize metrics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'errors': 0,
            'disk_writes': 0,
            'evictions': 0,
            'start_time': time.time()
        }

        logger.info(f"Cache service initialized with memory_size={memory_size}, "
                   f"disk_cache_dir={disk_cache_dir}, "
                   f"max_age_hours={max_age_hours}, "
                   f"compression={enable_compression}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from cache with metrics tracking.

        Args:
            key: Cache key
            default: Default value to return if key not found

        Returns:
            Cached value or default if not found
        """
        try:
            return self._lookup_in_cache_hierarchy(key, default)
        except Exception as e:
            return self._handle_cache_error('Cache error: ', e, default)

    def _lookup_in_cache_hierarchy(self, key: str, default) -> Any:
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            self.stats['hits'] += 1
            self.stats['memory_hits'] += 1
            return result

        # Try disk cache
        disk_path = self.disk_cache_dir / f"{key}.pickle"
        if disk_path.exists():
            # Check if file is too old
            file_age = time.time() - disk_path.stat().st_mtime
            if file_age > self.max_age_seconds:
                logger.debug(f"Cache entry expired: {key[:20]}...")
                try:
                    disk_path.unlink()  # Delete expired file
                    self.stats['evictions'] += 1
                except Exception as e:
                    logger.error(f"Failed to delete expired cache file: {e}")
                self.stats['misses'] += 1
                return default

            try:
                return self._read_disk_cached_item(disk_path, key)
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
                self.stats['errors'] += 1

        # Cache miss
        self.stats['misses'] += 1
        return default

    def _read_disk_cached_item(self, disk_path: Path, key: str):
        # Read from disk
        with open(disk_path, "rb") as f:
            if self.enable_compression:
                try:
                    # Try to read as compressed data
                    data = f.read()
                    result = pickle.loads(zlib.decompress(data))
                except Exception as e:
                    logger.warning(f"Failed to decompress cache file: {e}")
                    # Fallback to regular pickle if not compressed
                    f.seek(0)
                    result = pickle.load(f)
            else:
                result = pickle.load(f)

        # Move to memory cache
        self.memory_cache.put(key, result)
        self.stats['hits'] += 1
        self.stats['disk_hits'] += 1

        # Update file access time to keep track of usage
        disk_path.touch()

        return result

    def put(self, key: str, value: Any) -> bool:
        """
        Store item in both memory and disk cache.

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store in memory
            self.memory_cache.put(key, value)

            # Store on disk
            disk_path = self.disk_cache_dir / f"{key}.pickle"

            try:
                if self.enable_compression:
                    with disk_path.open("wb") as f:
                        compressed_data = zlib.compress(pickle.dumps(value), level=1)
                        f.write(compressed_data)
                else:
                    with disk_path.open("wb") as f:
                        pickle.dump(value, f, protocol=4)

                self.stats['disk_writes'] += 1
            except Exception as e:
                logger.warning(f"Failed to write to disk cache: {e}")
                self.stats['errors'] += 1
                return False

            # Run occasional maintenance
            if random.random() < 0.01:  # 1% chance to clean on each write
                self._clean_old_files()

            return True
        except Exception as e:
            return self._handle_cache_error('Cache storage error: ', e, False)

    def _handle_cache_error[T](self, arg0: str, e: BaseException, arg2: T) -> T:
        logger.error(f"{arg0}{e}")
        self.stats['errors'] += 1
        return arg2

    def _clean_old_files(self) -> None:
        """Clean up old cache files."""
        try:
            now = time.time()
            count = 0

            for cache_file in self.disk_cache_dir.glob("*.pickle"):
                file_age = now - cache_file.stat().st_mtime
                if file_age > self.max_age_seconds:
                    try:
                        cache_file.unlink()
                        count += 1
                        self.stats['evictions'] += 1
                    except Exception as e:
                        logger.error(f"Error cleaning cache file: {e}")
            if count > 0:
                logger.info(f"Cleaned up {count} old cache files")
        except Exception as e:
            logger.error(f"Error cleaning old cache files: {e}")

    def get_stats(self) -> dict[str, Any]:
        """
        Return cache performance statistics.

        Returns:
            Dictionary of cache statistics
        """
        total = self.stats['hits'] + self.stats['misses']
        uptime = time.time() - self.stats['start_time']

        stats = {
            **self.stats,
            'hit_rate': self.stats['hits'] / total if total > 0 else 0,
            'memory_hit_rate': self.stats['memory_hits'] / self.stats['hits']if self.stats['hits'] > 0 else 0,
            'total_requests': total,
            'uptime_hours': uptime / 3600,
            'requests_per_hour': total / (uptime / 3600) if uptime > 0 else 0
        }

        return stats

    def clear(self) -> None:
        """Clear both memory and disk cache."""
        # Clear memory cache
        self.memory_cache = LRUCache(self.memory_cache.max_size)

        # Clear disk cache
        try:
            for cache_file in self.disk_cache_dir.glob("*.pickle"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Error clearing cache file: {e}")
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")

        # Reset stats but keep start time
        start_time = self.stats['start_time']
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'errors': 0,
            'disk_writes': 0,
            'evictions': 0,
            'start_time': start_time
        }

# Create a global instance for easy import
cache_service = CacheService()
