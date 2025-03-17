import logging
import pickle
import random
import time
import zlib
from collections import OrderedDict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LRUFigureCache:
    """A size-limited Least Recently Used (LRU) cache."""
    
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        # Add/update and move to end
        self.cache[key] = value
        self.cache.move_to_end(key)
        # Remove the oldest if over the size limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def __contains__(self, key):
        return key in self.cache
    
    def __len__(self):
        return len(self.cache)

class IntegratedCacheSystem:
    """Complete caching system with multi-level storage and metrics."""
    
    def __init__(self, memory_size=100, disk_cache_dir=Path("./cache"), 
                 max_age_hours=72, enable_compression=True):
        self.memory_cache = LRUFigureCache(memory_size)
        self.disk_cache_dir = disk_cache_dir
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        self.enable_compression = enable_compression
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
    
    def get(self, key, default=None):
        """Get item from cache with metrics tracking."""
        try:
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
                        pass
                    self.stats['misses'] += 1
                    return default
                
                try:
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
                except Exception as e:
                    logger.warning(f"Failed to load from disk cache: {e}")
                    self.stats['errors'] += 1
            
            # Cache miss
            self.stats['misses'] += 1
            return default
            
        except Exception as e:
            logger.error(f"Cache error: {e}")
            self.stats['errors'] += 1
            return default
    
    def put(self, key, value):
        """Store item in both memory and disk cache."""
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
            logger.error(f"Cache storage error: {e}")
            self.stats['errors'] += 1
            return False
    
    def _clean_old_files(self):
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
                        pass
            
            if count > 0:
                logger.info(f"Cleaned up {count} old cache files")
        except Exception as e:
            logger.error(f"Error cleaning old cache files: {e}")
    
    def get_stats(self):
        """Return cache performance statistics."""
        total = self.stats['hits'] + self.stats['misses']
        uptime = time.time() - self.stats['start_time']
        
        stats = {
            **self.stats,
            'hit_rate': self.stats['hits'] / total if total > 0 else 0,
            'memory_hit_rate': self.stats['memory_hits'] / self.stats['hits'] if self.stats['hits'] > 0 else 0,
            'total_requests': total,
            'uptime_hours': uptime / 3600,
            'requests_per_hour': total / (uptime / 3600) if uptime > 0 else 0
        }
        
        return stats
    
    def clear(self):
        """Clear both memory and disk cache."""
        # Clear memory cache
        self.memory_cache = LRUFigureCache(self.memory_cache.max_size)
        
        # Clear disk cache
        try:
            for cache_file in self.disk_cache_dir.glob("*.pickle"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Error clearing cache file: {e}")
                    pass
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

# Create a global instance of the cache
cache_system = IntegratedCacheSystem(
    memory_size=100, 
    disk_cache_dir=Path("./cache"),
    max_age_hours=72,  # Cache entries expire after 3 days
    enable_compression=True
)
