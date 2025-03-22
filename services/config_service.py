import logging
from pathlib import Path
from typing import Any, Dict, Optional

import tomli

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ConfigService:
    """Service for loading and accessing application configuration."""
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ConfigService, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from TOML file."""
        config_path = config_path or 'config.toml'
        try:
            with open(config_path, 'rb') as f:
                self._config = tomli.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Provide default configuration if file loading fails
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {
            "app": {
                "name": "Powerlifting Data Visualization",
                "version": "1.0.0",
                "debug": True
            },
            "data": {
                "data_dir": "data",
                "parquet_file": "openpowerlifting.parquet",
                "csv_backup": "openpowerlifting_backup.csv",
                "last_updated_file": "last_updated.txt",
                "csv_url": "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip",
                "update_interval_days": 1
            },
            "cache": {
                "memory_size": 100,
                "disk_cache_dir": "./cache",
                "max_age_hours": 72,
                "enable_compression": True
            },
            "visualization": {
                "default_sample_size": 10000,
                "wilks_coefficients_male": [-216.0475144, 16.2606339, -0.002388645, -0.00113732, 7.01863e-06, -1.291e-08],
                "wilks_coefficients_female": [594.31747775582, -27.23842536447, 0.82112226871, -0.00930733913, 4.731582e-05, -9.054e-08],
                "bodyweight_tolerance": 5
            },
            "display": {
                "weight_conversion_factor": 2.20462
            }
        }
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key within section (optional)
            
        Returns:
            The configuration value, section dict, or None if not found
        """
        if section not in self._config:
            logger.warning(f"Configuration section '{section}' not found")
            return None
        
        if key is None:
            return self._config[section]
        
        if key not in self._config[section]:
            logger.warning(f"Configuration key '{key}' not found in section '{section}'")
            return None
        
        return self._config[section][key]
    
    def get_path(self, section: str, key: str) -> Path:
        """
        Get a configuration value as a Path object.
        
        Args:
            section: Configuration section
            key: Configuration key for a path value
            
        Returns:
            Path object
        """
        path_str = self.get(section, key)
        if path_str is None:
            logger.warning(f"Path for '{section}.{key}' not found, using default")
            return Path(".")
        
        return Path(path_str)
    
    def get_data_path(self, key: Optional[str] = None) -> Path:
        """
        Get a path within the data directory.
        
        Args:
            key: File name within the data directory (optional)
            
        Returns:
            Full path to the data directory or file
        """
        data_dir = self.get_path("data", "data_dir")
        if key is None:
            return data_dir
        
        file_name = self.get("data", key)
        if file_name is None:
            logger.warning(f"Data file '{key}' not found in configuration")
            return data_dir
        
        return data_dir / file_name

# Create a global instance for easy import
config = ConfigService()
