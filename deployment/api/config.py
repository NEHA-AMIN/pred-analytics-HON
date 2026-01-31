"""
API Configuration Settings

Centralized configuration for the FastAPI application.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    api_title: str = "E-Commerce Purchase Prediction API"
    api_version: str = "1.0.0"
    api_description: str = "Production ML API for predicting user purchase intent"
    
    # Server Settings
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    reload: bool = Field(False, env="API_RELOAD")
    workers: int = Field(4, env="API_WORKERS")
    
    # Model Settings
    model_dir: Path = Field(Path("models"), env="MODEL_DIR")
    default_model: Optional[str] = Field(None, env="DEFAULT_MODEL")
    
    # Prediction Settings
    max_batch_size: int = Field(1000, env="MAX_BATCH_SIZE")
    default_threshold: float = Field(0.5, env="DEFAULT_THRESHOLD")
    
    # CORS Settings
    cors_origins: list = Field(["*"], env="CORS_ORIGINS")
    cors_credentials: bool = Field(True, env="CORS_CREDENTIALS")
    cors_methods: list = Field(["*"], env="CORS_METHODS")
    cors_headers: list = Field(["*"], env="CORS_HEADERS")
    
    # Logging Settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(Path("logs/api.log"), env="LOG_FILE")
    
    # Performance Settings
    enable_caching: bool = Field(False, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(300, env="CACHE_TTL")
    
    # Monitoring Settings
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_port: int = Field(9090, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
