"""Application configuration management."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    api_title: str = "Feddit Sentiment Analysis API"
    api_version: str = "1.0.0"
    api_description: str = "RESTful API for sentiment analysis of Feddit comments"
    api_prefix: str = "/api/v1"

    # Feddit API Configuration
    feddit_base_url: str = "http://localhost:8080"
    feddit_timeout: int = 30
    feddit_max_retries: int = 3

    # Application Configuration
    default_comment_limit: int = 25
    max_comment_limit: int = 100
    cache_ttl_seconds: int = 3600  # 1 hour, time to live for sentiment analysis results

    # Development Configuration
    debug: bool = False

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


# Global settings instance
settings = Settings()
