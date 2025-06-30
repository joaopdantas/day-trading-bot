"""
Core configuration settings for the API
"""
from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MakesALot Trading API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # CORS
    CHROME_EXTENSION_URL: str = os.getenv(
        "CHROME_EXTENSION_URL",
        "chrome-extension://your-extension-id"
    )

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./makesalot.db"
    )

    # External Services
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")

    # Trading Settings
    DEFAULT_TIMEFRAME: str = "1d"
    MAX_HISTORICAL_DAYS: int = 365
    SUPPORTED_SYMBOLS: List[str] = ["MSFT", "AAPL", "GOOGL", "AMZN"]

    class Config:
        case_sensitive = True


settings = Settings()
