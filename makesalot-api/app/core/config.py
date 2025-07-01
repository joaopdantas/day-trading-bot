"""
Core configuration settings for the API
"""
import os

class Settings:
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MakesALot Trading API"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()