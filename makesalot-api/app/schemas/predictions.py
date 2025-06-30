"""
ML prediction schema definitions
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g. MSFT)")
    timeframe: Optional[str] = Field(
        "1d",
        description="Data timeframe (1m, 5m, 1h, 1d, etc)"
    )
    days: Optional[int] = Field(
        100,
        description="Number of historical days to analyze"
    )
    models: Optional[List[str]] = Field(
        ["ensemble"],
        description="ML models to use for prediction"
    )


class PredictionResponse(BaseModel):
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.now)
    prediction: str
    confidence: float
    price_target: Optional[float]
    timeframe: str
    models_used: List[str]
    technical_factors: Dict[str, float]


class ModelPerformanceResponse(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_size: int
    time_period: str
    last_updated: datetime
