"""
Technical analysis schema definitions
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class TechnicalAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g. MSFT)")
    timeframe: Optional[str] = Field(
        "1d",
        description="Data timeframe (1m, 5m, 1h, 1d, etc)"
    )
    days: Optional[int] = Field(
        100,
        description="Number of historical days to analyze"
    )
    indicators: Optional[List[str]] = Field(
        ["RSI", "MACD", "BB"],
        description="Technical indicators to include"
    )


class TechnicalAnalysisResponse(BaseModel):
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.now)
    trend: str
    signals: Dict[str, str]
    indicators: Dict[str, float]
    support_levels: List[float]
    resistance_levels: List[float]
    risk_reward: Optional[float]


class SupportResistanceResponse(BaseModel):
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.now)
    support_levels: List[float]
    resistance_levels: List[float]
    current_price: float
    nearest_support: float
    nearest_resistance: float
