"""
Technical analysis endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta

from app.core.config import settings
from app.schemas.technical import (
    TechnicalAnalysisRequest,
    TechnicalAnalysisResponse,
    SupportResistanceResponse
)
from app.services.technical import TechnicalAnalysisService
from app.services.data import DataService

router = APIRouter()


@router.post("/analyze", response_model=TechnicalAnalysisResponse)
async def analyze_symbol(request: TechnicalAnalysisRequest):
    """
    Perform technical analysis on a given symbol
    """
    try:
        # Get historical data
        data_service = DataService()
        df = await data_service.get_historical_data(
            symbol=request.symbol,
            timeframe=request.timeframe or settings.DEFAULT_TIMEFRAME,
            days=request.days or 100
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol {request.symbol}"
            )

        # Perform analysis
        ta_service = TechnicalAnalysisService()
        analysis = ta_service.analyze(
            df,
            indicators=request.indicators or ["RSI", "MACD", "BB"]
        )

        return analysis

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing {request.symbol}: {str(e)}"
        )


@router.get(
    "/support-resistance/{symbol}",
    response_model=SupportResistanceResponse
)
async def get_support_resistance(
    symbol: str,
    timeframe: Optional[str] = Query(None),
    days: Optional[int] = Query(100)
):
    """
    Get support and resistance levels for a symbol
    """
    try:
        # Get historical data
        data_service = DataService()
        df = await data_service.get_historical_data(
            symbol=symbol,
            timeframe=timeframe or settings.DEFAULT_TIMEFRAME,
            days=days
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol {symbol}"
            )

        # Calculate levels
        ta_service = TechnicalAnalysisService()
        levels = ta_service.calculate_support_resistance(df)

        return levels

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating levels for {symbol}: {str(e)}"
        )
