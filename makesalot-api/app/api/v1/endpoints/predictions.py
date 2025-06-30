"""
Machine Learning prediction endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta

from app.core.config import settings
from app.schemas.predictions import (
    PredictionRequest,
    PredictionResponse,
    ModelPerformanceResponse
)
from app.services.predictions import PredictionService
from app.services.data import DataService

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """
    Get price movement predictions for a symbol
    """
    try:
        # Get historical data for prediction
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

        # Generate prediction
        prediction_service = PredictionService()
        prediction = prediction_service.predict(
            df,
            models=request.models or ["ensemble"]
        )

        return prediction

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting {request.symbol}: {str(e)}"
        )


@router.get(
    "/performance/{model_name}",
    response_model=ModelPerformanceResponse
)
async def get_model_performance(
    model_name: str,
    symbol: Optional[str] = Query(None),
    days: Optional[int] = Query(30)
):
    """
    Get performance metrics for a specific model
    """
    try:
        prediction_service = PredictionService()
        performance = prediction_service.get_model_performance(
            model_name,
            symbol,
            days
        )

        return performance

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting performance for {model_name}: {str(e)}"
        )
