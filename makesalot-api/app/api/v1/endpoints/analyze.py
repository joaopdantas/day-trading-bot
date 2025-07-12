"""
Advanced Analysis Endpoints for MakesALot Trading API
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, validator

from ..endpoints.fetcher import get_data_api
from ..endpoints.strategies import MLTradingStrategy, TechnicalAnalysisStrategy, RSIDivergenceStrategy
from ..indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models
class AnalyzeRequest(BaseModel):
    symbol: str
    strategy: str = "ml_trading"
    days: int = 100
    timeframe: str = "1d"
    include_predictions: bool = True
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 1 or len(v) > 10:
            raise ValueError('Symbol must be 1-10 characters')
        return v.upper().strip()
    
    @validator('strategy')
    def validate_strategy(cls, v):
        allowed = ['ml_trading', 'technical', 'rsi_divergence']
        if v not in allowed:
            raise ValueError(f'Strategy must be one of: {allowed}')
        return v
    
    @validator('days')
    def validate_days(cls, v):
        if v < 10 or v > 1000:
            raise ValueError('Days must be between 10 and 1000')
        return v

class TechnicalIndicatorsResponse(BaseModel):
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float
    sma_20: float
    sma_50: float
    volume_ratio: float

class RecommendationResponse(BaseModel):
    action: str
    confidence: float
    reasoning: List[str]
    strategy_used: str

class AnalyzeResponse(BaseModel):
    symbol: str
    current_price: float
    change_percent: float
    trend: str
    recommendation: RecommendationResponse
    technical_indicators: TechnicalIndicatorsResponse
    support_resistance: Dict
    volume_analysis: Dict
    risk_assessment: str
    timestamp: str
    data_source: str

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_symbol(request: AnalyzeRequest):
    """
    Advanced symbol analysis with multiple strategies
    """
    try:
        logger.info(f"Analyzing {request.symbol} with {request.strategy} strategy")
        
        # Fetch market data
        data_fetcher = get_data_api('polygon')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        
        historical_data = data_fetcher.fetch_historical_data(
            symbol=request.symbol,
            interval=request.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if historical_data.empty:
            # Fallback to Yahoo Finance
            logger.warning(f"Polygon failed for {request.symbol}, trying Yahoo Finance")
            data_fetcher = get_data_api('yahoo_finance')
            historical_data = data_fetcher.fetch_historical_data(
                symbol=request.symbol,
                interval=request.timeframe,
                start_date=start_date,
                end_date=end_date
            )
        
        if historical_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol {request.symbol}"
            )
        
        # Calculate technical indicators
        technical_data = TechnicalIndicators.calculate_all_indicators(historical_data)
        
        # Add technical indicators to historical data
        latest_data = historical_data.iloc[-1].copy()
        for key, value in technical_data.items():
            if isinstance(value, (int, float)):
                latest_data[key] = value
        
        # Initialize and run strategy
        strategy = None
        if request.strategy == "ml_trading":
            strategy = MLTradingStrategy()
        elif request.strategy == "technical":
            strategy = TechnicalAnalysisStrategy()
        elif request.strategy == "rsi_divergence":
            strategy = RSIDivergenceStrategy()
        
        if not strategy:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {request.strategy}"
            )
        
        # Generate trading signal
        signal = strategy.generate_signal(latest_data, historical_data)
        
        # Calculate additional metrics
        current_price = float(latest_data['close'])
        prev_price = float(historical_data.iloc[-2]['close']) if len(historical_data) > 1 else current_price
        change_percent = ((current_price - prev_price) / prev_price) * 100
        
        # Determine overall trend
        sma_20 = technical_data.get('sma_20', current_price)
        sma_50 = technical_data.get('sma_50', current_price)
        
        if current_price > sma_20 > sma_50:
            trend = "bullish"
        elif current_price < sma_20 < sma_50:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Risk assessment
        rsi = technical_data.get('rsi', 50)
        bb_position = technical_data.get('bb_position', 0.5)
        volume_ratio = technical_data.get('volume_ratio', 1.0)
        
        if rsi > 80 or rsi < 20 or bb_position > 0.9 or bb_position < 0.1:
            risk = "high"
        elif rsi > 70 or rsi < 30 or volume_ratio > 2.0:
            risk = "medium"
        else:
            risk = "low"
        
        # Volume analysis
        current_volume = float(latest_data.get('volume', 0))
        avg_volume = technical_data.get('volume_sma', current_volume)
        
        volume_analysis = {
            "current": current_volume,
            "average_20d": avg_volume,
            "ratio": volume_ratio,
            "trend": "increasing" if volume_ratio > 1.2 else "decreasing" if volume_ratio < 0.8 else "stable"
        }
        
        # Build response
        response = AnalyzeResponse(
            symbol=request.symbol,
            current_price=current_price,
            change_percent=change_percent,
            trend=trend,
            recommendation=RecommendationResponse(
                action=signal.get('action', 'HOLD'),
                confidence=signal.get('confidence', 0.5),
                reasoning=signal.get('reasoning', ['Analysis completed']),
                strategy_used=request.strategy
            ),
            technical_indicators=TechnicalIndicatorsResponse(
                rsi=technical_data.get('rsi', 50),
                macd=technical_data.get('macd', 0),
                macd_signal=technical_data.get('macd_signal', 0),
                bb_upper=technical_data.get('bb_upper', current_price * 1.02),
                bb_middle=technical_data.get('bb_middle', current_price),
                bb_lower=technical_data.get('bb_lower', current_price * 0.98),
                bb_position=technical_data.get('bb_position', 0.5),
                sma_20=technical_data.get('sma_20', current_price),
                sma_50=technical_data.get('sma_50', current_price),
                volume_ratio=volume_ratio
            ),
            support_resistance=technical_data.get('support_resistance', {'support': [], 'resistance': []}),
            volume_analysis=volume_analysis,
            risk_assessment=risk,
            timestamp=datetime.now().isoformat(),
            data_source="makesalot_api"
        )
        
        logger.info(f"Analysis completed for {request.symbol}: {signal.get('action')} ({signal.get('confidence', 0):.2f})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/simple-analyze")
async def simple_analyze(
    symbol: str = Query(..., description="Stock symbol"),
    days: int = Query(100, ge=10, le=365, description="Number of days of data")
):
    """
    Simplified analysis endpoint for quick results
    """
    try:
        logger.info(f"Simple analysis for {symbol}")
        
        # Validate symbol
        symbol = symbol.upper().strip()
        if not symbol or len(symbol) > 10:
            raise HTTPException(status_code=400, detail="Invalid symbol")
        
        # Fetch data with fallback
        data_fetcher = None
        historical_data = pd.DataFrame()
        
        # Try multiple data sources
        api_sources = ['polygon', 'yahoo_finance', 'alpha_vantage']
        
        for api_name in api_sources:
            try:
                data_fetcher = get_data_api(api_name)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                historical_data = data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval="1d",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not historical_data.empty:
                    logger.info(f"Data fetched successfully from {api_name}")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to fetch from {api_name}: {e}")
                continue
        
        if historical_data.empty:
            # Generate mock data for demo purposes
            logger.warning(f"No real data available for {symbol}, generating mock data")
            historical_data = generate_mock_data(symbol, days)
        
        # Calculate basic indicators
        close_prices = historical_data['close']
        current_price = float(close_prices.iloc[-1])
        prev_price = float(close_prices.iloc[-2]) if len(close_prices) > 1 else current_price
        change_percent = ((current_price - prev_price) / prev_price) * 100
        
        # Calculate RSI
        rsi = TechnicalIndicators.calculate_rsi(close_prices)
        current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
        
        # Calculate moving averages
        sma_20 = TechnicalIndicators.calculate_sma(close_prices, 20)
        sma_50 = TechnicalIndicators.calculate_sma(close_prices, 50)
        current_sma_20 = float(sma_20.iloc[-1]) if not sma_20.empty else current_price
        current_sma_50 = float(sma_50.iloc[-1]) if not sma_50.empty else current_price
        
        # Simple recommendation logic
        recommendation = "HOLD"
        if current_rsi < 30:
            recommendation = "BUY"
        elif current_rsi > 70:
            recommendation = "SELL"
        elif current_price > current_sma_20 > current_sma_50:
            recommendation = "BUY"
        elif current_price < current_sma_20 < current_sma_50:
            recommendation = "SELL"
        
        # Determine trend
        if current_price > current_sma_20 > current_sma_50:
            trend = "bullish"
        elif current_price < current_sma_20 < current_sma_50:
            trend = "bearish"
        else:
            trend = "neutral"
        
        response = {
            "symbol": symbol,
            "current_price": current_price,
            "change_percent": change_percent,
            "trend": trend,
            "recommendation": recommendation,
            "rsi": current_rsi,
            "sma_20": current_sma_20,
            "sma_50": current_sma_50,
            "volume": int(historical_data['volume'].iloc[-1]) if 'volume' in historical_data.columns else 1000000,
            "timestamp": datetime.now().isoformat(),
            "data_source": "simple_api"
        }
        
        logger.info(f"Simple analysis completed for {symbol}: {recommendation}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple analysis error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Simple analysis failed: {str(e)}"
        )

def generate_mock_data(symbol: str, days: int) -> pd.DataFrame:
    """Generate realistic mock market data for testing"""
    import numpy as np
    
    # Base prices for known symbols
    base_prices = {
        'MSFT': 350, 'AAPL': 180, 'GOOGL': 140, 'AMZN': 150,
        'TSLA': 200, 'NVDA': 800, 'META': 300, 'NFLX': 400,
        'SPY': 450, 'QQQ': 380, 'VOO': 400
    }
    
    base_price = base_prices.get(symbol, 100 + np.random.random() * 200)
    
    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(start=end_date - timedelta(days=days), end=end_date, freq='D')
    
    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.02, len(dates))  # ~0.05% daily return, 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + np.random.random() * 0.02)
        low = min(open_price, close) * (1 - np.random.random() * 0.02)
        volume = int(500000 + np.random.random() * 2000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    logger.info(f"Generated {len(df)} days of mock data for {symbol}")
    return df