"""
Main FastAPI application entry point - SIMPLIFIED VERSION
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Pydantic models for requests/responses
class TechnicalAnalysisRequest(BaseModel):
    symbol: str
    timeframe: Optional[str] = "1d"
    days: Optional[int] = 100

class TechnicalAnalysisResponse(BaseModel):
    symbol: str
    current_price: float
    trend: str
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    recommendation: str

# Create FastAPI app
app = FastAPI(
    title="MakesALot Trading API",
    description="API for technical analysis and trading predictions",
    version="1.0.0"
)

# CORS configuration - Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Chrome extension ID
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None

def get_stock_data(symbol: str, days: int = 100):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
            
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

@app.get("/")
def root():
    return {
        "message": "MakesALot Trading API is online!",
        "version": "1.0.0",
        "endpoints": [
            "/docs - API documentation",
            "/api/v1/analyze - Technical analysis",
            "/health - Health check"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/analyze", response_model=TechnicalAnalysisResponse)
async def technical_analysis(request: TechnicalAnalysisRequest):
    """
    Perform technical analysis on a stock symbol
    """
    try:
        # Fetch stock data
        data = get_stock_data(request.symbol, request.days)
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Could not fetch data for symbol: {request.symbol}"
            )
        
        # Calculate indicators
        current_price = float(data['Close'].iloc[-1])
        sma_20 = float(data['Close'].rolling(20).mean().iloc[-1]) if len(data) >= 20 else None
        sma_50 = float(data['Close'].rolling(50).mean().iloc[-1]) if len(data) >= 50 else None
        rsi = calculate_rsi(data['Close'])
        
        # Determine trend
        trend = "neutral"
        if sma_20 and sma_50:
            if sma_20 > sma_50:
                trend = "bullish"
            elif sma_20 < sma_50:
                trend = "bearish"
        
        # Generate recommendation
        recommendation = "hold"
        if rsi:
            if rsi < 30:
                recommendation = "buy"
            elif rsi > 70:
                recommendation = "sell"
        
        return TechnicalAnalysisResponse(
            symbol=request.symbol.upper(),
            current_price=current_price,
            trend=trend,
            rsi=rsi,
            sma_20=sma_20,
            sma_50=sma_50,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing {request.symbol}: {str(e)}"
        )

@app.get("/api/v1/quote/{symbol}")
async def get_quote(symbol: str):
    """
    Get current quote for a symbol
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found"
            )
        
        return {
            "symbol": symbol.upper(),
            "name": info.get("longName", "N/A"),
            "price": info.get("currentPrice", 0),
            "change": info.get("regularMarketChange", 0),
            "changePercent": info.get("regularMarketChangePercent", 0),
            "volume": info.get("volume", 0),
            "marketCap": info.get("marketCap", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting quote for {symbol}: {str(e)}"
        )

# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Set to False for production
    )