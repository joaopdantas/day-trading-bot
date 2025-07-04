"""
MakesALot Trading API - Simplified Production Version
Compatible with Python 3.11 - optimized for Render deployment
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uvicorn
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== TECHNICAL INDICATORS (Custom Implementation) =====
class TechnicalIndicators:
    """Custom technical indicators implementation - Production ready"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev),
            'position': (prices - (sma - std * std_dev)) / (2 * std * std_dev)
        }
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        if len(df) >= 15:
            df['RSI'] = TechnicalIndicators.rsi(df['close'])
        else:
            df['RSI'] = 50
        
        # MACD
        if len(df) >= 26:
            macd_data = TechnicalIndicators.macd(df['close'])
            df['MACD'] = macd_data['macd']
            df['MACD_signal'] = macd_data['signal']
            df['MACD_histogram'] = macd_data['histogram']
        else:
            df['MACD'] = 0
            df['MACD_signal'] = 0
            df['MACD_histogram'] = 0
        
        # Bollinger Bands
        if len(df) >= 20:
            bb_data = TechnicalIndicators.bollinger_bands(df['close'])
            df['BB_upper'] = bb_data['upper']
            df['BB_middle'] = bb_data['middle']
            df['BB_lower'] = bb_data['lower']
            df['BB_position'] = bb_data['position']
        else:
            df['BB_upper'] = df['close']
            df['BB_middle'] = df['close']
            df['BB_lower'] = df['close']
            df['BB_position'] = 0.5
        
        # Stochastic
        if len(df) >= 14 and all(col in df.columns for col in ['high', 'low']):
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        else:
            df['Stoch_K'] = 50
            df['Stoch_D'] = 50
        
        # Volume indicators
        if 'volume' in df.columns and len(df) >= 20:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df

# ===== DATA MODELS =====
class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., MSFT, AAPL)")
    timeframe: Optional[str] = Field("1d", description="Data timeframe")
    days: Optional[int] = Field(100, description="Number of days of data")

class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    timeframe: Optional[str] = Field("1d", description="Data timeframe")

class QuoteResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str

# ===== SIMPLIFIED DATA FETCHER =====
class DataFetcher:
    """Simplified data fetcher using yfinance"""
    
    @staticmethod
    async def fetch_data(symbol: str, timeframe: str = "1d", days: int = 100) -> pd.DataFrame:
        """Fetch data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Calculate period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data
            df = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            if 'adj_close' in df.columns:
                df['close'] = df['adj_close']
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")

# ===== SIMPLE ML PREDICTOR =====
class SimplePredictor:
    """Simple ML predictor using RandomForest"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        
        # Price features
        features.append(df['close'].pct_change(1).fillna(0))
        features.append(df['close'].pct_change(5).fillna(0))
        features.append(df['close'].rolling(5).std().fillna(0))
        
        # Technical indicators
        if 'RSI' in df.columns:
            features.append(df['RSI'].fillna(50))
        if 'MACD' in df.columns:
            features.append(df['MACD'].fillna(0))
        if 'BB_position' in df.columns:
            features.append(df['BB_position'].fillna(0.5))
        
        # Volume features
        if 'volume_ratio' in df.columns:
            features.append(df['volume_ratio'].fillna(1))
        
        return np.column_stack(features)
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction"""
        try:
            if len(df) < 30:
                return {
                    "direction": "HOLD",
                    "confidence": 0.5,
                    "message": "Insufficient data for ML prediction"
                }
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Create target (future returns)
            y = df['close'].shift(-1) / df['close'] - 1
            y = y.dropna()
            X = X[:len(y)]
            
            if len(X) < 20:
                return {
                    "direction": "HOLD",
                    "confidence": 0.5,
                    "message": "Insufficient data"
                }
            
            # Train model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Make prediction
            latest_features = X[-1:] 
            latest_scaled = self.scaler.transform(latest_features)
            prediction = self.model.predict(latest_scaled)[0]
            
            # Convert to direction
            if prediction > 0.02:
                direction = "BUY"
                confidence = min(abs(prediction) * 10, 0.95)
            elif prediction < -0.02:
                direction = "SELL"
                confidence = min(abs(prediction) * 10, 0.95)
            else:
                direction = "HOLD"
                confidence = 0.5
            
            return {
                "direction": direction,
                "confidence": confidence,
                "predicted_return": prediction,
                "message": "ML prediction based on technical indicators"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "direction": "HOLD",
                "confidence": 0.5,
                "message": "Prediction failed, defaulting to HOLD"
            }

# ===== INITIALIZE COMPONENTS =====
data_fetcher = DataFetcher()
predictor = SimplePredictor()

# ===== CREATE FASTAPI APP =====
app = FastAPI(
    title="MakesALot Trading API",
    description="Professional Trading Analysis API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ===== CORS CONFIGURATION =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== API ENDPOINTS =====

@app.get("/")
def root():
    """API root with system status"""
    return {
        "message": "🚀 MakesALot Trading API",
        "version": "2.0.0",
        "status": "healthy",
        "features": [
            "Real-time stock data",
            "Technical analysis",
            "ML predictions",
            "Chart data"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "technical_analysis": "/api/v1/technical/analyze",
            "predictions": "/api/v1/predictions/predict",
            "chart_data": "/api/v1/chart/data/{symbol}",
            "quote": "/api/v1/quote/{symbol}"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MakesALot Trading API",
        "version": "2.0.0"
    }

@app.post("/api/v1/technical/analyze")
async def technical_analyze(request: AnalysisRequest):
    """Technical analysis endpoint"""
    try:
        # Fetch data
        df = await data_fetcher.fetch_data(request.symbol, request.timeframe, request.days)
        
        # Add technical indicators
        df_enhanced = TechnicalIndicators.add_all_indicators(df)
        
        current = df_enhanced.iloc[-1]
        previous = df_enhanced.iloc[-2] if len(df_enhanced) > 1 else current
        
        # Calculate metrics
        current_price = float(current['close'])
        change_percent = ((current_price - previous['close']) / previous['close']) * 100
        
        # Generate signals
        signals = {}
        
        if 'RSI' in df_enhanced.columns and pd.notna(current['RSI']):
            rsi = current['RSI']
            signals['RSI'] = 'BUY' if rsi < 30 else 'SELL' if rsi > 70 else 'HOLD'
        
        if 'MACD' in df_enhanced.columns and pd.notna(current['MACD']):
            macd = current['MACD']
            signals['MACD'] = 'BUY' if macd > 0 else 'SELL'
        
        if 'BB_position' in df_enhanced.columns and pd.notna(current['BB_position']):
            bb_pos = current['BB_position']
            signals['BB'] = 'BUY' if bb_pos < 0.2 else 'SELL' if bb_pos > 0.8 else 'HOLD'
        
        # Determine trend
        trend = "NEUTRAL"
        if 'SMA_20' in df_enhanced.columns and 'SMA_50' in df_enhanced.columns:
            sma20 = current.get('SMA_20')
            sma50 = current.get('SMA_50')
            if pd.notna(sma20) and pd.notna(sma50):
                if current_price > sma20 > sma50:
                    trend = "BULLISH"
                elif current_price < sma20 < sma50:
                    trend = "BEARISH"
        
        # Calculate support/resistance
        support_levels = []
        resistance_levels = []
        
        if len(df_enhanced) >= 20:
            recent_data = df_enhanced.tail(20)
            for i in range(2, len(recent_data) - 2):
                # Support levels
                if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]):
                    support_levels.append(float(recent_data['low'].iloc[i]))
                
                # Resistance levels  
                if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]):
                    resistance_levels.append(float(recent_data['high'].iloc[i]))
        
        return {
            "symbol": request.symbol,
            "current_price": round(current_price, 2),
            "change_percent": round(change_percent, 2),
            "trend": trend,
            "signals": signals,
            "indicators": [
                {
                    "name": "RSI",
                    "value": round(float(current.get('RSI', 50)), 2),
                    "signal": signals.get('RSI', 'HOLD')
                },
                {
                    "name": "MACD",
                    "value": round(float(current.get('MACD', 0)), 4),
                    "signal": signals.get('MACD', 'HOLD')
                },
                {
                    "name": "BB_position", 
                    "value": round(float(current.get('BB_position', 0.5)), 3),
                    "signal": signals.get('BB', 'HOLD')
                }
            ],
            "support_levels": sorted(support_levels, reverse=True)[:3],
            "resistance_levels": sorted(resistance_levels)[:3],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Technical analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predictions/predict")
async def predictions_predict(request: PredictionRequest):
    """ML prediction endpoint"""
    try:
        # Fetch data
        df = await data_fetcher.fetch_data(request.symbol, request.timeframe, 200)
        
        # Add indicators
        df_enhanced = TechnicalIndicators.add_all_indicators(df)
        
        # Make prediction
        prediction = predictor.predict(df_enhanced)
        
        return {
            "symbol": request.symbol,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/chart/data/{symbol}")
async def chart_data(
    symbol: str,
    period: str = Query("3m", description="Period: 1w, 1m, 3m, 6m, 1y"),
    interval: str = Query("1d", description="Interval: 1d, 1h, 30m")
):
    """Chart data endpoint"""
    try:
        # Map period to days
        period_mapping = {"1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
        days = period_mapping.get(period, 90)
        
        # Fetch data
        df = await data_fetcher.fetch_data(symbol, interval, days)
        
        current_price = float(df['close'].iloc[-1])
        start_price = float(df['close'].iloc[0])
        price_change = ((current_price - start_price) / start_price) * 100
        
        # Create chart data
        chart_data = []
        for idx, row in df.iterrows():
            chart_data.append({
                "date": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume']) if pd.notna(row['volume']) else 0
            })
        
        # Calculate statistics
        high_price = float(df['high'].max())
        low_price = float(df['low'].min())
        avg_volume = float(df['volume'].mean()) if 'volume' in df.columns else 0
        
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "high_price": round(high_price, 2),
            "low_price": round(low_price, 2),
            "avg_volume": int(avg_volume),
            "data_points": len(chart_data),
            "data": chart_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/quote/{symbol}", response_model=QuoteResponse)
async def get_quote(symbol: str):
    """Get real-time quote"""
    try:
        df = await data_fetcher.fetch_data(symbol, "1d", 5)
        
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        current_price = float(current['close'])
        prev_price = float(previous['close'])
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
        
        return QuoteResponse(
            symbol=symbol.upper(),
            price=round(current_price, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            volume=int(current['volume']) if pd.notna(current['volume']) else 0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Quote error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/symbols")
async def get_symbols():
    """Get list of supported symbols"""
    return {
        "popular_stocks": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "NFLX", "name": "Netflix Inc."}
        ],
        "indices": [
            {"symbol": "^GSPC", "name": "S&P 500"},
            {"symbol": "^IXIC", "name": "NASDAQ Composite"},
            {"symbol": "^DJI", "name": "Dow Jones Industrial Average"}
        ],
        "crypto": [
            {"symbol": "BTC-USD", "name": "Bitcoin"},
            {"symbol": "ETH-USD", "name": "Ethereum"}
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting MakesALot Trading API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )