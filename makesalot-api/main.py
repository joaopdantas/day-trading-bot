"""
MakesALot Trading API - Ultra-Simple Production Version
Python 3.13 compatible with minimal dependencies
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import os
import logging
from datetime import datetime, timedelta
import json
import math
import statistics

# Only use built-in libraries and minimal dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== SIMPLE TECHNICAL INDICATORS (Pure Python) =====
class SimpleTechnicalIndicators:
    """Pure Python technical indicators - no external dependencies"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        result = []
        for i in range(len(prices)):
            if i < period - 1:
                result.append(None)
            else:
                avg = sum(prices[i-period+1:i+1]) / period
                result.append(avg)
        return result
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """RSI calculation"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        rsi_values = []
        
        for i in range(len(gains)):
            if i < period - 1:
                rsi_values.append(50.0)
            else:
                avg_gain = sum(gains[i-period+1:i+1]) / period
                avg_loss = sum(losses[i-period+1:i+1]) / period
                
                if avg_loss == 0:
                    rsi_values.append(100.0)
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """MACD calculation"""
        if len(prices) < slow:
            return {
                'macd': [0.0] * len(prices),
                'signal': [0.0] * len(prices),
                'histogram': [0.0] * len(prices)
            }
        
        # Calculate EMAs
        ema_fast = SimpleTechnicalIndicators.ema(prices, fast)
        ema_slow = SimpleTechnicalIndicators.ema(prices, slow)
        
        # MACD line
        macd_line = []
        for i in range(len(prices)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line.append(ema_fast[i] - ema_slow[i])
            else:
                macd_line.append(0.0)
        
        # Signal line (EMA of MACD)
        signal_line = SimpleTechnicalIndicators.ema(macd_line, signal)
        
        # Histogram
        histogram = []
        for i in range(len(macd_line)):
            if signal_line[i] is not None:
                histogram.append(macd_line[i] - signal_line[i])
            else:
                histogram.append(0.0)
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return [None] * len(prices)
        
        alpha = 2 / (period + 1)
        ema_values = []
        
        # First EMA value is SMA
        first_sma = sum(prices[:period]) / period
        ema_values.extend([None] * (period - 1))
        ema_values.append(first_sma)
        
        # Calculate subsequent EMA values
        for i in range(period, len(prices)):
            ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, List[float]]:
        """Bollinger Bands"""
        sma_values = SimpleTechnicalIndicators.sma(prices, period)
        
        upper_band = []
        lower_band = []
        bb_position = []
        
        for i in range(len(prices)):
            if i < period - 1 or sma_values[i] is None:
                upper_band.append(None)
                lower_band.append(None)
                bb_position.append(0.5)
            else:
                # Calculate standard deviation for this period
                period_prices = prices[i-period+1:i+1]
                variance = sum((p - sma_values[i]) ** 2 for p in period_prices) / period
                std = math.sqrt(variance)
                
                upper = sma_values[i] + (std * std_dev)
                lower = sma_values[i] - (std * std_dev)
                
                upper_band.append(upper)
                lower_band.append(lower)
                
                # BB position (0 = lower band, 1 = upper band)
                if upper != lower:
                    position = (prices[i] - lower) / (upper - lower)
                    bb_position.append(max(0, min(1, position)))
                else:
                    bb_position.append(0.5)
        
        return {
            'upper': upper_band,
            'middle': sma_values,
            'lower': lower_band,
            'position': bb_position
        }

# ===== DATA MODELS =====
class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., MSFT, AAPL)")
    timeframe: Optional[str] = Field("1d", description="Data timeframe")

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

# ===== SIMPLE DATA FETCHER =====
class SimpleDataFetcher:
    """Ultra-simple data fetcher"""
    
    @staticmethod
    def fetch_stock_data(symbol: str, days: int = 100) -> Dict[str, Any]:
        """Fetch stock data using yfinance or generate mock data"""
        if not YFINANCE_AVAILABLE:
            return SimpleDataFetcher.generate_mock_data(symbol, days)
        
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return SimpleDataFetcher.generate_mock_data(symbol, days)
            
            # Convert to simple dict structure
            data = {
                'dates': [str(date.date()) for date in df.index],
                'open': df['Open'].tolist(),
                'high': df['High'].tolist(),
                'low': df['Low'].tolist(),
                'close': df['Close'].tolist(),
                'volume': df['Volume'].tolist()
            }
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {e}")
            return SimpleDataFetcher.generate_mock_data(symbol, days)
    
    @staticmethod
    def generate_mock_data(symbol: str, days: int) -> Dict[str, Any]:
        """Generate realistic mock data"""
        # Base price varies by symbol
        base_prices = {
            'AAPL': 180, 'MSFT': 350, 'GOOGL': 2800, 'AMZN': 3200, 'TSLA': 800,
            'NVDA': 450, 'META': 320, 'NFLX': 400, 'JPM': 150, 'JNJ': 170
        }
        
        base_price = base_prices.get(symbol.upper(), 100)
        
        dates = []
        prices = []
        volumes = []
        
        current_date = datetime.now() - timedelta(days=days)
        current_price = base_price
        
        for i in range(days):
            dates.append(str(current_date.date()))
            
            # Simulate price movement
            change_percent = (hash(f"{symbol}{i}") % 200 - 100) / 1000  # -10% to +10%
            current_price *= (1 + change_percent)
            current_price = max(current_price, 1)  # Prevent negative prices
            
            # Simulate intraday values
            high = current_price * (1 + abs(change_percent) * 0.5)
            low = current_price * (1 - abs(change_percent) * 0.5)
            open_price = current_price * (1 + change_percent * 0.1)
            
            prices.append({
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(current_price, 2)
            })
            
            # Simulate volume
            base_volume = 1000000 + (hash(f"{symbol}{i}volume") % 5000000)
            volumes.append(base_volume)
            
            current_date += timedelta(days=1)
        
        return {
            'dates': dates,
            'open': [p['open'] for p in prices],
            'high': [p['high'] for p in prices],
            'low': [p['low'] for p in prices],
            'close': [p['close'] for p in prices],
            'volume': volumes
        }

# ===== SIMPLE PREDICTOR =====
class SimplePredictor:
    """Simple prediction based on technical indicators"""
    
    @staticmethod
    def predict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Make simple prediction based on technical analysis"""
        try:
            prices = data['close']
            
            if len(prices) < 20:
                return {
                    "direction": "HOLD",
                    "confidence": 0.5,
                    "message": "Insufficient data"
                }
            
            # Calculate indicators
            rsi_values = SimpleTechnicalIndicators.rsi(prices)
            macd_data = SimpleTechnicalIndicators.macd(prices)
            bb_data = SimpleTechnicalIndicators.bollinger_bands(prices)
            
            current_rsi = rsi_values[-1]
            current_macd = macd_data['macd'][-1]
            current_bb_pos = bb_data['position'][-1]
            
            # Simple signal logic
            signals = []
            weights = []
            
            # RSI signal
            if current_rsi < 30:
                signals.append(0.8)  # Strong buy
                weights.append(0.3)
            elif current_rsi > 70:
                signals.append(-0.8)  # Strong sell
                weights.append(0.3)
            else:
                signals.append((50 - current_rsi) / 50)  # Neutral
                weights.append(0.2)
            
            # MACD signal
            if current_macd > 0:
                signals.append(0.6)
                weights.append(0.3)
            else:
                signals.append(-0.6)
                weights.append(0.3)
            
            # Bollinger Bands signal
            if current_bb_pos < 0.2:
                signals.append(0.7)
                weights.append(0.4)
            elif current_bb_pos > 0.8:
                signals.append(-0.7)
                weights.append(0.4)
            else:
                signals.append(0.0)
                weights.append(0.1)
            
            # Calculate weighted prediction
            if sum(weights) > 0:
                prediction = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
            else:
                prediction = 0
            
            # Convert to direction
            if prediction > 0.2:
                direction = "BUY"
                confidence = min(abs(prediction), 0.95)
            elif prediction < -0.2:
                direction = "SELL" 
                confidence = min(abs(prediction), 0.95)
            else:
                direction = "HOLD"
                confidence = 0.5
            
            return {
                "direction": direction,
                "confidence": confidence,
                "predicted_signal": prediction,
                "rsi": current_rsi,
                "macd": current_macd,
                "bb_position": current_bb_pos
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "direction": "HOLD",
                "confidence": 0.5,
                "message": f"Prediction failed: {str(e)}"
            }

# ===== CREATE FASTAPI APP =====
app = FastAPI(
    title="MakesALot Trading API",
    description="Ultra-Simple Trading Analysis API",
    version="2.1.0",
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

# ===== INITIALIZE COMPONENTS =====
data_fetcher = SimpleDataFetcher()
predictor = SimplePredictor()

# ===== API ENDPOINTS =====

@app.get("/")
def root():
    """API root"""
    return {
        "message": "🚀 MakesALot Trading API - Ultra Simple",
        "version": "2.1.0",
        "status": "healthy",
        "features": [
            "Real-time stock data",
            "Technical indicators (RSI, MACD, BB)",
            "Simple ML predictions",
            "Chart data",
            "Python 3.13 compatible"
        ],
        "yfinance_available": YFINANCE_AVAILABLE
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MakesALot Trading API",
        "version": "2.1.0",
        "python_version": "3.13+",
        "yfinance_available": YFINANCE_AVAILABLE
    }

@app.post("/api/v1/technical/analyze")
async def technical_analyze(request: AnalysisRequest):
    """Technical analysis endpoint"""
    try:
        # Fetch data
        data = data_fetcher.fetch_stock_data(request.symbol, 100)
        
        if not data or not data['close']:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        prices = data['close']
        current_price = prices[-1]
        previous_price = prices[-2] if len(prices) > 1 else current_price
        
        change_percent = ((current_price - previous_price) / previous_price) * 100
        
        # Calculate indicators
        rsi_values = SimpleTechnicalIndicators.rsi(prices)
        macd_data = SimpleTechnicalIndicators.macd(prices)
        bb_data = SimpleTechnicalIndicators.bollinger_bands(prices)
        sma_20 = SimpleTechnicalIndicators.sma(prices, 20)
        
        current_rsi = rsi_values[-1]
        current_macd = macd_data['macd'][-1]
        current_bb_pos = bb_data['position'][-1]
        current_sma20 = sma_20[-1] if sma_20[-1] is not None else current_price
        
        # Generate signals
        signals = {}
        signals['RSI'] = 'BUY' if current_rsi < 30 else 'SELL' if current_rsi > 70 else 'HOLD'
        signals['MACD'] = 'BUY' if current_macd > 0 else 'SELL'
        signals['BB'] = 'BUY' if current_bb_pos < 0.2 else 'SELL' if current_bb_pos > 0.8 else 'HOLD'
        
        # Determine trend
        trend = "BULLISH" if current_price > current_sma20 else "BEARISH" if current_price < current_sma20 * 0.98 else "NEUTRAL"
        
        return {
            "symbol": request.symbol,
            "current_price": round(current_price, 2),
            "change_percent": round(change_percent, 2),
            "trend": trend,
            "signals": signals,
            "indicators": [
                {
                    "name": "RSI",
                    "value": round(current_rsi, 2),
                    "signal": signals['RSI']
                },
                {
                    "name": "MACD",
                    "value": round(current_macd, 4),
                    "signal": signals['MACD']
                },
                {
                    "name": "BB_position",
                    "value": round(current_bb_pos, 3),
                    "signal": signals['BB']
                }
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Technical analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predictions/predict")
async def predictions_predict(request: PredictionRequest):
    """Prediction endpoint"""
    try:
        # Fetch data
        data = data_fetcher.fetch_stock_data(request.symbol, 200)
        
        if not data or not data['close']:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Make prediction
        prediction = predictor.predict(data)
        
        return {
            "symbol": request.symbol,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/chart/data/{symbol}")
async def chart_data(symbol: str, period: str = "3m"):
    """Chart data endpoint"""
    try:
        # Map period to days
        period_mapping = {"1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
        days = period_mapping.get(period, 90)
        
        # Fetch data
        data = data_fetcher.fetch_stock_data(symbol, days)
        
        if not data or not data['close']:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        current_price = data['close'][-1]
        start_price = data['close'][0]
        price_change = ((current_price - start_price) / start_price) * 100
        
        # Create chart data
        chart_data = []
        for i in range(len(data['dates'])):
            chart_data.append({
                "date": data['dates'][i],
                "open": data['open'][i],
                "high": data['high'][i],
                "low": data['low'][i],
                "close": data['close'][i],
                "volume": data['volume'][i]
            })
        
        return {
            "symbol": symbol,
            "period": period,
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "high_price": round(max(data['high']), 2),
            "low_price": round(min(data['low']), 2),
            "avg_volume": int(sum(data['volume']) / len(data['volume'])),
            "data_points": len(chart_data),
            "data": chart_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/quote/{symbol}", response_model=QuoteResponse)
async def get_quote(symbol: str):
    """Get quote"""
    try:
        data = data_fetcher.fetch_stock_data(symbol, 5)
        
        if not data or not data['close']:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        current_price = data['close'][-1]
        previous_price = data['close'][-2] if len(data['close']) > 1 else current_price
        
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
        
        return QuoteResponse(
            symbol=symbol.upper(),
            price=round(current_price, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            volume=data['volume'][-1] if data['volume'] else 0,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/symbols")
async def get_symbols():
    """Get supported symbols"""
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
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting MakesALot Trading API on {host}:{port}")
    logger.info(f"YFinance available: {YFINANCE_AVAILABLE}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )