"""
Main FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Simple API without complex imports for basic functionality
app = FastAPI(
    title="MakesALot Trading API",
    description="AI-powered trading analysis and prediction tool",
    version="1.0.0"
)

# CORS configuration for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your extension ID
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "MakesALot Trading API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Basic endpoints for extension
@app.post("/api/v1/technical/analyze")
async def analyze_symbol(request: dict):
    """Basic technical analysis endpoint"""
    symbol = request.get("symbol", "UNKNOWN")
    
    # Mock analysis response
    return {
        "symbol": symbol,
        "trend": "BULLISH",
        "signals": {"RSI": "BUY", "MACD": "HOLD", "BB": "BUY"},
        "indicators": [
            {"name": "RSI", "value": 65.5, "signal": "BUY"},
            {"name": "MACD", "value": 1.2, "signal": "HOLD"},
            {"name": "BB", "value": 0.8, "signal": "BUY"}
        ],
        "support_levels": [150.0, 148.5, 146.0],
        "resistance_levels": [155.0, 157.5, 160.0],
        "timestamp": "2024-01-01T12:00:00"
    }

@app.post("/api/v1/predictions/predict")
async def predict_symbol(request: dict):
    """Basic prediction endpoint"""
    symbol = request.get("symbol", "UNKNOWN")
    
    # Mock prediction response
    import random
    directions = ["BUY", "SELL", "HOLD"]
    direction = random.choice(directions)
    confidence = 0.6 + random.random() * 0.3
    
    return {
        "symbol": symbol,
        "prediction": {
            "direction": direction,
            "confidence": confidence,
            "probability": confidence
        },
        "model_performance": {"accuracy": 0.72},
        "timestamp": "2024-01-01T12:00:00"
    }

@app.get("/api/v1/chart/data/{symbol}")
async def get_chart_data(symbol: str, period: str = "3m"):
    """Basic chart data endpoint"""
    import random
    from datetime import datetime, timedelta
    
    # Generate mock chart data
    days = {"3m": 90, "6m": 180, "1y": 365}.get(period, 90)
    base_price = 100 + random.random() * 200
    
    chart_data = []
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i)
        price_change = (random.random() - 0.5) * 0.05
        price = base_price * (1 + price_change)
        base_price = price
        
        chart_data.append({
            "date": date.isoformat(),
            "open": price * (1 + (random.random() - 0.5) * 0.02),
            "high": price * (1 + random.random() * 0.03),
            "low": price * (1 - random.random() * 0.03),
            "close": price,
            "volume": random.randint(100000, 1000000)
        })
    
    current_price = chart_data[-1]["close"]
    previous_price = chart_data[-2]["close"]
    price_change = ((current_price - previous_price) / previous_price) * 100
    
    return {
        "symbol": symbol,
        "period": period,
        "current_price": current_price,
        "price_change": price_change,
        "data": chart_data,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )