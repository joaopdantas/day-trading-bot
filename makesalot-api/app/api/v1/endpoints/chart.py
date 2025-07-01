"""
Chart data endpoints
"""
from fastapi import APIRouter
import random
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/data/{symbol}")
async def get_chart_data(symbol: str, period: str = "3m"):
    """
    Get historical chart data for a symbol
    """
    
    # Convert period to days
    period_days = {
        "3m": 90,
        "6m": 180, 
        "1y": 365
    }.get(period, 90)
    
    # Generate mock chart data
    base_price = 100 + random.random() * 200
    chart_data = []
    
    for i in range(period_days):
        date = datetime.now() - timedelta(days=period_days-i)
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