"""
Machine Learning prediction endpoints
"""
from fastapi import APIRouter
import random
from datetime import datetime

router = APIRouter()

@router.post("/predict")
async def get_prediction(request: dict):
    """
    Get price movement predictions for a symbol
    """
    symbol = request.get("symbol", "UNKNOWN")
    
    # Mock prediction response
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
        "model_performance": {"accuracy": 0.72, "precision": 0.68},
        "timestamp": datetime.now().isoformat()
    }