"""
API integration tests
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from app.main import app
from app.core.config import settings

client = TestClient(app)


def test_technical_analysis():
    """Test technical analysis endpoint"""
    response = client.post(
        "/api/v1/technical/analyze",
        json={
            "symbol": "MSFT",
            "timeframe": "1d",
            "days": 100,
            "indicators": ["RSI", "MACD", "BB"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "trend" in data
    assert "signals" in data
    assert "indicators" in data


def test_ml_prediction():
    """Test ML prediction endpoint"""
    response = client.post(
        "/api/v1/predictions/predict",
        json={
            "symbol": "MSFT",
            "timeframe": "1d",
            "days": 100,
            "models": ["ensemble"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data


def test_support_resistance():
    """Test support/resistance endpoint"""
    response = client.get(
        "/api/v1/technical/support-resistance/MSFT",
        params={"timeframe": "1d"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "support_levels" in data
    assert "resistance_levels" in data


def test_feedback_submission():
    """Test feedback submission"""
    response = client.post(
        "/api/v1/feedback/submit",
        json={
            "type": "suggestion",
            "content": "Test feedback",
            "rating": 5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data


def test_error_handling():
    """Test error handling"""
    response = client.post(
        "/api/v1/technical/analyze",
        json={
            "symbol": "INVALID",
            "timeframe": "1d"
        }
    )
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_analytics_endpoint():
    """Test analytics endpoint"""
    response = client.get(
        "/api/v1/feedback/analytics",
        params={"days": 30}
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_users" in data
    assert "feedback_stats" in data
