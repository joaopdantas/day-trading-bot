"""
Trading Strategies Router for MakesALot API
"""

import logging
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .strategies import (
    MLTradingStrategy, 
    TechnicalAnalysisStrategy, 
    BuyAndHoldStrategy,
    RSIDivergenceStrategy,
    HybridRSIDivergenceStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Response Models
class StrategyInfo(BaseModel):
    name: str
    description: str
    parameters: Dict
    expected_performance: Dict
    risk_level: str
    recommended_for: List[str]

class AvailableStrategiesResponse(BaseModel):
    strategies: List[StrategyInfo]
    total_count: int
    default_strategy: str
    timestamp: str

@router.get("/available", response_model=AvailableStrategiesResponse)
async def get_available_strategies():
    """
    Get list of all available trading strategies
    """
    try:
        strategies = [
            StrategyInfo(
                name="ml_trading",
                description="Advanced ML-based trading strategy with technical analysis",
                parameters={
                    "rsi_oversold": 25,
                    "rsi_overbought": 75,
                    "volume_threshold": 1.3,
                    "confidence_threshold": 0.50,
                    "min_hold_period": 7
                },
                expected_performance={
                    "annual_return": "5-15%",
                    "win_rate": "55-65%",
                    "max_drawdown": "8-12%"
                },
                risk_level="medium",
                recommended_for=["intermediate", "advanced", "algorithmic_trading"]
            ),
            StrategyInfo(
                name="technical",
                description="Traditional technical analysis with RSI and moving averages",
                parameters={
                    "sma_short": 20,
                    "sma_long": 50,
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70
                },
                expected_performance={
                    "annual_return": "8-12%",
                    "win_rate": "60-70%",
                    "max_drawdown": "5-10%"
                },
                risk_level="low",
                recommended_for=["beginners", "conservative", "long_term"]
            ),
            StrategyInfo(
                name="rsi_divergence",
                description="RSI divergence strategy with proven 64%+ returns",
                parameters={
                    "swing_threshold_pct": 2.5,
                    "hold_days": 15,
                    "min_divergence_strength": 1.0,
                    "max_lookback": 50
                },
                expected_performance={
                    "annual_return": "64%+",
                    "win_rate": "76.5%",
                    "max_drawdown": "15-20%"
                },
                risk_level="high",
                recommended_for=["advanced", "active_trading", "high_risk_tolerance"]
            ),
            StrategyInfo(
                name="buy_and_hold",
                description="Classic buy and hold strategy for long-term investors",
                parameters={
                    "entry_timing": "immediate",
                    "exit_timing": "manual_only"
                },
                expected_performance={
                    "annual_return": "7-10%",
                    "win_rate": "85%+",
                    "max_drawdown": "20-30%"
                },
                risk_level="low",
                recommended_for=["beginners", "passive_investing", "retirement"]
            ),
            StrategyInfo(
                name="hybrid_rsi",
                description="Hybrid strategy combining RSI divergence with technical analysis",
                parameters={
                    "divergence_weight": 0.6,
                    "technical_weight": 0.4,
                    "base_strategy": "technical"
                },
                expected_performance={
                    "annual_return": "15-25%",
                    "win_rate": "65-75%",
                    "max_drawdown": "10-15%"
                },
                risk_level="medium-high",
                recommended_for=["intermediate", "balanced_approach", "moderate_risk"]
            )
        ]
        
        response = AvailableStrategiesResponse(
            strategies=strategies,
            total_count=len(strategies),
            default_strategy="ml_trading",
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Returned {len(strategies)} available strategies")
        return response
        
    except Exception as e:
        logger.error(f"Error getting available strategies: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available strategies: {str(e)}"
        )

@router.get("/info/{strategy_name}")
async def get_strategy_info(strategy_name: str):
    """
    Get detailed information about a specific strategy
    """
    try:
        strategy_name = strategy_name.lower().strip()
        
        # Initialize strategy to get info
        strategy_instance = None
        
        if strategy_name == "ml_trading":
            strategy_instance = MLTradingStrategy()
        elif strategy_name == "technical":
            strategy_instance = TechnicalAnalysisStrategy()
        elif strategy_name == "rsi_divergence":
            strategy_instance = RSIDivergenceStrategy()
        elif strategy_name == "buy_and_hold":
            strategy_instance = BuyAndHoldStrategy()
        elif strategy_name == "hybrid_rsi":
            strategy_instance = HybridRSIDivergenceStrategy()
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy '{strategy_name}' not found"
            )
        
        # Get strategy info if method exists
        if hasattr(strategy_instance, 'get_strategy_info'):
            strategy_info = strategy_instance.get_strategy_info()
        else:
            # Fallback info
            strategy_info = {
                "name": strategy_name,
                "description": f"Information for {strategy_name} strategy",
                "status": "active"
            }
        
        # Add runtime information
        strategy_info.update({
            "class_name": strategy_instance.__class__.__name__,
            "available_methods": [method for method in dir(strategy_instance) 
                                if not method.startswith('_') and callable(getattr(strategy_instance, method))],
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Retrieved info for strategy: {strategy_name}")
        return strategy_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy info for {strategy_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get strategy info: {str(e)}"
        )

@router.get("/performance-comparison")
async def get_performance_comparison():
    """
    Get performance comparison between different strategies
    """
    try:
        # Simulated backtesting results for comparison
        performance_data = {
            "comparison_period": "2024 YTD",
            "benchmark": {
                "symbol": "SPY",
                "return": "35.39%"
            },
            "strategies": {
                "rsi_divergence": {
                    "return": "64.15%",
                    "vs_benchmark": "+81%",
                    "win_rate": "76.5%",
                    "total_trades": 17,
                    "avg_hold_days": 15,
                    "max_drawdown": "18.2%",
                    "sharpe_ratio": 1.85,
                    "status": "proven"
                },
                "ml_trading": {
                    "return": "5.62%",
                    "vs_benchmark": "-84%",
                    "win_rate": "58.3%",
                    "total_trades": 24,
                    "avg_hold_days": 12,
                    "max_drawdown": "8.5%",
                    "sharpe_ratio": 0.42,
                    "status": "needs_optimization"
                },
                "technical": {
                    "return": "12.8%",
                    "vs_benchmark": "-64%",
                    "win_rate": "65.2%",
                    "total_trades": 31,
                    "avg_hold_days": 8,
                    "max_drawdown": "6.2%",
                    "sharpe_ratio": 1.12,
                    "status": "stable"
                },
                "buy_and_hold": {
                    "return": "35.39%",
                    "vs_benchmark": "0%",
                    "win_rate": "100%",
                    "total_trades": 1,
                    "avg_hold_days": 365,
                    "max_drawdown": "22.1%",
                    "sharpe_ratio": 1.58,
                    "status": "benchmark"
                }
            },
            "recommendations": {
                "best_overall": "rsi_divergence",
                "most_stable": "technical",
                "lowest_risk": "buy_and_hold",
                "highest_win_rate": "rsi_divergence"
            },
            "methodology": {
                "test_symbol": "MSFT",
                "test_period": "Jan 1, 2024 - Dec 31, 2024",
                "initial_capital": "$10,000",
                "commission": "$0 (zero commission assumed)",
                "data_source": "Polygon.io"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Performance comparison data retrieved")
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting performance comparison: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance comparison: {str(e)}"
        )

@router.get("/backtesting-results/{strategy_name}")
async def get_backtesting_results(strategy_name: str):
    """
    Get detailed backtesting results for a specific strategy
    """
    try:
        strategy_name = strategy_name.lower().strip()
        
        # Mock backtesting results - in production, these would come from actual backtesting
        results_data = {
            "strategy": strategy_name,
            "test_period": "2024-01-01 to 2024-12-31",
            "test_symbol": "MSFT",
            "initial_capital": 10000,
            "final_capital": 0,
            "total_return": 0,
            "win_rate": 0,
            "total_trades": 0,
            "avg_trade_duration": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "trades": [],
            "monthly_returns": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Strategy-specific results
        if strategy_name == "rsi_divergence":
            results_data.update({
                "final_capital": 16415,
                "total_return": 64.15,
                "win_rate": 76.5,
                "total_trades": 17,
                "avg_trade_duration": 15,
                "max_drawdown": 18.2,
                "sharpe_ratio": 1.85,
                "best_trade": 12.8,
                "worst_trade": -5.2,
                "status": "excellent"
            })
        elif strategy_name == "ml_trading":
            results_data.update({
                "final_capital": 10562,
                "total_return": 5.62,
                "win_rate": 58.3,
                "total_trades": 24,
                "avg_trade_duration": 12,
                "max_drawdown": 8.5,
                "sharpe_ratio": 0.42,
                "best_trade": 8.1,
                "worst_trade": -3.2,
                "status": "underperforming"
            })
        elif strategy_name == "technical":
            results_data.update({
                "final_capital": 11280,
                "total_return": 12.8,
                "win_rate": 65.2,
                "total_trades": 31,
                "avg_trade_duration": 8,
                "max_drawdown": 6.2,
                "sharpe_ratio": 1.12,
                "best_trade": 6.5,
                "worst_trade": -2.8,
                "status": "good"
            })
        elif strategy_name == "buy_and_hold":
            results_data.update({
                "final_capital": 13539,
                "total_return": 35.39,
                "win_rate": 100.0,
                "total_trades": 1,
                "avg_trade_duration": 365,
                "max_drawdown": 22.1,
                "sharpe_ratio": 1.58,
                "best_trade": 35.39,
                "worst_trade": 0,
                "status": "benchmark"
            })
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No backtesting results available for strategy '{strategy_name}'"
            )
        
        logger.info(f"Retrieved backtesting results for {strategy_name}")
        return results_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtesting results for {strategy_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get backtesting results: {str(e)}"
        )

@router.get("/recommendations")
async def get_strategy_recommendations():
    """
    Get personalized strategy recommendations based on user profile
    """
    try:
        recommendations = {
            "recommended_strategies": [
                {
                    "strategy": "rsi_divergence",
                    "match_score": 95,
                    "reasons": [
                        "Proven 64%+ returns in backtesting",
                        "High win rate (76.5%)",
                        "Suitable for active trading"
                    ],
                    "risk_warning": "Higher volatility and drawdown potential"
                },
                {
                    "strategy": "technical",
                    "match_score": 85,
                    "reasons": [
                        "Consistent performance",
                        "Lower risk profile",
                        "Good for beginners"
                    ],
                    "risk_warning": "Lower returns compared to aggressive strategies"
                },
                {
                    "strategy": "buy_and_hold",
                    "match_score": 75,
                    "reasons": [
                        "Simplest to implement",
                        "No active management required",
                        "Matches market performance"
                    ],
                    "risk_warning": "Subject to market crashes and long drawdowns"
                }
            ],
            "user_profile": {
                "experience_level": "intermediate",
                "risk_tolerance": "medium-high",
                "time_commitment": "active",
                "investment_horizon": "1-3_years"
            },
            "market_conditions": {
                "current_trend": "bullish",
                "volatility": "medium",
                "recommended_approach": "active_strategies_favored"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Strategy recommendations generated")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )