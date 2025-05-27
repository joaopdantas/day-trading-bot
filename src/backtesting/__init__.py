"""
Backtesting Module for Day Trading Bot.

This module provides comprehensive backtesting capabilities for trading strategies
using the optimized ML models (49% MAE improvement achieved).
"""

from .backtester import ProductionBacktester
from .strategies import MLTradingStrategy, TechnicalAnalysisStrategy, BuyAndHoldStrategy
from .portfolio import Portfolio
from .metrics import PerformanceMetrics

__all__ = [
    'ProductionBacktester',
    'MLTradingStrategy',
    'TechnicalAnalysisStrategy', 
    'BuyAndHoldStrategy',
    'Portfolio',
    'PerformanceMetrics'
]