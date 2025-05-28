"""
FIXED Backtesting Module for Day Trading Bot.

This module provides comprehensive backtesting capabilities for trading strategies
using optimized ML models and corrected calculations.
"""

from .backtester import ProductionBacktester
from .strategies import (
    MLTradingStrategy, 
    TechnicalAnalysisStrategy, 
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TradingStrategy
)
from .portfolio import Portfolio
from .metrics import PerformanceMetrics

__all__ = [
    'ProductionBacktester',
    'MLTradingStrategy',
    'TechnicalAnalysisStrategy', 
    'BuyAndHoldStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'TradingStrategy',
    'Portfolio',
    'PerformanceMetrics'
]

# Version info
__version__ = "2.0.0-fixed"
__author__ = "Day Trading Bot Team"
__description__ = "Fixed backtesting framework with correct calculations"