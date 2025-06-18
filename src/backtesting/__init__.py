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
    TradingStrategy,
    RSIDivergenceStrategy,      # NEW: Add RSI Divergence Strategy
    HybridRSIDivergenceStrategy
)
from .portfolio import Portfolio
from .portfolio_manager import (
    PortfolioManager,
    UltimatePortfolioRunner,
    create_default_strategy_config,
    create_multi_asset_strategy_config,
)
from .metrics import PerformanceMetrics

__all__ = [
    'ProductionBacktester',
    'MLTradingStrategy',
    'TechnicalAnalysisStrategy', 
    'BuyAndHoldStrategy',
    'TradingStrategy',
    'Portfolio',
    'PerformanceMetrics',
    'RSIDivergenceStrategy',      # NEW
    'HybridRSIDivergenceStrategy',
    'PortfolioManager',
    'UltimatePortfolioRunner',
    'create_default_strategy_config',
    'create_multi_asset_strategy_config',
]

# Version info
__version__ = "2.0.0-fixed"
__author__ = "Day Trading Bot Team"
__description__ = "Fixed backtesting framework with correct calculations"
