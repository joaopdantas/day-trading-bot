"""
Performance Metrics for Backtesting

Calculates comprehensive performance metrics including risk-adjusted returns,
drawdowns, Sharpe ratio, and trading statistics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for backtesting results.
    
    Calculates risk metrics, return metrics, and trading statistics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(
        self,
        daily_returns: List[float],
        portfolio_history: List[Dict],
        trade_history: List[Dict]
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            daily_returns: List of daily return percentages
            portfolio_history: List of portfolio value snapshots
            trade_history: List of executed trades
            
        Returns:
            Dictionary of performance metrics
        """
        if not daily_returns or not portfolio_history:
            return self._empty_metrics()
        
        returns_array = np.array(daily_returns)
        portfolio_values = [p['portfolio_value'] for p in portfolio_history]
        
        # Basic return metrics
        total_return = self._calculate_total_return(portfolio_values)
        annualized_return = self._calculate_annualized_return(returns_array)
        
        # Risk metrics
        volatility = self._calculate_volatility(returns_array)
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array, volatility)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Trading metrics
        trade_stats = self._calculate_trade_statistics(trade_history)
        
        # Risk-adjusted metrics
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
        sortino_ratio = self._calculate_sortino_ratio(returns_array)
        
        # Additional metrics
        win_rate = trade_stats.get('win_rate', 0)
        profit_factor = trade_stats.get('profit_factor', 0)
        
        return {
            # Return metrics
            'total_return': total_return,
            'annualized_return': annualized_return,
            'daily_return_mean': np.mean(returns_array),
            'daily_return_std': np.std(returns_array),
            
            # Risk metrics
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': self._calculate_var(returns_array, 0.05),
            'cvar_95': self._calculate_cvar(returns_array, 0.05),
            
            # Trading metrics
            'total_trades': len(trade_history),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': trade_stats.get('avg_trade_return', 0),
            'best_trade': trade_stats.get('best_trade', 0),
            'worst_trade': trade_stats.get('worst_trade', 0),
            
            # Portfolio metrics
            'final_portfolio_value': portfolio_values[-1] if portfolio_values else 0,
            'max_portfolio_value': max(portfolio_values) if portfolio_values else 0,
            'min_portfolio_value': min(portfolio_values) if portfolio_values else 0,
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {metric: 0.0 for metric in [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'profit_factor', 'total_trades'
        ]}
    
    def _calculate_total_return(self, portfolio_values: List[float]) -> float:
        """Calculate total return over the period"""
        if len(portfolio_values) < 2:
            return 0.0
        return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        
        # Assume daily returns, 252 trading days per year
        trading_days = 252
        total_return = np.prod(1 + returns) - 1
        periods = len(returns) / trading_days
        
        if periods <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / periods) - 1
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        if volatility == 0 or len(returns) == 0:
            return 0.0
        
        excess_return = self._calculate_annualized_return(returns) - self.risk_free_rate
        return excess_return / volatility
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_deviation = np.std(negative_returns) * np.sqrt(252)
        excess_return = self._calculate_annualized_return(returns) - self.risk_free_rate
        
        return excess_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return abs(np.min(drawdown))
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        return annualized_return / max_drawdown
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def _calculate_trade_statistics(self, trade_history: List[Dict]) -> Dict:
        """Calculate detailed trade statistics"""
        if not trade_history:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        # Group trades by symbol to calculate P&L
        trades_by_symbol = {}
        for trade in trade_history:
            symbol = trade['symbol']
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Calculate trade returns
        trade_returns = []
        for symbol, trades in trades_by_symbol.items():
            position_value = 0
            shares_held = 0
            
            for trade in trades:
                if trade['action'] == 'BUY':
                    position_value += trade['shares'] * trade['price']
                    shares_held += trade['shares']
                elif trade['action'] == 'SELL' and shares_held > 0:
                    # Calculate return for this trade
                    avg_cost = position_value / shares_held if shares_held > 0 else 0
                    trade_return = (trade['price'] - avg_cost) / avg_cost if avg_cost > 0 else 0
                    trade_returns.append(trade_return)
                    
                    # Update position
                    shares_sold = min(trade['shares'], shares_held)
                    position_value -= shares_sold * avg_cost
                    shares_held -= shares_sold
        
        if not trade_returns:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        trade_returns = np.array(trade_returns)
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0
        
        total_gains = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        total_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': np.mean(trade_returns),
            'best_trade': np.max(trade_returns),
            'worst_trade': np.min(trade_returns)
        }
    
    def generate_performance_report(self, metrics: Dict) -> str:
        """Generate a formatted performance report"""
        report = """
=== BACKTESTING PERFORMANCE REPORT ===

RETURN METRICS:
• Total Return: {total_return:.2%}
• Annualized Return: {annualized_return:.2%}
• Daily Return (Mean): {daily_return_mean:.4f}
• Daily Return (Std): {daily_return_std:.4f}

RISK METRICS:
• Volatility (Annual): {volatility:.2%}
• Sharpe Ratio: {sharpe_ratio:.2f}
• Sortino Ratio: {sortino_ratio:.2f}
• Calmar Ratio: {calmar_ratio:.2f}
• Maximum Drawdown: {max_drawdown:.2%}
• VaR (95%): {var_95:.2%}
• CVaR (95%): {cvar_95:.2%}

TRADING METRICS:
• Total Trades: {total_trades}
• Win Rate: {win_rate:.2%}
• Profit Factor: {profit_factor:.2f}
• Average Trade Return: {avg_trade_return:.2%}
• Best Trade: {best_trade:.2%}
• Worst Trade: {worst_trade:.2%}

PORTFOLIO METRICS:
• Final Portfolio Value: ${final_portfolio_value:,.2f}
• Maximum Portfolio Value: ${max_portfolio_value:,.2f}
• Minimum Portfolio Value: ${min_portfolio_value:,.2f}
        """.format(**metrics)
        
        return report.strip()