"""
COMPLETELY FIXED Performance Metrics for Backtesting

This replaces the broken src/backtesting/metrics.py with correct calculations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    FIXED comprehensive performance metrics calculator for backtesting results.
    
    The original had multiple calculation errors:
    - Wrong Sharpe ratio formula
    - Incorrect max drawdown calculation
    - Flawed VaR/CVaR calculations
    - Wrong trade statistics
    
    This version provides accurate financial metrics.
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
        FIXED comprehensive performance metrics calculation.
        
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
        total_return = self._calculate_total_return_fixed(portfolio_values)
        annualized_return = self._calculate_annualized_return_fixed(returns_array)
        
        # Risk metrics - FIXED
        volatility = self._calculate_volatility_fixed(returns_array)
        sharpe_ratio = self._calculate_sharpe_ratio_fixed(returns_array)
        max_drawdown = self._calculate_max_drawdown_fixed(portfolio_values)
        
        # Trading metrics - FIXED
        trade_stats = self._calculate_trade_statistics_fixed(trade_history)
        
        # Risk-adjusted metrics - FIXED
        calmar_ratio = self._calculate_calmar_ratio_fixed(annualized_return, max_drawdown)
        sortino_ratio = self._calculate_sortino_ratio_fixed(returns_array)
        
        # Risk metrics - FIXED
        var_95 = self._calculate_var_fixed(returns_array, 0.05)
        cvar_95 = self._calculate_cvar_fixed(returns_array, 0.05)
        
        return {
            # Return metrics
            'total_return': total_return,
            'annualized_return': annualized_return,
            'daily_return_mean': np.mean(returns_array) if len(returns_array) > 0 else 0,
            'daily_return_std': np.std(returns_array) if len(returns_array) > 0 else 0,
            
            # Risk metrics
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            
            # Trading metrics
            'total_trades': len(trade_history),
            'win_rate': trade_stats.get('win_rate', 0),
            'profit_factor': trade_stats.get('profit_factor', 0),
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
            'max_drawdown', 'win_rate', 'profit_factor', 'total_trades',
            'sortino_ratio', 'calmar_ratio', 'var_95', 'cvar_95'
        ]}
    
    def _calculate_total_return_fixed(self, portfolio_values: List[float]) -> float:
        """FIXED total return calculation"""
        if len(portfolio_values) < 2:
            return 0.0
        return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    def _calculate_annualized_return_fixed(self, returns: np.ndarray) -> float:
        """FIXED annualized return calculation"""
        if len(returns) == 0:
            return 0.0
        
        # Assume daily returns, 252 trading days per year
        trading_days_per_year = 252
        total_return = np.prod(1 + returns) - 1
        years = len(returns) / trading_days_per_year
        
        if years <= 0:
            return 0.0
        
        try:
            annualized = (1 + total_return) ** (1 / years) - 1
            return annualized
        except (OverflowError, ValueError):
            # Handle extreme values
            return np.mean(returns) * trading_days_per_year
    
    def _calculate_volatility_fixed(self, returns: np.ndarray) -> float:
        """FIXED volatility calculation (annualized)"""
        if len(returns) < 2:
            return 0.0
        return np.std(returns, ddof=1) * np.sqrt(252)  # Annualized with sample std
    
    def _calculate_sharpe_ratio_fixed(self, returns: np.ndarray) -> float:
        """FIXED Sharpe ratio calculation"""
        if len(returns) == 0:
            return 0.0
        
        # Calculate annualized return and volatility
        annualized_return = np.mean(returns) * 252
        volatility = self._calculate_volatility_fixed(returns)
        
        if volatility == 0:
            return 0.0
        
        # Excess return over risk-free rate
        excess_return = annualized_return - self.risk_free_rate
        return excess_return / volatility
    
    def _calculate_sortino_ratio_fixed(self, returns: np.ndarray) -> float:
        """FIXED Sortino ratio calculation (downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(252)
        annualized_return = np.mean(returns) * 252
        excess_return = annualized_return - self.risk_free_rate
        
        return excess_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_max_drawdown_fixed(self, portfolio_values: List[float]) -> float:
        """FIXED maximum drawdown calculation"""
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        
        # Calculate running maximum (peak)
        peak = np.maximum.accumulate(values)
        
        # Calculate drawdown from peak
        drawdown = (values - peak) / peak
        
        # Return absolute maximum drawdown
        return abs(np.min(drawdown))
    
    def _calculate_calmar_ratio_fixed(self, annualized_return: float, max_drawdown: float) -> float:
        """FIXED Calmar ratio calculation"""
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        return annualized_return / max_drawdown
    
    def _calculate_var_fixed(self, returns: np.ndarray, confidence_level: float) -> float:
        """FIXED Value at Risk calculation"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    def _calculate_cvar_fixed(self, returns: np.ndarray, confidence_level: float) -> float:
        """FIXED Conditional Value at Risk (Expected Shortfall) calculation"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var_fixed(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
    
    def _calculate_trade_statistics_fixed(self, trade_history: List[Dict]) -> Dict:
        """FIXED detailed trade statistics calculation"""
        if not trade_history:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        # Group trades into round trips (buy-sell pairs)
        round_trip_returns = []
        open_positions = {}
        
        for trade in trade_history:
            symbol = trade['symbol']
            action = trade['action']
            price = trade['price']
            shares = trade['shares']
            
            if action == 'BUY':
                if symbol not in open_positions:
                    open_positions[symbol] = []
                open_positions[symbol].append({
                    'price': price,
                    'shares': shares,
                    'date': trade['date']
                })
            
            elif action == 'SELL' and symbol in open_positions and open_positions[symbol]:
                # Calculate return for this trade
                shares_to_sell = shares
                total_cost = 0
                total_shares = 0
                
                while shares_to_sell > 0 and open_positions[symbol]:
                    position = open_positions[symbol][0]
                    shares_from_position = min(shares_to_sell, position['shares'])
                    
                    # Calculate cost basis for these shares
                    cost_basis = shares_from_position * position['price']
                    total_cost += cost_basis
                    total_shares += shares_from_position
                    
                    # Update or remove position
                    position['shares'] -= shares_from_position
                    if position['shares'] == 0:
                        open_positions[symbol].pop(0)
                    
                    shares_to_sell -= shares_from_position
                
                if total_shares > 0:
                    # Calculate return for this round trip
                    avg_cost = total_cost / total_shares
                    trade_return = (price - avg_cost) / avg_cost
                    round_trip_returns.append(trade_return)
        
        if not round_trip_returns:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        # Calculate statistics
        returns_array = np.array(round_trip_returns)
        winning_trades = returns_array[returns_array > 0]
        losing_trades = returns_array[returns_array < 0]
        
        win_rate = len(winning_trades) / len(returns_array) if len(returns_array) > 0 else 0
        
        total_gains = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        total_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': np.mean(returns_array),
            'best_trade': np.max(returns_array),
            'worst_trade': np.min(returns_array)
        }
    
    def generate_performance_report(self, metrics: Dict) -> str:
        """Generate a formatted performance report"""
        report = f"""
=== FIXED BACKTESTING PERFORMANCE REPORT ===

RETURN METRICS:
• Total Return: {metrics.get('total_return', 0):.2%}
• Annualized Return: {metrics.get('annualized_return', 0):.2%}
• Daily Return (Mean): {metrics.get('daily_return_mean', 0):.4f}
• Daily Return (Std): {metrics.get('daily_return_std', 0):.4f}

RISK METRICS:
• Volatility (Annual): {metrics.get('volatility', 0):.2%}
• Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
• Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
• Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}
• Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}
• VaR (95%): {metrics.get('var_95', 0):.2%}
• CVaR (95%): {metrics.get('cvar_95', 0):.2%}

TRADING METRICS:
• Total Trades: {metrics.get('total_trades', 0)}
• Win Rate: {metrics.get('win_rate', 0):.2%}
• Profit Factor: {metrics.get('profit_factor', 0):.2f}
• Average Trade Return: {metrics.get('avg_trade_return', 0):.2%}
• Best Trade: {metrics.get('best_trade', 0):.2%}
• Worst Trade: {metrics.get('worst_trade', 0):.2%}

PORTFOLIO METRICS:
• Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}
• Maximum Portfolio Value: ${metrics.get('max_portfolio_value', 0):,.2f}
• Minimum Portfolio Value: ${metrics.get('min_portfolio_value', 0):,.2f}
        """
        
        return report.strip()
    
    def calculate_rolling_metrics(
        self,
        daily_returns: List[float],
        window: int = 252
    ) -> Dict[str, List[float]]:
        """Calculate rolling performance metrics"""
        if len(daily_returns) < window:
            return {}
        
        returns_array = np.array(daily_returns)
        rolling_sharpe = []
        rolling_volatility = []
        rolling_return = []
        
        for i in range(window, len(returns_array)):
            window_returns = returns_array[i-window:i]
            
            # Rolling annualized return
            rolling_ann_return = np.mean(window_returns) * 252
            rolling_return.append(rolling_ann_return)
            
            # Rolling volatility
            rolling_vol = np.std(window_returns, ddof=1) * np.sqrt(252)
            rolling_volatility.append(rolling_vol)
            
            # Rolling Sharpe ratio
            if rolling_vol > 0:
                rolling_sharpe.append((rolling_ann_return - self.risk_free_rate) / rolling_vol)
            else:
                rolling_sharpe.append(0)
        
        return {
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_volatility,
            'rolling_sharpe': rolling_sharpe
        }
    