"""
Main Backtesting Engine

Integrates with the ultimate ML models (GRU with 49% MAE improvement)
for realistic trading simulation with transaction costs and risk management.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt

from .portfolio import Portfolio
from .strategies import TradingStrategy, MLTradingStrategy
from .metrics import PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionBacktester:
    """
    Production-grade backtesting engine for ML trading strategies.
    
    Features:
    - Realistic transaction costs
    - Position sizing and risk management
    - Stop loss and take profit
    - Performance analytics
    - Integration with ML models
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.2,
        commission_per_share: float = 0.01,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage of trade value
            max_position_size: Maximum position size as percentage of portfolio
            commission_per_share: Fixed commission per share
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.commission_per_share = commission_per_share
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Initialize components
        self.portfolio = Portfolio(initial_capital)
        self.strategy = None
        self.performance = PerformanceMetrics()
        
        # Results tracking
        self.trade_history = []
        self.portfolio_history = []
        self.daily_returns = []
        self.signals_history = []
        
        # Risk management
        self.position_entry_prices = {}
        
    def set_strategy(self, strategy: TradingStrategy):
        """Set the trading strategy to use"""
        self.strategy = strategy
        logger.info(f"Strategy set: {strategy.__class__.__name__}")
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        benchmark_symbol: str = 'SPY'
    ) -> Dict:
        """
        Run comprehensive backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            start_date: Start date for backtesting (optional)
            end_date: End date for backtesting (optional)
            benchmark_symbol: Benchmark symbol for comparison
            
        Returns:
            Dictionary with comprehensive results
        """
        if self.strategy is None:
            raise ValueError("No trading strategy set. Use set_strategy() first.")
        
        logger.info("Starting backtesting...")
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Total data points: {len(data)}")
        
        # Filter data by date range if specified
        if start_date or end_date:
            original_length = len(data)
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            logger.info(f"Filtered data: {original_length} -> {len(data)} points")
        
        # Initialize tracking
        self.portfolio.reset(self.initial_capital)
        self.strategy.reset()
        self.trade_history.clear()
        self.portfolio_history.clear()
        self.daily_returns.clear()
        self.signals_history.clear()
        self.position_entry_prices.clear()
        
        # Track benchmark performance
        benchmark_start = data['close'].iloc[0]
        
        # Main backtesting loop
        total_signals = 0
        executed_trades = 0
        
        for i, (date, row) in enumerate(data.iterrows()):
            # Skip first few days for technical indicators to stabilize
            if i < 20:
                self._record_portfolio_state(date, row)
                continue
            
            # Get historical data window for strategy
            historical_window = data.iloc[max(0, i-50):i+1]
            
            # Generate trading signal
            signal = self.strategy.generate_signal(row, historical_window)
            total_signals += 1
            
            # Record signal
            self.signals_history.append({
                'date': date,
                'signal': signal,
                'price': row['close']
            })
            
            # Check stop loss and take profit first
            self._check_risk_management(row, date)
            
            # Execute trades based on signal
            if signal['action'] in ['BUY', 'SELL']:
                if self._execute_signal(signal, row, date):
                    executed_trades += 1
            
            # Record portfolio state
            self._record_portfolio_state(date, row)
        
        # Calculate final results
        results = self._calculate_results(data, benchmark_start)
        
        logger.info(f"Backtesting completed!")
        logger.info(f"Total signals generated: {total_signals}")
        logger.info(f"Trades executed: {executed_trades}")
        logger.info(f"Total return: {results['total_return']:.2%}")
        logger.info(f"Benchmark return: {results['benchmark_return']:.2%}")
        logger.info(f"Alpha: {results['alpha']:.2%}")
        
        return results
    
    def _execute_signal(self, signal: Dict, row: pd.Series, date: pd.Timestamp) -> bool:
        """Execute a trading signal with risk management"""
        action = signal['action']
        symbol = signal.get('symbol', 'STOCK')
        confidence = signal.get('confidence', 1.0)
        price = row['close']
        
        trade_executed = False
        
        if action == 'BUY' and not self.portfolio.has_position(symbol):
            # Calculate position size based on confidence and risk management
            shares = self._calculate_position_size(price, confidence)
            
            if shares > 0:
                if self.portfolio.buy_stock(symbol, shares, price, self.transaction_cost):
                    self.position_entry_prices[symbol] = price
                    
                    self.trade_history.append({
                        'date': date,
                        'action': 'BUY',
                        'symbol': symbol,
                        'shares': shares,
                        'price': price,
                        'confidence': confidence,
                        'reasoning': signal.get('reasoning', []),
                        'portfolio_value': self.portfolio.get_total_value(price, symbol)
                    })
                    
                    logger.info(f"BUY: {shares} shares of {symbol} at ${price:.2f} (confidence: {confidence:.2f})")
                    trade_executed = True
        
        elif action == 'SELL' and self.portfolio.has_position(symbol):
            position = self.portfolio.get_position_info(symbol)
            if position:
                shares = position['shares']
                if self.portfolio.sell_stock(symbol, shares, price, self.transaction_cost):
                    # Remove from entry prices tracking
                    if symbol in self.position_entry_prices:
                        del self.position_entry_prices[symbol]
                    
                    self.trade_history.append({
                        'date': date,
                        'action': 'SELL',
                        'symbol': symbol,
                        'shares': shares,
                        'price': price,
                        'confidence': confidence,
                        'reasoning': signal.get('reasoning', []),
                        'portfolio_value': self.portfolio.get_total_value(price, symbol)
                    })
                    
                    logger.info(f"SELL: {shares} shares of {symbol} at ${price:.2f} (confidence: {confidence:.2f})")
                    trade_executed = True
        
        return trade_executed
    
    def _calculate_position_size(self, price: float, confidence: float) -> int:
        """Calculate position size based on risk management rules"""
        # Base position size on portfolio value and confidence
        portfolio_value = self.portfolio.get_total_value(price)
        max_investment = portfolio_value * self.max_position_size * confidence
        
        # Consider available cash
        available_cash = self.portfolio.get_available_cash()
        max_investment = min(max_investment, available_cash * 0.95)  # Leave some cash buffer
        
        # Calculate shares (accounting for transaction costs)
        effective_price = price * (1 + self.transaction_cost)
        shares = int(max_investment / effective_price)
        
        return max(0, shares)
    
    def _check_risk_management(self, row: pd.Series, date: pd.Timestamp):
        """Check stop loss and take profit conditions"""
        current_price = row['close']
        symbol = 'STOCK'  # Default symbol
        
        if symbol in self.position_entry_prices and self.portfolio.has_position(symbol):
            entry_price = self.position_entry_prices[symbol]
            price_change = (current_price - entry_price) / entry_price
            
            # Check stop loss
            if price_change <= -self.stop_loss_pct:
                position = self.portfolio.get_position_info(symbol)
                if position:
                    shares = position['shares']
                    if self.portfolio.sell_stock(symbol, shares, current_price, self.transaction_cost):
                        self.trade_history.append({
                            'date': date,
                            'action': 'SELL',
                            'symbol': symbol,
                            'shares': shares,
                            'price': current_price,
                            'confidence': 1.0,
                            'reasoning': [f'Stop loss triggered at {price_change:.2%}'],
                            'portfolio_value': self.portfolio.get_total_value(current_price, symbol)
                        })
                        
                        del self.position_entry_prices[symbol]
                        logger.info(f"STOP LOSS: Sold {shares} shares at ${current_price:.2f} ({price_change:.2%})")
            
            # Check take profit
            elif price_change >= self.take_profit_pct:
                position = self.portfolio.get_position_info(symbol)
                if position:
                    shares = position['shares']
                    if self.portfolio.sell_stock(symbol, shares, current_price, self.transaction_cost):
                        self.trade_history.append({
                            'date': date,
                            'action': 'SELL',
                            'symbol': symbol,
                            'shares': shares,
                            'price': current_price,
                            'confidence': 1.0,
                            'reasoning': [f'Take profit triggered at {price_change:.2%}'],
                            'portfolio_value': self.portfolio.get_total_value(current_price, symbol)
                        })
                        
                        del self.position_entry_prices[symbol]
                        logger.info(f"TAKE PROFIT: Sold {shares} shares at ${current_price:.2f} ({price_change:.2%})")
    
    def _record_portfolio_state(self, date: pd.Timestamp, row: pd.Series):
        """Record current portfolio state"""
        current_price = row['close']
        portfolio_value = self.portfolio.get_total_value(current_price)
        
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.portfolio.cash,
            'positions_value': portfolio_value - self.portfolio.cash,
            'price': current_price
        })
        
        # Calculate daily return
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def _calculate_results(self, data: pd.DataFrame, benchmark_start: float) -> Dict:
        """Calculate comprehensive backtesting results"""
        if not self.portfolio_history:
            return {'error': 'No portfolio history recorded'}
        
        # Basic returns
        initial_value = self.initial_capital
        final_value = self.portfolio_history[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value
        
        # Benchmark comparison (buy and hold)
        benchmark_end = data['close'].iloc[-1]
        benchmark_return = (benchmark_end - benchmark_start) / benchmark_start
        alpha = total_return - benchmark_return
        
        # Performance metrics
        metrics = self.performance.calculate_metrics(
            self.daily_returns,
            self.portfolio_history,
            self.trade_history
        )
        
        # Trading statistics
        buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        
        # Calculate additional statistics
        trading_days = len(self.portfolio_history)
        if trading_days > 0:
            average_daily_return = np.mean(self.daily_returns) if self.daily_returns else 0
            hit_rate = len([r for r in self.daily_returns if r > 0]) / len(self.daily_returns) if self.daily_returns else 0
        else:
            average_daily_return = 0
            hit_rate = 0
        
        # Combine results
        results = {
            # Basic performance
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'alpha': alpha,
            'trading_days': trading_days,
            
            # Trading statistics
            'total_trades': len(self.trade_history),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'average_daily_return': average_daily_return,
            'hit_rate': hit_rate,
            
            # Risk management
            'stop_losses_triggered': len([t for t in self.trade_history if 'Stop loss' in str(t.get('reasoning', []))]),
            'take_profits_triggered': len([t for t in self.trade_history if 'Take profit' in str(t.get('reasoning', []))]),
            
            # Performance metrics
            **metrics
        }
        
        return results
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        return pd.DataFrame(self.trade_history)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        return pd.DataFrame(self.portfolio_history)
    
    def get_signals_history(self) -> pd.DataFrame:
        """Get signals history as DataFrame"""
        return pd.DataFrame(self.signals_history)
    
    def create_performance_visualization(self, output_path: str = None) -> str:
        """Create comprehensive performance visualization"""
        if not self.portfolio_history:
            raise ValueError("No backtesting results to visualize")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtesting Performance Analysis', fontsize=16, fontweight='bold')
        
        # Convert to DataFrames
        portfolio_df = self.get_portfolio_history()
        
        # Plot 1: Portfolio Value Over Time
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df['date'], portfolio_df['portfolio_value'], 
                 label='Portfolio Value', linewidth=2, color='blue')
        ax1.plot(portfolio_df['date'], portfolio_df['price'] * (self.initial_capital / portfolio_df['price'].iloc[0]), 
                 label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)
        ax1.set_title('Portfolio Performance')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trade Signals
        ax2 = axes[0, 1]
        if self.trade_history:
            trades_df = self.get_trade_history()
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            ax2.plot(portfolio_df['date'], portfolio_df['price'], color='black', alpha=0.6, linewidth=1)
            
            if not buy_trades.empty:
                ax2.scatter(buy_trades['date'], buy_trades['price'], 
                           color='green', marker='^', s=100, label='Buy', alpha=0.8)
            if not sell_trades.empty:
                ax2.scatter(sell_trades['date'], sell_trades['price'], 
                           color='red', marker='v', s=100, label='Sell', alpha=0.8)
            
            ax2.set_title('Trading Signals')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Daily Returns Distribution
        ax3 = axes[1, 0]
        if self.daily_returns:
            ax3.hist(self.daily_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(np.mean(self.daily_returns), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(self.daily_returns):.4f}')
            ax3.set_title('Daily Returns Distribution')
            ax3.set_xlabel('Daily Return')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative Returns
        ax4 = axes[1, 1]
        if self.daily_returns:
            cumulative_returns = np.cumprod(1 + np.array(self.daily_returns)) - 1
            ax4.plot(portfolio_df['date'][1:], cumulative_returns, 
                     color='green', linewidth=2, label='Strategy')
            
            # Compare with benchmark
            price_returns = portfolio_df['price'].pct_change().dropna()
            benchmark_cumulative = np.cumprod(1 + price_returns) - 1
            ax4.plot(portfolio_df['date'][1:], benchmark_cumulative, 
                     color='gray', linewidth=2, alpha=0.7, label='Benchmark')
            
            ax4.set_title('Cumulative Returns Comparison')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Cumulative Return')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return "Visualization displayed"
    
    def generate_performance_report(self, results: Dict) -> str:
        """Generate a comprehensive performance report"""
        report = f"""
=== BACKTESTING PERFORMANCE REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO PERFORMANCE:
• Initial Capital: ${results['initial_capital']:,.2f}
• Final Value: ${results['final_value']:,.2f}
• Total Return: {results['total_return']:.2%}
• Benchmark Return: {results['benchmark_return']:.2%}
• Alpha (Excess Return): {results['alpha']:.2%}

RISK METRICS:
• Volatility (Annual): {results.get('volatility', 0):.2%}
• Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
• Sortino Ratio: {results.get('sortino_ratio', 0):.2f}
• Maximum Drawdown: {results.get('max_drawdown', 0):.2%}
• Calmar Ratio: {results.get('calmar_ratio', 0):.2f}

TRADING STATISTICS:
• Total Trades: {results['total_trades']}
• Buy Trades: {results['buy_trades']}
• Sell Trades: {results['sell_trades']}
• Win Rate: {results.get('win_rate', 0):.2%}
• Profit Factor: {results.get('profit_factor', 0):.2f}
• Average Daily Return: {results['average_daily_return']:.4f}
• Hit Rate (Positive Days): {results['hit_rate']:.2%}

RISK MANAGEMENT:
• Stop Losses Triggered: {results['stop_losses_triggered']}
• Take Profits Triggered: {results['take_profits_triggered']}

ADDITIONAL METRICS:
• Trading Days: {results['trading_days']}
• VaR (95%): {results.get('var_95', 0):.4f}
• CVaR (95%): {results.get('cvar_95', 0):.4f}
• Best Trade: {results.get('best_trade', 0):.2%}
• Worst Trade: {results.get('worst_trade', 0):.2%}

=== END REPORT ===
        """
        
        return report.strip()