"""
COMPLETELY FIXED Backtesting Engine
Fixed portfolio value calculations and signal execution
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
from .strategies import TradingStrategy, BuyAndHoldStrategy
from .metrics import PerformanceMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionBacktester:
    """COMPLETELY FIXED backtesting engine with corrected calculations"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.3,  # Increased for more meaningful trades
        commission_per_share: float = 0.01,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ):
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
        
        print(f"🔧 Fixed Backtester initialized with ${initial_capital:,.2f}")
        print(f"   Max position size: {max_position_size:.1%}")
        print(f"   Transaction cost: {transaction_cost:.3%}")
        
    def set_strategy(self, strategy: TradingStrategy):
        """Set the trading strategy to use"""
        self.strategy = strategy
        print(f"🎯 Strategy set: {strategy.__class__.__name__}")
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        benchmark_symbol: str = 'SPY'
    ) -> Dict:
        """Run FIXED comprehensive backtest"""
        
        if self.strategy is None:
            raise ValueError("No trading strategy set")
        
        print(f"\n🚀 Starting FIXED backtesting...")
        print(f"Data range: {data.index[0]} to {data.index[-1]}")
        print(f"Total data points: {len(data)}")
        
        # Filter data by date range
        if start_date or end_date:
            original_length = len(data)
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            print(f"Filtered: {original_length} -> {len(data)} points")
        
        # Initialize tracking
        self.portfolio.reset(self.initial_capital)
        self.strategy.reset()
        self.trade_history.clear()
        self.portfolio_history.clear()
        self.daily_returns.clear()
        self.signals_history.clear()
        self.position_entry_prices.clear()
        
        # FIXED: Calculate benchmark correctly
        benchmark_start_price = data['close'].iloc[0]
        benchmark_end_price = data['close'].iloc[-1]
        
        print(f"📊 Benchmark prices: ${benchmark_start_price:.2f} -> ${benchmark_end_price:.2f}")
        
        # Main backtesting loop
        total_signals = 0
        executed_trades = 0

        for i, (date, row) in enumerate(data.iterrows()):
            # CRITICAL FIX: Check if this is Buy & Hold strategy
            is_buy_and_hold = isinstance(self.strategy, BuyAndHoldStrategy)
            
            # Skip first few days for technical indicators UNLESS it's Buy & Hold
            if i < 5 and not is_buy_and_hold:
                self._record_portfolio_state(date, row)
                continue
            
            # For Buy & Hold: Allow immediate execution on first day
            if is_buy_and_hold and i == 0:
                # Generate signal immediately for Buy & Hold
                signal = self.strategy.generate_signal(row, data.iloc[0:1])
                total_signals += 1
                
                self.signals_history.append({
                    'date': date,
                    'signal': signal,
                    'price': row['close']
                })
                
                # Execute Buy & Hold signal immediately
                if signal['action'] == 'BUY':
                    execution_price = row.get('open', row['close'])
                    if self._execute_signal_fixed(signal, execution_price, date, row):
                        executed_trades += 1
                
                self._record_portfolio_state(date, row)
                continue
        
        # For other strategies: Get historical data window (no lookahead)
        historical_window = data.iloc[max(0, i-50):i]
        
        # Generate trading signal using PREVIOUS bar data (no lookahead bias)
        signal_row = data.iloc[i-1] if i > 0 else row
        signal = self.strategy.generate_signal(signal_row, historical_window)
        total_signals += 1
        
        # Record signal
        self.signals_history.append({
            'date': date,
            'signal': signal,
            'price': row['close']
        })
        
        # Check risk management BEFORE new signals (except for Buy & Hold)
        if not is_buy_and_hold:
            self._check_risk_management_fixed(row, date)
        else:
            # For Buy & Hold, still call risk management but it will be skipped
            self._check_risk_management_fixed(row, date)
        
        # Execute trades based on signal with proper timing
        if signal['action'] in ['BUY', 'SELL']:
            execution_price = row.get('open', row['close'])
            if self._execute_signal_fixed(signal, execution_price, date, row):
                executed_trades += 1
        
        # Record portfolio state
        self._record_portfolio_state(date, row)
            
            # Debug every 20 days
        if i % 20 == 0:
            portfolio_value = self.portfolio.get_total_value(current_price, 'STOCK')
            print(f"📅 Day {i}: ${current_price:.2f}, Portfolio: ${portfolio_value:.2f}")
        
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for signal_entry in self.signals_history:
            action = signal_entry['signal'].get('action', 'HOLD')
            signal_counts[action] = signal_counts.get(action, 0) + 1

        print(f"\n📊 Backtesting Summary:")
        print(f"   Total signals: {total_signals}")
        print(f"   BUY signals: {signal_counts['BUY']}")
        print(f"   SELL signals: {signal_counts['SELL']}")
        print(f"   HOLD signals: {signal_counts['HOLD']}")
        print(f"   Executed trades: {executed_trades}") 
        
        # Calculate results
        results = self._calculate_results_fixed(data, benchmark_start_price, benchmark_end_price)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Total return: {results['total_return']:.2%}")
        print(f"   Benchmark return: {results['benchmark_return']:.2%}")
        print(f"   Alpha: {results['alpha']:.2%}")
        
        return results
    
    def _execute_signal_fixed(self, signal: Dict, execution_price: float, date: pd.Timestamp, row: pd.Series) -> bool:
        """FIXED signal execution with proper position sizing"""
        
        action = signal['action']
        symbol = signal.get('symbol', 'STOCK')
        confidence = signal.get('confidence', 1.0)
        
        trade_executed = False
        current_price = row['close']
        
        print(f"🔄 Executing {action} signal at ${execution_price:.2f} (confidence: {confidence:.2f})")
        
        if action == 'BUY':
            # Check if we already have a position
            if not self.portfolio.has_position(symbol):
                # Calculate position size - FIXED
                shares = self._calculate_position_size_fixed(execution_price, confidence)
                
                print(f"   💡 Calculated {shares} shares for ${execution_price:.2f}")
                
                if shares > 0:
                    if self.portfolio.buy_stock(symbol, shares, execution_price, self.transaction_cost):
                        self.position_entry_prices[symbol] = execution_price
                        
                        self.trade_history.append({
                            'date': date,
                            'action': 'BUY',
                            'symbol': symbol,
                            'shares': shares,
                            'price': execution_price,
                            'confidence': confidence,
                            'reasoning': signal.get('reasoning', []),
                            'portfolio_value': self.portfolio.get_total_value(execution_price, symbol)
                        })
                        
                        print(f"✅ BUY EXECUTED: {shares} shares at ${execution_price:.2f}")
                        trade_executed = True
                else:
                    print(f"   ❌ No shares calculated - insufficient funds or constraints")
            else:
                print(f"   ⚠️ Already have position in {symbol}")
        
        elif action == 'SELL':
            if self.portfolio.has_position(symbol):
                position = self.portfolio.get_position_info(symbol)
                if position:
                    shares = position['shares']
                    if self.portfolio.sell_stock(symbol, shares, execution_price, self.transaction_cost):
                        # Remove from entry prices
                        if symbol in self.position_entry_prices:
                            del self.position_entry_prices[symbol]
                        
                        self.trade_history.append({
                            'date': date,
                            'action': 'SELL',
                            'symbol': symbol,
                            'shares': shares,
                            'price': execution_price,
                            'confidence': confidence,
                            'reasoning': signal.get('reasoning', []),
                            'portfolio_value': self.portfolio.get_total_value(execution_price, symbol)
                        })
                        
                        print(f"✅ SELL EXECUTED: {shares} shares at ${execution_price:.2f}")
                        trade_executed = True
            else:
                print(f"   ⚠️ No position to sell in {symbol}")
        
        return trade_executed
    
    def _calculate_position_size_fixed(self, price: float, confidence: float) -> int:
        """FIXED position size calculation with Buy & Hold special case"""
        
        portfolio_value = self.portfolio.get_total_value(price, 'STOCK')
        available_cash = self.portfolio.get_available_cash()
        
        print(f"   📊 Portfolio value: ${portfolio_value:.2f}, Available cash: ${available_cash:.2f}")
        
        # SPECIAL CASE: Buy & Hold should invest ALL available cash
        if isinstance(self.strategy, BuyAndHoldStrategy):
            max_investment = available_cash * 0.99  # Use 99% to account for fees
            print(f"   💎 Buy & Hold: Investing ALL available cash: ${max_investment:.2f}")
        else:
            # Regular strategies: use position sizing
            max_investment_by_portfolio = portfolio_value * self.max_position_size * confidence
            max_investment_by_cash = available_cash * 0.95  # Leave 5% buffer
            max_investment = min(max_investment_by_portfolio, max_investment_by_cash)
        
        print(f"   💰 Max investment: ${max_investment:.2f}")
        
        if max_investment <= 100:  # Minimum investment threshold
            print(f"   ❌ Investment too small: ${max_investment:.2f}")
            return 0
        
        # Calculate shares accounting for transaction costs
        effective_price = price * (1 + self.transaction_cost)
        shares = int(max_investment / effective_price)
        
        print(f"   🧮 Effective price: ${effective_price:.2f}, Shares: {shares}")
        
        return max(0, shares)
    
    def _check_risk_management_fixed(self, row: pd.Series, date: pd.Timestamp):
        """FIXED risk management with Buy & Hold exception"""
        current_price = row['close']
        symbol = 'STOCK'  # Default symbol
        
        # CRITICAL FIX: Skip risk management for Buy & Hold
        # Check if we're in a Buy & Hold strategy by looking at recent signals
        if hasattr(self, 'signals_history') and self.signals_history:
            recent_signal = self.signals_history[-1]['signal']
            if recent_signal.get('buy_and_hold', False):
                # This is Buy & Hold - NEVER trigger stop loss or take profit
                logger.info(f"💎 Buy & Hold: Skipping risk management at ${current_price:.2f}")
                return
        
        # Regular risk management for other strategies
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
                        logger.info(f"🛑 STOP LOSS: Sold {shares} shares at ${current_price:.2f} ({price_change:.2%})")
            
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
                        logger.info(f"🎯 TAKE PROFIT: Sold {shares} shares at ${current_price:.2f} ({price_change:.2%})")
    
    def _record_portfolio_state(self, date: pd.Timestamp, row: pd.Series):
        """Record current portfolio state with FIXED calculations"""
        
        current_price = row['close']
        portfolio_value = self.portfolio.get_total_value(current_price, 'STOCK')
        
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
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
    
    def _calculate_results_fixed(self, data: pd.DataFrame, benchmark_start: float, benchmark_end: float) -> Dict:
        """FIXED comprehensive results calculation"""
        
        if not self.portfolio_history:
            return {'error': 'No portfolio history recorded'}
        
        # Basic returns
        initial_value = self.initial_capital
        final_value = self.portfolio_history[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value
        
        # FIXED: Benchmark calculation (buy and hold the stock)
        benchmark_return = (benchmark_end - benchmark_start) / benchmark_start
        alpha = total_return - benchmark_return
        
        print(f"📊 Return Calculation:")
        print(f"   Initial: ${initial_value:.2f}")
        print(f"   Final: ${final_value:.2f}")
        print(f"   Strategy return: {total_return:.2%}")
        print(f"   Benchmark (${benchmark_start:.2f} -> ${benchmark_end:.2f}): {benchmark_return:.2%}")
        print(f"   Alpha: {alpha:.2%}")
        
        # Performance metrics
        metrics = self.performance.calculate_metrics(
            self.daily_returns,
            self.portfolio_history,
            self.trade_history
        )
        
        # Trading statistics
        buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        
        # Additional statistics
        trading_days = len(self.portfolio_history)
        if trading_days > 0 and self.daily_returns:
            average_daily_return = np.mean(self.daily_returns)
            hit_rate = len([r for r in self.daily_returns if r > 0]) / len(self.daily_returns)
        else:
            average_daily_return = 0
            hit_rate = 0
        
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
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        if not self.portfolio_history:
            return pd.DataFrame()
        return pd.DataFrame(self.portfolio_history)
    
    def get_signals_history(self) -> pd.DataFrame:
        """Get signals history as DataFrame"""
        if not self.signals_history:
            return pd.DataFrame()
        return pd.DataFrame(self.signals_history)
    
    def create_performance_visualization(self, output_path: str = None) -> str:
        """FIXED performance visualization"""
        if not self.portfolio_history:
            raise ValueError("No backtesting results to visualize")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('FIXED Backtesting Performance Analysis', fontsize=16, fontweight='bold')
        
        portfolio_df = self.get_portfolio_history()
        portfolio_df.set_index('date', inplace=True)
        
        # Plot 1: Portfolio Value Over Time - FIXED
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
                 label='Strategy Portfolio', linewidth=2, color='blue')
        
        # FIXED: Proper buy & hold calculation
        initial_price = portfolio_df['price'].iloc[0]
        shares_bought = self.initial_capital / initial_price
        buy_hold_values = shares_bought * portfolio_df['price']
        ax1.plot(portfolio_df.index, buy_hold_values, 
                 label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)
        
        ax1.set_title('Portfolio Performance (FIXED)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trading Signals
        ax2 = axes[0, 1]
        if self.trade_history:
            trades_df = self.get_trade_history()
            trades_df.set_index('date', inplace=True)
            
            ax2.plot(portfolio_df.index, portfolio_df['price'], color='black', alpha=0.6, linewidth=1)
            
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if not buy_trades.empty:
                ax2.scatter(buy_trades.index, buy_trades['price'], 
                           color='green', marker='^', s=100, label='Buy', alpha=0.8)
            if not sell_trades.empty:
                ax2.scatter(sell_trades.index, sell_trades['price'], 
                           color='red', marker='v', s=100, label='Sell', alpha=0.8)
            
            ax2.set_title('Trading Signals (FIXED)')
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
        
        # Plot 4: Cumulative Returns Comparison
        ax4 = axes[1, 1]
        if self.daily_returns:
            strategy_cumulative = np.cumprod(1 + np.array(self.daily_returns)) - 1
            ax4.plot(portfolio_df.index[1:len(strategy_cumulative)+1], strategy_cumulative, 
                     color='blue', linewidth=2, label='Strategy')
            
            # Benchmark cumulative returns
            price_returns = portfolio_df['price'].pct_change().dropna()
            benchmark_cumulative = np.cumprod(1 + price_returns) - 1
            ax4.plot(portfolio_df.index[1:len(benchmark_cumulative)+1], benchmark_cumulative, 
                     color='gray', linewidth=2, alpha=0.7, label='Buy & Hold')
            
            ax4.set_title('Cumulative Returns Comparison (FIXED)')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Cumulative Return')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return "Visualization displayed"
    
    def generate_performance_report(self, results: Dict) -> str:
        """Generate comprehensive performance report"""
        
        report = f"""
=== FIXED BACKTESTING PERFORMANCE REPORT ===
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