"""
TIME HORIZON COMPARISON SCRIPT
Compares trading performance across different time horizons:
- Short: 50 days
- Medium: 5-6 months (150 days)
- Long: 1 year (252 days)

Performance and efficiency analysis across timeframes
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

try:
    from src.backtesting import MLTradingStrategy, TechnicalAnalysisStrategy, BuyAndHoldStrategy
    from src.data.fetcher import get_data_api
    from src.indicators.technical import TechnicalIndicators
    PROJECT_AVAILABLE = True
    print("✅ Your project modules loaded successfully")
except ImportError as e:
    print(f"❌ Project modules not available: {e}")
    PROJECT_AVAILABLE = False
    import yfinance as yf


class TimeHorizonOptimizedStrategy:
    """Base strategy that adapts to different time horizons"""
    
    def __init__(self, time_horizon="medium", name="Time Horizon Strategy"):
        self.time_horizon = time_horizon
        self.name = f"{name} ({time_horizon.title()}-term)"
        
        # Adapt parameters based on time horizon
        if time_horizon == "short":
            self.sma_short = 5
            self.sma_long = 15
            self.rsi_period = 7
            self.confidence_threshold = 0.2
            self.position_hold_target = 5  # Target hold period in days
            self.rebalance_frequency = 1  # Daily
            
        elif time_horizon == "medium":
            self.sma_short = 20
            self.sma_long = 50
            self.rsi_period = 14
            self.confidence_threshold = 0.3
            self.position_hold_target = 30  # Target hold period
            self.rebalance_frequency = 5  # Weekly
            
        else:  # long
            self.sma_short = 50
            self.sma_long = 200
            self.rsi_period = 21
            self.confidence_threshold = 0.4
            self.position_hold_target = 90  # Target hold period
            self.rebalance_frequency = 20  # Monthly
        
        self.days_in_position = 0
        self.last_signal_date = None
        
    def reset(self):
        """Reset strategy state"""
        self.days_in_position = 0
        self.last_signal_date = None
    
    def generate_signal(self, current_data, historical_data):
        """Generate signal optimized for specific time horizon"""
        
        min_data_needed = max(self.sma_long, self.rsi_period) + 10
        if len(historical_data) < min_data_needed:
            return {'action': 'HOLD', 'confidence': 0.2, 'reasoning': ['Insufficient data for time horizon']}
        
        # Calculate indicators with time horizon specific parameters
        sma_short = historical_data['close'].tail(self.sma_short).mean()
        sma_long = historical_data['close'].tail(self.sma_long).mean()
        current_price = current_data['close']
        
        # RSI with horizon-specific period
        rsi = current_data.get('rsi', 50)
        
        # MACD (adapt span based on time horizon)
        if self.time_horizon == "short":
            macd_fast, macd_slow = 6, 13
        elif self.time_horizon == "medium":
            macd_fast, macd_slow = 12, 26
        else:
            macd_fast, macd_slow = 19, 39
        
        # Calculate custom MACD
        ema_fast = historical_data['close'].ewm(span=macd_fast).mean().iloc[-1]
        ema_slow = historical_data['close'].ewm(span=macd_slow).mean().iloc[-1]
        macd_line = ema_fast - ema_slow
        
        # Time horizon specific signal generation
        confidence = 0.3
        action = 'HOLD'
        reasoning = []
        
        # Trend analysis
        trend_strength = (sma_short - sma_long) / sma_long
        price_position = (current_price - sma_long) / sma_long
        
        if self.time_horizon == "short":
            # Short-term: Focus on momentum and quick reversals
            if rsi < 35 and trend_strength > -0.02:  # Oversold but not in strong downtrend
                action = 'BUY'
                confidence = (40 - rsi) / 40 * 0.8
                reasoning.append(f'Short-term oversold RSI: {rsi:.1f}')
                
            elif rsi > 65 and trend_strength < 0.02:  # Overbought
                action = 'SELL'
                confidence = (rsi - 60) / 40 * 0.7
                reasoning.append(f'Short-term overbought RSI: {rsi:.1f}')
                
            elif abs(macd_line) > 0.01:  # MACD momentum
                if macd_line > 0 and price_position > 0:
                    action = 'BUY'
                    confidence = min(0.6, abs(macd_line) * 50)
                    reasoning.append('Short-term MACD momentum')
                    
        elif self.time_horizon == "medium":
            # Medium-term: Balance trend following with mean reversion
            if sma_short > sma_long and price_position > -0.05:  # Uptrend
                if rsi < 50:  # Not overbought
                    action = 'BUY'
                    confidence = min(0.7, trend_strength * 10 + 0.4)
                    reasoning.append('Medium-term uptrend with RSI confirmation')
                    
            elif sma_short < sma_long and price_position < 0.05:  # Downtrend
                if rsi > 50:  # Not oversold
                    action = 'SELL'
                    confidence = min(0.7, abs(trend_strength) * 10 + 0.4)
                    reasoning.append('Medium-term downtrend with RSI confirmation')
                    
        else:  # long-term
            # Long-term: Focus on major trend changes and fundamentals
            if sma_short > sma_long * 1.02 and price_position > 0:  # Strong uptrend
                action = 'BUY'
                confidence = min(0.8, trend_strength * 15 + 0.5)
                reasoning.append('Long-term strong uptrend established')
                
            elif sma_short < sma_long * 0.98 and price_position < 0:  # Strong downtrend
                action = 'SELL'
                confidence = min(0.8, abs(trend_strength) * 15 + 0.5)
                reasoning.append('Long-term strong downtrend established')
        
        # Position holding logic based on time horizon
        current_date = current_data.name if hasattr(current_data, 'name') else datetime.now()
        
        if self.last_signal_date:
            self.days_in_position = (current_date - self.last_signal_date).days
            
            # Don't change position too frequently for longer time horizons
            if action != 'HOLD' and self.days_in_position < self.position_hold_target:
                if self.time_horizon != "short":  # Short-term can trade more frequently
                    action = 'HOLD'
                    confidence = 0.3
                    reasoning = [f'Position held for only {self.days_in_position} days, target: {self.position_hold_target}']
        
        # Update tracking
        if action != 'HOLD':
            self.last_signal_date = current_date
            self.days_in_position = 0
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold and action != 'HOLD':
            action = 'HOLD'
            reasoning.append(f'Confidence {confidence:.2f} below threshold {self.confidence_threshold}')
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'time_horizon': self.time_horizon,
            'days_in_position': self.days_in_position,
            'trend_strength': trend_strength
        }


class TimeHorizonPortfolioSimulator:
    """Portfolio simulator optimized for time horizon analysis"""
    
    def __init__(self, initial_capital=10000, time_horizon="medium"):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.entry_price = 0
        self.time_horizon = time_horizon
        
        # Time horizon specific parameters
        if time_horizon == "short":
            self.transaction_cost = 0.001  # 0.1% per trade (higher for frequent trading)
            self.max_position_size = 0.8  # Can use more capital for short-term
        elif time_horizon == "medium":
            self.transaction_cost = 0.0005  # 0.05% per trade
            self.max_position_size = 0.6
        else:  # long
            self.transaction_cost = 0.0002  # 0.02% per trade (lower for infrequent trading)
            self.max_position_size = 0.5  # More conservative for long-term
        
        # Enhanced tracking
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        self.max_drawdown = 0
        self.peak_value = initial_capital
        self.total_transaction_costs = 0
        self.position_holding_periods = []
        
    def reset(self):
        """Reset portfolio state"""
        self.cash = self.initial_capital
        self.shares = 0
        self.entry_price = 0
        self.trades.clear()
        self.portfolio_history.clear()
        self.daily_returns.clear()
        self.max_drawdown = 0
        self.peak_value = self.initial_capital
        self.total_transaction_costs = 0
        self.position_holding_periods.clear()
    
    def get_portfolio_value(self, current_price):
        return self.cash + (self.shares * current_price)
    
    def buy(self, price, signal_data):
        """Buy with time horizon optimized position sizing"""
        if self.shares == 0:
            confidence = signal_data.get('confidence', 0.5)
            
            # Position sizing based on time horizon and confidence
            if self.time_horizon == "short":
                base_ratio = 0.4 * confidence
            elif self.time_horizon == "medium":
                base_ratio = 0.5 * confidence  
            else:  # long
                base_ratio = 0.6 * confidence  # Long-term can be more committed
            
            investment_ratio = min(base_ratio, self.max_position_size)
            investment = self.cash * investment_ratio
            
            # Calculate transaction costs
            transaction_cost = investment * self.transaction_cost
            net_investment = investment - transaction_cost
            
            shares_to_buy = int(net_investment / price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price + transaction_cost
                self.cash -= cost
                self.shares += shares_to_buy
                self.entry_price = price
                self.total_transaction_costs += transaction_cost
                
                trade = {
                    'action': 'BUY',
                    'date': signal_data.get('date', datetime.now()),
                    'price': price,
                    'shares': shares_to_buy,
                    'confidence': confidence,
                    'investment_ratio': investment_ratio,
                    'transaction_cost': transaction_cost,
                    'reasoning': signal_data.get('reasoning', [])
                }
                self.trades.append(trade)
                return True
        return False
    
    def sell(self, price, signal_data):
        """Sell position"""
        if self.shares > 0:
            proceeds = self.shares * price
            transaction_cost = proceeds * self.transaction_cost
            net_proceeds = proceeds - transaction_cost
            
            # Calculate holding period
            last_buy = next((t for t in reversed(self.trades) if t['action'] == 'BUY'), None)
            holding_period = 0
            if last_buy:
                current_date = signal_data.get('date', datetime.now())
                holding_period = (current_date - last_buy['date']).days
                self.position_holding_periods.append(holding_period)
            
            profit = net_proceeds - (self.shares * self.entry_price)
            profit_pct = profit / (self.shares * self.entry_price)
            
            self.cash += net_proceeds
            self.total_transaction_costs += transaction_cost
            
            trade = {
                'action': 'SELL',
                'date': signal_data.get('date', datetime.now()),
                'price': price,
                'shares': self.shares,
                'confidence': signal_data.get('confidence', 0.5),
                'profit': profit,
                'profit_pct': profit_pct,
                'holding_period': holding_period,
                'transaction_cost': transaction_cost,
                'reasoning': signal_data.get('reasoning', [])
            }
            self.trades.append(trade)
            
            self.shares = 0
            self.entry_price = 0
            return profit_pct
        return 0
    
    def record_state(self, date, price):
        """Record daily portfolio state"""
        portfolio_value = self.get_portfolio_value(price)
        
        # Track max drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Record daily return
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        
        self.portfolio_history.append({
            'date': date,
            'price': price,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'shares': self.shares,
            'total_costs': self.total_transaction_costs
        })


class TimeHorizonComparisonFramework:
    """Framework for comparing performance across time horizons"""
    
    def __init__(self, test_symbol="MSFT", total_period=400):
        self.test_symbol = test_symbol
        self.total_period = total_period  # Need enough data for long-term analysis
        self.data = None
        self.results = {}
        
        # Define time horizons (calendar-based for 2024)
        self.time_horizons = {
            'short': {
                'days': 50,  # Jan 1 to mid-February (about 2 months of trading days)
                'name': 'Short-term (Jan - mid Feb, 50 days)',
                'description': 'January to mid-February period'
            },
            'medium': {
                'days': 167,  # Jan to June (about 6 months trading days)
                'name': 'Medium-term (Jan - June, ~167 days)', 
                'description': 'January to June period (5-6 months)'
            },
            'long': {
                'days': 365,  # Full calendar year
                'name': 'Long-term (Full Year, 365 days)',
                'description': 'Full calendar year'
            }
        }
    
    def load_data(self):
        """Load market data for testing"""
        
        print(f"📊 Loading data for {self.test_symbol}...")
        
        if PROJECT_AVAILABLE:
            try:
                # Use project's data fetcher
                api = get_data_api("yahoo_finance")
                self.data = api.fetch_historical_data(self.test_symbol, "1d")
                
                if self.data is None or self.data.empty:
                    print("⚠️ Yahoo Finance failed, trying Alpha Vantage...")
                    api = get_data_api("alpha_vantage")
                    self.data = api.fetch_historical_data(self.test_symbol, "1d")
                
                self.data = self.data.tail(self.total_period)
                self.data = TechnicalIndicators.add_all_indicators(self.data)
                print(f"✅ Loaded {len(self.data)} days using project methods")
                
            except Exception as e:
                print(f"⚠️ Project data fetcher failed: {e}")
                self._load_data_fallback()
        else:
            self._load_data_fallback()
        
        return self.data is not None and not self.data.empty
    
    def _load_data_fallback(self):
        """Fallback data loading"""
        try:
            stock = yf.Ticker(self.test_symbol)
            self.data = stock.history(period="2y")
            
            if not self.data.empty:
                self.data.columns = [col.lower() for col in self.data.columns]
                self.data = self.data.tail(self.total_period)
                self._add_basic_indicators()
                print(f"✅ Loaded {len(self.data)} days using fallback")
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
        
        return True
    
    def _add_basic_indicators(self):
        """Add basic technical indicators"""
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['close'].ewm(span=12).mean()
        exp2 = self.data['close'].ewm(span=26).mean()
        self.data['macd'] = exp1 - exp2
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        self.data['bb_middle'] = self.data['close'].rolling(window=20).mean()
        bb_std = self.data['close'].rolling(window=20).std()
        self.data['bb_upper'] = self.data['bb_middle'] + (bb_std * 2)
        self.data['bb_lower'] = self.data['bb_middle'] - (bb_std * 2)
    
    def test_time_horizon(self, horizon_name, test_days):
        """Test strategy for specific time horizon"""
        
        print(f"\n⏰ Testing: {self.time_horizons[horizon_name]['name']}")
        
        # Use YOUR project's ML strategy if available
        if PROJECT_AVAILABLE:
            try:
                strategy = MLTradingStrategy(confidence_threshold=0.15)
                strategy.name = f"Your AI System ({horizon_name.title()}-term)"
            except:
                strategy = TimeHorizonOptimizedStrategy(horizon_name, "Optimized Strategy")
        else:
            strategy = TimeHorizonOptimizedStrategy(horizon_name, "Optimized Strategy")
        
        portfolio = TimeHorizonPortfolioSimulator(10000, horizon_name)
        
        # Test on the specified period
        test_data = self.data.tail(test_days)
        
        strategy.reset()
        portfolio.reset()
        
        signals_generated = 0
        trades_executed = 0
        
        start_idx = 50  # Allow for indicator calculation
        
        for i in range(start_idx, len(test_data)):
            current_row = test_data.iloc[i]
            historical_data = test_data.iloc[max(0, i-100):i]
            
            # Generate signal
            signal = strategy.generate_signal(current_row, historical_data)
            signal['date'] = current_row.name  # Add date for tracking
            signals_generated += 1
            
            current_price = current_row['close']
            action = signal.get('action', 'HOLD')
            
            # Execute trades
            if action == 'BUY':
                if portfolio.buy(current_price, signal):
                    trades_executed += 1
            elif action == 'SELL':
                result = portfolio.sell(current_price, signal)
                if result is not None:
                    trades_executed += 1
            
            portfolio.record_state(current_row.name, current_price)
        
        # Calculate performance metrics
        if not portfolio.portfolio_history:
            print(f"   ❌ No portfolio history for {horizon_name}")
            return None
            
        final_value = portfolio.portfolio_history[-1]['portfolio_value']
        total_return = (final_value - portfolio.initial_capital) / portfolio.initial_capital
        
        # Benchmark comparison (buy & hold for same period)
        start_price = test_data['close'].iloc[start_idx]
        end_price = test_data['close'].iloc[-1]
        benchmark_return = (end_price - start_price) / start_price
        alpha = total_return - benchmark_return
        
        # Risk metrics
        if portfolio.daily_returns:
            volatility = np.std(portfolio.daily_returns) * np.sqrt(365)  # Annualized using 365 days
            sharpe_ratio = (np.mean(portfolio.daily_returns) * 365) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Trading efficiency metrics
        profitable_trades = len([t for t in portfolio.trades if t.get('profit', 0) > 0])
        total_sell_trades = len([t for t in portfolio.trades if t['action'] == 'SELL'])
        win_rate = profitable_trades / total_sell_trades if total_sell_trades > 0 else 0
        
        # Time horizon specific metrics
        avg_holding_period = np.mean(portfolio.position_holding_periods) if portfolio.position_holding_periods else 0
        trading_frequency = trades_executed / test_days if test_days > 0 else 0
        cost_drag = portfolio.total_transaction_costs / portfolio.initial_capital
        
        # Efficiency score (return per unit of risk and cost)
        efficiency_score = total_return / (volatility + cost_drag + 0.01) if (volatility + cost_drag) > 0 else 0
        
        results = {
            'horizon_name': horizon_name,
            'horizon_display': self.time_horizons[horizon_name]['name'],
            'test_days': test_days,
            'strategy_name': strategy.name,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'alpha': alpha,
            'final_value': final_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': portfolio.max_drawdown,
            'total_trades': trades_executed,
            'signals_generated': signals_generated,
            'win_rate': win_rate,
            'avg_holding_period': avg_holding_period,
            'trading_frequency': trading_frequency,
            'transaction_costs': portfolio.total_transaction_costs,
            'cost_drag': cost_drag,
            'efficiency_score': efficiency_score,
            'portfolio_history': portfolio.portfolio_history,
            'trades': portfolio.trades
        }
        
        print(f"   📈 Total Return: {total_return:.2%}")
        print(f"   📊 Alpha: {alpha:.2%}")
        print(f"   🔄 Total Trades: {trades_executed}")
        print(f"   ⏱️ Avg Holding: {avg_holding_period:.1f} days")
        print(f"   💰 Cost Drag: {cost_drag:.2%}")
        print(f"   🎯 Efficiency: {efficiency_score:.2f}")
        
        return results
    
    def run_time_horizon_comparison(self):
        """Run complete time horizon comparison using 2024 calendar periods"""
        
        print("⏰ TIME HORIZON COMPARISON ANALYSIS")
        print("=" * 60)
        print("Comparing performance across different time horizons using 2024 calendar periods:")
        print("• Short: 50 days (Jan - mid Feb)")  
        print("• Medium: 152-182 days (Jan - May/June)")
        print("• Long: 365 days (Jan - December)")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data")
            return
        
        # Ensure we have enough data for all time horizons
        if len(self.data) < 365:
            print(f"⚠️ Not enough data for full year analysis. Have {len(self.data)} days, need 365.")
            print("Using available data for proportional analysis...")
        
        # Update time horizons based on available data
        for horizon_name, horizon_info in self.time_horizons.items():
            if len(self.data) < horizon_info['days']:
                self.time_horizons[horizon_name]['days'] = min(horizon_info['days'], len(self.data))
                print(f"⚠️ Adjusted {horizon_name} period to {self.time_horizons[horizon_name]['days']} days")
        
        # Test each time horizon
        for horizon_name, horizon_info in self.time_horizons.items():
            test_days = horizon_info['days']
            
            # Ensure we have enough data
            if len(self.data) < test_days:
                print(f"⚠️ Insufficient data for {horizon_name} horizon. Skipping...")
                continue
                
            results = self.test_time_horizon(horizon_name, test_days)
            if results:
                self.results[horizon_name] = results
        
        if not self.results:
            print("❌ No results generated")
            return
        
        # Generate comprehensive analysis
        self._generate_time_horizon_analysis()
        self._create_time_horizon_visualizations()
        self._save_time_horizon_report()
        
        print("\n✅ Time horizon comparison completed!")
        print("📁 Check 'time_horizon_results' folder for outputs")
    
    def _generate_time_horizon_analysis(self):
        """Generate detailed time horizon analysis"""
        
        print("\n" + "=" * 80)
        print("⏰ TIME HORIZON PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to analyze")
            return
        
        # Performance summary table
        print(f"\n{'Time Horizon':<25} {'Return':<10} {'Alpha':<10} {'Sharpe':<8} {'Trades':<8} {'Efficiency'}")
        print("-" * 85)
        
        for horizon_name, results in self.results.items():
            return_pct = results['total_return'] * 100
            alpha_pct = results['alpha'] * 100
            sharpe = results['sharpe_ratio']
            trades = results['total_trades']
            efficiency = results['efficiency_score']
            
            print(f"{results['horizon_display']:<25} {return_pct:>6.2f}% {alpha_pct:>7.2f}% {sharpe:>7.2f} {trades:>6} {efficiency:>9.2f}")
        
        # Best performing horizon
        best_horizon = max(self.results.items(), key=lambda x: x[1]['total_return'])
        best_name = best_horizon[0]
        best_results = best_horizon[1]
        
        print(f"\n🏆 BEST PERFORMING HORIZON: {best_results['horizon_display']}")
        print(f"   📈 Return: {best_results['total_return']:.2%}")
        print(f"   📊 Alpha: {best_results['alpha']:.2%}")
        print(f"   🎯 Efficiency Score: {best_results['efficiency_score']:.2f}")
        
        # Most efficient horizon (risk-adjusted)
        most_efficient = max(self.results.items(), key=lambda x: x[1]['efficiency_score'])
        eff_name = most_efficient[0]
        eff_results = most_efficient[1]
        
        if eff_name != best_name:
            print(f"\n🎯 MOST EFFICIENT HORIZON: {eff_results['horizon_display']}")
            print(f"   ⚡ Efficiency Score: {eff_results['efficiency_score']:.2f}")
            print(f"   📊 Risk-Adjusted Performance Superior")
        
        # Detailed horizon analysis
        print(f"\n📊 DETAILED HORIZON ANALYSIS:")
        print("-" * 40)
        
        for horizon_name, results in self.results.items():
            print(f"\n{results['horizon_display']}:")
            print(f"  📅 Test Period: {results['test_days']} days")
            print(f"  📈 Total Return: {results['total_return']:.2%}")
            print(f"  📊 vs Benchmark: {results['alpha']:.2%} alpha")
            print(f"  📉 Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"  📊 Volatility: {results['volatility']:.2%}")
            print(f"  🔄 Total Trades: {results['total_trades']}")
            print(f"  🎯 Win Rate: {results['win_rate']:.2%}")
            print(f"  ⏱️ Avg Holding: {results['avg_holding_period']:.1f} days")
            print(f"  📈 Trading Freq: {results['trading_frequency']:.3f} trades/day")
            print(f"  💰 Cost Drag: {results['cost_drag']:.2%}")
            print(f"  ⚡ Efficiency: {results['efficiency_score']:.2f}")
        
        # Calendar period insights
        print(f"\n📅 CALENDAR PERIOD INSIGHTS:")
        print("-" * 40)
        
        if 'short' in self.results:
            short = self.results['short']
            print(f"🗓️ Early Year Performance (Jan-Feb):")
            print(f"   • {short['total_return']:.2%} return in {short['test_days']} days")
            print(f"   • Annualized: {short['total_return'] * (365/short['test_days']):.2%}")
            print(f"   • High frequency: {short['trading_frequency']:.3f} trades/day")
        
        if 'medium' in self.results:
            medium = self.results['medium']
            print(f"🗓️ First Half Year Performance (Jan-June):")
            print(f"   • {medium['total_return']:.2%} return in {medium['test_days']} days")
            print(f"   • Seasonal trends captured")
            print(f"   • Balanced approach: {medium['avg_holding_period']:.1f} day avg hold")
        
        if 'long' in self.results:
            long_r = self.results['long']
            print(f"🗓️ Full Year Performance:")
            print(f"   • {long_r['total_return']:.2%} annual return")
            print(f"   • Complete market cycles captured")
            print(f"   • Patient approach: {long_r['avg_holding_period']:.1f} day avg hold")
        
        # Performance patterns
        self._analyze_performance_patterns()
    
    def _analyze_performance_patterns(self):
        """Analyze patterns across time horizons"""
        
        print(f"\n🔍 PERFORMANCE PATTERNS:")
        print("-" * 40)
        
        # Return scaling analysis
        returns = [(h, r['total_return']) for h, r in self.results.items()]
        returns.sort(key=lambda x: self.time_horizons[x[0]]['days'])
        
        print("📈 Return Scaling:")
        for i, (horizon, return_val) in enumerate(returns):
            days = self.time_horizons[horizon]['days']
            annualized_return = return_val * (365 / days)
            print(f"   {horizon.title()}: {return_val:.2%} ({days} days) → {annualized_return:.2%} annualized")
        
        # Trading efficiency analysis
        print("\n⚡ Trading Efficiency:")
        for horizon, results in self.results.items():
            efficiency = results['efficiency_score']
            cost_impact = results['cost_drag']
            print(f"   {horizon.title()}: {efficiency:.2f} efficiency, {cost_impact:.2%} cost drag")
        
        # Risk-return profile
        print("\n⚖️ Risk-Return Profile:")
        for horizon, results in self.results.items():
            risk_adj_return = results['total_return'] / max(results['volatility'], 0.01)
            print(f"   {horizon.title()}: {risk_adj_return:.2f} return/risk ratio")
        
        # Optimal horizon recommendation
        print(f"\n🎯 OPTIMAL HORIZON RECOMMENDATION:")
        
        # Score each horizon on multiple factors
        horizon_scores = {}
        for horizon, results in self.results.items():
            score = (
                results['total_return'] * 0.3 +  # 30% weight on returns
                results['efficiency_score'] * 0.25 +  # 25% weight on efficiency
                results['sharpe_ratio'] * 0.2 +  # 20% weight on Sharpe ratio
                (1 - results['max_drawdown']) * 0.15 +  # 15% weight on low drawdown
                results['win_rate'] * 0.1  # 10% weight on win rate
            )
            horizon_scores[horizon] = score
        
        optimal_horizon = max(horizon_scores.items(), key=lambda x: x[1])
        opt_name = optimal_horizon[0]
        opt_score = optimal_horizon[1]
        
        print(f"   🏆 Recommended: {self.time_horizons[opt_name]['name']}")
        print(f"   📊 Overall Score: {opt_score:.3f}")
        print(f"   💡 This horizon provides the best balance of returns, efficiency, and risk management")
    
    def _create_time_horizon_visualizations(self):
        """Create comprehensive time horizon visualizations"""
        
        print("📊 Creating time horizon visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f'Time Horizon Analysis - {self.test_symbol}', fontsize=16, fontweight='bold')
        
        # Color mapping for consistency
        colors = {'short': 'red', 'medium': 'orange', 'long': 'blue'}
        
        # Plot 1: Portfolio Performance Over Time
        ax1 = axes[0, 0]
        
        for horizon, results in self.results.items():
            history = pd.DataFrame(results['portfolio_history'])
            color = colors.get(horizon, 'gray')
            
            # Normalize to start from the same date for comparison
            if not history.empty:
                ax1.plot(range(len(history)), history['portfolio_value'], 
                        label=f"{horizon.title()} ({len(history)} days)", 
                        linewidth=2, color=color, alpha=0.8)
        
        ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Return vs Risk Scatter
        ax2 = axes[0, 1]
        
        for horizon, results in self.results.items():
            color = colors.get(horizon, 'gray')
            ax2.scatter(results['volatility']*100, results['total_return']*100, 
                        s=results['efficiency_score']*100, color=color, alpha=0.7,
                        label=f"{horizon.title()}")
            
            # Add text labels
            ax2.annotate(horizon.title(), 
                        (results['volatility']*100, results['total_return']*100),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_title('Return vs Risk (bubble size = efficiency)')
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Total Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance Metrics Comparison
        ax3 = axes[1, 0]
        
        metrics = ['Total Return %', 'Alpha %', 'Sharpe Ratio', 'Win Rate %']
        horizon_names = list(self.results.keys())
        
        metric_data = {
            'Total Return %': [self.results[h]['total_return']*100 for h in horizon_names],
            'Alpha %': [self.results[h]['alpha']*100 for h in horizon_names],
            'Sharpe Ratio': [self.results[h]['sharpe_ratio'] for h in horizon_names],
            'Win Rate %': [self.results[h]['win_rate']*100 for h in horizon_names]
        }
        
        x = np.arange(len(horizon_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2) * width
            bars = ax3.bar(x + offset, metric_data[metric], width, 
                            label=metric, alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([h.title() for h in horizon_names])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trading Activity Analysis
        ax4 = axes[1, 1]
        
        activity_metrics = ['Total Trades', 'Avg Hold Days', 'Cost Drag %']
        activity_data = {
            'Total Trades': [self.results[h]['total_trades'] for h in horizon_names],
            'Avg Hold Days': [self.results[h]['avg_holding_period'] for h in horizon_names],
            'Cost Drag %': [self.results[h]['cost_drag']*100 for h in horizon_names]
        }
        
        x = np.arange(len(horizon_names))
        
        for i, metric in enumerate(activity_metrics):
            offset = (i - len(activity_metrics)/2) * width
            bars = ax4.bar(x + offset, activity_data[metric], width, 
                            label=metric, alpha=0.7)
        
        ax4.set_title('Trading Activity Analysis')
        ax4.set_xticks(x)
        ax4.set_xticklabels([h.title() for h in horizon_names])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Efficiency Scores
        ax5 = axes[2, 0]
        
        efficiency_scores = [self.results[h]['efficiency_score'] for h in horizon_names]
        bars = ax5.bar(horizon_names, efficiency_scores, 
                        color=[colors.get(h, 'gray') for h in horizon_names], alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        ax5.set_title('Efficiency Scores by Time Horizon')
        ax5.set_ylabel('Efficiency Score')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Calendar Period Summary
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Create summary text
        summary_text = f"""
    TIME HORIZON COMPARISON SUMMARY
    (Calendar Year 2024 Analysis)

    """
        
        for horizon, results in self.results.items():
            period_desc = self.time_horizons[horizon]['description']
            summary_text += f"""
    {horizon.upper()} TERM ({period_desc}):
    - Period: {results['test_days']} days
    - Return: {results['total_return']:.2%}
    - Alpha: {results['alpha']:.2%}
    - Trades: {results['total_trades']}
    - Efficiency: {results['efficiency_score']:.2f}
    """
        
        # Add best performer
        best_horizon = max(self.results.items(), key=lambda x: x[1]['total_return'])
        summary_text += f"\n🏆 BEST PERFORMER: {best_horizon[0].upper()}"
        summary_text += f"\nReturn: {best_horizon[1]['total_return']:.2%}"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        results_dir = os.path.join(os.path.dirname(__file__), 'time_horizon_results')
        os.makedirs(results_dir, exist_ok=True)
        
        plot_file = os.path.join(results_dir, f'time_horizon_analysis_{self.test_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_file}")
        
        plt.show()
    
    def _save_time_horizon_report(self):
        """Save detailed time horizon comparison report"""
        
        results_dir = os.path.join(os.path.dirname(__file__), 'time_horizon_results')
        os.makedirs(results_dir, exist_ok=True)
        
        report_file = os.path.join(results_dir, f'time_horizon_report_{self.test_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(report_file, 'w') as f:
            f.write("TIME HORIZON COMPARISON ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Symbol: {self.test_symbol}\n")
            f.write(f"Analysis Type: Calendar Year 2024 Based\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TIME HORIZON DEFINITIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("Short-term: 50 days (January to mid-February)\n")
            f.write("Medium-term: ~167 days (January to June, 5-6 months)\n") 
            f.write("Long-term: 365 days (Full calendar year)\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Horizon':<15} {'Days':<6} {'Return':<10} {'Alpha':<10} {'Sharpe':<8} {'Trades':<8} {'Efficiency'}\n")
            f.write("-" * 75 + "\n")
            
            for horizon, results in self.results.items():
                f.write(f"{horizon.title():<15} {results['test_days']:<6} "
                        f"{results['total_return']*100:>6.2f}% {results['alpha']*100:>7.2f}% "
                        f"{results['sharpe_ratio']:>7.2f} {results['total_trades']:>6} "
                        f"{results['efficiency_score']:>9.2f}\n")
            
            f.write("\nDETAILED ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            for horizon, results in self.results.items():
                f.write(f"\n{results['horizon_display']}:\n")
                f.write(f"  Test Period: {results['test_days']} calendar days\n")
                f.write(f"  Strategy: {results['strategy_name']}\n")
                f.write(f"  Total Return: {results['total_return']:.2%}\n")
                f.write(f"  Benchmark Return: {results['benchmark_return']:.2%}\n")
                f.write(f"  Alpha (vs benchmark): {results['alpha']:.2%}\n")
                f.write(f"  Volatility (annualized): {results['volatility']:.2%}\n")
                f.write(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
                f.write(f"  Maximum Drawdown: {results['max_drawdown']:.2%}\n")
                f.write(f"  Total Trades Executed: {results['total_trades']}\n")
                f.write(f"  Signals Generated: {results['signals_generated']}\n")
                f.write(f"  Trade Execution Rate: {results['total_trades']/results['signals_generated']:.2%}\n")
                f.write(f"  Win Rate: {results['win_rate']:.2%}\n")
                f.write(f"  Average Holding Period: {results['avg_holding_period']:.1f} days\n")
                f.write(f"  Trading Frequency: {results['trading_frequency']:.3f} trades/day\n")
                f.write(f"  Total Transaction Costs: ${results['transaction_costs']:.2f}\n")
                f.write(f"  Cost Drag: {results['cost_drag']:.2%}\n")
                f.write(f"  Efficiency Score: {results['efficiency_score']:.2f}\n")
                
                # Annualized metrics
                days = results['test_days']
                annualized_return = results['total_return'] * (365 / days)
                f.write(f"  Annualized Return: {annualized_return:.2%}\n")
            
            # Best performer analysis
            best_horizon = max(self.results.items(), key=lambda x: x[1]['total_return'])
            f.write(f"\nBEST PERFORMING HORIZON:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Winner: {best_horizon[1]['horizon_display']}\n")
            f.write(f"Total Return: {best_horizon[1]['total_return']:.2%}\n")
            f.write(f"Alpha: {best_horizon[1]['alpha']:.2%}\n")
            f.write(f"Efficiency Score: {best_horizon[1]['efficiency_score']:.2f}\n")
            
            # Calendar insights
            f.write(f"\nCALENDAR PERIOD INSIGHTS:\n")
            f.write("-" * 40 + "\n")
            
            if 'short' in self.results:
                short = self.results['short']
                f.write(f"Early Year Performance (Jan-Feb):\n")
                f.write(f"  - Strong/weak start to the year with {short['total_return']:.2%} return\n")
                f.write(f"  - High activity period with {short['total_trades']} trades in {short['test_days']} days\n")
                f.write(f"  - Annualized pace: {short['total_return'] * (365/short['test_days']):.2%}\n\n")
            
            if 'medium' in self.results:
                medium = self.results['medium']
                f.write(f"First Half Performance (Jan-June):\n") 
                f.write(f"  - Six-month return of {medium['total_return']:.2%}\n")
                f.write(f"  - Captures seasonal trends and earnings cycles\n")
                f.write(f"  - Balanced trading approach: {medium['avg_holding_period']:.1f} day avg hold\n\n")
            
            if 'long' in self.results:
                long_r = self.results['long']
                f.write(f"Full Year Performance:\n")
                f.write(f"  - Annual return: {long_r['total_return']:.2%}\n")
                f.write(f"  - Complete market cycle analysis\n")
                f.write(f"  - Patient strategy: {long_r['avg_holding_period']:.1f} day average hold\n")
                f.write(f"  - Cost efficient: {long_r['cost_drag']:.2%} total cost drag\n")
            
            # Recommendations
            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            
            # Find most efficient
            most_efficient = max(self.results.items(), key=lambda x: x[1]['efficiency_score'])
            
            f.write(f"1. OPTIMAL TIME HORIZON: {most_efficient[1]['horizon_display']}\n")
            f.write(f"   - Best efficiency score: {most_efficient[1]['efficiency_score']:.2f}\n")
            f.write(f"   - Balances returns, risk, and costs effectively\n\n")
            
            f.write(f"2. TRADING STRATEGY INSIGHTS:\n")
            if 'short' in self.results and self.results['short']['total_trades'] > 20:
                f.write(f"   - Short-term: High activity may lead to overtrading\n")
            if 'long' in self.results:
                f.write(f"   - Long-term: Patient approach with {self.results['long']['avg_holding_period']:.0f} day holds\n")
            
            f.write(f"\n3. CALENDAR SEASONALITY:\n")
            f.write(f"   - Consider seasonal patterns when choosing time horizons\n")
            f.write(f"   - Early year vs full year performance may vary significantly\n")
            f.write(f"   - Medium-term captures earnings seasons and quarterly cycles\n")
        
        print(f"📄 Report saved to: {report_file}")


def main():
    """Main function to run time horizon comparison"""
    
    print("⏰ TIME HORIZON COMPARISON ANALYSIS")
    print("Comparing performance across calendar-based time periods")
    print("=" * 60)
    
    # Get test symbol from user
    symbol = input("Enter symbol to analyze (default: MSFT): ").strip().upper() or "MSFT"
    
    # Ask for data period
    print("\nData period options:")
    print("1. Standard (400 days) - recommended")
    print("2. Extended (500 days) - more historical context")
    print("3. Minimal (365 days) - exact year")
    
    period_choice = input("Choose data period (1-3, default: 1): ").strip() or "1"
    
    period_map = {
        "1": 400,
        "2": 500, 
        "3": 365
    }
    
    total_period = period_map.get(period_choice, 400)
    
    print(f"\nAnalyzing {symbol} over {total_period} days with calendar-based horizons...")
    
    # Initialize framework
    framework = TimeHorizonComparisonFramework(test_symbol=symbol, total_period=total_period)
    
    # Run comparison
    framework.run_time_horizon_comparison()
    
    print(f"\n✅ Time horizon analysis completed for {symbol}!")
    print("📁 Check 'time_horizon_results' folder for detailed analysis")
    print("\n💡 Key Insights:")
    
    if framework.results:
        best_horizon = max(framework.results.items(), key=lambda x: x[1]['total_return'])
        most_efficient = max(framework.results.items(), key=lambda x: x[1]['efficiency_score'])
        
        print(f"   🏆 Best Return: {best_horizon[1]['horizon_display']} ({best_horizon[1]['total_return']:.2%})")
        print(f"   ⚡ Most Efficient: {most_efficient[1]['horizon_display']} ({most_efficient[1]['efficiency_score']:.2f})")
        
        # Calendar insights
        if 'short' in framework.results and 'long' in framework.results:
            short_annual = framework.results['short']['total_return'] * (365/framework.results['short']['test_days'])
            long_annual = framework.results['long']['total_return']
            
            if short_annual > long_annual * 1.2:
                print(f"   📈 Early year momentum detected - short-term outpacing annual")
            elif long_annual > short_annual * 0.8:
                print(f"   📊 Consistent performance - long-term strategy validated")


if __name__ == "__main__":
    main()