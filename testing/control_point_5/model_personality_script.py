"""
MODEL PERSONALITY COMPARISON SCRIPT
Compares Speculative (Aggressive & Optimistic) vs Conservative (Cautious & Denial-based) models
Same capital, same timeframe, same market conditions
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
    from src.backtesting import MLTradingStrategy, TechnicalAnalysisStrategy
    from src.data.fetcher import get_data_api
    from src.indicators.technical import TechnicalIndicators
    PROJECT_AVAILABLE = True
    print("✅ Your project modules loaded successfully")
except ImportError as e:
    print(f"❌ Project modules not available: {e}")
    PROJECT_AVAILABLE = False
    import yfinance as yf


class SpeculativeModel:
    """
    Aggressive & Optimistic Trading Model
    
    Characteristics:
    - Lower confidence threshold (takes more risks)
    - Amplifies positive signals
    - FOMO-driven buying
    - Quick to enter positions
    - Higher position sizing
    """
    
    def __init__(self):
        self.name = "Speculative Model (Aggressive & Optimistic)"
        self.confidence_threshold = 0.15  # Low threshold = more trades
        self.risk_tolerance = 0.9  # High risk tolerance
        self.fomo_factor = 1.3  # Amplifies bullish signals
        self.position_size_multiplier = 1.4  # Larger positions
        self.recent_performance = []  # Track recent wins/losses
        
    def reset(self):
        """Reset model state"""
        self.recent_performance = []
        
    def generate_signal(self, current_data, historical_data):
        """Generate trading signal with aggressive/optimistic bias"""
        
        if len(historical_data) < 20:
            return {'action': 'HOLD', 'confidence': 0.3, 'reasoning': ['Insufficient data']}
        
        # Basic technical analysis
        rsi = current_data.get('rsi', 50)
        macd = current_data.get('macd', 0)
        macd_signal = current_data.get('macd_signal', 0)
        bb_upper = current_data.get('bb_upper', current_data['close'] * 1.02)
        bb_lower = current_data.get('bb_lower', current_data['close'] * 0.98)
        
        current_price = current_data['close']
        sma_20 = historical_data['close'].tail(20).mean()
        price_momentum = (current_price - sma_20) / sma_20
        
        # Aggressive signal generation
        base_confidence = 0.5
        action = 'HOLD'
        reasoning = []
        
        # FOMO-driven buying (lower RSI threshold)
        if rsi < 45:  # More aggressive than typical 30
            action = 'BUY'
            base_confidence = (50 - rsi) / 50 * self.fomo_factor  # Amplify confidence
            reasoning.append(f'RSI {rsi:.1f} - Speculative buy opportunity')
            
            # Extra bullishness if recent momentum
            if price_momentum > 0:
                base_confidence *= 1.2
                reasoning.append('Price momentum - FOMO activated!')
        
        # Quick profit-taking (lower RSI threshold for selling)
        elif rsi > 65:  # More aggressive than typical 70
            action = 'SELL'
            base_confidence = (rsi - 50) / 50 * 0.8  # Moderate confidence for selling
            reasoning.append(f'RSI {rsi:.1f} - Quick profit taking')
        
        # MACD momentum trading
        elif macd > macd_signal and macd > 0:
            action = 'BUY'
            base_confidence = 0.6 * self.fomo_factor
            reasoning.append('MACD bullish momentum - jumping in!')
        
        # Breakout trading (aggressive)
        elif current_price > bb_upper:
            action = 'BUY'  # Buy breakouts aggressively
            base_confidence = 0.7 * self.fomo_factor
            reasoning.append('Bollinger Band breakout - momentum trade!')
        
        # Recent wins boost confidence (overconfidence bias)
        if len(self.recent_performance) >= 3:
            recent_wins = sum(1 for p in self.recent_performance[-3:] if p > 0)
            if recent_wins >= 2:
                base_confidence *= 1.3  # Overconfidence after wins
                reasoning.append('Recent wins - feeling confident!')
        
        # Optimistic bias - always see the bright side
        if action == 'HOLD' and rsi < 55:
            action = 'BUY'
            base_confidence = 0.4
            reasoning.append('Optimistic outlook - market will recover!')
        
        # Cap confidence and add speculative note
        final_confidence = min(1.0, base_confidence)
        
        if action != 'HOLD':
            reasoning.append('Speculative model - high risk, high reward!')
        
        return {
            'action': action,
            'confidence': final_confidence,
            'reasoning': reasoning,
            'position_size_multiplier': self.position_size_multiplier,
            'model_type': 'SPECULATIVE'
        }
    
    def record_trade_result(self, profit_loss):
        """Record trade result for psychological modeling"""
        self.recent_performance.append(profit_loss)
        # Keep only last 5 trades
        if len(self.recent_performance) > 5:
            self.recent_performance.pop(0)


class ConservativeModel:
    """
    Cautious & Denial-based Trading Model
    
    Characteristics:
    - High confidence threshold (very picky)
    - Second-guesses good opportunities
    - Fear-driven decision making
    - Smaller position sizes
    - Denial of negative signals
    """
    
    def __init__(self):
        self.name = "Conservative Model (Cautious & Denial-based)"
        self.confidence_threshold = 0.6  # High threshold = fewer trades
        self.risk_tolerance = 0.3  # Low risk tolerance
        self.denial_factor = 0.7  # Reduces signal strength
        self.position_size_multiplier = 0.6  # Smaller positions
        self.recent_losses = []  # Track recent losses for fear
        
    def reset(self):
        """Reset model state"""
        self.recent_losses = []
        
    def generate_signal(self, current_data, historical_data):
        """Generate trading signal with conservative/denial bias"""
        
        if len(historical_data) < 30:  # Need more data to feel confident
            return {'action': 'HOLD', 'confidence': 0.2, 'reasoning': ['Need more data to be sure']}
        
        # Basic technical analysis (same data as speculative)
        rsi = current_data.get('rsi', 50)
        macd = current_data.get('macd', 0)
        macd_signal = current_data.get('macd_signal', 0)
        bb_upper = current_data.get('bb_upper', current_data['close'] * 1.02)
        bb_lower = current_data.get('bb_lower', current_data['close'] * 0.98)
        
        current_price = current_data['close']
        sma_50 = historical_data['close'].tail(50).mean()  # Longer term view
        price_vs_sma = (current_price - sma_50) / sma_50
        
        # Conservative signal generation
        base_confidence = 0.3  # Start with low confidence
        action = 'HOLD'  # Default to holding
        reasoning = []
        
        # Very conservative buying (extreme oversold only)
        if rsi < 25:  # Much more conservative than typical 30
            action = 'BUY'
            base_confidence = (30 - rsi) / 30 * self.denial_factor
            reasoning.append(f'RSI {rsi:.1f} - Extremely oversold, cautious entry')
            
            # But second-guess if recent losses
            if len(self.recent_losses) >= 2:
                base_confidence *= 0.5  # Cut confidence in half
                reasoning.append('Recent losses - very hesitant to buy')
        
        # Conservative selling (moderate overbought)
        elif rsi > 75:  # More conservative threshold
            action = 'SELL'
            base_confidence = (rsi - 70) / 30 * 0.8
            reasoning.append(f'RSI {rsi:.1f} - Conservative profit taking')
        
        # MACD confirmation (need strong signals)
        elif macd > macd_signal and macd > 0.02:  # Need significant MACD
            if rsi < 30:  # Double confirmation needed
                action = 'BUY'
                base_confidence = 0.4 * self.denial_factor
                reasoning.append('MACD + RSI confirmation - cautious buy')
            else:
                reasoning.append('MACD bullish but waiting for better entry')
        
        # Denial of breakouts (fear of false breakouts)
        elif current_price > bb_upper:
            action = 'HOLD'  # Don't chase breakouts
            reasoning.append('BB breakout but could be false - staying cautious')
        
        # Fear after losses
        if len(self.recent_losses) >= 3:
            if action == 'BUY':
                base_confidence *= 0.3  # Major confidence reduction
                reasoning.append('Multiple recent losses - extreme caution!')
        
        # Denial bias - ignore moderately negative signals
        if rsi > 45 and rsi < 65:
            reasoning.append('Market seems okay - no action needed')
        
        # Conservative position sizing
        if action != 'HOLD' and base_confidence < self.confidence_threshold:
            action = 'HOLD'
            reasoning.append(f'Signal not strong enough (conf: {base_confidence:.2f} < {self.confidence_threshold})')
        
        # Always question the decision
        if action != 'HOLD':
            reasoning.append('Conservative model - better safe than sorry')
        
        return {
            'action': action,
            'confidence': min(0.8, base_confidence),  # Cap confidence lower
            'reasoning': reasoning,
            'position_size_multiplier': self.position_size_multiplier,
            'model_type': 'CONSERVATIVE'
        }
    
    def record_trade_result(self, profit_loss):
        """Record trade result, especially losses for fear modeling"""
        if profit_loss < 0:
            self.recent_losses.append(profit_loss)
            # Keep only last 5 losses
            if len(self.recent_losses) > 5:
                self.recent_losses.pop(0)


class PersonalityPortfolioSimulator:
    """Portfolio simulator that accounts for personality-based position sizing"""
    
    def __init__(self, initial_capital=10000, model_name="Model"):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.entry_price = 0
        self.model_name = model_name
        
        # Enhanced tracking
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        self.max_drawdown = 0
        self.peak_value = initial_capital
        self.psychological_state = "NEUTRAL"
        
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
        self.psychological_state = "NEUTRAL"
    
    def get_portfolio_value(self, current_price):
        return self.cash + (self.shares * current_price)
    
    def buy(self, price, signal_data):
        """Buy with personality-based position sizing"""
        if self.shares == 0:
            confidence = signal_data.get('confidence', 0.5)
            position_multiplier = signal_data.get('position_size_multiplier', 1.0)
            
            # Base investment ratio
            base_ratio = 0.4  # Conservative base
            
            # Adjust based on personality
            investment_ratio = base_ratio * confidence * position_multiplier
            investment_ratio = min(investment_ratio, 0.8)  # Cap at 80%
            
            investment = self.cash * investment_ratio
            shares_to_buy = int(investment / price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                self.cash -= cost
                self.shares += shares_to_buy
                self.entry_price = price
                
                trade = {
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'confidence': confidence,
                    'investment_ratio': investment_ratio,
                    'reasoning': signal_data.get('reasoning', [])
                }
                self.trades.append(trade)
                
                # Update psychological state
                if 'FOMO' in str(signal_data.get('reasoning', [])):
                    self.psychological_state = "EXCITED"
                elif 'cautious' in str(signal_data.get('reasoning', [])).lower():
                    self.psychological_state = "ANXIOUS"
                
                return True
        return False
    
    def sell(self, price, signal_data):
        """Sell position"""
        if self.shares > 0:
            proceeds = self.shares * price
            profit = proceeds - (self.shares * self.entry_price)
            profit_pct = profit / (self.shares * self.entry_price)
            
            self.cash += proceeds
            
            trade = {
                'action': 'SELL',
                'price': price,
                'shares': self.shares,
                'confidence': signal_data.get('confidence', 0.5),
                'profit': profit,
                'profit_pct': profit_pct,
                'reasoning': signal_data.get('reasoning', [])
            }
            self.trades.append(trade)
            
            # Update psychological state based on result
            if profit > 0:
                self.psychological_state = "CONFIDENT" if "SPECULATIVE" in self.model_name else "RELIEVED"
            else:
                self.psychological_state = "DISAPPOINTED" if "SPECULATIVE" in self.model_name else "FEARFUL"
            
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
            'psychological_state': self.psychological_state
        })


class PersonalityComparisonFramework:
    """Framework for comparing model personalities"""
    
    def __init__(self, test_symbol="MSFT", test_period=252):
        self.test_symbol = test_symbol
        self.test_period = test_period
        self.data = None
        self.results = {}
        
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
                
                self.data = self.data.tail(self.test_period)
                self.data = TechnicalIndicators.add_all_indicators(self.data)
                print(f"✅ Loaded {len(self.data)} days using project methods")
                
            except Exception as e:
                print(f"⚠️ Project data fetcher failed: {e}")
                self._load_data_fallback()
        else:
            self._load_data_fallback()
    
    def _load_data_fallback(self):
        """Fallback data loading"""
        try:
            stock = yf.Ticker(self.test_symbol)
            self.data = stock.history(period="1y")
            
            if not self.data.empty:
                self.data.columns = [col.lower() for col in self.data.columns]
                self.data = self.data.tail(self.test_period)
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
    
    def test_model_personality(self, model, portfolio):
        """Test a specific model personality"""
        
        print(f"\n🎭 Testing: {model.name}")
        
        model.reset()
        portfolio.reset()
        
        signals_generated = 0
        trades_executed = 0
        
        for i in range(50, len(self.data)):  # Start after 50 days for indicators
            current_row = self.data.iloc[i]
            historical_data = self.data.iloc[max(0, i-100):i]
            
            # Generate signal
            signal = model.generate_signal(current_row, historical_data)
            signals_generated += 1
            
            current_price = current_row['close']
            action = signal.get('action', 'HOLD')
            
            # Execute trades
            trade_result = None
            if action == 'BUY':
                if portfolio.buy(current_price, signal):
                    trades_executed += 1
            elif action == 'SELL':
                trade_result = portfolio.sell(current_price, signal)
                if trade_result is not None:
                    trades_executed += 1
                    # Record result for psychological modeling
                    if hasattr(model, 'record_trade_result'):
                        model.record_trade_result(trade_result)
            
            portfolio.record_state(current_row.name, current_price)
        
        # Calculate performance metrics
        final_value = portfolio.portfolio_history[-1]['portfolio_value']
        total_return = (final_value - portfolio.initial_capital) / portfolio.initial_capital
        
        # Risk metrics
        if portfolio.daily_returns:
            volatility = np.std(portfolio.daily_returns) * np.sqrt(252)
            sharpe_ratio = (np.mean(portfolio.daily_returns) * 252) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Trading metrics
        profitable_trades = len([t for t in portfolio.trades if t.get('profit', 0) > 0])
        total_sell_trades = len([t for t in portfolio.trades if t['action'] == 'SELL'])
        win_rate = profitable_trades / total_sell_trades if total_sell_trades > 0 else 0
        
        # Personality-specific metrics
        psychological_states = [h['psychological_state'] for h in portfolio.portfolio_history]
        dominant_state = max(set(psychological_states), key=psychological_states.count)
        
        avg_position_size = np.mean([t.get('investment_ratio', 0) for t in portfolio.trades if t['action'] == 'BUY']) if portfolio.trades else 0
        
        results = {
            'model_name': model.name,
            'total_return': total_return,
            'final_value': final_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': portfolio.max_drawdown,
            'total_trades': trades_executed,
            'signals_generated': signals_generated,
            'win_rate': win_rate,
            'avg_position_size': avg_position_size,
            'dominant_psychological_state': dominant_state,
            'portfolio_history': portfolio.portfolio_history,
            'trades': portfolio.trades,
            'model_type': signal.get('model_type', 'UNKNOWN')
        }
        
        print(f"   📈 Total Return: {total_return:.2%}")
        print(f"   🔄 Total Trades: {trades_executed}")
        print(f"   📊 Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   🎭 Dominant State: {dominant_state}")
        
        return results
    
    def run_personality_comparison(self):
        """Run complete personality comparison"""
        
        print("🎭 MODEL PERSONALITY COMPARISON")
        print("=" * 60)
        print("Comparing Speculative vs Conservative trading personalities")
        print("Same capital, same timeframe, same market conditions")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data")
            return
        
        # Initialize models
        speculative_model = SpeculativeModel()
        conservative_model = ConservativeModel()
        
        # Initialize portfolios
        speculative_portfolio = PersonalityPortfolioSimulator(10000, "Speculative")
        conservative_portfolio = PersonalityPortfolioSimulator(10000, "Conservative")
        
        # Test both models
        spec_results = self.test_model_personality(speculative_model, speculative_portfolio)
        cons_results = self.test_model_personality(conservative_model, conservative_portfolio)
        
        self.results['Speculative Model'] = spec_results
        self.results['Conservative Model'] = cons_results
        
        # Generate comparison analysis
        self._generate_comparison_analysis()
        self._create_comparison_visualizations()
        self._save_comparison_report()
        
        print("\n✅ Personality comparison completed!")
        print("📁 Check 'personality_comparison_results' folder for outputs")
    
    def _generate_comparison_analysis(self):
        """Generate detailed comparison analysis"""
        
        print("\n" + "=" * 60)
        print("📊 PERSONALITY COMPARISON RESULTS")
        print("=" * 60)
        
        spec = self.results['Speculative Model']
        cons = self.results['Conservative Model']
        
        # Performance comparison
        print(f"\n{'Metric':<25} {'Speculative':<15} {'Conservative':<15} {'Difference'}")
        print("-" * 70)
        
        metrics = [
            ('Total Return', 'total_return', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Max Drawdown', 'max_drawdown', '%'),
            ('Volatility', 'volatility', '%'),
            ('Total Trades', 'total_trades', ''),
            ('Win Rate', 'win_rate', '%'),
            ('Avg Position Size', 'avg_position_size', '%')
        ]
        
        for metric_name, metric_key, suffix in metrics:
            spec_val = spec.get(metric_key, 0)
            cons_val = cons.get(metric_key, 0)
            
            if suffix == '%':
                spec_str = f"{spec_val*100:>6.2f}%"
                cons_str = f"{cons_val*100:>6.2f}%"
                diff = (spec_val - cons_val) * 100
                diff_str = f"{diff:+.2f}%"
            else:
                spec_str = f"{spec_val:>8.2f}"
                cons_str = f"{cons_val:>8.2f}"
                diff = spec_val - cons_val
                diff_str = f"{diff:+.2f}"
            
            print(f"{metric_name:<25} {spec_str:<15} {cons_str:<15} {diff_str}")
        
        # Behavioral analysis
        print(f"\n🎭 BEHAVIORAL ANALYSIS:")
        print(f"Speculative Dominant State: {spec['dominant_psychological_state']}")
        print(f"Conservative Dominant State: {cons['dominant_psychological_state']}")
        
        # Winner determination
        spec_return = spec['total_return']
        cons_return = cons['total_return']
        
        if spec_return > cons_return:
            winner = "SPECULATIVE"
            margin = (spec_return - cons_return) * 100
            print(f"\n🏆 WINNER: {winner} Model by {margin:.2f}%")
        else:
            winner = "CONSERVATIVE"
            margin = (cons_return - spec_return) * 100
            print(f"\n🏆 WINNER: {winner} Model by {margin:.2f}%")
        
        # Risk-adjusted analysis
        spec_risk_adj = spec['total_return'] / max(spec['volatility'], 0.01)
        cons_risk_adj = cons['total_return'] / max(cons['volatility'], 0.01)
        
        if spec_risk_adj > cons_risk_adj:
            print(f"🎯 RISK-ADJUSTED WINNER: SPECULATIVE (better risk-adjusted returns)")
        else:
            print(f"🎯 RISK-ADJUSTED WINNER: CONSERVATIVE (better risk-adjusted returns)")
    
    def _create_comparison_visualizations(self):
        """Create visualization comparing both models"""
        
        print("📊 Creating comparison visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Model Personality Comparison: Speculative vs Conservative', fontsize=16, fontweight='bold')
        
        spec = self.results['Speculative Model']
        cons = self.results['Conservative Model']
        
        # Plot 1: Portfolio Performance Over Time
        ax1 = axes[0, 0]
        
        spec_history = pd.DataFrame(spec['portfolio_history'])
        cons_history = pd.DataFrame(cons['portfolio_history'])
        
        ax1.plot(spec_history['date'], spec_history['portfolio_value'], 
                label='Speculative', linewidth=2, color='red', alpha=0.8)
        ax1.plot(cons_history['date'], cons_history['portfolio_value'], 
                label='Conservative', linewidth=2, color='blue', alpha=0.8)
        ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance Metrics Radar Chart (simplified bar chart)
        ax2 = axes[0, 1]
        
        metrics = ['Return', 'Sharpe', 'Win Rate']
        spec_values = [spec['total_return']*100, spec['sharpe_ratio']*10, spec['win_rate']*100]
        cons_values = [cons['total_return']*100, cons['sharpe_ratio']*10, cons['win_rate']*100]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x - width/2, spec_values, width, label='Speculative', color='red', alpha=0.7)
        ax2.bar(x + width/2, cons_values, width, label='Conservative', color='blue', alpha=0.7)
        
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trading Activity
        ax3 = axes[1, 0]
        
        activity_metrics = ['Total Trades', 'Signals Generated', 'Win Rate %']
        spec_activity = [spec['total_trades'], spec['signals_generated'], spec['win_rate']*100]
        cons_activity = [cons['total_trades'], cons['signals_generated'], cons['win_rate']*100]
        
        x = np.arange(len(activity_metrics))
        ax3.bar(x - width/2, spec_activity, width, label='Speculative', color='red', alpha=0.7)
        ax3.bar(x + width/2, cons_activity, width, label='Conservative', color='blue', alpha=0.7)
        
        ax3.set_title('Trading Activity Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(activity_metrics, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Risk Metrics
        ax4 = axes[1, 1]
        
        risk_metrics = ['Volatility %', 'Max Drawdown %', 'Avg Position %']
        spec_risk = [spec['volatility']*100, spec['max_drawdown']*100, spec['avg_position_size']*100]
        cons_risk = [cons['volatility']*100, cons['max_drawdown']*100, cons['avg_position_size']*100]
       
        x = np.arange(len(risk_metrics))
        ax4.bar(x - width/2, spec_risk, width, label='Speculative', color='red', alpha=0.7)
        ax4.bar(x + width/2, cons_risk, width, label='Conservative', color='blue', alpha=0.7)
        
        ax4.set_title('Risk Metrics Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(risk_metrics, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Psychological States
        ax5 = axes[2, 0]
        
        spec_states = [h['psychological_state'] for h in spec['portfolio_history']]
        cons_states = [h['psychological_state'] for h in cons['portfolio_history']]
        
        all_states = list(set(spec_states + cons_states))
        spec_state_counts = [spec_states.count(state) for state in all_states]
        cons_state_counts = [cons_states.count(state) for state in all_states]
        
        x = np.arange(len(all_states))
        ax5.bar(x - width/2, spec_state_counts, width, label='Speculative', color='red', alpha=0.7)
        ax5.bar(x + width/2, cons_state_counts, width, label='Conservative', color='blue', alpha=0.7)
        
        ax5.set_title('Psychological States Distribution')
        ax5.set_xticks(x)
        ax5.set_xticklabels(all_states, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary Statistics
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Create summary text
        summary_text = f"""
    MODEL PERSONALITY COMPARISON SUMMARY

    SPECULATIVE MODEL (Aggressive & Optimistic):
    - Total Return: {spec['total_return']:.2%}
    - Sharpe Ratio: {spec['sharpe_ratio']:.2f}
    - Total Trades: {spec['total_trades']}
    - Win Rate: {spec['win_rate']:.2%}
    - Max Drawdown: {spec['max_drawdown']:.2%}
    - Dominant State: {spec['dominant_psychological_state']}

    CONSERVATIVE MODEL (Cautious & Denial-based):
    - Total Return: {cons['total_return']:.2%}
    - Sharpe Ratio: {cons['sharpe_ratio']:.2f}
    - Total Trades: {cons['total_trades']}
    - Win Rate: {cons['win_rate']:.2%}
    - Max Drawdown: {cons['max_drawdown']:.2%}
    - Dominant State: {cons['dominant_psychological_state']}

    WINNER: {'SPECULATIVE' if spec['total_return'] > cons['total_return'] else 'CONSERVATIVE'}
    Margin: {abs(spec['total_return'] - cons['total_return']):.2%}
    """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        results_dir = os.path.join(os.path.dirname(__file__), 'personality_comparison_results')
        os.makedirs(results_dir, exist_ok=True)
        
        plot_file = os.path.join(results_dir, f'personality_comparison_{self.test_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_file}")
        
        plt.show()
    
    def _save_comparison_report(self):
        """Save detailed comparison report"""
        
        results_dir = os.path.join(os.path.dirname(__file__), 'personality_comparison_results')
        os.makedirs(results_dir, exist_ok=True)
        
        report_file = os.path.join(results_dir, f'personality_comparison_report_{self.test_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(report_file, 'w') as f:
            f.write("MODEL PERSONALITY COMPARISON REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Symbol: {self.test_symbol}\n")
            f.write(f"Test Period: {self.test_period} days\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TESTING CONDITIONS:\n")
            f.write("- Same initial capital: $10,000\n")
            f.write("- Same market data and timeframe\n")
            f.write("- Same technical indicators\n")
            f.write("- Different decision-making personalities\n\n")
            
            # Model descriptions
            f.write("MODEL DESCRIPTIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("SPECULATIVE MODEL (Aggressive & Optimistic):\n")
            f.write("• Lower confidence threshold (0.15 vs 0.6)\n")
            f.write("• FOMO-driven buying behavior\n")
            f.write("• Amplifies positive signals (1.3x multiplier)\n")
            f.write("• Larger position sizes (1.4x multiplier)\n")
            f.write("• Overconfidence after wins\n")
            f.write("• Quick to enter positions\n\n")
            
            f.write("CONSERVATIVE MODEL (Cautious & Denial-based):\n")
            f.write("• Higher confidence threshold (0.6 vs 0.15)\n")
            f.write("• Fear-driven decision making\n")
            f.write("• Reduces signal strength (0.7x multiplier)\n")
            f.write("• Smaller position sizes (0.6x multiplier)\n")
            f.write("• Second-guesses opportunities\n")
            f.write("• Denial of negative signals\n\n")
            
            # Detailed results
            for model_name, results in self.results.items():
                f.write(f"{model_name.upper()} RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Return: {results['total_return']:.2%}\n")
                f.write(f"Final Portfolio Value: ${results['final_value']:.2f}\n")
                f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
                f.write(f"Volatility: {results['volatility']:.2%}\n")
                f.write(f"Max Drawdown: {results['max_drawdown']:.2%}\n")
                f.write(f"Total Trades: {results['total_trades']}\n")
                f.write(f"Signals Generated: {results['signals_generated']}\n")
                f.write(f"Trading Frequency: {results['total_trades']/results['signals_generated']:.2%}\n")
                f.write(f"Win Rate: {results['win_rate']:.2%}\n")
                f.write(f"Average Position Size: {results['avg_position_size']:.2%}\n")
                f.write(f"Dominant Psychological State: {results['dominant_psychological_state']}\n\n")
            
            # Comparative analysis
            spec = self.results['Speculative Model']
            cons = self.results['Conservative Model']
            
            f.write("COMPARATIVE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Return Difference: {(spec['total_return'] - cons['total_return'])*100:.2f}%\n")
            f.write(f"Risk Difference (Volatility): {(spec['volatility'] - cons['volatility'])*100:.2f}%\n")
            f.write(f"Trading Activity: Speculative ({spec['total_trades']}) vs Conservative ({cons['total_trades']})\n")
            f.write(f"Position Sizing: Speculative ({spec['avg_position_size']:.2%}) vs Conservative ({cons['avg_position_size']:.2%})\n\n")
            
            # Winner and conclusions
            if spec['total_return'] > cons['total_return']:
                winner = "SPECULATIVE"
                margin = (spec['total_return'] - cons['total_return']) * 100
            else:
                winner = "CONSERVATIVE"
                margin = (cons['total_return'] - spec['total_return']) * 100
            
            f.write("CONCLUSIONS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"WINNER: {winner} Model\n")
            f.write(f"Performance Margin: {margin:.2f}%\n\n")
            
            if winner == "SPECULATIVE":
                f.write("ANALYSIS: The aggressive, optimistic approach paid off in this market environment.\n")
                f.write("Key factors:\n")
                f.write("• Higher risk tolerance led to larger gains\n")
                f.write("• FOMO-driven buying captured momentum\n")
                f.write("• Larger position sizes amplified returns\n")
                f.write("• More frequent trading captured opportunities\n\n")
            else:
                f.write("ANALYSIS: The conservative, cautious approach proved more effective.\n")
                f.write("Key factors:\n")
                f.write("• Better risk management preserved capital\n")
                f.write("• Lower drawdowns provided stability\n")
                f.write("• Selective trading avoided bad positions\n")
                f.write("• Conservative position sizing limited losses\n\n")
            
            # Risk-adjusted analysis
            spec_risk_adj = spec['total_return'] / max(spec['volatility'], 0.01)
            cons_risk_adj = cons['total_return'] / max(cons['volatility'], 0.01)
            
            f.write("RISK-ADJUSTED ANALYSIS:\n")
            f.write(f"Speculative Risk-Adjusted Return: {spec_risk_adj:.2f}\n")
            f.write(f"Conservative Risk-Adjusted Return: {cons_risk_adj:.2f}\n")
            
            if spec_risk_adj > cons_risk_adj:
                f.write("RISK-ADJUSTED WINNER: SPECULATIVE Model\n")
            else:
                f.write("RISK-ADJUSTED WINNER: CONSERVATIVE Model\n")
            
            f.write("\nThe risk-adjusted analysis considers both returns and volatility,\n")
            f.write("providing a more complete picture of performance quality.\n")
        
        print(f"📄 Report saved to: {report_file}")


def main():
    """Main function to run personality comparison"""
    
    print("🎭 MODEL PERSONALITY COMPARISON")
    print("Testing Speculative vs Conservative trading personalities")
    print("=" * 60)
    
    # Get test symbol from user
    symbol = input("Enter symbol to test (default: MSFT): ").strip().upper() or "MSFT"
    
    # Initialize framework
    framework = PersonalityComparisonFramework(test_symbol=symbol, test_period=252)
    
    # Run comparison
    framework.run_personality_comparison()
    
    print(f"\n✅ Personality comparison completed for {symbol}!")
    print("📁 Check 'personality_comparison_results' folder for detailed analysis")


if __name__ == "__main__":
    main()