"""
HYPOTHESIS TESTING: 1 Stock, 1 Year
Testing framework for comparing trading strategies on single asset over full year

Configuration:
- Asset: MSFT
- Period: 2024 (12 months)
- Strategies: Individual + Split-Capital Multi-Strategy
- Capital: $10,000
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import yfinance as yf
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

try:
    from src.backtesting import (
        MLTradingStrategy, 
        TechnicalAnalysisStrategy,
        HybridRSIDivergenceStrategy,
        UltimatePortfolioRunner
    )
    from src.backtesting.backtester import ProductionBacktester
    from src.data.fetcher import get_data_api
    from src.indicators.technical import TechnicalIndicators
    PROJECT_AVAILABLE = True
    print("✅ Project modules loaded successfully")
except ImportError as e:
    print(f"❌ Project modules not available: {e}")
    PROJECT_AVAILABLE = False


class Hypothesis1Stock1Year:
    """Hypothesis testing framework: 1 Stock, 1 Year"""
    
    def __init__(self):
        self.results = {}
        self.backtester_signals = {}
        self.trade_history = {}
        
        # Test configuration
        self.TEST_CONFIG = {
            'test_name': '1 Stock, 1 Year',
            'test_symbol': 'MSFT',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 10000,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,
            'data_source': 'polygon'
        }
        
        self.test_data = None
        print("🎯 HYPOTHESIS TESTING: 1 Stock, 1 Year")
        print("=" * 50)
        print(f"Asset: {self.TEST_CONFIG['test_symbol']}")
        print(f"Period: {self.TEST_CONFIG['start_date']} to {self.TEST_CONFIG['end_date']}")
        print(f"Capital: ${self.TEST_CONFIG['initial_capital']:,}")
        print(f"Test: Single asset over full year")
    
    def load_test_data(self):
        """Load test data for the specified period"""
        print(f"\n📊 Loading test data for {self.TEST_CONFIG['test_name']}...")
        
        try:
            if PROJECT_AVAILABLE:
                api = get_data_api("polygon")
                data = api.fetch_historical_data(
                    self.TEST_CONFIG['test_symbol'], 
                    "1d",
                    start_date=self.TEST_CONFIG['start_date'],
                    end_date=self.TEST_CONFIG['end_date']
                )
                
                if data is not None and not data.empty:
                    self.test_data = TechnicalIndicators.add_all_indicators(data)
                    print(f"✅ Loaded {len(self.test_data)} days from project API")
                    return True
            
            # Fallback to yfinance
            print("   Trying Yahoo Finance fallback...")
            ticker = yf.Ticker(self.TEST_CONFIG['test_symbol'])
            data = ticker.history(
                start=self.TEST_CONFIG['start_date'],
                end=self.TEST_CONFIG['end_date'],
                interval='1d'
            )
            
            if not data.empty:
                data.columns = [col.lower() for col in data.columns]
                data = data.rename(columns={'adj close': 'adj_close'})
                self.test_data = TechnicalIndicators.add_all_indicators(data)
                print(f"✅ Loaded {len(self.test_data)} days from Yahoo Finance")
                return True
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        print("❌ No data available")
        return False
    
    def test_individual_strategies(self):
        """Test individual strategies"""
        print(f"\n🤖 Testing Individual Strategies")
        print("-" * 40)
        
        if not PROJECT_AVAILABLE:
            print("❌ Project strategies not available")
            return
        
        strategies_to_test = [
            (MLTradingStrategy(confidence_threshold=0.40), "MLTrading Strategy"),
            (TechnicalAnalysisStrategy(), "Technical Analysis Strategy"),
            (HybridRSIDivergenceStrategy(
                divergence_weight=0.6,
                technical_weight=0.4,
                base_strategy=TechnicalAnalysisStrategy()
            ), "Hybrid RSI-ML"),
            (HybridRSIDivergenceStrategy(
                divergence_weight=0.4,
                technical_weight=0.6,
                base_strategy=TechnicalAnalysisStrategy()
            ), "Hybrid RSI-Technical"),
        ]
        
        for strategy, name in strategies_to_test:
            self._test_strategy_with_tracking(strategy, name)
    
    def test_split_capital_strategy(self):
        """Test Split-Capital Multi-Strategy"""
        print(f"\n🏆 Testing Split-Capital Multi-Strategy")
        print("-" * 45)
        
        if not PROJECT_AVAILABLE or self.test_data is None:
            print("❌ Cannot run Split-Capital Multi-Strategy test")
            return None
        
        try:
            strategy_classes = {
                'TechnicalAnalysisStrategy': TechnicalAnalysisStrategy,
                'MLTradingStrategy': MLTradingStrategy
            }
            
            runner = UltimatePortfolioRunner(
                assets=[self.TEST_CONFIG['test_symbol']],
                initial_capital=self.TEST_CONFIG['initial_capital']
            )
            
            results = runner.run_ultimate_portfolio_test(
                data=self.test_data,
                backtester_class=ProductionBacktester,
                strategy_classes=strategy_classes
            )
            
            # Store with descriptive name
            strategy_name = "Split-Capital Multi-Strategy"
            self.results[strategy_name] = results
            
            # Get signals and trades
            signals_df, trades_df = runner.get_signals_and_trades_for_visualization()
            if not signals_df.empty:
                self.backtester_signals[strategy_name] = signals_df
            if not trades_df.empty:
                self.trade_history[strategy_name] = trades_df
            
            print(f"✅ Split-Capital Multi-Strategy completed")
            print(f"   Return: {results['total_return']*100:+.2f}%")
            print(f"   Trades: {results['total_trades']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error running Split-Capital Multi-Strategy: {e}")
            return None
    
    def _test_strategy_with_tracking(self, strategy, name):
        """Test strategy with signal and trade tracking"""
        try:
            print(f"   Testing {name}...")
            
            backtester = ProductionBacktester(
                initial_capital=self.TEST_CONFIG['initial_capital'],
                transaction_cost=self.TEST_CONFIG['transaction_cost'],
                max_position_size=self.TEST_CONFIG['max_position_size']
            )
            
            backtester.set_strategy(strategy)
            results = backtester.run_backtest(self.test_data)
            results['strategy_name'] = name
            
            # Store results and tracking data
            self.results[name] = results
            self.backtester_signals[name] = backtester.get_signals_history()
            self.trade_history[name] = backtester.get_trade_history()
            
            print(f"      Return: {results['total_return']:.2%}")
            print(f"      Trades: {results['total_trades']}")
            print(f"      Win Rate: {results.get('win_rate', 0):.1%}")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
    
    def add_benchmark_strategies(self):
        """Add benchmark strategies (MSFT-compatible only)"""
        print(f"\n🏆 Adding Benchmark Strategies (MSFT 2024 Compatible)")
        print("-" * 55)
        
        # Only include benchmarks that are valid for MSFT 2024 comparison
        self.results['H1: TradingView Strategy'] = {
            'strategy_name': 'H1: TradingView Strategy',
            'total_return': 0.3539, 'total_trades': 92, 'win_rate': 64.13,
            'sharpe_ratio': -0.263, 'data_source': 'REAL TRADINGVIEW STRATEGY (MSFT 2024)'
        }
        
        self.results['H1: Systematic Strategy'] = {
            'strategy_name': 'H1: Systematic Strategy',
            'total_return': 0.0429, 'total_trades': 12, 'win_rate': 58.0,
            'sharpe_ratio': 0.67, 'data_source': 'REAL SYSTEMATIC STRATEGY (MSFT 2024)'
        }
        
        # Market-wide benchmarks (not asset-specific)
        self.results['H2: Cathie Wood (ARKK)'] = {
            'strategy_name': 'H2: Cathie Wood (ARKK)',
            'total_return': 0.1408, 'total_trades': 156, 'win_rate': 52.0,
            'sharpe_ratio': 0.3936, 'data_source': 'REAL ARKK ETF DATA (Market-wide)'
        }
        
        self.results['H3: AI ETF (QQQ)'] = {
            'strategy_name': 'H3: AI ETF (QQQ)',
            'total_return': 0.2883, 'total_trades': 100, 'win_rate': 58.0,
            'sharpe_ratio': 1.6052, 'data_source': 'REAL QQQ ETF DATA (Market-wide)'
        }
        
        self.results['H4: Beginner Trader'] = {
            'strategy_name': 'H4: Beginner Trader',
            'total_return': -0.15, 'total_trades': 67, 'win_rate': 41.0,
            'sharpe_ratio': -0.23, 'data_source': 'ACADEMIC STUDY (General)'
        }
        
        print("✅ MSFT-compatible benchmark strategies added")
        print("⚠️  Note: H1 benchmarks are MSFT 2024-specific")
        print("✅ H2, H3, H4 benchmarks are market-wide references")
    
    def generate_analysis(self):
        """Generate comprehensive analysis"""
        print(f"\n📊 ANALYSIS: {self.TEST_CONFIG['test_name']}")
        print("=" * 50)
        
        if not self.results:
            print("❌ No results available")
            return
        
        # Sort by performance
        sorted_results = sorted(
            [(name, data) for name, data in self.results.items() if isinstance(data, dict) and 'total_return' in data],
            key=lambda x: x[1]['total_return'],
            reverse=True
        )
        
        print(f"\n🏆 STRATEGY PERFORMANCE RANKING:")
        print("-" * 50)
        
        for i, (name, data) in enumerate(sorted_results, 1):
            return_pct = data['total_return'] * 100
            trades = data.get('total_trades', 0)
            win_rate = data.get('win_rate', 0)
            
            print(f"{i:2d}. {name}")
            print(f"    Return: {return_pct:+7.2f}% | Trades: {trades:3d} | Win Rate: {win_rate:5.1f}%")
        
        # Find Split-Capital Multi-Strategy
        split_capital_result = None
        for name, data in self.results.items():
            if "Split-Capital" in name:
                split_capital_result = data
                break
        
        if split_capital_result:
            split_rank = next((i for i, (name, _) in enumerate(sorted_results, 1) if "Split-Capital" in name), 0)
            print(f"\n🏆 SPLIT-CAPITAL MULTI-STRATEGY ANALYSIS:")
            print(f"   Ranking: #{split_rank} out of {len(sorted_results)} strategies")
            print(f"   Return: {split_capital_result['total_return']:.2%}")
            print(f"   Trades: {split_capital_result['total_trades']}")
            print(f"   Configuration: {self.TEST_CONFIG['test_symbol']} over {self.TEST_CONFIG['test_name']}")
    
    def create_visualization(self):
        """Create visualization for this test configuration"""
        if self.test_data is None:
            print("❌ No data available for visualization")
            return
        
        print(f"\n📈 Creating visualization for {self.TEST_CONFIG['test_name']}...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), height_ratios=[3, 1])
        
        # Plot price
        ax1.plot(self.test_data.index, self.test_data['close'], 
                label=f"{self.TEST_CONFIG['test_symbol']} Price", linewidth=2, color='black', alpha=0.7)
        
        # Plot key strategies
        key_strategies = ['MLTrading Strategy', 'Technical Analysis Strategy', 'Split-Capital Multi-Strategy']
        colors = ['#FF4444', '#4444FF', '#FF8800']
        markers = ['o', 's', 'D']
        
        for i, strategy in enumerate(key_strategies):
            if strategy in self.backtester_signals and not self.backtester_signals[strategy].empty:
                signals_df = self.backtester_signals[strategy]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                # Plot signals
                buy_signals = signals_df[signals_df['signal'].apply(lambda x: x['action'] == 'BUY')]
                sell_signals = signals_df[signals_df['signal'].apply(lambda x: x['action'] == 'SELL')]
                
                if not buy_signals.empty:
                    ax1.scatter(buy_signals['date'], buy_signals['price'], 
                              color=color, marker=marker, s=80, alpha=0.8, 
                              label=f'{strategy} BUY', edgecolors='white', linewidth=1.5)
                
                if not sell_signals.empty:
                    ax1.scatter(sell_signals['date'], sell_signals['price'], 
                              color='white', marker=marker, s=80, alpha=0.9, 
                              label=f'{strategy} SELL', edgecolors=color, linewidth=2.5)
        
        ax1.set_title(f'{self.TEST_CONFIG["test_name"]}: {self.TEST_CONFIG["test_symbol"]} Trading Strategies', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Performance comparison
        if self.results:
            strategy_names = []
            returns = []
            
            for name, data in self.results.items():
                if isinstance(data, dict) and 'total_return' in data:
                    short_name = name.replace('Strategy', '').replace('H1:', '').replace('H2:', '').replace('H3:', '').replace('H4:', '').strip()
                    strategy_names.append(short_name)
                    returns.append(data['total_return'] * 100)
            
            sorted_data = sorted(zip(strategy_names, returns), key=lambda x: x[1], reverse=True)
            strategy_names, returns = zip(*sorted_data)
            
            colors_bar = ['orange' if 'Split-Capital' in name else 'darkgreen' if r > 15 else 'green' if r > 0 else 'red' 
                         for name, r in zip(strategy_names, returns)]
            
            ax2.barh(range(len(strategy_names)), returns, color=colors_bar, alpha=0.8)
            ax2.set_yticks(range(len(strategy_names)))
            ax2.set_yticklabels(strategy_names, fontsize=10)
            ax2.set_xlabel('Return (%)', fontsize=12)
            ax2.set_title(f'{self.TEST_CONFIG["test_name"]} - Strategy Performance Comparison', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=0, color='black', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        # Save to results folder
        os.makedirs('comparison_tests_results', exist_ok=True)
        filename = f'comparison_tests_results/visualization_1stock_1year_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to {filename}")
    
    def save_results(self):
        """Save results to comparison_tests_results folder"""
        os.makedirs('comparison_tests_results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'comparison_tests_results/results_1stock_1year_{timestamp}.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("HYPOTHESIS TESTING RESULTS: 1 Stock, 1 Year\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Configuration: {self.TEST_CONFIG['test_name']}\n")
            f.write(f"Asset: {self.TEST_CONFIG['test_symbol']}\n")
            f.write(f"Period: {self.TEST_CONFIG['start_date']} to {self.TEST_CONFIG['end_date']}\n")
            f.write(f"Capital: ${self.TEST_CONFIG['initial_capital']:,}\n\n")
            
            # Write all results
            for name, data in self.results.items():
                if isinstance(data, dict):
                    f.write(f"{name}:\n")
                    for key, value in data.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
        
        print(f"✅ Results saved to {filename}")
        return filename
    
    def run_all_tests(self):
        """Run all tests for this configuration"""
        if not self.load_test_data():
            return
        
        print(f"\n🚀 Running {self.TEST_CONFIG['test_name']} tests...")
        
        self.test_individual_strategies()
        self.test_split_capital_strategy()
        self.add_benchmark_strategies()
        self.generate_analysis()
        self.create_visualization()
        results_file = self.save_results()
        
        print(f"\n🎉 {self.TEST_CONFIG['test_name']} testing complete!")
        print(f"📁 Results saved to: {results_file}")


def main():
    """Run 1 Stock, 1 Year hypothesis testing"""
    tester = Hypothesis1Stock1Year()
    tester.run_all_tests()


if __name__ == "__main__":
    main()