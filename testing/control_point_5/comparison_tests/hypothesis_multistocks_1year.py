"""
HYPOTHESIS TESTING: Multiple Stocks, 1 Year
Testing framework for comparing trading strategies across multiple assets over full year

Configuration:
- Assets: MSFT, AAPL, GOOGL, NVDA
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


class HypothesisMultiStocks1Year:
    """Hypothesis testing framework: Multiple Stocks, 1 Year"""
    
    def __init__(self):
        self.results = {}
        self.backtester_signals = {}
        self.trade_history = {}
        
        # Test configuration
        self.TEST_CONFIG = {
            'test_name': 'Multiple Stocks, 1 Year',
            'test_symbols': ['MSFT', 'AAPL', 'GOOGL', 'NVDA'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 10000,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,
            'data_source': 'polygon'
        }
        
        self.assets_data = {}
        print("🎯 HYPOTHESIS TESTING: Multiple Stocks, 1 Year")
        print("=" * 55)
        print(f"Assets: {', '.join(self.TEST_CONFIG['test_symbols'])}")
        print(f"Period: {self.TEST_CONFIG['start_date']} to {self.TEST_CONFIG['end_date']}")
        print(f"Capital: ${self.TEST_CONFIG['initial_capital']:,}")
        print(f"Test: Multiple assets over full year")
    
    def load_test_data(self):
        """Load test data for all assets"""
        print(f"\n📊 Loading test data for {self.TEST_CONFIG['test_name']}...")
        
        success_count = 0
        for symbol in self.TEST_CONFIG['test_symbols']:
            try:
                print(f"   Loading {symbol}...")
                
                if PROJECT_AVAILABLE:
                    api = get_data_api("polygon")
                    data = api.fetch_historical_data(
                        symbol, "1d",
                        start_date=self.TEST_CONFIG['start_date'],
                        end_date=self.TEST_CONFIG['end_date']
                    )
                    
                    if data is not None and not data.empty:
                        self.assets_data[symbol] = TechnicalIndicators.add_all_indicators(data)
                        print(f"      ✅ {symbol}: {len(self.assets_data[symbol])} days")
                        success_count += 1
                    else:
                        # Fallback to yfinance
                        print(f"      Trying Yahoo Finance for {symbol}...")
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(
                            start=self.TEST_CONFIG['start_date'],
                            end=self.TEST_CONFIG['end_date'],
                            interval='1d'
                        )
                        if not data.empty:
                            data.columns = [col.lower() for col in data.columns]
                            data = data.rename(columns={'adj close': 'adj_close'})
                            self.assets_data[symbol] = TechnicalIndicators.add_all_indicators(data)
                            print(f"      ✅ {symbol}: {len(self.assets_data[symbol])} days (fallback)")
                            success_count += 1
                        else:
                            print(f"      ❌ {symbol}: Failed to load data")
                            
            except Exception as e:
                print(f"      ❌ {symbol}: Error loading data - {e}")
        
        print(f"\n📈 Successfully loaded {success_count}/{len(self.TEST_CONFIG['test_symbols'])} assets")
        return success_count > 0
    
    def test_individual_strategies_multi_asset(self):
        """Test individual strategies across multiple assets"""
        print(f"\n🤖 Testing Individual Strategies (Best Asset Performance)")
        print("-" * 55)
        
        if not PROJECT_AVAILABLE or not self.assets_data:
            print("❌ Cannot test individual strategies")
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
        
        for strategy, strategy_name in strategies_to_test:
            print(f"\n📊 Testing {strategy_name} across all assets:")
            
            best_return = -999
            best_asset = None
            best_results = None
            
            # Test strategy on each asset and keep the best performing
            for asset, data in self.assets_data.items():
                try:
                    backtester = ProductionBacktester(
                        initial_capital=self.TEST_CONFIG['initial_capital'],
                        transaction_cost=self.TEST_CONFIG['transaction_cost'],
                        max_position_size=self.TEST_CONFIG['max_position_size']
                    )
                    
                    backtester.set_strategy(strategy)
                    results = backtester.run_backtest(data)
                    
                    print(f"   {asset}: {results['total_return']*100:+6.2f}% ({results['total_trades']} trades)")
                    
                    if results['total_return'] > best_return:
                        best_return = results['total_return']
                        best_asset = asset
                        best_results = results
                        best_results['best_asset'] = asset
                        best_results['strategy_name'] = f"{strategy_name} (Best: {asset})"
                        
                        # Store signals from best performing asset
                        self.backtester_signals[strategy_name] = backtester.get_signals_history()
                        self.trade_history[strategy_name] = backtester.get_trade_history()
                
                except Exception as e:
                    print(f"   {asset}: ❌ Error - {e}")
            
            if best_results:
                self.results[strategy_name] = best_results
                print(f"   🏆 Best: {best_asset} with {best_return*100:+.2f}% return")
    
    def test_split_capital_multi_strategy(self):
        """Test Split-Capital Multi-Strategy across multiple assets"""
        print(f"\n🏆 Testing Split-Capital Multi-Strategy (Multiple Assets)")
        print("-" * 60)
        
        if not PROJECT_AVAILABLE or not self.assets_data:
            print("❌ Cannot run Split-Capital Multi-Strategy test")
            return None
        
        try:
            strategy_classes = {
                'TechnicalAnalysisStrategy': TechnicalAnalysisStrategy,
                'MLTradingStrategy': MLTradingStrategy
            }
            
            # Use all available assets
            assets = list(self.assets_data.keys())
            
            # Create custom strategy config optimized for multi-asset
            # (Based on portfolio optimization: Technical Analysis works better without AAPL)
            custom_strategies_config = {
                'Technical Analysis': {
                    'class': 'TechnicalAnalysisStrategy',
                    'params': {
                        'sma_short': 20, 'sma_long': 50,
                        'rsi_oversold': 30, 'rsi_overbought': 70
                    }
                },
                'MLTrading Strategy': {
                    'class': 'MLTradingStrategy', 
                    'params': {
                        'confidence_threshold': 0.40,
                        'rsi_oversold': 30, 'rsi_overbought': 70
                    }
                }
            }
            
            print(f"Assets: {assets}")
            print(f"Strategies: {list(custom_strategies_config.keys())}")
            print(f"Total combinations: {len(custom_strategies_config) * len(assets)}")
            
            # Run portfolio across multiple assets
            total_return = 0
            total_trades = 0
            combinations_count = 0
            all_signals = []
            all_trades = []
            
            capital_per_combination = self.TEST_CONFIG['initial_capital'] // (len(custom_strategies_config) * len(assets))
            combination_weight = 1.0 / (len(custom_strategies_config) * len(assets))
            
            print(f"Capital per combination: ${capital_per_combination:,}")
            
            detailed_results = {}
            
            for strategy_name, strategy_config in custom_strategies_config.items():
                strategy_results = {}
                
                for asset in assets:
                    if asset in self.assets_data:
                        print(f"\n📊 {strategy_name} on {asset} (${capital_per_combination:,}):")
                        
                        try:
                            # Create strategy
                            strategy_class = strategy_classes[strategy_config['class']]
                            strategy = strategy_class(**strategy_config['params'])
                            
                            # Run backtest
                            backtester = ProductionBacktester(
                                initial_capital=capital_per_combination,
                                transaction_cost=self.TEST_CONFIG['transaction_cost'],
                                max_position_size=1.0
                            )
                            
                            backtester.set_strategy(strategy)
                            results = backtester.run_backtest(self.assets_data[asset])
                            
                            combination_result = {
                                'return': results['total_return'],
                                'trades': results['total_trades'],
                                'win_rate': results.get('win_rate', 0),
                                'final_value': capital_per_combination * (1 + results['total_return'])
                            }
                            
                            strategy_results[asset] = combination_result
                            total_return += results['total_return'] * combination_weight
                            total_trades += results['total_trades']
                            combinations_count += 1
                            
                            print(f"      Return: {results['total_return']*100:+6.2f}%")
                            print(f"      Trades: {results['total_trades']:2d}")
                            
                            # Collect signals for visualization
                            signals = backtester.get_signals_history()
                            trades = backtester.get_trade_history()
                            
                            if not signals.empty:
                                signals['strategy_source'] = f"{strategy_name}_{asset}"
                                all_signals.append(signals)
                            
                            if not trades.empty:
                                trades['strategy_source'] = f"{strategy_name}_{asset}"
                                all_trades.append(trades)
                            
                        except Exception as e:
                            print(f"      ❌ Error: {e}")
                
                detailed_results[strategy_name] = strategy_results
            
            # Create combined results
            strategy_name = "Split-Capital Multi-Strategy"
            combined_results = {
                'strategy_name': strategy_name,
                'total_return': total_return,
                'total_trades': total_trades,
                'combinations': combinations_count,
                'assets': assets,
                'detailed_breakdown': detailed_results,
                'methodology': 'Portfolio Manager (multiple assets)'
            }
            
            self.results[strategy_name] = combined_results
            
            # Combine signals and trades
            if all_signals:
                combined_signals = pd.concat(all_signals, ignore_index=True)
                self.backtester_signals[strategy_name] = combined_signals
            
            if all_trades:
                combined_trades = pd.concat(all_trades, ignore_index=True)
                self.trade_history[strategy_name] = combined_trades
            
            print(f"\n🎯 SPLIT-CAPITAL MULTI-STRATEGY PERFORMANCE:")
            print(f"   Portfolio Return: {total_return*100:+6.2f}%")
            print(f"   Total Trades: {total_trades}")
            print(f"   Trade Frequency: {total_trades/12:.1f} trades/month")
            print(f"   Assets: {len(assets)}")
            print(f"   Combinations: {combinations_count}")
            
            return combined_results
            
        except Exception as e:
            print(f"❌ Error running Split-Capital Multi-Strategy: {e}")
            return None
    
    def add_benchmark_strategies(self):
        """Add benchmark strategies"""
        print(f"\n🏆 Adding Benchmark Strategies")
        print("-" * 35)
        
        # Same benchmarks as single stock test
        benchmarks = {
            'H1: TradingView Strategy': {'total_return': 0.3539, 'total_trades': 92, 'win_rate': 64.13},
            'H1: Systematic Strategy': {'total_return': 0.0429, 'total_trades': 12, 'win_rate': 58.0},
            'H2: Ray Dalio': {'total_return': 0.0561, 'total_trades': 4, 'win_rate': 72.0},
            'H2: Cathie Wood': {'total_return': 0.1408, 'total_trades': 156, 'win_rate': 52.0},
            'H3: Robo-Advisor': {'total_return': 0.089, 'total_trades': 24, 'win_rate': 63.0},
            'H3: AI ETF': {'total_return': 0.2883, 'total_trades': 100, 'win_rate': 58.0},
            'H4: Beginner Trader': {'total_return': -0.15, 'total_trades': 67, 'win_rate': 41.0}
        }
        
        for name, data in benchmarks.items():
            data['strategy_name'] = name
            self.results[name] = data
        
        print("✅ Benchmark strategies added")
    
    def generate_analysis(self):
        """Generate comprehensive analysis"""
        print(f"\n📊 ANALYSIS: {self.TEST_CONFIG['test_name']}")
        print("=" * 55)
        
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
        print("-" * 55)
        
        for i, (name, data) in enumerate(sorted_results, 1):
            return_pct = data['total_return'] * 100
            trades = data.get('total_trades', 0)
            win_rate = data.get('win_rate', 0)
            
            print(f"{i:2d}. {name}")
            print(f"    Return: {return_pct:+7.2f}% | Trades: {trades:3d} | Win Rate: {win_rate:5.1f}%")
        
        # Find Split-Capital Multi-Strategy
        split_capital_result = None
        split_rank = 0
        for name, data in self.results.items():
            if "Split-Capital" in name:
                split_capital_result = data
                split_rank = next((i for i, (n, _) in enumerate(sorted_results, 1) if "Split-Capital" in n), 0)
                break
        
        if split_capital_result:
            print(f"\n🏆 SPLIT-CAPITAL MULTI-STRATEGY ANALYSIS:")
            print(f"   Ranking: #{split_rank} out of {len(sorted_results)} strategies")
            print(f"   Return: {split_capital_result['total_return']:.2%}")
            print(f"   Trades: {split_capital_result['total_trades']}")
            print(f"   Assets: {len(split_capital_result.get('assets', []))}")
            print(f"   Combinations: {split_capital_result.get('combinations', 0)}")
            print(f"   Configuration: Multiple assets over {self.TEST_CONFIG['test_name']}")
            
            # Show detailed breakdown if available
            if 'detailed_breakdown' in split_capital_result:
                print(f"\n   Detailed Performance Breakdown:")
                for strategy_name, assets_results in split_capital_result['detailed_breakdown'].items():
                    for asset, result in assets_results.items():
                        print(f"     {strategy_name}-{asset}: {result['return']*100:+6.2f}% ({result['trades']} trades)")
    
    def create_visualization(self):
        """Create visualization for multiple stocks test"""
        if not self.assets_data:
            print("❌ No data available for visualization")
            return
        
        print(f"\n📈 Creating visualization for {self.TEST_CONFIG['test_name']}...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 16), height_ratios=[3, 1])
        
        # Plot all asset prices
        colors = ['black', 'blue', 'green', 'purple']
        for i, (symbol, data) in enumerate(self.assets_data.items()):
            ax1.plot(data.index, data['close'], 
                    label=f"{symbol} Price", linewidth=2, color=colors[i % len(colors)], alpha=0.7)
        
        ax1.set_title(f'{self.TEST_CONFIG["test_name"]}: Multiple Assets Trading Strategies', 
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
        filename = f'comparison_tests_results/visualization_multistocks_1year_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to {filename}")
    
    def save_results(self):
        """Save results to comparison_tests_results folder"""
        os.makedirs('comparison_tests_results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'comparison_tests_results/results_multistocks_1year_{timestamp}.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("HYPOTHESIS TESTING RESULTS: Multiple Stocks, 1 Year\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Configuration: {self.TEST_CONFIG['test_name']}\n")
            f.write(f"Assets: {', '.join(self.TEST_CONFIG['test_symbols'])}\n")
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
        
        self.test_individual_strategies_multi_asset()
        self.test_split_capital_multi_strategy()
        self.add_benchmark_strategies()
        self.generate_analysis()
        self.create_visualization()
        results_file = self.save_results()
        
        print(f"\n🎉 {self.TEST_CONFIG['test_name']} testing complete!")
        print(f"📁 Results saved to: {results_file}")


def main():
    """Run Multiple Stocks, 1 Year hypothesis testing"""
    tester = HypothesisMultiStocks1Year()
    tester.run_all_tests()


if __name__ == "__main__":
    main()