"""
HYPOTHESIS TESTING: Single Stock, 3 Months
Testing framework using reusable hypothesis_framework module

Configuration:
- Asset: MSFT (single stock focus)
- Period: Q4 2024 (3 months: October-December)
- Strategies: Individual + Split-Capital Multi-Strategy
- Capital: $10,000
- Benchmarks: Complete H1-H4 framework (period-adjusted)
- MINIMUM VIABLE PERIOD: 3 months is the minimum acceptable testing period
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

# Import reusable framework
from hypothesis_framework import (
    add_benchmarks_to_results,
    add_hypothesis_test_analysis, 
    get_test_description,
    get_date_range_for_test,
    get_assets_for_test,
    calculate_time_period_months
)

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

class Hypothesis1Stock3Months:
    """Hypothesis testing framework: Single Stock, 3 Months - MINIMUM VIABLE PERIOD"""
    
    def __init__(self):
        # Test configuration using framework
        self.test_type = '1stock_3months'
        start_date, end_date = get_date_range_for_test(self.test_type)
        assets = get_assets_for_test(self.test_type)
        
        self.TEST_CONFIG = {
            'test_name': get_test_description(self.test_type),
            'test_symbols': assets,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': 10000,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,
            'time_months': calculate_time_period_months(self.test_type)
        }
        
        self.results = {}
        self.backtester_signals = {}
        self.trade_history = {}
        self.test_data = {}
        
        print(f"🎯 HYPOTHESIS TESTING: {self.TEST_CONFIG['test_name']} 📊")
        print("=" * 60)
        print(f"📈 MINIMUM VIABLE PERIOD: 3 months testing!")
        print(f"Assets: {', '.join(self.TEST_CONFIG['test_symbols'])}")
        print(f"Period: {self.TEST_CONFIG['start_date']} to {self.TEST_CONFIG['end_date']}")
        print(f"Capital: ${self.TEST_CONFIG['initial_capital']:,}")
    
    def load_test_data(self):
        """Load test data for specified assets and period"""
        print(f"\n📊 Loading test data...")
        
        success_count = 0
        for symbol in self.TEST_CONFIG['test_symbols']:
            try:
                print(f"   Loading {symbol}...")
                
                if PROJECT_AVAILABLE:
                    # TRY 1: Polygon API (primary)
                    try:
                        api = get_data_api("polygon")
                        data = api.fetch_historical_data(
                            symbol, "1d",
                            start_date=self.TEST_CONFIG['start_date'],
                            end_date=self.TEST_CONFIG['end_date']
                        )
                        
                        if data is not None and not data.empty:
                            self.test_data[symbol] = TechnicalIndicators.add_all_indicators(data)
                            print(f"      ✅ {symbol}: {len(self.test_data[symbol])} days (Polygon)")
                            success_count += 1
                            continue
                        else:
                            print(f"      ⚠️ Polygon returned empty data for {symbol}")
                    except Exception as e:
                        print(f"      ⚠️ Polygon API failed for {symbol}: {e}")
                    
                    # TRY 2: Alpha Vantage API (backup)
                    try:
                        print(f"      Trying Alpha Vantage for {symbol}...")
                        api = get_data_api("alpha_vantage")
                        data = api.fetch_historical_data(
                            symbol, "1d",
                            start_date=self.TEST_CONFIG['start_date'],
                            end_date=self.TEST_CONFIG['end_date']
                        )
                        
                        if data is not None and not data.empty:
                            self.test_data[symbol] = TechnicalIndicators.add_all_indicators(data)
                            print(f"      ✅ {symbol}: {len(self.test_data[symbol])} days (Alpha Vantage)")
                            success_count += 1
                            continue
                        else:
                            print(f"      ⚠️ Alpha Vantage returned empty data for {symbol}")
                    except Exception as e:
                        print(f"      ⚠️ Alpha Vantage API failed for {symbol}: {e}")
                
                # TRY 3: Yahoo Finance (always available)
                print(f"      Trying Yahoo Finance for {symbol}...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=self.TEST_CONFIG['start_date'],
                    end=self.TEST_CONFIG['end_date'],
                    interval='1d'
                )
                
                if not data.empty:
                    # Fix column naming for consistency
                    data = data.rename(columns={
                    'Close': 'close', 
                    'Volume': 'volume',
                    'Open': 'open', 
                    'High': 'high', 
                    'Low': 'low',
                    'Adj Close': 'adj_close'
                })
                
                # Add indicators
                self.test_data[symbol] = TechnicalIndicators.add_all_indicators(data)
                
                print(f"      ✅ {symbol}: {len(self.test_data[symbol])} days (Yahoo Finance)")
                success_count += 1
        
            except Exception as e:
                print(f"      ❌ {symbol}: Error loading data - {e}")
        
        print(f"\n📈 Successfully loaded {success_count}/{len(self.TEST_CONFIG['test_symbols'])} assets")
        return success_count > 0
    
    def _test_strategy_with_tracking(self, strategy, strategy_name, data):
        """Test strategy and track results"""
        try:
            backtester = ProductionBacktester(
                initial_capital=self.TEST_CONFIG['initial_capital'],
                transaction_cost=self.TEST_CONFIG['transaction_cost'],
                max_position_size=self.TEST_CONFIG['max_position_size']
            )
            
            backtester.set_strategy(strategy)
            results = backtester.run_backtest(data)
            
            # Store results
            self.results[strategy_name] = results
            self.backtester_signals[strategy_name] = backtester.get_signals_history()
            self.trade_history[strategy_name] = backtester.get_trade_history()
            
            print(f"   ✅ {strategy_name}: {results['total_return']*100:+6.2f}% ({results['total_trades']} trades)")
            
        except Exception as e:
            print(f"   ❌ {strategy_name}: Error - {e}")
    
    def test_individual_strategies(self):
        """Test individual strategies on single stock - 3 MONTHS VERSION"""
        print(f"\n🤖 Testing Individual Strategies (Single Stock) - 3 MONTHS")
        print("-" * 55)
        
        if not PROJECT_AVAILABLE:
            print("❌ Project strategies not available")
            return
        
        # Use first (primary) asset for single stock test
        primary_symbol = self.TEST_CONFIG['test_symbols'][0]
        if primary_symbol not in self.test_data:
            print(f"❌ No data for {primary_symbol}")
            return
        
        strategies_to_test = [
            (MLTradingStrategy(confidence_threshold=0.40), "MLTrading Strategy"),
            (TechnicalAnalysisStrategy(), "Technical Analysis Strategy"),
        ]
        
        # CONDITIONAL: Only add hybrid strategies if project modules are fully available
        if PROJECT_AVAILABLE:
            try:
                # Test if HybridRSIDivergenceStrategy works
                test_strategy = HybridRSIDivergenceStrategy(
                    divergence_weight=0.6,
                    technical_weight=0.4,
                    base_strategy=TechnicalAnalysisStrategy()
                )
                
                # Add hybrid strategies if they work
                strategies_to_test.extend([
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
                ])
                print("   ✅ Including Hybrid RSI strategies")
                
            except Exception as e:
                print(f"   ⚠️ Skipping Hybrid RSI strategies due to error: {e}")
        
        for strategy, strategy_name in strategies_to_test:
            self._test_strategy_with_tracking(strategy, strategy_name, self.test_data[primary_symbol])
        
        print(f"\n📊 Individual strategies completed: {len(self.results)} successful strategies")
    
    def test_split_capital_strategy(self):
        """Test Split-Capital Multi-Strategy on single stock - 3 MONTHS"""
        print(f"\n🏆 Testing Split-Capital Multi-Strategy (Single Stock) - 3 MONTHS")
        print("-" * 65)
        
        if not PROJECT_AVAILABLE:
            print("❌ Cannot run Split-Capital Multi-Strategy test")
            return None
        
        # Use the primary symbol for single stock testing
        primary_symbol = self.TEST_CONFIG['test_symbols'][0]
        if primary_symbol not in self.test_data:
            print(f"❌ No data for {primary_symbol}")
            return None
        
        try:
            strategy_classes = {
                'TechnicalAnalysisStrategy': TechnicalAnalysisStrategy,
                'MLTradingStrategy': MLTradingStrategy
            }
            
            print(f"\nAsset: {primary_symbol}")
            print(f"Strategies: {list(strategy_classes.keys())}")
            print(f"Total Capital: ${self.TEST_CONFIG['initial_capital']:,}")
            
            # Create UltimatePortfolioRunner with single asset
            runner = UltimatePortfolioRunner(
                assets=[primary_symbol],
                initial_capital=self.TEST_CONFIG['initial_capital']
            )
            
            # Pass single asset data
            results = runner.run_ultimate_portfolio_test(
                data={primary_symbol: self.test_data[primary_symbol]},
                backtester_class=ProductionBacktester,
                strategy_classes=strategy_classes
            )
            
            # Store with descriptive name
            strategy_name = "Split-Capital Multi-Strategy"
            self.results[strategy_name] = results
            
            # Get signals and trades from UltimatePortfolioRunner
            try:
                signals_df, trades_df = runner.get_signals_and_trades_for_visualization()
                if not signals_df.empty:
                    self.backtester_signals[strategy_name] = signals_df
                    print(f"   📊 Collected {len(signals_df)} signals")
                if not trades_df.empty:
                    self.trade_history[strategy_name] = trades_df
                    print(f"   💼 Collected {len(trades_df)} trades")
            except Exception as e:
                print(f"   Warning: Could not extract signals/trades: {e}")
            
            time_months = self.TEST_CONFIG['time_months']
            monthly_freq = results['total_trades'] / time_months
            
            print(f"\n🎯 SPLIT-CAPITAL MULTI-STRATEGY PERFORMANCE (3 MONTHS, SINGLE STOCK):")
            print(f"   Portfolio Return: {results['total_return']*100:+6.2f}%")
            print(f"   Total Trades: {results['total_trades']}")
            print(f"   Trade Frequency: {monthly_freq:.1f} trades/month")
            print(f"   Final Value: ${results.get('final_value', 0):,.2f}")
            print(f"   Asset: {primary_symbol}")
            print(f"   Strategies: {len(strategy_classes)}")
            print(f"   Win Rate: {results.get('win_rate', 0)*100:.1f}%")
            print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error running Split-Capital Multi-Strategy: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_analysis(self):
        """Generate comprehensive analysis using framework - 3 MONTHS"""
        print(f"\n📊 ANALYSIS: {self.TEST_CONFIG['test_name']}")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results available")
            return
        
        # Sort by performance
        sorted_results = sorted(
            [(name, data) for name, data in self.results.items() if isinstance(data, dict) and 'total_return' in data],
            key=lambda x: x[1]['total_return'],
            reverse=True
        )
        
        print(f"\n🏆 STRATEGY PERFORMANCE RANKING (3 MONTHS, SINGLE STOCK):")
        print("-" * 65)
        
        time_months = self.TEST_CONFIG['time_months']
        
        for i, (name, data) in enumerate(sorted_results, 1):
            return_pct = data['total_return'] * 100
            trades = data.get('total_trades', 0)
            win_rate = data.get('win_rate', 0)
            monthly_trades = trades / time_months
            
            print(f"{i:2d}. {name}")
            print(f"    Return: {return_pct:+7.2f}% | Trades: {trades:3d} ({monthly_trades:.1f}/mo) | Win Rate: {win_rate:5.1f}%")
        
        # Use framework for hypothesis analysis
        add_hypothesis_test_analysis(self.results, "Split-Capital Multi-Strategy")
    
    def create_visualization(self):
        """Create visualization for 3 months single stock test"""
        if not self.test_data:
            print("❌ No data available for visualization")
            return
        
        print(f"\n📈 Creating visualization...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), height_ratios=[3, 1])
        
        # Plot price for single asset
        primary_symbol = self.TEST_CONFIG['test_symbols'][0]
        if primary_symbol in self.test_data:
            data = self.test_data[primary_symbol]
            ax1.plot(data.index, data['close'], 
                    label=f"{primary_symbol} Price", linewidth=2, color='blue', alpha=0.8)
        
        ax1.set_title(f'{self.TEST_CONFIG["test_name"]}: Trading Strategies', 
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
            
            colors_bar = ['orange' if 'Split-Capital' in name else 'darkgreen' if r > 10 else 'green' if r > 0 else 'red' 
                         for name, r in zip(strategy_names, returns)]
            
            ax2.barh(range(len(strategy_names)), returns, color=colors_bar, alpha=0.8)
            ax2.set_yticks(range(len(strategy_names)))
            ax2.set_yticklabels(strategy_names, fontsize=10)
            ax2.set_xlabel('Return (%)', fontsize=12)
            ax2.set_title(f'{self.TEST_CONFIG["test_name"]} - Performance Comparison', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=0, color='black', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        # Save to results folder
        os.makedirs('comparison_tests_results', exist_ok=True)
        filename = f'comparison_tests_results/visualization_{self.test_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to {filename}")
    
    def save_results(self):
        """Save results to comparison_tests_results folder - 3 MONTHS"""
        os.makedirs('comparison_tests_results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'comparison_tests_results/results_{self.test_type}_{timestamp}.txt'
        
        # Add encoding='utf-8' to handle emoji characters
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"HYPOTHESIS TESTING RESULTS: {self.TEST_CONFIG['test_name']}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Configuration: {self.test_type}\n")
            f.write(f"Assets: {', '.join(self.TEST_CONFIG['test_symbols'])}\n")
            f.write(f"Period: {self.TEST_CONFIG['start_date']} to {self.TEST_CONFIG['end_date']}\n")
            f.write(f"Capital: ${self.TEST_CONFIG['initial_capital']:,}\n")
            f.write(f"Duration: {self.TEST_CONFIG['time_months']} months\n")
            f.write(f"NOTE: 3 months is the MINIMUM VIABLE PERIOD for algorithmic testing\n\n")
            
            # Write all results - convert complex objects to safe strings
            for name, data in self.results.items():
                if isinstance(data, dict):
                    f.write(f"{name}:\n")
                    for key, value in data.items():
                        # Handle complex objects that might contain emojis or problematic characters
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            f.write(f"  {key}: {value}\n")
                        else:
                            # Convert to string and clean up
                            clean_value = str(value).encode('ascii', 'ignore').decode('ascii')
                            f.write(f"  {key}: {clean_value}\n")
                    f.write("\n")
        
        print(f"✅ Results saved to {filename}")
        return filename
    
    def run_all_tests(self):
        """Run all tests for 3 months single stock configuration"""

        print(f"\n🔍 INITIAL CONFIG DEBUG:")
        print(f"   Test type: {self.test_type}")
        print(f"   Framework assets: {get_assets_for_test(self.test_type)}")
        print(f"   Config assets: {self.TEST_CONFIG['test_symbols']}")
        
        if not self.load_test_data():
            return
        
        print(f"\n🚀 Running {self.TEST_CONFIG['test_name']} tests...")
        
        # Test strategies
        self.test_individual_strategies()
        self.test_split_capital_strategy()
        
        # Add benchmarks using framework
        add_benchmarks_to_results(self.results, self.test_type)
        
        # Generate analysis
        self.generate_analysis()
        self.create_visualization()
        results_file = self.save_results()
        
        print(f"\n🎉 {self.TEST_CONFIG['test_name']} testing complete!")
        print(f"📁 Results saved to: {results_file}")
        print(f"\n📊 MINIMUM VIABLE PERIOD TESTING:")
        print(f"   ✅ 3 months is the minimum acceptable period for reliable algorithmic trading")
        print(f"   ⚠️ Consider 6+ months for production trading decisions")


def main():
    """Run Single Stock, 3 Months hypothesis testing"""
    tester = Hypothesis1Stock3Months()
    tester.run_all_tests()


if __name__ == "__main__":
    main()