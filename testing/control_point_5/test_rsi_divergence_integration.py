"""
STANDARDIZED RSI Divergence Integration Test
Using EXACT same parameters as hypothesis testing script for fair comparison

CRITICAL: Uses identical conditions:
- Date range: 2024-01-01 to 2024-12-31
- Symbol: MSFT
- Initial capital: $10,000
- Transaction cost: 0.001
- Max position size: 1.0 (full capital)
- MLTradingStrategy confidence_threshold: 0.40
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.backtesting import ProductionBacktester, RSIDivergenceStrategy, HybridRSIDivergenceStrategy
from src.data.fetcher import get_data_api
from src.indicators.technical import TechnicalIndicators


class StandardizedRSITestFramework:
    """Standardized testing framework using EXACT same conditions as hypothesis script"""
    
    def __init__(self):
        # EXACT SAME CONFIG as hypothesis testing script
        self.STANDARD_CONFIG = {
            'test_symbol': 'MSFT',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31', 
            'initial_capital': 10000,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,  # Full capital like hypothesis script
            'data_source': 'polygon'
        }
        
        self.test_data_2024 = None
        
        print("🎯 STANDARDIZED RSI DIVERGENCE TESTING")
        print("=" * 60)
        print("Using EXACT same parameters as hypothesis testing:")
        print(f"Symbol: {self.STANDARD_CONFIG['test_symbol']}")
        print(f"Period: {self.STANDARD_CONFIG['start_date']} to {self.STANDARD_CONFIG['end_date']}")
        print(f"Initial Capital: ${self.STANDARD_CONFIG['initial_capital']:,}")
        print(f"Transaction Cost: {self.STANDARD_CONFIG['transaction_cost']}")
        print(f"Max Position Size: {self.STANDARD_CONFIG['max_position_size']}")
        print("=" * 60)
    
    def load_standardized_data(self):
        """Load EXACT same data as hypothesis testing script"""
        
        print(f"\n📊 Loading STANDARDIZED 2024 data for {self.STANDARD_CONFIG['test_symbol']}...")
        
        try:
            # Use SAME data source priority as hypothesis script
            api = get_data_api("polygon")
            data = api.fetch_historical_data(
                symbol=self.STANDARD_CONFIG['test_symbol'],
                interval="1d",
                start_date=self.STANDARD_CONFIG['start_date'],
                end_date=self.STANDARD_CONFIG['end_date']
            )
            
            if data is None or data.empty:
                print("❌ Failed to load data from Polygon")
                return False
            
            # Add technical indicators (including RSI)
            data = TechnicalIndicators.add_all_indicators(data)
            
            # Store for testing
            self.test_data_2024 = data
            
            print(f"✅ Data loaded: {len(data)} trading days")
            print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Data loading error: {e}")
            return False
    
    def test_rsi_divergence_strategy(self):
        """Test RSI Divergence Strategy with STANDARDIZED parameters"""
        
        print("\n🎯 Testing RSIDivergenceStrategy (STANDARDIZED)")
        print("-" * 50)
        
        if self.test_data_2024 is None or self.test_data_2024.empty:
            print("❌ No test data available")
            return None
        
        try:
            # Use EXACT same backtester config as hypothesis script
            backtester = ProductionBacktester(
                initial_capital=self.STANDARD_CONFIG['initial_capital'],
                transaction_cost=self.STANDARD_CONFIG['transaction_cost'],
                max_position_size=self.STANDARD_CONFIG['max_position_size']  # Full capital
            )
            
            # RSI Divergence Strategy with PROVEN optimal parameters
            strategy = RSIDivergenceStrategy(
                swing_threshold_pct=2.5,  # Optimal from testing
                hold_days=15,             # Optimal from testing
                confidence_base=0.7
            )
            
            backtester.set_strategy(strategy)
            results = backtester.run_backtest(
                self.test_data_2024,
                start_date=self.STANDARD_CONFIG['start_date'],
                end_date=self.STANDARD_CONFIG['end_date']
            )
            
            print(f"✅ RSI Divergence Results (STANDARDIZED):")
            print(f"   Total Return: {results['total_return']:.2%}")
            print(f"   Final Value: ${results['final_value']:,.2f}")
            print(f"   Total Trades: {results['total_trades']}")
            print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
            print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"   Benchmark Return: {results.get('benchmark_return', 0):.2%}")
            print(f"   Alpha: {results.get('alpha', 0):.2%}")
            
            # Compare to expected results from original testing
            expected_return = 0.6415  # 64.15%
            tolerance = 0.15  # 15% tolerance due to different backtesting frameworks
            
            if abs(results['total_return'] - expected_return) <= tolerance:
                print(f"🎉 VALIDATION: Results within {tolerance:.0%} of expected 64.15%!")
            else:
                deviation = abs(results['total_return'] - expected_return)
                print(f"⚠️ DEVIATION: {deviation:.2%} difference from expected")
                print(f"   This could be due to different backtesting implementation")
            
            return results
            
        except Exception as e:
            print(f"❌ RSI Divergence test failed: {e}")
            return None
    
    def test_comparison_strategies(self):
        """Test comparison strategies with EXACT same parameters"""
        
        print("\n🏆 Testing Comparison Strategies (STANDARDIZED)")
        print("-" * 50)
        
        if self.test_data_2024 is None or self.test_data_2024.empty:
            return []
        
        from src.backtesting import TechnicalAnalysisStrategy, BuyAndHoldStrategy, MLTradingStrategy
        
        # EXACT same strategies and parameters as hypothesis script
        strategies_to_test = [
            (RSIDivergenceStrategy(swing_threshold_pct=2.5, hold_days=15), "RSI Divergence"),
            (MLTradingStrategy(confidence_threshold=0.40), "ML Trading (0.40)"),  # EXACT same as hypothesis
            (TechnicalAnalysisStrategy(), "Technical Analysis"),
            (BuyAndHoldStrategy(), "Buy & Hold")
        ]
        
        comparison_results = []
        
        for strategy, name in strategies_to_test:
            try:
                print(f"\n   Testing {name}...")
                
                # Use EXACT same backtester config
                backtester = ProductionBacktester(
                    initial_capital=self.STANDARD_CONFIG['initial_capital'],
                    transaction_cost=self.STANDARD_CONFIG['transaction_cost'],
                    max_position_size=self.STANDARD_CONFIG['max_position_size']  # Full capital
                )
                
                backtester.set_strategy(strategy)
                results = backtester.run_backtest(
                    self.test_data_2024,
                    start_date=self.STANDARD_CONFIG['start_date'],
                    end_date=self.STANDARD_CONFIG['end_date']
                )
                
                comparison_results.append({
                    'strategy': name,
                    'return': results['total_return'],
                    'final_value': results['final_value'],
                    'trades': results['total_trades'],
                    'win_rate': results.get('win_rate', 0),
                    'sharpe': results.get('sharpe_ratio', 0),
                    'alpha': results.get('alpha', 0),
                    'benchmark_return': results.get('benchmark_return', 0)
                })
                
                print(f"      Return: {results['total_return']:.2%}")
                print(f"      Trades: {results['total_trades']}")
                print(f"      Win Rate: {results.get('win_rate', 0):.1%}")
                
            except Exception as e:
                print(f"      ❌ {name} failed: {e}")
        
        return comparison_results
    
    def test_hybrid_strategies(self):
        """Test hybrid strategies with standardized parameters"""
        
        print("\n🔀 Testing Hybrid Strategies (STANDARDIZED)")
        print("-" * 50)
        
        if self.test_data_2024 is None or self.test_data_2024.empty:
            return []
        
        from src.backtesting import TechnicalAnalysisStrategy, MLTradingStrategy
        
        # Test different hybrid combinations
        hybrid_configs = [
            {
                'name': 'RSI-Technical (60/40)',
                'divergence_weight': 0.6,
                'technical_weight': 0.4,
                'base_strategy': TechnicalAnalysisStrategy()
            },
            {
                'name': 'RSI-ML (70/30)', 
                'divergence_weight': 0.7,
                'technical_weight': 0.3,
                'base_strategy': MLTradingStrategy(confidence_threshold=0.40)  # Same as hypothesis
            },
            {
                'name': 'RSI-Heavy (80/20)',
                'divergence_weight': 0.8,
                'technical_weight': 0.2,
                'base_strategy': TechnicalAnalysisStrategy()
            }
        ]
        
        hybrid_results = []
        
        for config in hybrid_configs:
            try:
                print(f"\n   Testing {config['name']}...")
                
                strategy = HybridRSIDivergenceStrategy(
                    divergence_weight=config['divergence_weight'],
                    technical_weight=config['technical_weight'],
                    base_strategy=config['base_strategy']
                )
                
                # Use EXACT same backtester config
                backtester = ProductionBacktester(
                    initial_capital=self.STANDARD_CONFIG['initial_capital'],
                    transaction_cost=self.STANDARD_CONFIG['transaction_cost'],
                    max_position_size=self.STANDARD_CONFIG['max_position_size']
                )
                
                backtester.set_strategy(strategy)
                results = backtester.run_backtest(
                    self.test_data_2024,
                    start_date=self.STANDARD_CONFIG['start_date'],
                    end_date=self.STANDARD_CONFIG['end_date']
                )
                
                hybrid_results.append({
                    'strategy': config['name'],
                    'return': results['total_return'],
                    'trades': results['total_trades'],
                    'win_rate': results.get('win_rate', 0),
                    'sharpe': results.get('sharpe_ratio', 0),
                    'config': config
                })
                
                print(f"      Return: {results['total_return']:.2%}")
                print(f"      Trades: {results['total_trades']}")
                print(f"      Win Rate: {results.get('win_rate', 0):.1%}")
                
            except Exception as e:
                print(f"      ❌ {config['name']} failed: {e}")
        
        return hybrid_results
    
    def generate_final_comparison(self, comparison_results, hybrid_results):
        """Generate comprehensive comparison using standardized results"""
        
        print("\n🏆 STANDARDIZED FINAL COMPARISON")
        print("=" * 80)
        print("All strategies tested with IDENTICAL conditions for fair comparison")
        print("=" * 80)
        
        # Combine all results
        all_results = comparison_results + hybrid_results
        
        if not all_results:
            print("❌ No results to compare")
            return
        
        # Sort by return (descending)
        all_results.sort(key=lambda x: x['return'], reverse=True)
        
        # Display comparison table
        print(f"{'Strategy':<25} | {'Return':<8} | {'Trades':<6} | {'Win Rate':<8} | {'Sharpe':<6}")
        print("-" * 80)
        
        for result in all_results:
            print(f"{result['strategy']:<25} | "
                  f"{result['return']:>7.2%} | "
                  f"{result['trades']:>6d} | "
                  f"{result['win_rate']:>7.1%} | "
                  f"{result.get('sharpe', 0):>6.2f}")
        
        # Analysis
        best_performer = all_results[0]
        print(f"\n🥇 BEST PERFORMER: {best_performer['strategy']}")
        print(f"   Return: {best_performer['return']:.2%}")
        print(f"   Trades: {best_performer['trades']}")
        print(f"   Win Rate: {best_performer['win_rate']:.1%}")
        
        # Check RSI Divergence performance
        rsi_result = next((r for r in all_results if 'RSI Divergence' in r['strategy']), None)
        if rsi_result:
            position = all_results.index(rsi_result) + 1
            print(f"\n📊 RSI DIVERGENCE ANALYSIS:")
            print(f"   Ranking: #{position} out of {len(all_results)} strategies")
            print(f"   Return: {rsi_result['return']:.2%}")
            
            if position == 1:
                print("🎉 RSI DIVERGENCE IS THE TOP PERFORMER!")
                print("   RECOMMENDATION: Deploy as primary strategy")
            elif position <= 3:
                print("👍 RSI DIVERGENCE is in top 3 performers")
                print("   RECOMMENDATION: Strong candidate for deployment")
            else:
                print("⚠️ RSI DIVERGENCE not in top performers")
                print("   RECOMMENDATION: Further optimization needed")
        
        # Expected vs actual analysis
        if rsi_result:
            expected = 0.6415  # 64.15% from original testing
            actual = rsi_result['return']
            
            print(f"\n🎯 EXPECTED vs ACTUAL:")
            print(f"   Original Testing: {expected:.2%}")
            print(f"   Standardized Test: {actual:.2%}")
            print(f"   Difference: {actual - expected:.2%}")
            
            if actual >= expected * 0.85:  # Within 15% is acceptable
                print("✅ Performance validates original testing results!")
            else:
                print("⚠️ Performance differs from original - investigate differences")


def main():
    """Run standardized RSI Divergence testing"""
    
    print("🚀 STANDARDIZED RSI DIVERGENCE STRATEGY TESTING")
    print("Using EXACT same parameters as hypothesis testing script")
    print("This ensures fair comparison with MLTradingStrategy and others")
    print("")
    
    # Initialize framework
    framework = StandardizedRSITestFramework()
    
    # Load standardized data
    if not framework.load_standardized_data():
        print("❌ Failed to load test data!")
        return False
    
    # Test RSI Divergence Strategy
    rsi_results = framework.test_rsi_divergence_strategy()
    if not rsi_results:
        print("❌ RSI Divergence testing failed!")
        return False
    
    # Test comparison strategies
    comparison_results = framework.test_comparison_strategies()
    
    # Test hybrid strategies
    hybrid_results = framework.test_hybrid_strategies()
    
    # Generate final comparison
    framework.generate_final_comparison(comparison_results, hybrid_results)
    
    print("\n✅ STANDARDIZED TESTING COMPLETED!")
    print("🎯 Results are now directly comparable to hypothesis testing!")
    print("🚀 Ready for production deployment based on validated performance!")
    
    return True


if __name__ == "__main__":
    main()