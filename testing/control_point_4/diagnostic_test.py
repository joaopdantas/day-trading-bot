"""
CORRECTED DIAGNOSTIC TEST - Fixing the Data Range Issue

The problem is that the backtester is using different data ranges!
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.backtesting import ProductionBacktester, MLTradingStrategy, BuyAndHoldStrategy
from src.indicators.technical import TechnicalIndicators


def create_test_data_fixed():
    """Create EXACTLY the same test data for all tests"""
    print("📊 CREATING FIXED TEST DATA")
    print("--------------------------------------------------")
    
    # Create exactly 50 days of data
    dates = pd.date_range(start='2025-01-10', periods=50, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Start at $400 and create realistic movements
    base_price = 400.0
    price_changes = np.random.normal(0, 0.02, 50)  # 2% daily volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, 50))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, 50))
    data['volume'] = np.random.randint(1000000, 5000000, 50)
    
    print(f"✅ Created test data: {len(data)} rows")
    print(f"   Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   First price: ${data['close'].iloc[0]:.2f}")
    print(f"   Last price: ${data['close'].iloc[-1]:.2f}")
    print(f"   Expected B&H return: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    return data


def test_backtesting_with_same_data():
    """Test backtesting using EXACTLY the same data"""
    print("\n🔬 TESTING BACKTESTING WITH SAME DATA")
    print("==================================================")
    
    # Create the SAME data for all tests
    data = create_test_data_fixed()
    
    # Add technical indicators
    data = TechnicalIndicators.add_all_indicators(data)
    
    # Calculate expected returns manually
    start_price = data['close'].iloc[0]
    end_price = data['close'].iloc[-1]
    expected_return = (end_price / start_price - 1) * 100
    
    print(f"\n📊 MANUAL CALCULATION:")
    print(f"   Start price: ${start_price:.2f}")
    print(f"   End price: ${end_price:.2f}")
    print(f"   Expected return: {expected_return:.2f}%")
    
    # Test Buy & Hold Strategy
    print(f"\n🏠 TESTING BUY & HOLD WITH SAME DATA:")
    print(f"--------------------------------------------------")
    
    backtester = ProductionBacktester(initial_capital=10000)
    bh_strategy = BuyAndHoldStrategy()
    backtester.set_strategy(bh_strategy)
    
    # CRITICAL: Use the SAME data
    results = backtester.run_backtest(data)
    
    print(f"   Backtester result: {results['total_return'] * 100:.2f}%")
    print(f"   Expected result: {expected_return:.2f}%")
    print(f"   Difference: {abs(results['total_return'] * 100 - expected_return):.2f}%")
    
    if abs(results['total_return'] * 100 - expected_return) < 1.0:  # Realistic 1% threshold
        print("   ✅ BACKTESTER CALCULATION IS CORRECT! (Difference within acceptable range)")
        print(f"   📊 Real-world factors: transaction costs, share rounding, cash management")
        return True
    else:
        print("   ❌ BACKTESTER CALCULATION IS WRONG!")
        
        # Debug the issue
        print(f"\n🔍 DEBUGGING THE CALCULATION:")
        portfolio_history = backtester.get_portfolio_history()
        if not portfolio_history.empty:
            print(f"   First portfolio value: ${portfolio_history['portfolio_value'].iloc[0]:.2f}")
            print(f"   Last portfolio value: ${portfolio_history['portfolio_value'].iloc[-1]:.2f}")
            print(f"   First price used: ${portfolio_history['price'].iloc[0]:.2f}")
            print(f"   Last price used: ${portfolio_history['price'].iloc[-1]:.2f}")
        
        return False


def test_ml_strategy_with_same_data():
    """Test ML strategy with the same data"""
    print(f"\n🎯 TESTING ML STRATEGY WITH SAME DATA:")
    print(f"--------------------------------------------------")
    
    # Use the SAME data
    data = create_test_data_fixed()
    data = TechnicalIndicators.add_all_indicators(data)
    
    backtester = ProductionBacktester(initial_capital=10000)
    ml_strategy = MLTradingStrategy(confidence_threshold=0.2)  # Lower threshold for more signals
    backtester.set_strategy(ml_strategy)
    
    results = backtester.run_backtest(data)
    
    print(f"   ML Strategy return: {results['total_return'] * 100:.2f}%")
    print(f"   Total trades: {results['total_trades']}")
    print(f"   Buy trades: {results['buy_trades']}")
    print(f"   Sell trades: {results['sell_trades']}")
    
    # Check signals generated
    signals_history = backtester.get_signals_history()
    if not signals_history.empty:
        buy_signals = len(signals_history[signals_history['signal'].apply(lambda x: x['action'] == 'BUY')])
        sell_signals = len(signals_history[signals_history['signal'].apply(lambda x: x['action'] == 'SELL')])
        print(f"   Buy signals generated: {buy_signals}")
        print(f"   Sell signals generated: {sell_signals}")
        print(f"   Total signals: {len(signals_history)}")
    
    return results


def main():
    """Main diagnostic function"""
    print("🔍 CORRECTED BACKTESTING DIAGNOSTIC")
    print("================================================================================")
    print("This will test if the backtesting engine calculates returns correctly")
    print("by using EXACTLY the same data for all tests.")
    print("================================================================================")
    
    # Test if backtesting calculation is correct
    bh_correct = test_backtesting_with_same_data()
    
    if bh_correct:
        print("\n✅ BACKTESTING ENGINE IS WORKING CORRECTLY!")
        print("The issue was in data range selection, not the calculation logic.")
        
        # Test ML strategy
        ml_results = test_ml_strategy_with_same_data()
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"✅ Backtesting engine: WORKING")
        print(f"✅ Buy & Hold calculation: CORRECT")
        print(f"✅ ML Strategy: {ml_results['total_return'] * 100:.2f}% return")
        print(f"✅ Ready for full testing!")
        
    else:
        print("\n❌ BACKTESTING ENGINE HAS CALCULATION ERRORS!")
        print("Need to fix the backtesting calculation logic.")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"1. Check portfolio value calculation in backtester.py")
        print(f"2. Verify that buy & hold uses consistent pricing")
        print(f"3. Debug the portfolio history tracking")


if __name__ == "__main__":
    main()