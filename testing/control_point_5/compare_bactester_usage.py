"""
COMPARE HOW BACKTESTER IS USED
Find the difference between working diagnostic test and failing hypothesis test
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

def test_diagnostic_method():
    """Test using the EXACT method from diagnostic_test.py that works"""
    
    print("🧪 TESTING DIAGNOSTIC METHOD (KNOWN TO WORK)")
    print("=" * 60)
    
    try:
        from src.backtesting import ProductionBacktester, MLTradingStrategy
        from src.indicators.technical import TechnicalIndicators
        import pandas as pd
        import numpy as np
        
        # Create test data EXACTLY like diagnostic_test.py
        dates = pd.date_range(start='2025-01-10', periods=50, freq='D')
        np.random.seed(42)  # Same seed as diagnostic
        
        base_price = 400.0
        price_changes = np.random.normal(0, 0.02, 50)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, 50))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, 50))
        data['volume'] = np.random.randint(1000000, 5000000, 50)
        
        # Add indicators
        data = TechnicalIndicators.add_all_indicators(data)
        
        print(f"✅ Test data created: {len(data)} rows")
        print(f"   Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
        
        # Test backtester EXACTLY like diagnostic
        backtester = ProductionBacktester(initial_capital=10000)
        ml_strategy = MLTradingStrategy(confidence_threshold=0.2)
        backtester.set_strategy(ml_strategy)
        
        # Run backtest
        results = backtester.run_backtest(data)
        
        print(f"\n📊 DIAGNOSTIC METHOD RESULTS:")
        print(f"   Total Return: {results.get('total_return', 0):.2%}")
        print(f"   Total Trades: {results.get('total_trades', 0)}")
        print(f"   Buy Trades: {results.get('buy_trades', 0)}")
        print(f"   Sell Trades: {results.get('sell_trades', 0)}")
        
        if results.get('total_trades', 0) > 0:
            print(f"✅ DIAGNOSTIC METHOD WORKS! Backtester executes trades")
            return True
        else:
            print(f"❌ Even diagnostic method shows 0 trades")
            return False
            
    except Exception as e:
        print(f"❌ Error in diagnostic method: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hypothesis_method():
    """Test using the method from hypothesis test that fails"""
    
    print(f"\n🧪 TESTING HYPOTHESIS METHOD (FAILS)")
    print("=" * 60)
    
    try:
        from src.data.fetcher import get_data_api
        from src.indicators.technical import TechnicalIndicators
        from src.backtesting import ProductionBacktester, MLTradingStrategy
        import pandas as pd
        
        # Get real data EXACTLY like hypothesis test
        api = get_data_api("alpha_vantage")
        full_data = api.fetch_historical_data("MSFT", "1d")
        
        # Filter for 2024 EXACTLY like hypothesis test
        start_date = pd.Timestamp('2024-01-01')
        end_date = pd.Timestamp('2024-12-31')
        
        data_2024 = full_data[
            (full_data.index >= start_date) & 
            (full_data.index <= end_date)
        ]
        
        data_2024 = TechnicalIndicators.add_all_indicators(data_2024)
        
        print(f"✅ Real 2024 data loaded: {len(data_2024)} rows")
        print(f"   Date range: {data_2024.index[0]} to {data_2024.index[-1]}")
        
        # Test backtester EXACTLY like hypothesis test
        backtester = ProductionBacktester(
            initial_capital=10000,
            transaction_cost=0.001,
            max_position_size=0.3
        )
        
        strategy = MLTradingStrategy(confidence_threshold=0.15)
        backtester.set_strategy(strategy)
        
        # Run backtest
        results = backtester.run_backtest(data_2024)
        
        print(f"\n📊 HYPOTHESIS METHOD RESULTS:")
        print(f"   Total Return: {results.get('total_return', 0):.2%}")
        print(f"   Total Trades: {results.get('total_trades', 0)}")
        print(f"   Buy Trades: {results.get('buy_trades', 0)}")
        print(f"   Sell Trades: {results.get('sell_trades', 0)}")
        
        if results.get('total_trades', 0) > 0:
            print(f"✅ HYPOTHESIS METHOD ALSO WORKS!")
            return True
        else:
            print(f"❌ HYPOTHESIS METHOD FAILS - 0 trades")
            
            # Check what's different
            print(f"\n🔍 INVESTIGATING DIFFERENCES:")
            print(f"   Data type: {type(data_2024)}")
            print(f"   Data columns: {list(data_2024.columns)}")
            print(f"   Has RSI: {'rsi' in data_2024.columns}")
            print(f"   RSI range: {data_2024['rsi'].min():.1f} to {data_2024['rsi'].max():.1f}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error in hypothesis method: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_methods():
    """Compare the two methods to find the difference"""
    
    print("🔍 BACKTESTER USAGE COMPARISON")
    print("=" * 60)
    print("Finding why diagnostic works but hypothesis test fails")
    print("=" * 60)
    
    diagnostic_works = test_diagnostic_method()
    hypothesis_works = test_hypothesis_method()
    
    print(f"\n📊 COMPARISON RESULTS:")
    print("=" * 40)
    print(f"Diagnostic Method:  {'✅ WORKS' if diagnostic_works else '❌ FAILS'}")
    print(f"Hypothesis Method:  {'✅ WORKS' if hypothesis_works else '❌ FAILS'}")
    
    if diagnostic_works and hypothesis_works:
        print(f"\n🎉 BOTH METHODS WORK!")
        print(f"   The backtester is not broken")
        print(f"   Issue might be in test script configuration")
        
    elif diagnostic_works and not hypothesis_works:
        print(f"\n🔍 DIAGNOSTIC WORKS, HYPOTHESIS FAILS")
        print(f"   Key differences to investigate:")
        print(f"   1. Data source: Synthetic vs Real MSFT data")
        print(f"   2. Data size: 50 days vs 252 days")  
        print(f"   3. Confidence threshold: 0.2 vs 0.15")
        print(f"   4. Data structure or column names")
        
        return "data_difference"
        
    elif not diagnostic_works and not hypothesis_works:
        print(f"\n❌ BOTH METHODS FAIL")
        print(f"   The backtester itself is broken")
        print(f"   Need to fix the application backtester")
        
        return "backtester_broken"
        
    else:
        print(f"\n🤔 UNEXPECTED RESULT")
        return "unknown"

if __name__ == "__main__":
    result = compare_methods()
    
    print(f"\n🎯 RECOMMENDATION:")
    if result == "data_difference":
        print(f"   The backtester works, but real 2024 data causes issues")
        print(f"   Need to debug why real data fails but synthetic data works")
    elif result == "backtester_broken":
        print(f"   The backtester is broken and needs to be fixed")
    else:
        print(f"   Both methods work - check test script configuration")