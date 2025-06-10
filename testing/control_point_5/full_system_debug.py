"""
FULL SYSTEM DEBUG: End-to-End Trading System Check
This will simulate the exact process used in hypothesis testing
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

def full_system_debug():
    """Debug the entire system end-to-end"""
    
    print("🔍 FULL SYSTEM DEBUG - REPLICATING HYPOTHESIS TESTING")
    print("=" * 70)
    
    try:
        # Step 1: Import exactly like hypothesis testing
        from src.backtesting import MLTradingStrategy
        from src.backtesting.backtester import ProductionBacktester
        from src.data.fetcher import get_data_api
        from src.indicators.technical import TechnicalIndicators
        
        print("✅ All imports successful")
        
        # Step 2: Create strategy with 0.05 threshold
        print(f"\n🎯 STEP 2: Creating MLTradingStrategy with confidence_threshold=0.05")
        strategy = MLTradingStrategy(confidence_threshold=0.05)
        print(f"   Strategy created with threshold: {strategy.confidence_threshold}")
        
        # Step 3: Get some real data (small sample)
        print(f"\n🎯 STEP 3: Loading real MSFT data")
        api = get_data_api("alpha_vantage")
        data = api.fetch_historical_data("MSFT", "1d")
        
        if data is None or data.empty:
            print("❌ No data - using synthetic data")
            # Create synthetic test data with clear signals
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 400,
                'high': np.random.randn(100).cumsum() + 405,
                'low': np.random.randn(100).cumsum() + 395,
                'close': np.random.randn(100).cumsum() + 400,
                'volume': np.random.randint(1000000, 5000000, 100)
            }, index=dates)
        else:
            # Use last 100 days
            data = data.tail(100)
            print(f"   Data loaded: {len(data)} days from {data.index[0].date()} to {data.index[-1].date()}")
        
        # Step 4: Add technical indicators
        print(f"\n🎯 STEP 4: Adding technical indicators")
        data = TechnicalIndicators.add_all_indicators(data)
        print(f"   RSI range: {data['rsi'].min():.1f} to {data['rsi'].max():.1f}")
        
        # Find oversold/overbought periods
        oversold = data[data['rsi'] < 30]
        overbought = data[data['rsi'] > 70]
        print(f"   Oversold periods (RSI < 30): {len(oversold)}")
        print(f"   Overbought periods (RSI > 70): {len(overbought)}")
        
        # Step 5: Test signal generation manually
        print(f"\n🎯 STEP 5: Testing signal generation manually")
        total_signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_above_threshold = 0
        
        for i in range(20, min(50, len(data))):  # Test first 30 days
            current_row = data.iloc[i]
            historical_data = data.iloc[max(0, i-20):i]
            
            signal = strategy.generate_signal(current_row, historical_data)
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.0)
            
            total_signals[action] += 1
            
            if confidence >= 0.05:
                confidence_above_threshold += 1
                
            if action != 'HOLD':
                print(f"   Day {i}: {action} at ${current_row['close']:.2f} (RSI: {current_row['rsi']:.1f}, confidence: {confidence:.3f})")
        
        print(f"\n📊 SIGNAL SUMMARY:")
        print(f"   BUY signals: {total_signals['BUY']}")
        print(f"   SELL signals: {total_signals['SELL']}")
        print(f"   HOLD signals: {total_signals['HOLD']}")
        print(f"   Confidence >= 0.05: {confidence_above_threshold}")
        
        # Step 6: Test with backtester (like hypothesis testing)
        print(f"\n🎯 STEP 6: Testing with ProductionBacktester")
        backtester = ProductionBacktester(
            initial_capital=10000,
            transaction_cost=0.001,
            max_position_size=0.3
        )
        
        # Reset strategy and set in backtester
        strategy.reset()
        backtester.set_strategy(strategy)
        
        # Run backtest
        results = backtester.run_backtest(data)
        
        print(f"   📈 Total Return: {results['total_return']:.2%}")
        print(f"   🔄 Total Trades: {results['total_trades']}")
        print(f"   📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   🎯 Win Rate: {results['win_rate']:.2%}")
        
        # Step 7: Check what actually happened
        print(f"\n🎯 STEP 7: Analyzing what happened")
        
        # Get signals history from backtester
        signals_history = backtester.get_signals_history()
        if not signals_history.empty:
            executed_signals = signals_history[signals_history['signal'].apply(lambda x: x['action'] != 'HOLD')]
            print(f"   Total signals generated: {len(signals_history)}")
            print(f"   Non-HOLD signals: {len(executed_signals)}")
            
            if len(executed_signals) > 0:
                print(f"   First few executed signals:")
                for i, (_, row) in enumerate(executed_signals.head(5).iterrows()):
                    signal = row['signal']
                    print(f"     {signal['action']} on {row['date'].date()} - confidence: {signal['confidence']:.3f}")
        else:
            print(f"   ❌ No signals history available")
        
        # Step 8: Final diagnosis
        print(f"\n🎯 STEP 8: DIAGNOSIS")
        if results['total_trades'] == 27:
            print(f"   ❌ PROBLEM: Still getting 27 trades - confidence threshold not working!")
            print(f"   🔍 Check if there's a hardcoded threshold somewhere")
            print(f"   🔍 Check if strategy is being overridden after creation")
        elif results['total_trades'] > 27:
            print(f"   ✅ SUCCESS: Getting {results['total_trades']} trades (more than 27)")
            print(f"   🎯 Optimization is working!")
        else:
            print(f"   ⚠️  UNEXPECTED: Getting {results['total_trades']} trades (less than 27)")
            
    except Exception as e:
        print(f"❌ Error during full system debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    full_system_debug()