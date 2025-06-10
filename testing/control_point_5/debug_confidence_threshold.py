"""
DEBUG SCRIPT: Check Confidence Threshold Issue
This will help identify where the confidence threshold is being overridden
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

def debug_confidence_threshold():
    """Debug confidence threshold issues"""
    
    print("🔍 DEBUGGING CONFIDENCE THRESHOLD ISSUE")
    print("=" * 60)
    
    try:
        from src.backtesting import MLTradingStrategy
        
        # Test 1: Check what gets initialized with explicit parameter
        print("\n🎯 TEST 1: Explicit confidence_threshold=0.05")
        strategy1 = MLTradingStrategy(confidence_threshold=0.05)
        print(f"   Strategy confidence_threshold: {strategy1.confidence_threshold}")
        
        # Test 2: Check default value
        print("\n🎯 TEST 2: Default initialization")
        strategy2 = MLTradingStrategy()
        print(f"   Default confidence_threshold: {strategy2.confidence_threshold}")
        
        # Test 3: Check if the parameter is actually being used
        print("\n🎯 TEST 3: Testing signal generation with low confidence")
        
        # Create dummy data that should trigger signals
        import pandas as pd
        import numpy as np
        
        # Create test data with extreme RSI values
        test_data = pd.Series({
            'close': 100.0,
            'rsi': 25.0,  # Very oversold - should trigger BUY
            'volume': 1000000,
            'macd': 0.5,
            'macd_signal': 0.3
        })
        
        historical_data = pd.DataFrame({
            'close': [95, 96, 97, 98, 99, 100],
            'volume': [1000000] * 6
        })
        
        # Test with 0.05 threshold
        strategy_low = MLTradingStrategy(confidence_threshold=0.05)
        signal_low = strategy_low.generate_signal(test_data, historical_data)
        
        print(f"   Low threshold (0.05) signal: {signal_low['action']}")
        print(f"   Signal confidence: {signal_low['confidence']:.3f}")
        print(f"   Strategy threshold: {strategy_low.confidence_threshold}")
        
        # Test with 0.15 threshold
        strategy_high = MLTradingStrategy(confidence_threshold=0.15)
        signal_high = strategy_high.generate_signal(test_data, historical_data)
        
        print(f"   High threshold (0.15) signal: {signal_high['action']}")
        print(f"   Signal confidence: {signal_high['confidence']:.3f}")
        print(f"   Strategy threshold: {strategy_high.confidence_threshold}")
        
        # Test 4: Check if signals meet threshold
        print(f"\n🎯 TEST 4: Signal vs Threshold Analysis")
        if signal_low['confidence'] >= 0.05:
            print(f"   ✅ Signal confidence {signal_low['confidence']:.3f} >= 0.05 threshold")
        else:
            print(f"   ❌ Signal confidence {signal_low['confidence']:.3f} < 0.05 threshold")
            
        if signal_high['confidence'] >= 0.15:
            print(f"   ✅ Signal confidence {signal_high['confidence']:.3f} >= 0.15 threshold")
        else:
            print(f"   ❌ Signal confidence {signal_high['confidence']:.3f} < 0.15 threshold")
        
        # Test 5: Check the actual signal logic inside generate_signal
        print(f"\n🎯 TEST 5: Examining Signal Logic")
        print(f"   RSI value: {test_data['rsi']} (should be < 30 for BUY)")
        
        # Let's manually check what the signal generation is doing
        print(f"   Manual check: RSI {test_data['rsi']} < 30? {test_data['rsi'] < 30}")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_confidence_threshold()