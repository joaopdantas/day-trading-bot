"""
EMERGENCY DEBUG: Signal Generation Issues

Create this debug script to identify why ML strategy isn't trading:
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.backtesting import MLTradingStrategy
from src.data.fetcher import get_data_api
from src.indicators.technical import TechnicalIndicators

def debug_signal_generation():
    """Debug why ML strategy isn't generating signals"""
    
    print("🔍 DEBUGGING ML STRATEGY SIGNAL GENERATION")
    print("=" * 60)
    
    # Get real MSFT data
    try:
        api = get_data_api("alpha_vantage")
        df = api.fetch_historical_data("MSFT", "1d")
        
        if df is None or df.empty:
            print("❌ No data retrieved")
            return
            
        # Take recent data
        df = df.tail(100)
        df = TechnicalIndicators.add_all_indicators(df)
        
        print(f"✅ Data: {len(df)} rows")
        print(f"   RSI range: {df['rsi'].min():.1f} to {df['rsi'].max():.1f}")
        print(f"   Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
        
        # Test different confidence thresholds
        thresholds = [0.1, 0.25, 0.5, 0.75]
        
        for threshold in thresholds:
            print(f"\n🎯 Testing confidence threshold: {threshold}")
            
            strategy = MLTradingStrategy(confidence_threshold=threshold)
            signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for i in range(20, len(df)):
                row = df.iloc[i]
                historical = df.iloc[max(0, i-50):i]
                
                signal = strategy.generate_signal(row, historical)
                action = signal.get('action', 'HOLD')
                signals[action] += 1
                
                if action != 'HOLD':
                    print(f"   {action} at ${row['close']:.2f} (RSI: {row['rsi']:.1f}, confidence: {signal.get('confidence', 0):.2f})")
            
            print(f"   Total signals: BUY={signals['BUY']}, SELL={signals['SELL']}, HOLD={signals['HOLD']}")
        
        # Check RSI extremes specifically
        print(f"\n📊 RSI ANALYSIS:")
        oversold = df[df['rsi'] < 30]
        overbought = df[df['rsi'] > 70]
        
        print(f"   Oversold periods (RSI < 30): {len(oversold)}")
        print(f"   Overbought periods (RSI > 70): {len(overbought)}")
        
        if len(oversold) > 0:
            print(f"   Example oversold: RSI {oversold['rsi'].iloc[0]:.1f} at ${oversold['close'].iloc[0]:.2f}")
        if len(overbought) > 0:
            print(f"   Example overbought: RSI {overbought['rsi'].iloc[0]:.1f} at ${overbought['close'].iloc[0]:.2f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_signal_generation()