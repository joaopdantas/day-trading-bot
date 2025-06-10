"""
DIAGNOSE TRADE EXECUTION ISSUE
Find out why signals are generated but not executed
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.backtesting import MLTradingStrategy
from src.backtesting.backtester import ProductionBacktester
from src.indicators.technical import TechnicalIndicators

def diagnose_execution_issue():
    """Diagnose why signals are not being executed"""
    
    print("🔍 DIAGNOSING TRADE EXECUTION ISSUE")
    print("=" * 60)
    
    # Get test data
    ticker = yf.Ticker("MSFT")
    data = ticker.history(period="2mo")  # 2 months of data
    
    if data is None or data.empty:
        print("❌ No data retrieved")
        return
        
    # Convert to expected format
    data = data.reset_index()
    data.columns = [col.lower() for col in data.columns]
    data.set_index('date', inplace=True)
    
    # Use last 60 days for better indicators, then take last 30 for testing
    data = data.tail(60)
    
    # Add indicators with error handling
    try:
        data = TechnicalIndicators.add_all_indicators(data)
        print("✅ Technical indicators added successfully")
    except Exception as e:
        print(f"⚠️ Indicator error: {e}")
        # Add basic indicators manually to continue
        data['rsi'] = 50.0  # Default neutral RSI
        data['macd'] = 0.0
        data['macd_signal'] = 0.0
        data['bb_upper'] = data['close'] * 1.02
        data['bb_lower'] = data['close'] * 0.98
        data['bb_middle'] = data['close']
        data['volume_ratio'] = 1.0
        print("✅ Using basic default indicators")
    
    # Now use last 30 days for actual testing
    data = data.tail(30)
    
    print(f"✅ Data loaded: {len(data)} days")
    
    # Create strategy and backtester
    strategy = MLTradingStrategy(
        rsi_oversold=35,
        rsi_overbought=65,
        confidence_threshold=0.15
    )
    
    backtester = ProductionBacktester(
        initial_capital=10000,
        transaction_cost=0.001,
        max_position_size=0.20
    )
    
    backtester.set_strategy(strategy)
    
    print(f"\n🎯 BACKTESTER SETTINGS:")
    print(f"   Initial capital: ${backtester.initial_capital:,}")
    print(f"   Max position size: {backtester.max_position_size:.1%}")
    print(f"   Transaction cost: {backtester.transaction_cost:.3%}")
    
    # Manual step-by-step execution to see what happens
    print(f"\n📊 MANUAL STEP-BY-STEP EXECUTION:")
    print("-" * 60)
    
    # Reset everything
    backtester.portfolio.reset(backtester.initial_capital)
    strategy.reset()
    
    signals_generated = 0
    trades_executed = 0
    
    for i in range(10, min(20, len(data))):  # Test 10 days, start from day 10
        row = data.iloc[i]
        historical = data.iloc[max(0, i-20):i]
        date = row.name
        
        # Generate signal
        signal = strategy.generate_signal(row, historical)
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        price = row['close']
        
        if action != 'HOLD':
            signals_generated += 1
            
            print(f"\n📅 Day {i}: {date.date()}")
            print(f"   💰 Price: ${price:.2f}")
            print(f"   🎯 Signal: {action} (Score: {signal.get('technical_score', 0):.1f}, Confidence: {confidence:.2f})")
            
            # Check portfolio state BEFORE execution
            portfolio_value = backtester.portfolio.get_total_value(price, 'STOCK')
            available_cash = backtester.portfolio.get_available_cash()
            has_position = backtester.portfolio.has_position('STOCK')
            position_info = backtester.portfolio.get_position_info('STOCK')
            
            print(f"   📊 Portfolio state:")
            print(f"      Total value: ${portfolio_value:.2f}")
            print(f"      Available cash: ${available_cash:.2f}")
            print(f"      Has position: {has_position}")
            if position_info:
                print(f"      Position: {position_info['shares']} shares at ${position_info['avg_price']:.2f}")
            
            # Try to execute manually
            if action == 'BUY':
                # Check if backtester would allow this BUY
                can_buy = not has_position  # Assuming this is the logic
                print(f"   🟢 BUY execution check:")
                print(f"      Can buy (no position): {can_buy}")
                
                if can_buy:
                    # Calculate position size
                    shares = backtester._calculate_position_size_fixed(price, confidence)
                    print(f"      Calculated shares: {shares}")
                    
                    if shares > 0:
                        # Try to execute
                        success = backtester.portfolio.buy_stock('STOCK', shares, price, backtester.transaction_cost)
                        if success:
                            trades_executed += 1
                            print(f"      ✅ EXECUTED: Bought {shares} shares")
                        else:
                            print(f"      ❌ FAILED: Portfolio.buy_stock returned False")
                    else:
                        print(f"      ❌ FAILED: Calculated 0 shares")
                else:
                    print(f"      ❌ BLOCKED: Already has position")
                    
            elif action == 'SELL':
                # Check if backtester would allow this SELL
                can_sell = has_position
                print(f"   🔴 SELL execution check:")
                print(f"      Can sell (has position): {can_sell}")
                
                if can_sell and position_info:
                    shares = position_info['shares']
                    print(f"      Shares to sell: {shares}")
                    
                    # Try to execute
                    success = backtester.portfolio.sell_stock('STOCK', shares, price, backtester.transaction_cost)
                    if success:
                        trades_executed += 1
                        print(f"      ✅ EXECUTED: Sold {shares} shares")
                    else:
                        print(f"      ❌ FAILED: Portfolio.sell_stock returned False")
                else:
                    print(f"      ❌ BLOCKED: No position to sell")
    
    print(f"\n📋 EXECUTION SUMMARY:")
    print(f"   Signals generated: {signals_generated}")
    print(f"   Trades executed: {trades_executed}")
    print(f"   Execution rate: {trades_executed/signals_generated*100 if signals_generated > 0 else 0:.1f}%")
    
    # Now run the actual backtester to compare
    print(f"\n🔄 RUNNING ACTUAL BACKTESTER:")
    backtester.portfolio.reset(backtester.initial_capital)
    strategy.reset()
    
    results = backtester.run_backtest(data)
    
    print(f"   Backtester trades: {results['total_trades']}")
    print(f"   Manual trades: {trades_executed}")
    
    if results['total_trades'] != trades_executed:
        print(f"   🚨 DISCREPANCY: Manual vs Backtester execution differs!")
    else:
        print(f"   ✅ CONSISTENT: Manual and backtester results match")

if __name__ == "__main__":
    diagnose_execution_issue()