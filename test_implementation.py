"""
Test script for Day Trading Bot implementation.

This script tests the data fetching, preprocessing, and technical indicators functionality.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import our modules
from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition

# Load environment variables
load_dotenv()

def test_data_pipeline():
    """Test the complete data pipeline from fetching to technical analysis."""
    
    print("Starting Day Trading Bot Test")
    print("-" * 50)
    
    # 1. Initialize data fetcher
    print("Initializing data fetcher...")
    api = get_data_api(api_name="alpha_vantage")
    
    # 2. Fetch historical data
    symbol = "AAPL"
    print(f"Fetching historical data for {symbol}...")
    df = api.fetch_historical_data(
        symbol=symbol,
        interval="1d",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    if df.empty:
        print("Error: Failed to retrieve data!")
        return
    
    print(f"Retrieved {len(df)} data points.")
    print("\nSample data:")
    print(df.head())
    
    # 3. Preprocess data
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.clean_data(df)
    df_processed = preprocessor.add_time_features(df_processed)
    df_processed = preprocessor.calculate_returns(df_processed)
    
    print("\nPreprocessed data sample:")
    print(df_processed[['open', 'high', 'low', 'close', 'volume', 'daily_return']].head())
    
    # 4. Calculate technical indicators
    print("\nCalculating technical indicators...")
    df_with_indicators = TechnicalIndicators.add_all_indicators(df_processed)
    
    print("\nTechnical indicators sample:")
    indicator_cols = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_20']
    available_cols = [col for col in indicator_cols if col in df_with_indicators.columns]
    print(df_with_indicators[available_cols].tail())
    
    # 5. Recognize patterns
    print("\nRecognizing candlestick patterns...")
    df_with_patterns = PatternRecognition.recognize_candlestick_patterns(df_with_indicators)
    df_with_patterns = PatternRecognition.detect_trend(df_with_patterns)
    
    pattern_cols = [col for col in df_with_patterns.columns if col.startswith('pattern_')]
    trend_cols = ['uptrend', 'downtrend', 'golden_cross', 'death_cross']
    
    print("\nDetected patterns:")
    pattern_counts = df_with_patterns[pattern_cols].sum()
    for pattern, count in pattern_counts.items():
        if count > 0:
            print(f"- {pattern}: {count} occurrences")
    
    # 6. Plot results
    try:
        print("\nGenerating price chart with indicators...")
        plt.figure(figsize=(12, 8))
        
        # Price and moving averages
        plt.subplot(3, 1, 1)
        plt.plot(df_with_indicators.index, df_with_indicators['close'], label='Close Price')
        if 'sma_20' in df_with_indicators.columns:
            plt.plot(df_with_indicators.index, df_with_indicators['sma_20'], label='SMA 20')
        if 'sma_50' in df_with_indicators.columns:
            plt.plot(df_with_indicators.index, df_with_indicators['sma_50'], label='SMA 50')
        plt.title(f'{symbol} Price Chart')
        plt.legend()
        
        # RSI
        if 'rsi' in df_with_indicators.columns:
            plt.subplot(3, 1, 2)
            plt.plot(df_with_indicators.index, df_with_indicators['rsi'])
            plt.axhline(y=70, color='r', linestyle='-')
            plt.axhline(y=30, color='g', linestyle='-')
            plt.title('RSI')
        
        # MACD
        if all(x in df_with_indicators.columns for x in ['macd', 'macd_signal']):
            plt.subplot(3, 1, 3)
            plt.plot(df_with_indicators.index, df_with_indicators['macd'], label='MACD')
            plt.plot(df_with_indicators.index, df_with_indicators['macd_signal'], label='Signal Line')
            plt.bar(df_with_indicators.index, df_with_indicators['macd_histogram'], width=0.5, alpha=0.3)
            plt.title('MACD')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('test_chart.png')
        print("Chart saved as 'test_chart.png'")
    except Exception as e:
        print(f"Error generating chart: {e}")
    
    print("\nTest completed successfully!")
    return df_with_indicators

if __name__ == "__main__":
    result_df = test_data_pipeline()