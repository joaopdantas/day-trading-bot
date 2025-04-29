"""
Beginner-Friendly Visualization for MakesALot Trading Bot.
Creates simple buy/sell signal visualization.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition

def validate_trading_signal(df, index, pattern_name):
    """Simplified signal validation with balanced buy/sell conditions."""
    rsi = df.loc[index, 'rsi'] if 'rsi' in df.columns else None
    macd = df.loc[index, 'macd'] if 'macd' in df.columns else None
    macd_signal = df.loc[index, 'macd_signal'] if 'macd_signal' in df.columns else None
    price = df.loc[index, 'close']
    
    # Calculate short-term trend
    prev_idx = df.index.get_loc(index) - 3  # Look back 3 periods
    if prev_idx >= 0:
        trend = (price - df.iloc[prev_idx]['close']) / df.iloc[prev_idx]['close'] * 100
    else:
        trend = 0

    # Buy signal conditions (less strict)
    if any(x in pattern_name.lower() for x in ['bullish', 'hammer', 'doji']):
        if ((rsi and rsi < 40) or  # Relaxed RSI threshold
            (macd and macd_signal and macd > macd_signal) or  # MACD crossover
            trend < -3):  # Downtrend
            return True, "BUY"

    # Sell signal conditions (less strict)
    elif any(x in pattern_name.lower() for x in ['bearish', 'shooting']):
        if ((rsi and rsi > 60) or  # Relaxed RSI threshold
            (macd and macd_signal and macd < macd_signal) or  # MACD crossover
            trend > 3):  # Uptrend
            return True, "SELL"

    return False, ""

def fetch_and_process_data(symbol="MSFT", days=100):
    """Fetch and process market data."""
    api = get_data_api("alpha_vantage")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = api.fetch_historical_data(
        symbol=symbol, 
        interval="1d",
        start_date=start_date,
        end_date=end_date
    )
    
    if df.empty:
        print("Failed to retrieve data. Check API key and connection.")
        return None
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_features(df)
    df_indicators = TechnicalIndicators.add_all_indicators(df_processed)
    df_with_patterns = PatternRecognition.recognize_candlestick_patterns(df_indicators)
    
    return df_with_patterns

def create_technical_insight_chart(df, output_file="./technical_insight_chart2.png"):
    """Create simplified technical chart with buy/sell signals."""
    if df is None or df.empty:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price and moving averages
    ax.plot(df.index, df['close'], label='Price', color='blue', linewidth=2)
    if 'sma_20' in df.columns:
        ax.plot(df.index, df['sma_20'], label='20MA', color='orange', linewidth=1)
    if 'sma_50' in df.columns:
        ax.plot(df.index, df['sma_50'], label='50MA', color='red', linewidth=1)

    # Plot trading signals
    pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
    for col in pattern_cols:
        pattern_dates = df.index[df[col] == 1]
        
        for d in pattern_dates:
            is_valid, signal = validate_trading_signal(df, d, col)
            
            if is_valid:
                price = df.loc[d, 'close']
                color = 'green' if signal == "BUY" else 'red'
                marker = '^' if signal == "BUY" else 'v'
                
                # Plot signal marker and label
                ax.scatter(d, price, color=color, marker=marker, s=150, zorder=5)
                ax.annotate(signal, 
                    xy=(d, price),
                    xytext=(0, 10 if signal == "BUY" else -10),
                    textcoords='offset points',
                    ha='center',
                    color=color,
                    fontweight='bold')

    ax.set_title(f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} Trading Signals", 
                fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trading signals chart saved to {output_file}")

if __name__ == "__main__":
    print("Fetching and processing data...")
    df = fetch_and_process_data(symbol="MSFT", days=100)
    
    if df is not None:
        print("Creating trading signals chart...")
        create_technical_insight_chart(df)