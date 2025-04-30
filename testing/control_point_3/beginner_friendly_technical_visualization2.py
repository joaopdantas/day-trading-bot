"""
Enhanced Beginner-Friendly Visualization for MakesALot Trading Bot.
Creates simplified trading chart with reliable signals using DatasetPreparation.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition, DatasetPreparation

def fetch_and_process_data(symbol="MSFT", days=100):
    """Fetch and process market data."""
    print(f"Fetching data for {symbol} for the past {days} days...")
    
    # Try Alpha Vantage first
    api = get_data_api("alpha_vantage")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = api.fetch_historical_data(
        symbol=symbol, 
        interval="1d",
        start_date=start_date,
        end_date=end_date
    )
    
    # If Alpha Vantage fails, try Yahoo Finance
    if df is None or df.empty:
        print("Alpha Vantage data retrieval failed. Trying Yahoo Finance...")
        api = get_data_api("yahoo_finance")
        df = api.fetch_historical_data(
            symbol=symbol, 
            interval="1d",
            start_date=start_date,
            end_date=end_date
        )
    
    if df is None or df.empty:
        print("Failed to retrieve data. Check API keys and internet connection.")
        return None
    
    print(f"Successfully retrieved {len(df)} days of market data.")
    
    try:
        # Process data
        print("Processing data and calculating indicators...")
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_features(df)
        df = TechnicalIndicators.add_all_indicators(df)
        df = PatternRecognition.recognize_candlestick_patterns(df)
        df = PatternRecognition.detect_trend(df)
        
        # Generate trading signals
        print("Generating trading signals...")
        signal_df = DatasetPreparation.create_target_labels(
            df,
            horizon=5,          # Look 5 days ahead
            threshold=0.01,     # Base threshold for returns
            min_risk_reward_ratio=1.5,  # Minimum risk/reward ratio
            volume_filter=True  # Apply volume filter
        )
        
        # Merge signals with main dataframe
        for col in signal_df.columns:
            df[col] = signal_df[col]
        
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def create_simplified_signal_chart(df, output_file="testing/control_point_3/enhanced_simplified_chart2.png"):
    """Create simplified, beginner-friendly chart with reliable buy/sell signals."""
    if df is None or df.empty:
        print("No data available for visualization.")
        return
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price and moving averages
        ax.plot(df.index, df['close'], label='Price', color='blue', linewidth=2.5)
        
        if 'sma_20' in df.columns:
            ax.plot(df.index, df['sma_20'], label='20-day Average', color='orange', linewidth=1.5)
        
        if 'sma_50' in df.columns:
            ax.plot(df.index, df['sma_50'], label='50-day Average', color='red', linewidth=1.5)

        # Highlight trend periods if available
        if 'uptrend' in df.columns and 'downtrend' in df.columns:
            # Find spans of uptrends
            uptrend_changes = df['uptrend'].diff().fillna(0)
            uptrend_starts = df.index[uptrend_changes == 1].tolist()
            uptrend_ends = df.index[uptrend_changes == -1].tolist()
            
            # Handle case where trend starts at beginning of data
            if df['uptrend'].iloc[0] == 1:
                uptrend_starts.insert(0, df.index[0])
            
            # Handle case where trend continues to end of data
            if df['uptrend'].iloc[-1] == 1:
                uptrend_ends.append(df.index[-1])
            
            # Shade uptrend areas
            for start, end in zip(uptrend_starts, uptrend_ends):
                ax.axvspan(start, end, alpha=0.15, color='green', label='_nolegend_')
            
            # Find spans of downtrends
            downtrend_changes = df['downtrend'].diff().fillna(0)
            downtrend_starts = df.index[downtrend_changes == 1].tolist()
            downtrend_ends = df.index[downtrend_changes == -1].tolist()
            
            # Handle edge cases
            if df['downtrend'].iloc[0] == 1:
                downtrend_starts.insert(0, df.index[0])
                
            if df['downtrend'].iloc[-1] == 1:
                downtrend_ends.append(df.index[-1])
            
            # Shade downtrend areas
            for start, end in zip(downtrend_starts, downtrend_ends):
                ax.axvspan(start, end, alpha=0.15, color='red', label='_nolegend_')

        # Plot signals based on target_label
        if 'target_label' in df.columns:
            # Buy signals
            buy_signals = df[df['target_label'] == 1]
            if not buy_signals.empty:
                ax.scatter(buy_signals.index, buy_signals['close'], 
                           marker='^', color='green', s=180, label='Buy Signal', zorder=5)
                
                # Add simplified labels
                for idx in buy_signals.index:
                    expected_return = buy_signals.loc[idx, 'expected_return'] * 100 if 'expected_return' in buy_signals.columns else 0
                    confidence = buy_signals.loc[idx, 'signal_probability'] * 100 if 'signal_probability' in buy_signals.columns else 0
                    
                    # Simplified label text
                    label = f"BUY\n{confidence:.0f}% confident"
                    
                    ax.annotate(label,
                               xy=(idx, buy_signals.loc[idx, 'close']),
                               xytext=(0, 20),
                               textcoords='offset points',
                               ha='center',
                               color='darkgreen',
                               fontweight='bold',
                               bbox=dict(facecolor='white', edgecolor='green', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Sell signals
            sell_signals = df[df['target_label'] == -1]
            if not sell_signals.empty:
                ax.scatter(sell_signals.index, sell_signals['close'], 
                           marker='v', color='red', s=180, label='Sell Signal', zorder=5)
                
                # Add simplified labels
                for idx in sell_signals.index:
                    expected_return = sell_signals.loc[idx, 'expected_return'] * 100 if 'expected_return' in sell_signals.columns else 0
                    confidence = sell_signals.loc[idx, 'signal_probability'] * 100 if 'signal_probability' in sell_signals.columns else 0
                    
                    # Simplified label text
                    label = f"SELL\n{confidence:.0f}% confident"
                    
                    ax.annotate(label,
                               xy=(idx, sell_signals.loc[idx, 'close']),
                               xytext=(0, -20),
                               textcoords='offset points',
                               ha='center',
                               color='darkred',
                               fontweight='bold',
                               bbox=dict(facecolor='white', edgecolor='red', alpha=0.8, boxstyle='round,pad=0.5'))

        # Add explanatory box
        explanation = (
            "TRADING SIGNALS:\n"
            "• Green ▲: AI-generated buy signal\n"
            "• Red ▼: AI-generated sell signal\n"
            "• Green areas: Market uptrends\n"
            "• Red areas: Market downtrends\n"
            "• Orange line: Short-term price trend (20 days)\n"
            "• Red line: Long-term price trend (50 days)"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9)
        ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=props)

        # Format chart
        symbol = "Stock" if 'symbol' not in df.columns else df['symbol'].iloc[0]
        ax.set_title(f"{symbol} Price Chart with AI Trading Signals", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Calculate performance stats
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        trend = "UPWARD" if end_price > start_price else "DOWNWARD"
        
        # Add performance summary
        summary = f"Performance: {trend} TREND, {percent_change:.1f}% change"
        fig.text(0.5, 0.01, summary, ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Simplified trading signals chart saved to {output_file}")
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        plt.close()

if __name__ == "__main__":
    print("Starting enhanced simplified visualization...")
    df = fetch_and_process_data(symbol="MSFT", days=100)
    
    if df is not None:
        print("Creating simplified trading signals chart...")
        create_simplified_signal_chart(df)