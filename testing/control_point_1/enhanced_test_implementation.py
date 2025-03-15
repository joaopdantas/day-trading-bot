"""
Enhanced test script for Day Trading Bot with more readable visualizations.

This script tests the data fetching, preprocessing, and technical indicators functionality
and creates more beginner-friendly visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from dotenv import load_dotenv

# Add the project root directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
sys.path.insert(0, project_root)

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
    
    # 6. Create enhanced visualizations
    pattern_counts = df_with_patterns[pattern_cols].sum()
    create_enhanced_visualizations(df_with_patterns, symbol, pattern_counts)
    
    print("\nTest completed successfully!")
    return df_with_patterns

def create_enhanced_visualizations(df, symbol, pattern_counts):
    """
    Create enhanced, beginner-friendly visualizations.
    
    Args:
        df: DataFrame with market data and indicators
        symbol: Stock symbol
        pattern_counts: Series containing the count of detected patterns
    """
    try:
        print("\nGenerating enhanced visualizations...")
        
        # Set style
        plt.style.use('ggplot')
        
        # Create main figure with multiple panels
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], figure=fig)
        
        # Format dates
        date_format = mdates.DateFormatter('%b %Y')
        
        # ---- PANEL 1: Price and Moving Averages ----
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df['close'], linewidth=2, label='Price', color='#1f77b4')
        
        # Add moving averages
        if 'sma_20' in df.columns:
            ax1.plot(df.index, df['sma_20'], linewidth=1.5, label='20-Day Average', color='#ff7f0e')
        if 'sma_50' in df.columns:
            ax1.plot(df.index, df['sma_50'], linewidth=1.5, label='50-Day Average', color='#2ca02c')
        
        # Add Bollinger Bands if available
        if all(x in df.columns for x in ['bb_upper', 'bb_lower']):
            ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='#1f77b4', alpha=0.1)
            ax1.plot(df.index, df['bb_upper'], '--', linewidth=1, color='#1f77b4', alpha=0.5, label='Upper Band')
            ax1.plot(df.index, df['bb_lower'], '--', linewidth=1, color='#1f77b4', alpha=0.5, label='Lower Band')
        
        # Highlight pattern occurrences
        if 'pattern_bullish_engulfing' in df.columns:
            bullish_engulfing_dates = df[df['pattern_bullish_engulfing'] == 1].index
            ax1.scatter(bullish_engulfing_dates, df.loc[bullish_engulfing_dates, 'close'], 
                       marker='^', color='green', s=100, label='Bullish Pattern')
        
        if 'pattern_bearish_engulfing' in df.columns:
            bearish_engulfing_dates = df[df['pattern_bearish_engulfing'] == 1].index
            ax1.scatter(bearish_engulfing_dates, df.loc[bearish_engulfing_dates, 'close'], 
                       marker='v', color='red', s=100, label='Bearish Pattern')
        
        # Add trend zones
        if 'uptrend' in df.columns and 'downtrend' in df.columns:
            # Find start and end of uptrends
            uptrend_changes = df['uptrend'].diff().fillna(0)
            uptrend_starts = df[uptrend_changes == 1].index
            uptrend_ends = df[uptrend_changes == -1].index
            
            # Shade uptrend areas
            for i in range(len(uptrend_starts)):
                start = uptrend_starts[i]
                if i < len(uptrend_ends) and uptrend_ends[i] > start:
                    end = uptrend_ends[i]
                else:
                    end = df.index[-1]
                ax1.axvspan(start, end, alpha=0.2, color='green')
        
            # Add annotations for significant crossovers
            if 'golden_cross' in df.columns and 'death_cross' in df.columns:
                golden_cross_dates = df[df['golden_cross'] == 1].index
                for date in golden_cross_dates:
                    ax1.axvline(x=date, color='green', linestyle='--', alpha=0.7)
                    price = df.loc[date, 'close']
                    ax1.annotate('Buy Signal', xy=(date, price), xytext=(date, price*0.95),
                               arrowprops=dict(facecolor='green', alpha=0.5),
                               color='green', fontsize=10)
                
                death_cross_dates = df[df['death_cross'] == 1].index
                for date in death_cross_dates:
                    ax1.axvline(x=date, color='red', linestyle='--', alpha=0.7)
                    price = df.loc[date, 'close']
                    ax1.annotate('Sell Signal', xy=(date, price), xytext=(date, price*1.05),
                               arrowprops=dict(facecolor='red', alpha=0.5),
                               color='red', fontsize=10)
        
        # Add title and labels
        ax1.set_title(f"{symbol} Price Chart with Trading Signals (2023)", fontsize=16, pad=10)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.legend(loc='best')
        ax1.xaxis.set_major_formatter(date_format)
        
        # Add explanation text
        ax1.text(0.01, 0.01, 
                "Moving Averages: Show average price over time\n"
                "Bollinger Bands: Price typically stays within these bands\n"
                "Green Zones: Uptrend periods (prices generally rising)\n"
                "Buy/Sell Signals: When short-term average crosses long-term", 
                transform=ax1.transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # ---- PANEL 2: RSI ----
        if 'rsi' in df.columns:
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.plot(df.index, df['rsi'], color='#d62728', linewidth=1.5)
            ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            ax2.fill_between(df.index, df['rsi'], 70, where=(df['rsi'] >= 70), 
                           color='r', alpha=0.3)
            ax2.fill_between(df.index, df['rsi'], 30, where=(df['rsi'] <= 30), 
                           color='g', alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI", fontsize=12)
            ax2.xaxis.set_major_formatter(date_format)
            
            # Add explanation text
            ax2.text(0.01, 0.05, 
                    "RSI (Relative Strength Index): Measures momentum\n"
                    "Above 70: Potentially overbought (consider selling)\n"
                    "Below 30: Potentially oversold (consider buying)", 
                    transform=ax2.transAxes, fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # ---- PANEL 3: MACD ----
        if all(x in df.columns for x in ['macd', 'macd_signal']):
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.plot(df.index, df['macd'], label='MACD Line', color='#1f77b4', linewidth=1.5)
            ax3.plot(df.index, df['macd_signal'], label='Signal Line', color='#ff7f0e', linewidth=1.5)
            
            # Color the histogram bars based on value
            positive = df['macd_histogram'] > 0
            negative = df['macd_histogram'] <= 0
            
            ax3.bar(df.index[positive], df.loc[positive, 'macd_histogram'], color='green', alpha=0.5, width=1.5)
            ax3.bar(df.index[negative], df.loc[negative, 'macd_histogram'], color='red', alpha=0.5, width=1.5)
            
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax3.set_ylabel("MACD", fontsize=12)
            ax3.legend(loc='upper right')
            ax3.xaxis.set_major_formatter(date_format)
            
            # Add explanation text
            ax3.text(0.01, 0.05, 
                    "MACD (Moving Average Convergence Divergence): Shows momentum changes\n"
                    "Blue Line Above Orange: Bullish momentum (potential buy)\n"
                    "Blue Line Below Orange: Bearish momentum (potential sell)\n"
                    "Green/Red Bars: Strength of bullish/bearish momentum", 
                    transform=ax3.transAxes, fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # ---- PANEL 4: Volume ----
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        volume_colors = np.where(df['daily_return'].values > 0, 'green', 'red')
        ax4.bar(df.index, df['volume'], color=volume_colors, alpha=0.7, width=1.5)
        ax4.set_ylabel("Volume", fontsize=12)
        ax4.xaxis.set_major_formatter(date_format)
        
        # Add explanation text
        ax4.text(0.01, 0.6, 
                "Volume: Number of shares traded each day\n"
                "Green: Volume on days with price increase\n"
                "Red: Volume on days with price decrease\n"
                "High Volume: More traders participating (stronger moves)", 
                transform=ax4.transAxes, fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add common x-axis label
        fig.text(0.5, 0.04, "Date", ha='center', fontsize=14)
        
        # Add summary section at the bottom
        performance = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        avg_volume = df['volume'].mean()
        volatility = df['daily_return'].std() * 100
        
        summary_text = (
            f"Summary for {symbol} in 2023:\n"
            f"• Performance: {performance:.2f}% {('↑' if performance > 0 else '↓')}\n"
            f"• Average Daily Volume: {avg_volume:.0f} shares\n"
            f"• Volatility: {volatility:.2f}%\n"
            f"• Bullish Patterns Detected: {sum(pattern_counts)}\n"
            f"• Golden Crosses (Buy Signals): {df['golden_cross'].sum()}\n"
            f"• Death Crosses (Sell Signals): {df['death_cross'].sum()}"
        )
        
        fig.text(0.5, 0.01, summary_text, ha='center', fontsize=12, 
                bbox=dict(facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save the enhanced chart
        plt.savefig('enhanced_analysis.png', dpi=300, bbox_inches='tight')
        print("Enhanced chart saved as 'enhanced_analysis.png'")
        
        # Create a separate simple summary chart for beginners
        create_beginner_summary(df, symbol)
        
    except Exception as e:
        print(f"Error generating enhanced visualizations: {e}")

def create_beginner_summary(df, symbol):
    """Create a simplified summary chart for beginners."""
    try:
        # Create a new figure
        plt.figure(figsize=(12, 8))
        
        # Calculate monthly returns
        monthly_returns = df.resample('M')['close'].last().pct_change() * 100
        monthly_returns = monthly_returns.dropna()
        
        # Plot monthly performance
        ax = monthly_returns.plot(kind='bar', color=np.where(monthly_returns.values > 0, 'green', 'red'))
        
        # Add labels and title
        plt.title(f"{symbol} Monthly Performance in 2023 - Simple View", fontsize=16)
        plt.ylabel("Monthly Return (%)", fontsize=14)
        plt.xlabel("Month", fontsize=14)
        
        # Add value labels on bars
        for i, value in enumerate(monthly_returns.values):
            color = 'white' if abs(value) > 5 else 'black'
            plt.text(i, value + (1 if value > 0 else -1), 
                    f"{value:.1f}%", 
                    ha='center', va='center', fontsize=10,
                    color=color, fontweight='bold')
        
        # Add a dashed line at y=0
        plt.axhline(y=0, linestyle='--', color='gray', alpha=0.7)
        
        # Add explanation
        explanation = (
            "This chart shows AAPL's monthly performance in 2023.\n"
            "• Green bars: Months when the stock price increased\n"
            "• Red bars: Months when the stock price decreased\n"
            "• The percentage shows how much the price changed each month"
        )
        
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('beginner_summary.png', dpi=300, bbox_inches='tight')
        print("Beginner summary chart saved as 'beginner_summary.png'")
        
    except Exception as e:
        print(f"Error generating beginner summary: {e}")

if __name__ == "__main__":
    result_df = test_data_pipeline()