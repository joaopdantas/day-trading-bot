"""
Beginner-Friendly Visualization for MakesALot Trading Bot.

This script creates easy-to-understand visualizations of technical analysis
with explanations suitable for trading beginners.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition


def fetch_and_process_data(symbol="MSFT", days=100):
    """Fetch and process data for visualization."""
    # Initialize data fetcher
    api = get_data_api("alpha_vantage")
    
    # Fetch historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = api.fetch_historical_data(
        symbol=symbol, 
        interval="1d",
        start_date=start_date,
        end_date=end_date
    )
    
    if df.empty:
        print("Failed to retrieve data. Please check your API key and connection.")
        return None
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_features(df)
    
    # Add technical indicators
    df_indicators = TechnicalIndicators.add_all_indicators(df_processed)
    
    # Add pattern recognition
    df_with_patterns = PatternRecognition.recognize_candlestick_patterns(df_indicators)
    df_with_patterns = PatternRecognition.detect_support_resistance(df_with_patterns)
    df_with_patterns = PatternRecognition.detect_trend(df_with_patterns)
    
    return df_with_patterns


def create_beginner_friendly_chart(df, output_file="beginner_friendly_chart.png"):
    """Create a beginner-friendly chart with explanations."""
    if df is None or df.empty:
        print("No data available for visualization.")
        return
    
    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.patch.set_facecolor('white')
    
    # Create custom colormap for backgrounds
    buy_zone_color = LinearSegmentedColormap.from_list('buy_zone', ['white', '#e6ffe6'])
    sell_zone_color = LinearSegmentedColormap.from_list('sell_zone', ['white', '#ffe6e6'])
    
    # Set title with more info
    symbol = df.get('symbol', ['MSFT'])[0] if 'symbol' in df.columns else 'MSFT'
    fig.suptitle(f'MakesALot Trading Analysis for {symbol}', fontsize=16, fontweight='bold')
    
    # ------ PANEL 1: Price Chart with Patterns ------
    ax1 = axs[0]
    ax1.set_title("Price Chart with Trading Signals", fontsize=14)
    
    # Plot price
    ax1.plot(df.index, df['close'], color='#0066cc', linewidth=2, label='Stock Price')
    
    # Add moving averages if available
    if 'sma_20' in df.columns:
        ax1.plot(df.index, df['sma_20'], color='#ff9900', linewidth=1.5, 
                 label='20-Day Average (Short-Term Trend)')
    
    if 'sma_50' in df.columns:
        ax1.plot(df.index, df['sma_50'], color='#cc3300', linewidth=1.5, 
                 label='50-Day Average (Long-Term Trend)')
    
    # Highlight buy/sell zones based on moving average crossovers
    if 'uptrend' in df.columns and 'downtrend' in df.columns:
        # Find areas of uptrend
        uptrend_mask = df['uptrend'] == 1
        if uptrend_mask.any():
            # Create spans for uptrends
            uptrend_changes = df['uptrend'].diff().fillna(0)
            uptrend_starts = df.index[uptrend_changes == 1].tolist()
            uptrend_ends = df.index[uptrend_changes == -1].tolist()
            
            # Handle case where trend starts at beginning of data
            if df['uptrend'].iloc[0] == 1:
                uptrend_starts.insert(0, df.index[0])
            
            # Handle case where trend continues to end of data
            if df['uptrend'].iloc[-1] == 1:
                uptrend_ends.append(df.index[-1])
            
            # Create shaded regions for uptrends
            for start, end in zip(uptrend_starts, uptrend_ends):
                ax1.axvspan(start, end, alpha=0.2, color='green', label='_nolegend_')
                # Add "Buying Zone" text if span is wide enough
                span_days = (end - start).days
                if span_days > 5:
                    mid_point = start + (end - start) / 2
                    y_pos = df['close'].max() * 0.95
                    ax1.text(mid_point, y_pos, "Buying Zone", ha='center', 
                             color='darkgreen', fontweight='bold', fontsize=10)
        
        # Find areas of downtrend
        downtrend_mask = df['downtrend'] == 1
        if downtrend_mask.any():
            # Create spans for downtrends
            downtrend_changes = df['downtrend'].diff().fillna(0)
            downtrend_starts = df.index[downtrend_changes == 1].tolist()
            downtrend_ends = df.index[downtrend_changes == -1].tolist()
            
            # Handle case where trend starts at beginning of data
            if df['downtrend'].iloc[0] == 1:
                downtrend_starts.insert(0, df.index[0])
            
            # Handle case where trend continues to end of data
            if df['downtrend'].iloc[-1] == 1:
                downtrend_ends.append(df.index[-1])
            
            # Create shaded regions for downtrends
            for start, end in zip(downtrend_starts, downtrend_ends):
                ax1.axvspan(start, end, alpha=0.2, color='red', label='_nolegend_')
                # Add "Selling Zone" text if span is wide enough
                span_days = (end - start).days
                if span_days > 5:
                    mid_point = start + (end - start) / 2
                    y_pos = df['close'].max() * 0.95
                    ax1.text(mid_point, y_pos, "Selling Zone", ha='center', 
                             color='darkred', fontweight='bold', fontsize=10)
    
    # Highlight patterns
    pattern_plotted = {
        'bullish': False,
        'bearish': False
    }
    
    for col in df.columns:
        if 'pattern_' in col and df[col].sum() > 0:
            pattern_type = 'bullish' if 'bullish' in col or 'hammer' in col or 'doji' in col else 'bearish'
            pattern_dates = df.index[df[col] == 1]
            pattern_prices = df.loc[pattern_dates, 'close']
            
            if len(pattern_dates) > 0 and not pattern_plotted[pattern_type]:
                if pattern_type == 'bullish':
                    marker = '^'
                    color = 'green'
                    size = 120
                    name = 'Bullish Pattern (Buy Signal)'
                else:
                    marker = 'v'
                    color = 'red'
                    size = 120
                    name = 'Bearish Pattern (Sell Signal)'
                
                # Plot the patterns
                ax1.scatter(pattern_dates, pattern_prices, 
                           marker=marker, color=color, s=size, label=name, zorder=5)
                
                # Add annotations for the first few patterns
                for i, (date, price) in enumerate(zip(pattern_dates, pattern_prices)):
                    if i < 3:  # Only annotate first 3 patterns to avoid clutter
                        action = "BUY" if pattern_type == 'bullish' else "SELL"
                        ax1.annotate(f"{action} Signal", 
                                   xy=(date, price),
                                   xytext=(10, 20 if pattern_type == 'bullish' else -20),
                                   textcoords="offset points",
                                   arrowprops=dict(arrowstyle="->", color=color),
                                   color=color,
                                   fontweight='bold')
                
                pattern_plotted[pattern_type] = True
    
    # Add support and resistance levels
    if 'support_level' in df.columns:
        support_levels = df['support_level'].dropna().unique()
        for level in support_levels:
            ax1.axhline(y=level, color='green', linestyle='--', linewidth=1.5, 
                       label='Support Level (Potential Buy Zone)')
            ax1.text(df.index[-1], level, f"  Support: ${level:.2f}", 
                   va='center', ha='left', color='green', fontweight='bold')
    
    if 'resistance_level' in df.columns:
        resistance_levels = df['resistance_level'].dropna().unique()
        for level in resistance_levels:
            ax1.axhline(y=level, color='red', linestyle='--', linewidth=1.5, 
                       label='Resistance Level (Potential Sell Zone)')
            ax1.text(df.index[-1], level, f"  Resistance: ${level:.2f}", 
                   va='center', ha='left', color='red', fontweight='bold')
    
    # Add explanation box for price chart
    explanation_text = (
        "PRICE CHART EXPLAINED:\n"
        "• Blue Line: Stock price over time\n"
        "• Orange Line: Short-term average (20 days)\n"
        "• Red Line: Long-term average (50 days)\n"
        "• Green Areas: Potential buying opportunities\n"
        "• Red Areas: Potential selling opportunities\n"
        "• Green ▲: Buy signals from patterns\n"
        "• Red ▼: Sell signals from patterns"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(0.02, 0.02, explanation_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    # Format price chart
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.grid(True, alpha=0.3)
    
    # ------ PANEL 2: RSI with Explanation ------
    ax2 = axs[1]
    ax2.set_title("RSI: Momentum Indicator", fontsize=14)
    
    if 'rsi' in df.columns:
        # Plot RSI
        ax2.plot(df.index, df['rsi'], color='purple', linewidth=2, label='RSI')
        
        # Add reference lines
        ax2.axhline(y=70, color='red', linestyle='-', label='Overbought (70)')
        ax2.axhline(y=30, color='green', linestyle='-', label='Oversold (30)')
        ax2.axhline(y=50, color='gray', linestyle='--', label='Neutral (50)')
        
        # Color zones
        ax2.fill_between(df.index, 70, 100, color='red', alpha=0.1)
        ax2.fill_between(df.index, 0, 30, color='green', alpha=0.1)
        
        # Add labels to the zones
        ax2.text(df.index[len(df.index)//2], 80, "OVERBOUGHT - Consider Selling", 
               ha='center', color='darkred', fontweight='bold')
        ax2.text(df.index[len(df.index)//2], 15, "OVERSOLD - Consider Buying", 
               ha='center', color='darkgreen', fontweight='bold')
        
        # Add explanation box for RSI
        rsi_explanation = (
            "RSI EXPLAINED:\n"
            "• RSI (Relative Strength Index) measures price momentum\n"
            "• RSI above 70 may indicate the stock is overbought (overvalued)\n"
            "• RSI below 30 may indicate the stock is oversold (undervalued)\n"
            "• RSI direction shows momentum (rising = strengthening, falling = weakening)"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax2.text(0.02, 0.05, rsi_explanation, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=props)
    
        # Format RSI chart
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI Value", fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "RSI data not available with current sample size", 
               ha='center', va='center', fontsize=12, transform=ax2.transAxes)
    
    # ------ PANEL 3: Bollinger Bands with Explanation ------
    ax3 = axs[2]
    ax3.set_title("Bollinger Bands: Volatility Indicator", fontsize=14)
    
    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        # Plot price
        ax3.plot(df.index, df['close'], color='blue', linewidth=2, label='Price')
        
        # Plot Bollinger Bands
        ax3.plot(df.index, df['bb_upper'], 'r--', linewidth=1.5, label='Upper Band (Potential Sell Zone)')
        ax3.plot(df.index, df['bb_middle'], 'g--', linewidth=1.5, label='Middle Band (Average)')
        ax3.plot(df.index, df['bb_lower'], 'r--', linewidth=1.5, label='Lower Band (Potential Buy Zone)')
        
        # Shade area between bands
        ax3.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='gray', alpha=0.1)
        
        # Add annotations for crossings
        price_crosses_upper = ((df['close'] > df['bb_upper']) & 
                              (df['close'].shift(1) <= df['bb_upper'].shift(1)))
        price_crosses_lower = ((df['close'] < df['bb_lower']) & 
                              (df['close'].shift(1) >= df['bb_lower'].shift(1)))
        
        upper_cross_dates = df.index[price_crosses_upper]
        lower_cross_dates = df.index[price_crosses_lower]
        
        for date in upper_cross_dates:
            price = df.loc[date, 'close']
            ax3.annotate("Potential Sell Signal", 
                       xy=(date, price), 
                       xytext=(10, 15),
                       textcoords="offset points",
                       arrowprops=dict(arrowstyle="->", color='red'),
                       color='red',
                       fontweight='bold')
        
        for date in lower_cross_dates:
            price = df.loc[date, 'close']
            ax3.annotate("Potential Buy Signal", 
                       xy=(date, price), 
                       xytext=(10, -15),
                       textcoords="offset points",
                       arrowprops=dict(arrowstyle="->", color='green'),
                       color='green',
                       fontweight='bold')
        
        # Add explanation box for Bollinger Bands
        bb_explanation = (
            "BOLLINGER BANDS EXPLAINED:\n"
            "• Bollinger Bands show price volatility and potential reversal points\n"
            "• Middle band: 20-day moving average (normal price)\n"
            "• Upper band: Potential resistance (price may be too high)\n"
            "• Lower band: Potential support (price may be too low)\n"
            "• When price touches or crosses a band, it often reverses direction"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax3.text(0.02, 0.05, bb_explanation, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=props)
        
        # Format Bollinger Bands chart
        ax3.legend(loc='upper right')
        ax3.set_ylabel("Price ($)", fontsize=12)
        ax3.set_xlabel("Date", fontsize=12)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Bollinger Bands data not available with current sample size", 
               ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    
    # Add summary info
    price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    action_color = 'green' if price_change > 0 else 'red'
    
    last_price = df['close'].iloc[-1]
    start_price = df['close'].iloc[0]
    price_trend = "UPWARD" if last_price > start_price else "DOWNWARD"
    
    current_trend = "UPTREND" if (df['uptrend'].iloc[-1] == 1) else "DOWNTREND" if (df['downtrend'].iloc[-1] == 1) else "NEUTRAL"
    
    # Determine recommendation
    recommendation = "HOLD"
    reasoning = ""
    
    if 'rsi' in df.columns:
        last_rsi = df['rsi'].iloc[-1]
        if last_rsi > 70:
            recommendation = "SELL/REDUCE"
            reasoning = f"RSI is high at {last_rsi:.1f}, suggesting potential overvaluation."
        elif last_rsi < 30:
            recommendation = "BUY/ACCUMULATE"
            reasoning = f"RSI is low at {last_rsi:.1f}, suggesting potential undervaluation."
    
    # Override based on trend if RSI is neutral
    if reasoning == "" and current_trend == "UPTREND":
        recommendation = "BUY/HOLD"
        reasoning = "Stock is in an uptrend, suggesting potential continued growth."
    elif reasoning == "" and current_trend == "DOWNTREND":
        recommendation = "HOLD/REDUCE"
        reasoning = "Stock is in a downtrend, suggesting caution."
    
    # Override based on pattern signals
    recent_bullish = False
    recent_bearish = False
    
    for col in df.columns:
        if 'pattern_' in col:
            if 'bullish' in col or 'hammer' in col or 'doji' in col:
                if df[col].iloc[-5:].sum() > 0:
                    recent_bullish = True
            else:
                if df[col].iloc[-5:].sum() > 0:
                    recent_bearish = True
    
    if recent_bullish and not recent_bearish:
        recommendation = "BUY/ACCUMULATE"
        reasoning = "Recent bullish pattern signals suggest potential price increase."
    elif recent_bearish and not recent_bullish:
        recommendation = "SELL/REDUCE"
        reasoning = "Recent bearish pattern signals suggest potential price decrease."
    
    # Add overall insight and recommendation
    summary_text = (
        f"MARKET INSIGHT SUMMARY:\n\n"
        f"• {symbol} has shown a {price_trend} trend over this period ({price_change:.1f}%)\n"
        f"• Current technical trend assessment: {current_trend}\n"
        f"• Opening price: ${start_price:.2f}\n"
        f"• Current price: ${last_price:.2f}\n\n"
        f"RECOMMENDATION: {recommendation}\n"
        f"Reasoning: {reasoning}"
    )
    
    # Add a text box with the summary
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=12, bbox=props)
    
    # Add MakesALot branding
    fig.text(0.02, 0.01, "MakesALot Trading Bot - Beginner-Friendly Analysis", 
             color='darkblue', fontweight='bold', fontsize=10, ha='left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    os.makedirs('testing/control_point_2', exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Beginner-friendly chart saved to {output_file}")


if __name__ == "__main__":
    # Fetch and process data
    print("Fetching and processing data...")
    df = fetch_and_process_data(symbol="MSFT", days=100)
    
    if df is not None:
        # Create beginner-friendly visualization
        print("Creating beginner-friendly visualization...")
        output_file = 'testing/control_point_2/beginner_friendly_chart.png'
        create_beginner_friendly_chart(df, output_file)