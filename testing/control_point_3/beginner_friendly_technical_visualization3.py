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
    """Validate trading signals using multiple technical indicators."""
    try:
        # Get current and previous data points
        loc = df.index.get_loc(index)
        prev_idx = max(0, loc - 3)

        # Get all required indicators
        price = df.loc[index, 'close']
        prev_price = df.iloc[prev_idx]['close']
        rsi = df.loc[index, 'rsi'] if 'rsi' in df.columns else None
        macd = df.loc[index, 'macd'] if 'macd' in df.columns else None
        macd_signal = df.loc[index,
                             'macd_signal'] if 'macd_signal' in df.columns else None
        volume = df.loc[index, 'volume'] if 'volume' in df.columns else None
        avg_volume = df['volume'].rolling(window=20).mean(
        ).loc[index] if 'volume' in df.columns else None

        # Calculate trend
        price_change = ((price - prev_price) / prev_price) * 100

        signal_strength = 0
        conditions_met = []

        # Buy conditions
        if any(x in pattern_name.lower() for x in ['bullish', 'hammer', 'doji']):
            # RSI oversold condition
            if rsi and rsi < 40:
                signal_strength += 2
                conditions_met.append("RSI")

            # MACD crossover
            if macd and macd_signal and macd > macd_signal:
                signal_strength += 2
                conditions_met.append("MACD")

            # Volume confirmation
            if volume and avg_volume and volume > avg_volume * 1.5:
                signal_strength += 1
                conditions_met.append("VOL")

            # Downtrend reversal
            if price_change < -2:
                signal_strength += 1
                conditions_met.append("TREND")

            if signal_strength >= 3:
                return True, "BUY", conditions_met

        # Sell conditions
        elif any(x in pattern_name.lower() for x in ['bearish', 'shooting']):
            # RSI overbought condition
            if rsi and rsi > 60:
                signal_strength += 2
                conditions_met.append("RSI")

            # MACD crossover
            if macd and macd_signal and macd < macd_signal:
                signal_strength += 2
                conditions_met.append("MACD")

            # Volume confirmation
            if volume and avg_volume and volume > avg_volume * 1.5:
                signal_strength += 1
                conditions_met.append("VOL")

            # Uptrend reversal
            if price_change > 2:
                signal_strength += 1
                conditions_met.append("TREND")

            if signal_strength >= 3:
                return True, "SELL", conditions_met

        return False, "", []
    except Exception as e:
        print(f"Error validating signal: {e}")
        return False, "", []


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
    
    if df is None or df.empty:
        print("Failed to retrieve data. Check API key and connection.")
        return None
    
    try:
        # Process data and add indicators
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_features(df)
        df = TechnicalIndicators.add_all_indicators(df)
        df = PatternRecognition.recognize_candlestick_patterns(df)
        
        # Add support and resistance levels
        df = PatternRecognition.detect_support_resistance(df)
        
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def create_technical_insight_chart(df, output_file="./technical_insight_chart.png"):
    """Create enhanced technical analysis chart with support/resistance and signals."""
    if df is None or df.empty:
        print("No data available.")
        return

    try:
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create figure with secondary y-axis for volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)

        # Plot price and moving averages
        ax1.plot(df.index, df['close'], label='Price',
                 color='blue', linewidth=2)
        if 'sma_20' in df.columns:
            ax1.plot(df.index, df['sma_20'], label='20MA',
                     color='orange', linewidth=1)
        if 'sma_50' in df.columns:
            ax1.plot(df.index, df['sma_50'],
                     label='50MA', color='red', linewidth=1)

        # Plot support and resistance levels with zones
        if 'support_level' in df.columns:
            support_levels = df['support_level'].dropna().unique()
            for level in support_levels:
                ax1.axhline(y=level, color='green', linestyle='--', alpha=0.5)
                ax1.fill_between(df.index, level-level*0.005, level+level*0.005,
                                 color='green', alpha=0.1)
                ax1.text(df.index[-1], level, f'S: {level:.2f}', color='green',
                         va='bottom', ha='right')

        if 'resistance_level' in df.columns:
            resistance_levels = df['resistance_level'].dropna().unique()
            for level in resistance_levels:
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5)
                ax1.fill_between(df.index, level-level*0.005, level+level*0.005,
                                 color='red', alpha=0.1)
                ax1.text(df.index[-1], level, f'R: {level:.2f}', color='red',
                         va='top', ha='right')

        # Plot signals with improved annotations
        pattern_cols = [
            col for col in df.columns if col.startswith('pattern_')]
        for col in pattern_cols:
            pattern_dates = df.index[df[col] == 1]

            for d in pattern_dates:
                is_valid, signal, conditions = validate_trading_signal(
                    df, d, col)  # Updated to handle 3 return values

                if is_valid:
                    price = df.loc[d, 'close']
                    color = 'green' if signal == "BUY" else 'red'
                    marker = '^' if signal == "BUY" else 'v'

                    # Plot signal marker with conditions
                    ax1.scatter(d, price, color=color,
                                marker=marker, s=150, zorder=5)
                    ax1.annotate(f'{signal}\n{"+".join(conditions)}',
                                 xy=(d, price),
                                 xytext=(0, 15 if signal == "BUY" else -15),
                                 textcoords='offset points',
                                 ha='center',
                                 va='center',
                                 color=color,
                                 fontweight='bold',
                                 fontsize=9,
                                 bbox=dict(facecolor='white', edgecolor=color, alpha=0.7, pad=0.5))

        # Plot volume in lower subplot
        if 'volume' in df.columns:
            volume_colors = ['green' if c >= o else 'red'
                             for c, o in zip(df['close'], df['open'])]
            ax2.bar(df.index, df['volume'], color=volume_colors, alpha=0.5)
            ax2.set_ylabel('Volume', fontsize=10)

            # Add volume MA
            volume_ma = df['volume'].rolling(window=20).mean()
            ax2.plot(df.index, volume_ma, color='blue', linewidth=1, alpha=0.8,
                     label='Volume MA(20)')
            ax2.legend(loc='upper right')

        # Enhance chart formatting
        ax1.set_title(f"Trading Signals ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})",
                      fontsize=14, pad=20)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax1.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enhanced technical chart saved to {output_file}")

    except Exception as e:
        print(f"Error creating chart: {e}")
        plt.close()


if __name__ == "__main__":
    print("Fetching and processing data...")
    df = fetch_and_process_data(symbol="MSFT", days=100)

    if df is not None:
        print("Creating trading signals chart...")
        create_technical_insight_chart(df)
