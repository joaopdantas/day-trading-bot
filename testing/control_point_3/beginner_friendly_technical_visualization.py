"""
Enhanced Beginner-Friendly Visualization for MakesALot Trading Bot.
Creates technical analysis visualization with reliable signals using DatasetPreparation.
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
from src.indicators.technical import TechnicalIndicators, PatternRecognition, DatasetPreparation

def fetch_and_process_data(symbol="MSFT", days=100):
    """Fetch and process market data with technical indicators."""
    try:
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
            print("No data available.")
            return None

        # Process data and add indicators
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_features(df)
        df = TechnicalIndicators.add_all_indicators(df)
        df = PatternRecognition.recognize_candlestick_patterns(df)
        df = PatternRecognition.detect_support_resistance(df)
        df = PatternRecognition.detect_trend(df)
        
        # Generate trading signals using improved method
        signals_df = DatasetPreparation.create_target_labels(
            df, 
            horizon=5, 
            threshold=0.01,
            min_risk_reward_ratio=1.5
        )
        
        # Merge signals back into main dataframe
        for col in signals_df.columns:
            df[col] = signals_df[col]
            
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def create_technical_insight_chart(df, output_file="testing/control_point_3/enhanced_technical_insight_chart1.png"):
    """Create enhanced technical analysis chart with support/resistance and reliable signals."""
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
            ax1.plot(df.index, df['sma_20'], label='20-day MA',
                     color='orange', linewidth=1)
        if 'sma_50' in df.columns:
            ax1.plot(df.index, df['sma_50'],
                     label='50-day MA', color='red', linewidth=1)

        # Plot support and resistance levels with zones
        if 'support_level' in df.columns:
            support_levels = df['support_level'].dropna().unique()
            for level in support_levels:
                ax1.axhline(y=level, color='green', linestyle='--', alpha=0.5)
                ax1.fill_between(df.index, level-level*0.005, level+level*0.005,
                                 color='green', alpha=0.1)
                ax1.text(df.index[-1], level, f'Support: {level:.2f}', color='green',
                         va='bottom', ha='right')

        if 'resistance_level' in df.columns:
            resistance_levels = df['resistance_level'].dropna().unique()
            for level in resistance_levels:
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5)
                ax1.fill_between(df.index, level-level*0.005, level+level*0.005,
                                 color='red', alpha=0.1)
                ax1.text(df.index[-1], level, f'Resistance: {level:.2f}', color='red',
                         va='top', ha='right')

        # Plot signals based on target_label
        if 'target_label' in df.columns:
            # Buy signals
            buy_indices = df[df['target_label'] == 1].index
            if not buy_indices.empty:
                buy_prices = df.loc[buy_indices, 'close']
                ax1.scatter(buy_indices, buy_prices, marker='^', color='green', 
                            s=150, label='Buy Signal', zorder=5)
                
                # Add probability annotations for each buy signal
                if 'signal_probability' in df.columns:
                    for idx in buy_indices:
                        prob = df.loc[idx, 'signal_probability']
                        exp_return = df.loc[idx, 'expected_return'] * 100 if 'expected_return' in df.columns else 0
                        
                        annotation = f"BUY\nConf: {prob:.2f}\nExp: {exp_return:.1f}%"
                        ax1.annotate(annotation,
                                    xy=(idx, df.loc[idx, 'close']),
                                    xytext=(0, 20),
                                    textcoords='offset points',
                                    ha='center',
                                    va='bottom',
                                    color='darkgreen',
                                    fontweight='bold',
                                    fontsize=9,
                                    bbox=dict(facecolor='white', edgecolor='green', alpha=0.7, pad=0.5))
            
            # Sell signals
            sell_indices = df[df['target_label'] == -1].index
            if not sell_indices.empty:
                sell_prices = df.loc[sell_indices, 'close']
                ax1.scatter(sell_indices, sell_prices, marker='v', color='red', 
                            s=150, label='Sell Signal', zorder=5)
                
                # Add probability annotations for each sell signal
                if 'signal_probability' in df.columns:
                    for idx in sell_indices:
                        prob = df.loc[idx, 'signal_probability']
                        exp_return = df.loc[idx, 'expected_return'] * 100 if 'expected_return' in df.columns else 0
                        
                        annotation = f"SELL\nConf: {prob:.2f}\nExp: {exp_return:.1f}%"
                        ax1.annotate(annotation,
                                    xy=(idx, df.loc[idx, 'close']),
                                    xytext=(0, -20),
                                    textcoords='offset points',
                                    ha='center',
                                    va='top',
                                    color='darkred',
                                    fontweight='bold',
                                    fontsize=9,
                                    bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, pad=0.5))

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
            ax2.legend(loc='upper left')

        # Add explanatory text box
        explanation = """
TRADING SIGNALS EXPLANATION:
• Green triangles ▲: Buy signals with high confidence
• Red triangles ▼: Sell signals with high confidence
• Conf: Signal confidence level (0-1)
• Exp: Expected return percentage
• Signal calculation uses multiple factors including technical
  indicators, price patterns, and risk/reward analysis
        """
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax1.text(0.02, 0.02, explanation, transform=ax1.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='left', bbox=props)

        # Enhance chart formatting
        ax1.set_title(f"Advanced Trading Signals ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})",
                      fontsize=14, pad=20)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enhanced technical chart saved to {output_file}")

    except Exception as e:
        print(f"Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        plt.close()


if __name__ == "__main__":
    print("Fetching and processing data...")
    df = fetch_and_process_data(symbol="MSFT", days=100)

    if df is not None:
        print("Creating enhanced technical analysis visualization...")
        create_technical_insight_chart(df)