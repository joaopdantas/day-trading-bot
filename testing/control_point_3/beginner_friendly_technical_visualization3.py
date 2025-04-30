"""
Enhanced Technical Visualization for MakesALot Trading Bot.
Creates comprehensive trading dashboard with machine learning enhanced signals.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.patches import Rectangle, FancyArrowPatch

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition, DatasetPreparation

def fetch_and_process_data(symbol="MSFT", days=100):
    """Fetch and process market data with advanced analytics."""
    print(f"Fetching and processing data for {symbol}...")
    
    # Try multiple data sources in case one fails
    api_sources = ["alpha_vantage", "yahoo_finance"]
    df = None
    
    for api_name in api_sources:
        try:
            api = get_data_api(api_name)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            df = api.fetch_historical_data(
                symbol=symbol,
                interval="1d",
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                print(f"Successfully retrieved data from {api_name}")
                break
        except Exception as e:
            print(f"Error fetching data from {api_name}: {e}")
    
    if df is None or df.empty:
        print("Failed to retrieve data from all sources. Check API keys and connection.")
        return None
    
    try:
        print(f"Processing {len(df)} days of market data...")
        
        # Add symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
            
        # Process data and add indicators
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_features(df)
        df = TechnicalIndicators.add_all_indicators(df)
        df = PatternRecognition.recognize_candlestick_patterns(df)
        df = PatternRecognition.detect_support_resistance(df)
        df = PatternRecognition.detect_trend(df)
        
        # Generate trading signals using the enhanced ML-based approach
        print("Generating ML-enhanced trading signals...")
        signal_df = DatasetPreparation.create_target_labels(
            df,
            horizon=5,                # Look 5 days ahead for returns
            threshold=0.01,           # 1% minimum expected return
            volatility_window=20,     # 20-day window for volatility
            min_risk_reward_ratio=1.5,# Risk/reward filter
            volume_filter=True,       # Apply volume confirmation
            confirm_with_indicators=True  # Use technical indicators for confirmation
        )
        
        # Merge signals with main dataframe
        for col in signal_df.columns:
            df[col] = signal_df[col]
            
        # Fill NaNs for visualization purposes
        df = df.ffill().bfill()
        
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comprehensive_dashboard(df, output_file="testing/control_point_3/enhanced_trading_dashboard3.png"):
    """Create comprehensive trading dashboard with advanced signals and analysis."""
    if df is None or df.empty:
        print("No data available for dashboard creation.")
        return
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 12), dpi=100)
        
        # Define gridspec for advanced layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])
        
        # Main price chart
        ax_price = fig.add_subplot(gs[0, 0])
        
        # Volume panel
        ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)
        
        # Indicators panel
        ax_indicators = fig.add_subplot(gs[2, 0], sharex=ax_price)
        
        # Summary panel
        ax_summary = fig.add_subplot(gs[:, 1])
        
        # ----- 1. PRICE CHART -----
        ax_price.set_title(f"{df['symbol'].iloc[0]} Price Chart with AI Trading Signals", 
                          fontsize=14, fontweight='bold')
        
        # Plot price
        ax_price.plot(df.index, df['close'], label='Price', color='blue', linewidth=2)
        
        # Plot moving averages
        if 'sma_20' in df.columns:
            ax_price.plot(df.index, df['sma_20'], label='20-day MA', 
                         color='orange', linewidth=1.5, alpha=0.8)
        
        if 'sma_50' in df.columns:
            ax_price.plot(df.index, df['sma_50'], label='50-day MA', 
                         color='red', linewidth=1.5, alpha=0.8)
        
        # Plot Bollinger Bands
        if all(x in df.columns for x in ['bb_upper', 'bb_middle', 'bb_lower']):
            ax_price.plot(df.index, df['bb_upper'], '--', color='gray', alpha=0.5, 
                         linewidth=1, label='Upper Band')
            ax_price.plot(df.index, df['bb_lower'], '--', color='gray', alpha=0.5, 
                         linewidth=1, label='Lower Band')
            ax_price.fill_between(df.index, df['bb_upper'], df['bb_lower'], 
                                color='blue', alpha=0.05)
        
        # Plot support and resistance levels
        if 'support_level' in df.columns:
            support_levels = df['support_level'].dropna().unique()
            for i, level in enumerate(support_levels):
                ax_price.axhline(y=level, color='green', linestyle='--', alpha=0.5, 
                               linewidth=1.5)
                ax_price.fill_between(df.index, level-level*0.005, level+level*0.005, 
                                    color='green', alpha=0.1)
                # Only label first support level to avoid cluttering
                if i == 0:
                    ax_price.text(df.index[-1], level, f' Support', va='center', 
                                color='green', fontweight='bold')
        
        if 'resistance_level' in df.columns:
            resistance_levels = df['resistance_level'].dropna().unique()
            for i, level in enumerate(resistance_levels):
                ax_price.axhline(y=level, color='red', linestyle='--', alpha=0.5, 
                               linewidth=1.5)
                ax_price.fill_between(df.index, level-level*0.005, level+level*0.005, 
                                    color='red', alpha=0.1)
                # Only label first resistance level to avoid cluttering
                if i == 0:
                    ax_price.text(df.index[-1], level, f' Resistance', va='center', 
                                color='red', fontweight='bold')
        
        # Plot optimized buy/sell signals
        if 'target_label' in df.columns:
            # Buy signals
            buy_signals = df[df['target_label'] == 1]
            if not buy_signals.empty:
                ax_price.scatter(buy_signals.index, buy_signals['close'], 
                               marker='^', color='green', s=150, zorder=5, 
                               label='Buy Signal')
                
                # Add annotations for high-confidence signals only
                for idx in buy_signals.index:
                    if 'signal_probability' in df.columns:
                        confidence = buy_signals.loc[idx, 'signal_probability']
                        if confidence >= 0.7:  # Only annotate high confidence signals
                            exp_return = buy_signals.loc[idx, 'expected_return'] * 100 if 'expected_return' in buy_signals.columns else 0
                            
                            ax_price.annotate(f"BUY\n{confidence:.2f}\n+{exp_return:.1f}%",
                                         xy=(idx, buy_signals.loc[idx, 'close']),
                                         xytext=(0, 25),
                                         textcoords='offset points',
                                         ha='center',
                                         color='darkgreen',
                                         fontweight='bold',
                                         fontsize=9,
                                         bbox=dict(facecolor='white', edgecolor='green', alpha=0.7, pad=0.5))
            
            # Sell signals
            sell_signals = df[df['target_label'] == -1]
            if not sell_signals.empty:
                ax_price.scatter(sell_signals.index, sell_signals['close'], 
                               marker='v', color='red', s=150, zorder=5, 
                               label='Sell Signal')
                
                # Add annotations for high-confidence signals only
                for idx in sell_signals.index:
                    if 'signal_probability' in df.columns:
                        confidence = sell_signals.loc[idx, 'signal_probability']
                        if confidence >= 0.7:  # Only annotate high confidence signals
                            exp_return = sell_signals.loc[idx, 'expected_return'] * 100 if 'expected_return' in sell_signals.columns else 0
                            
                            ax_price.annotate(f"SELL\n{confidence:.2f}\n{exp_return:.1f}%",
                                         xy=(idx, sell_signals.loc[idx, 'close']),
                                         xytext=(0, -25),
                                         textcoords='offset points',
                                         ha='center',
                                         color='darkred',
                                         fontweight='bold',
                                         fontsize=9,
                                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, pad=0.5))
        
        # Format price chart
        ax_price.set_ylabel('Price ($)', fontsize=12)
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        
        # ----- 2. VOLUME CHART -----
        if 'volume' in df.columns:
            # Color volume bars by price direction
            volume_colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
            ax_volume.bar(df.index, df['volume'], color=volume_colors, alpha=0.6, width=0.8)
            
            # Add volume moving average
            volume_ma = df['volume'].rolling(window=20).mean()
            ax_volume.plot(df.index, volume_ma, color='blue', linewidth=1.5, 
                         label='20-day Avg', alpha=0.8)
            
            # Highlight significant volume days
            if 'target_label' in df.columns:
                significant_volume = df[(df['volume'] > volume_ma * 1.5) & 
                                      ((df['target_label'] == 1) | (df['target_label'] == -1))]
                                      
                if not significant_volume.empty:
                    ax_volume.scatter(significant_volume.index, significant_volume['volume'],
                                   marker='*', color='blue', s=100, zorder=5, 
                                   label='Signal Volume')
            
            ax_volume.set_ylabel('Volume', fontsize=12)
            ax_volume.legend(loc='upper left')
            ax_volume.grid(True, alpha=0.3)
        
        # ----- 3. INDICATORS CHART -----
        # Plot RSI
        if 'rsi' in df.columns:
            ax_indicators.plot(df.index, df['rsi'], color='purple', linewidth=1.5, label='RSI')
            ax_indicators.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax_indicators.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax_indicators.fill_between(df.index, 70, 100, color='red', alpha=0.1)
            ax_indicators.fill_between(df.index, 0, 30, color='green', alpha=0.1)
            ax_indicators.text(df.index[0], 72, 'Overbought', color='darkred', fontsize=9)
            ax_indicators.text(df.index[0], 28, 'Oversold', color='darkgreen', fontsize=9)
            ax_indicators.set_ylim(0, 100)
        
        # Plot MACD histogram in the background
        if all(x in df.columns for x in ['macd', 'macd_signal', 'macd_histogram']):
            # Normalize MACD histogram to fit in RSI scale
            max_hist = max(abs(df['macd_histogram'].min()), abs(df['macd_histogram'].max()))
            if max_hist > 0:
                normalized_hist = df['macd_histogram'] * 25 / max_hist + 50
                
                # Color bars based on direction
                colors = ['green' if x > 0 else 'red' for x in df['macd_histogram']]
                ax_indicators.bar(df.index, normalized_hist - 50, bottom=50, color=colors, 
                                 alpha=0.3, width=0.8, label='MACD Hist')
        
        ax_indicators.set_ylabel('Indicators', fontsize=12)
        ax_indicators.legend(loc='upper left')
        ax_indicators.grid(True, alpha=0.3)
        
        # Set common x-axis format
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_volume.get_xticklabels(), visible=False)
        ax_indicators.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax_indicators.set_xlabel('Date', fontsize=12)
        
        # ----- 4. SUMMARY PANEL -----
        ax_summary.axis('off')  # Turn off axes
        
        # Create summary header
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "Stock"
        ax_summary.text(0.5, 0.98, f"{symbol} Analysis", 
                      fontsize=14, fontweight='bold', ha='center', va='top')
        
        # Calculate performance metrics
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100
        max_price = df['close'].max()
        min_price = df['close'].min()
        
        # Count signals
        buy_count = (df['target_label'] == 1).sum() if 'target_label' in df.columns else 0
        sell_count = (df['target_label'] == -1).sum() if 'target_label' in df.columns else 0
        
        # Calculate average metrics
        avg_rsi = df['rsi'].mean() if 'rsi' in df.columns else 0
        
        # Get current trend
        if 'uptrend' in df.columns and 'downtrend' in df.columns:
            if df['uptrend'].iloc[-1] == 1:
                current_trend = "UPTREND"
                trend_color = 'green'
            elif df['downtrend'].iloc[-1] == 1:
                current_trend = "DOWNTREND"
                trend_color = 'red'
            else:
                current_trend = "SIDEWAYS"
                trend_color = 'gray'
        else:
            if price_change > 5:
                current_trend = "UPTREND"
                trend_color = 'green'
            elif price_change < -5:
                current_trend = "DOWNTREND"
                trend_color = 'red'
            else:
                current_trend = "SIDEWAYS"
                trend_color = 'gray'
        
        # Create metrics summary
        summary_text = [
            f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            f"Trading Days: {len(df)}",
            f"Price Change: {price_change:.2f}%",
            f"Starting Price: ${start_price:.2f}",
            f"Current Price: ${end_price:.2f}",
            f"High: ${max_price:.2f}",
            f"Low: ${min_price:.2f}",
            f"Current Trend: {current_trend}",
            f"Buy Signals: {buy_count}",
            f"Sell Signals: {sell_count}",
            f"Avg RSI: {avg_rsi:.1f}"
        ]
        
        # Position the metrics
        y_pos = 0.9
        for text in summary_text:
            if "Current Trend" in text:
                parts = text.split(": ")
                ax_summary.text(0.05, y_pos, parts[0] + ": ", fontsize=11, ha='left')
                ax_summary.text(0.5, y_pos, parts[1], fontsize=11, ha='left', 
                              color=trend_color, fontweight='bold')
            elif "Price Change" in text:
                parts = text.split(": ")
                color = 'green' if price_change > 0 else 'red'
                ax_summary.text(0.05, y_pos, parts[0] + ": ", fontsize=11, ha='left')
                ax_summary.text(0.5, y_pos, parts[1], fontsize=11, ha='left', 
                              color=color, fontweight='bold')
            else:
                ax_summary.text(0.05, y_pos, text, fontsize=11, ha='left')
            y_pos -= 0.05
        
        # Add trading recommendation
        y_pos -= 0.05
        ax_summary.text(0.05, y_pos, "Trading Recommendation:", 
                      fontsize=12, fontweight='bold', ha='left')
        y_pos -= 0.05
        
        # Generate recommendation based on signals and trend
        if 'target_label' in df.columns and len(df) > 0:
            recent_signal = df['target_label'].iloc[-5:].sum()  # Check last 5 days
            
            if recent_signal > 0:
                recommendation = "BUY / ACCUMULATE"
                rec_color = 'green'
                reasoning = "Recent buy signals indicate potential upside."
            elif recent_signal < 0:
                recommendation = "SELL / REDUCE"
                rec_color = 'red'
                reasoning = "Recent sell signals indicate potential downside."
            else:
                if current_trend == "UPTREND":
                    recommendation = "HOLD / MONITOR BUY"
                    rec_color = 'green'
                    reasoning = "Uptrend continues but no recent signal."
                elif current_trend == "DOWNTREND":
                    recommendation = "HOLD / MONITOR SELL"
                    rec_color = 'red'
                    reasoning = "Downtrend continues but no recent signal."
                else:
                    recommendation = "HOLD / NEUTRAL"
                    rec_color = 'blue'
                    reasoning = "No clear trend or signals detected."
        else:
            recommendation = "INSUFFICIENT DATA"
            rec_color = 'gray'
            reasoning = "Not enough data for reliable recommendation."
        
        ax_summary.text(0.05, y_pos, recommendation, fontsize=14, 
                      fontweight='bold', color=rec_color, ha='left')
        y_pos -= 0.05
        ax_summary.text(0.05, y_pos, reasoning, fontsize=11, ha='left', fontstyle='italic')
        
        # Add explanatory notes
        y_pos -= 0.1
        ax_summary.text(0.05, y_pos, "TRADING SIGNALS INFO:", 
                      fontsize=11, fontweight='bold', ha='left')
        y_pos -= 0.05
        
        notes = [
            "• Signals use advanced ML algorithm",
            "• Based on 5-day future return prediction",
            "• Combines technical indicators and patterns",
            "• Includes risk/reward and volume analysis",
            "• Signals require confirmation (3+ factors)"
        ]
        
        for note in notes:
            ax_summary.text(0.05, y_pos, note, fontsize=10, ha='left')
            y_pos -= 0.03
        
        # Draw border around summary
        border = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, 
                             edgecolor='gray', linewidth=1.5, alpha=0.7,
                             transform=ax_summary.transAxes)
        ax_summary.add_patch(border)
        
        # Save the dashboard
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enhanced trading dashboard saved to {output_file}")
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()
        plt.close()


if __name__ == "__main__":
    print("Fetching and processing data...")
    df = fetch_and_process_data(symbol="MSFT", days=100)

    if df is not None:
        print("Creating enhanced trading dashboard...")
        create_comprehensive_dashboard(df)