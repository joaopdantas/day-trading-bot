import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from src.data.fetcher import get_data_api
from src.indicators.technical import SignalGeneration, TechnicalIndicators, PatternRecognition

def create_signal_visualization(symbol="MSFT", days=100):
    """Create visualization with real market data."""
    print(f"Fetching real-time market data for {symbol}...")
    
    # Get data from Yahoo Finance (more reliable for historical data)
    api = get_data_api("yahoo_finance")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch real market data
    df = api.fetch_historical_data(
        symbol=symbol,
        interval="1d",
        start_date=start_date,
        end_date=end_date
    )
    
    if df.empty:
        print("Failed to get Yahoo Finance data, trying Alpha Vantage...")
        api = get_data_api("alpha_vantage")
        df = api.fetch_historical_data(
            symbol=symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date
        )
    
    if df.empty:
        print("Failed to fetch market data. Please check your API keys.")
        return
    
    print(f"Successfully fetched {len(df)} days of market data")
    
    try:
        # Ensure the output directory exists
        output_dir = os.path.join('testing', 'control_point_3')
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare plotting data with uppercase columns for mplfinance
        plot_data = df.copy()
        plot_data = plot_data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        plot_data.index.name = 'Date'
        
        # Add technical indicators
        df_indicators = TechnicalIndicators.add_all_indicators(df)
        for col in ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 'macd_histogram']:
            if col in df_indicators.columns:
                plot_data[col] = df_indicators[col]
        
        # Custom style configuration
        mc = mpf.make_marketcolors(
            up='forestgreen',
            down='crimson',
            edge='inherit',
            wick='inherit',
            volume={'up': 'forestgreen', 'down': 'crimson'},
            ohlc='i'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle=':',
            y_on_right=False,
            gridcolor='gray',
            figcolor='white',
            facecolor='white',
            edgecolor='black'
        )

        # Additional plot overlays
        apds = []
        
        # Add moving averages
        if 'sma_20' in plot_data.columns:
            apds.append(mpf.make_addplot(plot_data['sma_20'], color='blue', width=0.8, label='SMA 20'))
        if 'sma_50' in plot_data.columns:
            apds.append(mpf.make_addplot(plot_data['sma_50'], color='orange', width=0.8, label='SMA 50'))
        
        # Add RSI
        if 'rsi' in plot_data.columns:
            apds.append(mpf.make_addplot(plot_data['rsi'], panel=2, color='purple', ylabel='RSI'))
        
        # Add MACD
        if all(x in plot_data.columns for x in ['macd', 'macd_signal', 'macd_histogram']):
            apds.extend([
                mpf.make_addplot(plot_data['macd'], panel=3, color='blue', ylabel='MACD'),
                mpf.make_addplot(plot_data['macd_signal'], panel=3, color='orange'),
                mpf.make_addplot(plot_data['macd_histogram'], type='bar', panel=3, color='gray', alpha=0.3)
            ])
        
        # Create a new figure with specific dimensions
        plt.close('all')  # Close any existing figures
        fig = plt.figure(figsize=(15, 12), dpi=100)
        fig.patch.set_facecolor('white')
        
        # Create the visualization with adjusted panel ratios
        fig, axes = mpf.plot(
            plot_data,
            type='candle',
            style=s,
            volume=True,
            addplot=apds,
            panel_ratios=(4, 1, 1, 1),
            figsize=(15, 12),
            title=f'\n{symbol} Technical Analysis Dashboard',
            returnfig=True,
            warn_too_much_data=10000  # Increase the data warning threshold
        )
        
        # Add panel titles with adjusted spacing
        axes[0].set_title('Candlestick Chart with Moving Averages', fontsize=12, y=1.0)
        axes[1].set_title('Trading Volume', fontsize=12, y=1.0)
        
        # Add RSI lines and title if available
        if len(axes) > 2:
            axes[2].set_title('RSI (Relative Strength Index)', fontsize=12, y=1.0)
            axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
            axes[2].text(plot_data.index[0], 70, 'Overbought (70)', va='bottom', fontsize=10)
            axes[2].text(plot_data.index[0], 30, 'Oversold (30)', va='top', fontsize=10)
        
        # Add MACD title if available
        if len(axes) > 3:
            axes[3].set_title('MACD (Moving Average Convergence Divergence)', fontsize=12, y=1.0)
        
        # Add explanatory text with adjusted position
        explanation = (
            "INDICATORS EXPLANATION:\n"
            "• Green candles: Price closed higher than open\n"
            "• Red candles: Price closed lower than open\n"
            "• Blue line: 20-day moving average (short-term trend)\n"
            "• Orange line: 50-day moving average (long-term trend)\n"
            "• RSI > 70: Potentially overbought\n"
            "• RSI < 30: Potentially oversold\n"
            "• MACD: Blue line crossing above orange = bullish signal"
        )
        
        # Create a text box for explanation with proper spacing
        plt.gcf().text(0.02, 0.02, explanation, fontsize=10,
                       bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                       verticalalignment='bottom')
        
        # Add data source and timestamp with adjusted position
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.gcf().text(0.98, 0.02, f"Generated: {timestamp}\nData Source: Alpha Vantage",
                       fontsize=8, ha='right',
                       bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                       verticalalignment='bottom')
        
        # Fine-tune the layout
        plt.subplots_adjust(
            left=0.1,    # Left margin
            right=0.9,   # Right margin
            top=0.95,    # Top margin
            bottom=0.15, # Bottom margin to accommodate explanation
            hspace=0.5   # Vertical space between subplots
        )
        
        # Save with high quality
        output_path = os.path.join(output_dir, 'signal_visualization.png')
        plt.savefig(output_path,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.3,
                    format='png',
                    transparent=False)
        
        print(f"Visualization saved successfully to: {output_path}")
        plt.close('all')  # Ensure all figures are closed
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        plt.close('all')  # Clean up in case of error

if __name__ == '__main__':
    create_signal_visualization("AAPL", days=100)  # Using AAPL to avoid rate limiting