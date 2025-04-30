"""
Test script for the enhanced DatasetPreparation class.

This script tests the improved buy/sell signal generation logic in the DatasetPreparation class,
evaluates its performance using real market data, and visualizes the results.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition, DatasetPreparation
from src.indicators.technical import SignalGeneration  # For comparison against standard method

# Create output directory
os.makedirs('testing/control_point_3', exist_ok=True)

def fetch_and_process_data(symbol="MSFT", days=365):
    """Fetch and process market data with technical indicators."""
    print(f"Fetching and processing data for {symbol}...")
    
    # Get data from API
    api = get_data_api("alpha_vantage")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch historical data
    df = api.fetch_historical_data(
        symbol=symbol,
        interval="1d",
        start_date=start_date,
        end_date=end_date
    )
    
    if df.empty:
        print("Failed to retrieve data from Alpha Vantage. Trying Yahoo Finance...")
        api = get_data_api("yahoo_finance")
        df = api.fetch_historical_data(
            symbol=symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date
        )
    
    if df.empty:
        print("Failed to retrieve data. Please check your API keys and connection.")
        return None
    
    print(f"Successfully fetched {len(df)} days of data")
    
    # Process data
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_features(df)
    
    # Add technical indicators
    df_indicators = TechnicalIndicators.add_all_indicators(df_processed)
    
    # Add pattern recognition
    df_with_patterns = PatternRecognition.recognize_candlestick_patterns(df_indicators)
    df_with_patterns = PatternRecognition.detect_support_resistance(df_with_patterns)
    df_with_patterns = PatternRecognition.detect_trend(df_with_patterns)
    
    return df_with_patterns


def evaluate_signal_quality(actual_returns, signals, name="Signal Evaluation"):
    """
    Evaluate the quality of trading signals based on actual future returns.
    
    Args:
        actual_returns: Series of actual future returns
        signals: Series of trading signals (1 for buy, -1 for sell, 0 for hold)
        name: Name of the signal set for display purposes
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Create binary classification for evaluation
    y_true = (actual_returns > 0).astype(int)
    y_pred = (signals > 0).astype(int)
    
    # Filter out hold signals (0) for evaluation
    mask = signals != 0
    if mask.sum() == 0:
        print(f"No active signals found in {name}")
        return {}
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Calculate metrics
    try:
        acc = accuracy_score(y_true_filtered, y_pred_filtered)
        prec = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
        rec = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
        f1 = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)
        
        # Calculate average return for signals
        avg_return_all = actual_returns.mean()
        avg_return_signals = actual_returns[mask].mean()
        
        # Calculate win rate (percentage of profitable trades)
        win_rate = (actual_returns[signals == 1] > 0).mean()
        loss_rate = (actual_returns[signals == -1] < 0).mean()
        
        print(f"\n{name}:")
        print(f"Number of signals: {mask.sum()} out of {len(signals)} days ({mask.sum()/len(signals)*100:.1f}%)")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average return (all days): {avg_return_all:.4f}")
        print(f"Average return (signal days): {avg_return_signals:.4f}")
        print(f"Win rate (buy signals): {win_rate:.4f}")
        print(f"Win rate (sell signals): {loss_rate:.4f}")
        
        return {
            "name": name,
            "signal_count": mask.sum(),
            "signal_percentage": mask.sum()/len(signals)*100,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "avg_return_all": avg_return_all,
            "avg_return_signals": avg_return_signals,
            "win_rate_buy": win_rate,
            "win_rate_sell": loss_rate
        }
    except Exception as e:
        print(f"Error evaluating signals: {e}")
        return {}


def create_trading_simulation(df):
    """
    Create a trading simulation to compare signal-based trading with buy & hold.
    
    Args:
        df: DataFrame with market data and signals
        
    Returns:
        DataFrame with cumulative returns
    """
    print("\n=== Creating Trading Simulation ===")
    try:
        # Check if we have required columns
        required_cols = ['close', 'target_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing required columns for trading simulation: {missing_cols}")
            return df
        
        # Create a copy for simulation
        sim_df = df.copy()
        
        # Initialize portfolio values (start with $10,000 for each strategy)
        initial_capital = 10000
        sim_df['buy_hold_value'] = initial_capital
        sim_df['signal_strategy_value'] = initial_capital
        
        # Buy and hold strategy (invest everything at the beginning)
        start_price = sim_df['close'].iloc[0]
        shares = initial_capital / start_price
        sim_df['buy_hold_value'] = shares * sim_df['close']
        
        # Signal-based trading strategy
        cash = initial_capital
        shares = 0
        position = 0  # 0 = cash, 1 = long
        
        # Run simulation
        for i, row in sim_df.iterrows():
            # Execute trades based on signals
            if row['target_label'] == 1 and position == 0:  # Buy signal and in cash
                shares = cash / row['close']
                cash = 0
                position = 1
            elif row['target_label'] == -1 and position == 1:  # Sell signal and in position
                cash = shares * row['close']
                shares = 0
                position = 0
            
            # Update portfolio value
            sim_df.loc[i, 'signal_strategy_value'] = cash + (shares * row['close'])
        
        # Calculate returns and statistics
        sim_df['buy_hold_return'] = sim_df['buy_hold_value'] / initial_capital - 1
        sim_df['signal_strategy_return'] = sim_df['signal_strategy_value'] / initial_capital - 1
        
        # Calculate drawdowns
        for col in ['buy_hold_value', 'signal_strategy_value']:
            # Calculate running maximum
            sim_df[f'{col}_max'] = sim_df[col].cummax()
            # Calculate drawdown percentage
            sim_df[f'{col}_drawdown'] = (sim_df[f'{col}_max'] - sim_df[col]) / sim_df[f'{col}_max'] * 100
        
        # Print performance metrics
        final_day = sim_df.iloc[-1]
        print("\nTrading Simulation Results:")
        print(f"Simulation Period: {sim_df.index[0]} to {sim_df.index[-1]}")
        print(f"Buy & Hold Return: {final_day['buy_hold_return']*100:.2f}%")
        print(f"Signal Strategy Return: {final_day['signal_strategy_return']*100:.2f}%")
        
        # Calculate max drawdowns
        max_drawdowns = {
            'Buy & Hold': sim_df['buy_hold_value_drawdown'].max(),
            'Signal Strategy': sim_df['signal_strategy_value_drawdown'].max()
        }
        
        print("\nMaximum Drawdowns:")
        for method, drawdown in max_drawdowns.items():
            print(f"{method}: {drawdown:.2f}%")
        
        # Create performance visualization
        plt.figure(figsize=(14, 10))
        
        # Plot portfolio values
        plt.subplot(2, 1, 1)
        plt.plot(sim_df.index, sim_df['buy_hold_value'], label='Buy & Hold', color='gray')
        plt.plot(sim_df.index, sim_df['signal_strategy_value'], label='Signal Strategy', color='green')
        plt.title('Portfolio Value Comparison')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        plt.plot(sim_df.index, sim_df['buy_hold_value_drawdown'], label='Buy & Hold', color='gray')
        plt.plot(sim_df.index, sim_df['signal_strategy_value_drawdown'], label='Signal Strategy', color='green')
        plt.title('Portfolio Drawdowns')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('testing/control_point_3/trading_simulation.png', dpi=300)
        plt.close()
        
        print(f"Trading simulation visualization saved to testing/control_point_3/trading_simulation.png")
        
        return sim_df
        
    except Exception as e:
        print(f"Error in trading simulation: {e}")
        return df


def test_signal_generation(df):
    """
    Test the signal generation method and evaluate its performance.
    
    Args:
        df: DataFrame with market data and technical indicators
        
    Returns:
        DataFrame with signals and evaluation metrics
    """
    print("\n=== Testing Signal Generation ===")
    try:
        # Create a copy for testing
        result_df = df.copy()
        
        # Generate signals using the target labels method
        signal_results = DatasetPreparation.create_target_labels(df)
        
        if not signal_results.empty:
            # Add signals to result DataFrame
            for col in signal_results.columns:
                result_df[col] = signal_results[col]
        else:
            print("Failed to generate signals")
            return df
        
        # Count signals
        buy_count = (result_df['target_label'] == 1).sum()
        sell_count = (result_df['target_label'] == -1).sum()
        
        print("\nSignal Statistics:")
        print(f"Buy signals: {buy_count}")
        print(f"Sell signals: {sell_count}")
        print(f"Total signals: {buy_count + sell_count} out of {len(df)} days ({(buy_count + sell_count)/len(df)*100:.1f}%)")
        
        # Evaluate signal quality using future returns
        if 'weighted_future_return' in result_df.columns:
            metrics = evaluate_signal_quality(
                result_df['weighted_future_return'], 
                result_df['target_label'],
                "Signal Quality Evaluation"
            )
        
        # Visualize signals on price chart
        plt.figure(figsize=(14, 8))
        
        # Plot price
        plt.plot(result_df.index, result_df['close'], color='blue', linewidth=1.5)
        
        # Plot buy signals
        buy_indices = result_df[result_df['target_label'] == 1].index
        plt.scatter(buy_indices, result_df.loc[buy_indices, 'close'], 
                   marker='^', color='green', s=120, label='Buy Signal')
        
        # Plot sell signals
        sell_indices = result_df[result_df['target_label'] == -1].index
        plt.scatter(sell_indices, result_df.loc[sell_indices, 'close'],
                   marker='v', color='red', s=120, label='Sell Signal')
        
        # Add probability information if available
        if 'signal_probability' in result_df.columns:
            # Add a text annotation for high-probability signals
            for idx in buy_indices.union(sell_indices):
                prob = result_df.loc[idx, 'signal_probability']
                if prob > 0.8:  # Only annotate high-probability signals
                    plt.annotate(f"{prob:.2f}", 
                               xy=(idx, result_df.loc[idx, 'close']),
                               xytext=(0, 10 if result_df.loc[idx, 'target_label'] == 1 else -10),
                               textcoords="offset points",
                               ha='center',
                               fontsize=8)
        
        plt.title('Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('testing/control_point_3/signal_visualization.png', dpi=300)
        plt.close()
        
        print(f"Signal visualization saved to testing/control_point_3/signal_visualization.png")

        # Run trading simulation
        simulation_df = create_trading_simulation(result_df)

        return result_df

    except Exception as e:
        print(f"Error testing signal generation: {e}")
        return df


def main():
    """Main function to run the test script."""
    print("=== Testing DatasetPreparation Class ===")
    
    # Get stock symbol from command line argument if provided
    symbol = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    
    print(f"Testing with {symbol} data for the past {days} days...")
    
    # Fetch and process data
    df = fetch_and_process_data(symbol, days)
    
    if df is None or df.empty:
        print("Failed to fetch and process data. Exiting.")
        sys.exit(1)
    
    # Test signal generation
    result_df = test_signal_generation(df)
    
    print("\n=== Test Completed ===")
    print(f"Results saved to testing/control_point_3/ directory")


if __name__ == "__main__":
    main()