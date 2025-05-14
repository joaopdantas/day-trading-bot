"""
Test Refactored ML Module - Training Initial Models

This script tests the refactored ML module by training different neural network
architectures (LSTM, GRU, CNN) on stock market data and comparing their performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import from the refactored module structure
from src.models import PredictionModel
from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition

# Create output directory
os.makedirs('testing/refactored_models', exist_ok=True)

def fetch_and_process_data(symbol="MSFT", days=365):
    """Fetch and process market data for model training."""
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
        print("Failed to retrieve data. Check API keys and connection.")
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
        
        # Check for any remaining NaN values after processing
        if df.isna().any().any():
            print("Processed data contains NaN values. Filling remaining NaNs...")
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_model(df, model_type, output_dir):
    """Train and evaluate a model with the refactored structure."""
    print(f"\n=== Training {model_type.upper()} Model ===")
    
    # Model parameters based on type
    model_params = {
        'lstm': {
            'lstm_units': [64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.0005
        },
        'gru': {
            'gru_units': [64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.0005
        },
        'cnn': {
            'filters': [32, 16, 8],
            'kernel_sizes': [5, 3, 3],
            'pool_sizes': [2, 2, 2],
            'dropout_rate': 0.3,
            'learning_rate': 0.0005
        }
    }
    
    # Create model with appropriate parameters
    model = PredictionModel(
        model_type=model_type,
        model_params=model_params[model_type],
        model_dir=output_dir
    )
    
    # Configure data preparation
    sequence_length = 20
    target_column = 'close'
    
    # Select feature columns
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
        'sma_20', 'ema_20', 'sma_50'
    ]
    
    # Filter to columns that actually exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Prepare data
    data_dict = model.prepare_data(
        df=df,
        sequence_length=sequence_length,
        target_column=target_column,
        feature_columns=feature_columns,
        target_horizon=1,
        train_size=0.7,
        val_size=0.15,
        scale_data=True,
        differencing=True
    )
    
    if not data_dict:
        print(f"Failed to prepare data for {model_type} model")
        return None
    
    # Extract prepared data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    print(f"Training {model_type} model with {len(X_train)} sequences...")
    
    # Handle any NaN values
    for arr_name, arr in [('X_train', X_train), ('y_train', y_train), 
                          ('X_val', X_val), ('y_val', y_val),
                          ('X_test', X_test), ('y_test', y_test)]:
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            print(f"Warning: {arr_name} contains {nan_count} NaN values. Replacing with 0.")
            data_dict[arr_name] = np.nan_to_num(arr, nan=0.0)
    
    try:
        # Train model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=50,
            batch_size=16,
            patience=10,
            reduce_lr_factor=0.5,
            reduce_lr_patience=3
        )
        
        # Evaluate model
        print(f"\nEvaluating {model_type} model...")
        metrics = model.evaluate(
            X_test=X_test,
            y_test=y_test,
            output_dir=os.path.join(output_dir, f'{model_type}_evaluation')
        )
        
        # Save model
        model.save(model_name=f"{model_type}_price_predictor")
        print(f"{model_type.upper()} model saved to {output_dir}")
        
        return metrics
    except Exception as e:
        print(f"Error during {model_type} model training: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(results, output_dir):
    """Compare the performance of different models."""
    print("\n=== Comparing Model Performance ===")
    
    if not results:
        print("No model results to compare.")
        return None
    
    # Create a comparison dataframe
    comparison_data = []
    
    for model_name, metrics in results.items():
        if metrics is None:
            print(f"Skipping {model_name} in comparison due to failed training")
            continue
            
        # Create a clean metrics dictionary, handling NaN values
        clean_metrics = {}
        missing_metrics = []
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or value > 1e10):
                # For MAPE, use a high default
                if metric == 'mape':
                    clean_metrics[metric] = 1000.0
                    missing_metrics.append(metric)
                # For direction_accuracy, use 50% (random)
                elif metric == 'direction_accuracy':
                    clean_metrics[metric] = 50.0
                    missing_metrics.append(metric)
                # For r2, if extremely negative, cap it
                elif metric == 'r2' and value < -10:
                    clean_metrics[metric] = -10.0
                    missing_metrics.append(metric)
                else:
                    # Skip truly problematic metrics
                    missing_metrics.append(metric)
            else:
                clean_metrics[metric] = value
                
        if missing_metrics:
            print(f"Note: {model_name} has estimated values for: {', '.join(missing_metrics)}")
            
        # Only add to comparison if we have the minimum necessary metrics
        required_metrics = ['rmse', 'mae']
        if all(metric in clean_metrics for metric in required_metrics):
            metric_row = {'model': model_name}
            metric_row.update(clean_metrics)
            comparison_data.append(metric_row)
        else:
            print(f"Skipping {model_name} in comparison due to missing required metrics")
    
    if not comparison_data:
        print("No valid models to compare.")
        return None
        
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Model comparison saved to {comparison_path}")
    
    # Create visualization
    metrics_to_plot = ['rmse', 'mae', 'direction_accuracy']
    plot_comparison(comparison_df, metrics_to_plot, output_dir)
    
    # Determine best model
    try:
        best_model = comparison_df.loc[comparison_df['direction_accuracy'].idxmax(), 'model']
        best_accuracy = comparison_df.loc[comparison_df['direction_accuracy'].idxmax(), 'direction_accuracy']
        print(f"Best model based on direction accuracy: {best_model} ({best_accuracy:.2f}%)")
        return best_model
    except Exception as e:
        print(f"Could not determine best model: {e}")
        return None

def plot_comparison(df, metrics_to_plot, output_dir):
    """Create comparison visualizations."""
    plt.figure(figsize=(15, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i+1)
        
        # Sort by metric value (ascending for errors, descending for accuracy)
        if metric == 'direction_accuracy':
            sorted_df = df.sort_values(metric, ascending=False)
            colors = plt.cm.Greens(np.linspace(0.6, 0.3, len(sorted_df)))
            title = f'{metric.replace("_", " ").title()} (Higher is better)'
        else:
            sorted_df = df.sort_values(metric, ascending=True)
            colors = plt.cm.Reds(np.linspace(0.3, 0.6, len(sorted_df)))
            title = f'{metric.upper()} (Lower is better)'
            
        # Create bar chart
        bars = plt.bar(sorted_df['model'], sorted_df[metric], color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.4f}', ha='center', fontsize=9)
        
        plt.title(title)
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    print(f"Model comparison visualization saved to {os.path.join(output_dir, 'model_comparison.png')}")

def main(symbol="MSFT"):
    """Main function to train and evaluate models."""
    print(f"=== Testing Refactored ML Module with {symbol} Data ===")
    output_dir = os.path.join('testing/refactored_models', symbol)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Fetch and process data
    df = fetch_and_process_data(symbol=symbol, days=500)
    
    if df is None or df.empty:
        print("Failed to fetch and process data. Exiting.")
        return
    
    # Step 2: Train each model type
    model_results = {}
    
    for model_type in ['lstm', 'gru', 'cnn']:
        print(f"\nTesting {model_type.upper()} model...")
        try:
            metrics = train_model(df, model_type, output_dir)
            if metrics:
                model_results[model_type.upper()] = metrics
        except Exception as e:
            print(f"{model_type.upper()} model training failed: {e}")
    
    # Step 3: Compare models
    if model_results:
        best_model = compare_models(model_results, output_dir)
        
        # Create summary report
        with open(os.path.join(output_dir, 'testing_summary.txt'), 'w') as f:
            f.write(f"REFACTORED ML MODULE TESTING SUMMARY\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Tested with {symbol} data\n")
            f.write(f"Data points: {len(df)}\n")
            f.write(f"Period: {df.index[0]} to {df.index[-1]}\n\n")
            
            f.write("MODEL RESULTS:\n")
            for model, metrics in model_results.items():
                f.write(f"\n{model}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            if best_model:
                f.write(f"\nBest model: {best_model}\n")
        
        print(f"\nTesting completed successfully!")
        print(f"Results saved to {output_dir}")
    else:
        print("\nNo models were successfully trained.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test refactored ML module')
    parser.add_argument('--symbol', type=str, default='MSFT', help='Stock symbol to analyze')
    
    args = parser.parse_args()
    main(args.symbol)