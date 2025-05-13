"""
Train and Evaluate Initial Prediction Models - Control Point 4

This script tests the machine learning model implementation by:
1. Loading and preprocessing stock market data
2. Creating training, validation and test sets
3. Training different neural network architectures (LSTM, GRU, CNN)
4. Evaluating model performance with appropriate metrics
5. Creating visualizations of predictions vs. actual prices
6. Saving the trained models for future use
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition
from src.models.ml import PredictionModel

# Create output directory
os.makedirs('testing/control_point_4', exist_ok=True)

def fetch_and_process_data(symbol="MSFT", days=365):
    """Fetch and process market data for model training."""
    print(f"Fetching and processing data for {symbol}...")
    
    # Get data from APIs
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
        print("Failed to retrieve data. Please check your API keys and connection.")
        return None
    
    print(f"Successfully fetched {len(df)} days of data")
    
    # Check for NaN values in the raw data
    if df.isna().any().any():
        print("Raw data contains NaN values. Filling with forward/backward fill...")
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Process data
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_features(df)
    
    # Add technical indicators
    df_with_indicators = TechnicalIndicators.add_all_indicators(df_processed)
    
    # Add pattern recognition
    df_with_patterns = PatternRecognition.recognize_candlestick_patterns(df_with_indicators)
    df_with_patterns = PatternRecognition.detect_support_resistance(df_with_patterns)
    df_with_patterns = PatternRecognition.detect_trend(df_with_patterns)
    
    # Add symbol column if not present
    if 'symbol' not in df_with_patterns.columns:
        df_with_patterns['symbol'] = symbol
    
    # Check for any remaining NaN values after processing
    if df_with_patterns.isna().any().any():
        print("Processed data contains NaN values. Filling remaining NaNs...")
        df_with_patterns = df_with_patterns.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df_with_patterns

def train_lstm_model(df, output_dir):
    """Train and evaluate an LSTM model with improved stability."""
    print("\n=== Training LSTM Model ===")
    
    # Create LSTM model with more conservative parameters
    lstm_model = PredictionModel(
        model_type='lstm',
        model_params={
            'lstm_units': [64, 32],
            'dropout_rate': 0.3,  # Increased dropout for stability
            'learning_rate': 0.0005  # Lower learning rate for stability
        },
        model_dir=output_dir
    )
    
    # Configure data preparation
    sequence_length = 20
    target_column = 'close'
    
    # Select feature columns (use a mix of price, technical indicators, and patterns)
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
        'sma_20', 'ema_20', 'sma_50'
    ]
    
    # Filter to columns that actually exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Prepare data with more stability-focused options
    data_dict = lstm_model.prepare_data(
        df=df,
        sequence_length=sequence_length,
        target_column=target_column,
        feature_columns=feature_columns,
        target_horizon=1,  # Shorter horizon for stability
        train_size=0.7,
        val_size=0.15,
        scale_data=True,
        differencing=True  # Use differencing for better stationarity
    )
    
    if not data_dict:
        print("Failed to prepare data for LSTM model")
        return None
    
    # Extract prepared data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    print(f"Training LSTM model with {len(X_train)} sequences...")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    # Check for NaN values in data
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print("Warning: Training data contains NaN values. Replacing with zeros...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
    
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        print("Warning: Validation data contains NaN values. Replacing with zeros...")
        X_val = np.nan_to_num(X_val, nan=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0)
    
    if np.isnan(X_test).any() or np.isnan(y_test).any():
        print("Warning: Test data contains NaN values. Replacing with zeros...")
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)
    
    try:
        # Train model with more conservative parameters
        lstm_history = lstm_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=50,  # Fewer epochs to prevent overfitting
            batch_size=16,  # Smaller batch size for better stability
            patience=10,
            reduce_lr_factor=0.5,  # More gradual learning rate reduction
            reduce_lr_patience=3
        )
        
        # Evaluate model - handle potential NaN values
        print("\nEvaluating LSTM model...")
        try:
            metrics = lstm_model.evaluate(
                X_test=X_test,
                y_test=y_test,
                output_dir=os.path.join(output_dir, 'lstm_evaluation')
            )
        except ValueError as e:
            print(f"Evaluation error: {e}")
            print("Model produced NaN values during prediction. Using fallback metrics.")
            metrics = {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan'),
                'mape': float('nan'),
                'direction_accuracy': 0.0
            }
        
        # Save model with metadata
        try:
            lstm_model.save(
                model_name="lstm_price_predictor",
                metadata={"symbol": df['symbol'].iloc[0], "metrics": metrics}
            )
            print(f"LSTM model saved to {os.path.join(output_dir, 'lstm_price_predictor.h5')}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
        # Plot training history if available and not NaN
        try:
            if lstm_history and not all(np.isnan(lstm_history['loss'])):
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(lstm_history['loss'], label='Training Loss', color='blue')
                plt.plot(lstm_history['val_loss'], label='Validation Loss', color='orange')
                plt.title('LSTM Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.plot(lstm_history['mae'], label='Training MAE', color='blue')
                plt.plot(lstm_history['val_mae'], label='Validation MAE', color='orange')
                plt.title('LSTM Mean Absolute Error')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'lstm_training_history.png'), dpi=300)
                plt.close()
            else:
                print("Training produced NaN values, skipping history plotting")
        except Exception as e:
            print(f"Error plotting training history: {e}")
        
        return metrics
    except Exception as e:
        print(f"Error during LSTM model training: {e}")
        return None

def train_gru_model(df, output_dir):
    """Train and evaluate a GRU model with improved stability."""
    print("\n=== Training GRU Model ===")
    
    # Create GRU model with more conservative parameters
    gru_model = PredictionModel(
        model_type='gru',
        model_params={
            'gru_units': [64, 32],
            'dropout_rate': 0.3,  # Increased dropout for stability
            'learning_rate': 0.0005  # Lower learning rate for stability
        },
        model_dir=output_dir
    )
    
    # Configure data preparation
    sequence_length = 20
    target_column = 'close'
    
    # Select feature columns (use a mix of price, technical indicators, and patterns)
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
        'sma_20', 'ema_20', 'sma_50'
    ]
    
    # Filter to columns that actually exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Prepare data with more stability-focused options
    data_dict = gru_model.prepare_data(
        df=df,
        sequence_length=sequence_length,
        target_column=target_column,
        feature_columns=feature_columns,
        target_horizon=1,  # Shorter horizon for stability
        train_size=0.7,
        val_size=0.15,
        scale_data=True,
        differencing=True  # Use differencing for better stationarity
    )
    
    if not data_dict:
        print("Failed to prepare data for GRU model")
        return None
    
    # Extract prepared data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    print(f"Training GRU model with {len(X_train)} sequences...")
    
    # Check for NaN values in data
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print("Warning: Training data contains NaN values. Replacing with zeros...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
    
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        print("Warning: Validation data contains NaN values. Replacing with zeros...")
        X_val = np.nan_to_num(X_val, nan=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0)
    
    if np.isnan(X_test).any() or np.isnan(y_test).any():
        print("Warning: Test data contains NaN values. Replacing with zeros...")
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)
    
    try:
        # Train model with more conservative parameters
        gru_history = gru_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=50,  # Fewer epochs to prevent overfitting
            batch_size=16,  # Smaller batch size for better stability
            patience=10,
            reduce_lr_factor=0.5,  # More gradual learning rate reduction
            reduce_lr_patience=3
        )
        
        # Evaluate model - handle potential NaN values
        print("\nEvaluating GRU model...")
        try:
            metrics = gru_model.evaluate(
                X_test=X_test,
                y_test=y_test,
                output_dir=os.path.join(output_dir, 'gru_evaluation')
            )
        except ValueError as e:
            print(f"Evaluation error: {e}")
            print("Model produced NaN values during prediction. Using fallback metrics.")
            metrics = {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan'),
                'mape': float('nan'),
                'direction_accuracy': 0.0
            }
        
        # Save model with metadata
        try:
            gru_model.save(
                model_name="gru_price_predictor",
                metadata={"symbol": df['symbol'].iloc[0], "metrics": metrics}
            )
            print(f"GRU model saved to {os.path.join(output_dir, 'gru_price_predictor.h5')}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
        # Plot training history if available and not NaN
        try:
            if gru_history and not all(np.isnan(gru_history['loss'])):
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(gru_history['loss'], label='Training Loss', color='blue')
                plt.plot(gru_history['val_loss'], label='Validation Loss', color='orange')
                plt.title('GRU Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.plot(gru_history['mae'], label='Training MAE', color='blue')
                plt.plot(gru_history['val_mae'], label='Validation MAE', color='orange')
                plt.title('GRU Mean Absolute Error')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'gru_training_history.png'), dpi=300)
                plt.close()
            else:
                print("Training produced NaN values, skipping history plotting")
        except Exception as e:
            print(f"Error plotting training history: {e}")
        
        return metrics
    except Exception as e:
        print(f"Error during GRU model training: {e}")
        return None

def train_cnn_model(df, output_dir):
    """Train and evaluate a 1D CNN model with improved stability."""
    print("\n=== Training CNN Model ===")
    
    # Create CNN model with more conservative parameters
    cnn_model = PredictionModel(
        model_type='cnn',
        model_params={
            'filters': [32, 16, 8],  # Smaller filters for stability
            'kernel_sizes': [5, 3, 3],
            'pool_sizes': [2, 2, 2],
            'dropout_rate': 0.3,
            'learning_rate': 0.0005  # Lower learning rate for stability
        },
        model_dir=output_dir
    )
    
    # Configure data preparation
    sequence_length = 20
    target_column = 'close'
    
    # Select feature columns (use a mix of price, technical indicators, and patterns)
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
        'sma_20', 'ema_20', 'sma_50'
    ]
    
    # Filter to columns that actually exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Prepare data with more stability-focused options
    data_dict = cnn_model.prepare_data(
        df=df,
        sequence_length=sequence_length,
        target_column=target_column,
        feature_columns=feature_columns,
        target_horizon=1,  # Shorter horizon for stability
        train_size=0.7,
        val_size=0.15,
        scale_data=True,
        differencing=True  # Use differencing for better stationarity
    )
    
    if not data_dict:
        print("Failed to prepare data for CNN model")
        return None
    
    # Extract prepared data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    print(f"Training CNN model with {len(X_train)} sequences...")
    
    # Check for NaN values in data
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print("Warning: Training data contains NaN values. Replacing with zeros...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
    
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        print("Warning: Validation data contains NaN values. Replacing with zeros...")
        X_val = np.nan_to_num(X_val, nan=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0)
    
    if np.isnan(X_test).any() or np.isnan(y_test).any():
        print("Warning: Test data contains NaN values. Replacing with zeros...")
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)
    
    try:
        # Train model with more conservative parameters
        cnn_history = cnn_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=50,  # Fewer epochs to prevent overfitting
            batch_size=16,  # Smaller batch size for better stability
            patience=10,
            reduce_lr_factor=0.5,  # More gradual learning rate reduction
            reduce_lr_patience=3
        )
        
        # Evaluate model - handle potential NaN values
        print("\nEvaluating CNN model...")
        try:
            metrics = cnn_model.evaluate(
                X_test=X_test,
                y_test=y_test,
                output_dir=os.path.join(output_dir, 'cnn_evaluation')
            )
        except ValueError as e:
            print(f"Evaluation error: {e}")
            print("Model produced NaN values during prediction. Using fallback metrics.")
            metrics = {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan'),
                'mape': float('nan'),
                'direction_accuracy': 0.0
            }
        
        # Save model with metadata
        try:
            cnn_model.save(
                model_name="cnn_price_predictor",
                metadata={"symbol": df['symbol'].iloc[0], "metrics": metrics}
            )
            print(f"CNN model saved to {os.path.join(output_dir, 'cnn_price_predictor.h5')}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
        # Plot training history if available and not NaN
        try:
            if cnn_history and not all(np.isnan(cnn_history['loss'])):
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(cnn_history['loss'], label='Training Loss', color='blue')
                plt.plot(cnn_history['val_loss'], label='Validation Loss', color='orange')
                plt.title('CNN Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.plot(cnn_history['mae'], label='Training MAE', color='blue')
                plt.plot(cnn_history['val_mae'], label='Validation MAE', color='orange')
                plt.title('CNN Mean Absolute Error')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'cnn_training_history.png'), dpi=300)
                plt.close()
            else:
                print("Training produced NaN values, skipping history plotting")
        except Exception as e:
            print(f"Error plotting training history: {e}")
        
        return metrics
    except Exception as e:
        print(f"Error during CNN model training: {e}")
        return None

def compare_models(model_results, output_dir):
    """Create a comparison of model performance."""
    print("\n=== Comparing Model Performance ===")
    
    if not model_results:
        print("No model results to compare.")
        return None, None
    
    # Create a comparison dataframe
    comparison_data = []
    
    for model_name, metrics in model_results.items():
        # Skip any models with NaN metrics
        if all(np.isnan(list(value for value in metrics.values() if isinstance(value, (int, float))))):
            print(f"Skipping {model_name} in comparison due to NaN metrics")
            continue
            
        metric_row = {'model': model_name}
        metric_row.update(metrics)
        comparison_data.append(metric_row)
    
    if not comparison_data:
        print("No valid models to compare.")
        return None, None
        
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Model comparison saved to {comparison_path}")
    
    # Create a bar chart comparing key metrics
    metrics_to_plot = ['rmse', 'mae', 'direction_accuracy']
    
    plt.figure(figsize=(15, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i+1)
        
        # Sort by metric value
        sorted_df = comparison_df.sort_values(metric)
        
        # Determine bar colors (lower is better for error metrics, higher is better for direction accuracy)
        if metric == 'direction_accuracy':
            colors = ['#5cb85c', '#5bc0de', '#d9534f']  # green to red
        else:
            colors = ['#5cb85c', '#5bc0de', '#d9534f'][::-1]  # red to green
            
        # Create bar chart
        bars = plt.bar(sorted_df['model'], sorted_df[metric], color=colors[:len(sorted_df)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.4f}', ha='center', fontsize=9)
        
        # Add title and labels
        if metric == 'direction_accuracy':
            plt.title(f'Direction Accuracy (%) - Higher is better')
            plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Random')
            plt.legend()
        else:
            plt.title(f'{metric.upper()} - Lower is better')
            
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    print(f"Model comparison visualization saved to {os.path.join(output_dir, 'model_comparison.png')}")
    
    # Determine the best model based on direction accuracy
    try:
        best_model = comparison_df.loc[comparison_df['direction_accuracy'].idxmax(), 'model']
        print(f"Best model based on direction accuracy: {best_model}")
    except Exception as e:
        print(f"Could not determine best model: {e}")
        best_model = None
    
    return comparison_df, best_model

def main(symbol="MSFT"):
    """Main function to train and evaluate models."""
    print(f"=== Training Initial Prediction Models for {symbol} ===")
    output_dir = os.path.join('testing/control_point_4', symbol)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Fetch and process data
    df = fetch_and_process_data(symbol=symbol, days=500)
    
    if df is None or df.empty:
        print("Failed to fetch and process data. Exiting.")
        return
    
    # Step 2: Train each model type
    model_results = {}
    
    # Train LSTM model
    print("\nAttempting to train LSTM model...")
    try:
        lstm_metrics = train_lstm_model(df, output_dir)
        if lstm_metrics and not (isinstance(lstm_metrics, dict) and 
                               all(np.isnan(list(value for value in lstm_metrics.values() 
                                                if isinstance(value, (int, float)))))):
            model_results['LSTM'] = lstm_metrics
        else:
            print("LSTM model training produced invalid metrics, skipping this model")
    except Exception as e:
        print(f"LSTM model training failed: {e}")
    
    # Train GRU model
    print("\nAttempting to train GRU model...")
    try:
        gru_metrics = train_gru_model(df, output_dir)
        if gru_metrics and not (isinstance(gru_metrics, dict) and 
                              all(np.isnan(list(value for value in gru_metrics.values() 
                                               if isinstance(value, (int, float)))))):
            model_results['GRU'] = gru_metrics
        else:
            print("GRU model training produced invalid metrics, skipping this model")
    except Exception as e:
        print(f"GRU model training failed: {e}")
    
    # Train CNN model
    print("\nAttempting to train CNN model...")
    try:
        cnn_metrics = train_cnn_model(df, output_dir)
        if cnn_metrics and not (isinstance(cnn_metrics, dict) and 
                              all(np.isnan(list(value for value in cnn_metrics.values() 
                                               if isinstance(value, (int, float)))))):
            model_results['CNN'] = cnn_metrics
        else:
            print("CNN model training produced invalid metrics, skipping this model")
    except Exception as e:
        print(f"CNN model training failed: {e}")
    
    # Step 3: Compare model performance if we have valid results
    if model_results:
        comparison_df, best_model = compare_models(model_results, output_dir)
        
        if comparison_df is not None:
            # Display the comparison summary
            print("\n=== Model Comparison Summary ===")
            print(comparison_df)
            
            # Create a summary report file
            with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
                f.write(f"TRAINING SUMMARY FOR {symbol}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Data points: {len(df)}\n")
                f.write(f"Training period: {df.index[0]} to {df.index[-1]}\n\n")
                f.write("MODEL PERFORMANCE METRICS:\n")
                f.write(f"{comparison_df.to_string()}\n\n")
                
                if best_model:
                    f.write(f"Best model: {best_model}\n")
                    try:
                        f.write(f"Best direction accuracy: {comparison_df.loc[comparison_df['model'] == best_model, 'direction_accuracy'].values[0]:.2f}%\n")
                        f.write(f"Best RMSE: {comparison_df.loc[comparison_df['model'] == best_model, 'rmse'].values[0]:.4f}\n")
                    except Exception as e:
                        f.write(f"Error getting best model metrics: {e}\n")
                else:
                    f.write("Could not determine best model.\n")
            
            if best_model:
                print(f"Best model: {best_model}")
                print(f"Best direction accuracy: {comparison_df.loc[comparison_df['model'] == best_model, 'direction_accuracy'].values[0]:.2f}%")
                print(f"Complete results saved to {os.path.join(output_dir, 'training_summary.txt')}")
            else:
                print("Could not determine best model.")
        else:
            print("\nNo valid models were available for comparison.")
    else:
        print("\nNo models were successfully trained with valid metrics.")
        
    print(f"\nAll model training and evaluation completed for {symbol}.")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate initial prediction models')
    parser.add_argument('--symbol', type=str, default='MSFT', help='Stock symbol to analyze')
    
    args = parser.parse_args()
    main(args.symbol)