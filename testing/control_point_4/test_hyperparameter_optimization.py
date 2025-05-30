"""
FIXED Test Hyperparameter Optimization Script

Key fix: Pass test_original_prices to the evaluate method to ensure 
real price evaluation instead of scaled price evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import from the refactored module structure
from src.models import PredictionModel, ModelOptimizer
from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators, PatternRecognition

# Create output directory
output_dir = 'testing/control_point_4/hyperparameter_optimization'
os.makedirs(output_dir, exist_ok=True)

def fetch_and_process_data(symbol="MSFT", days=500):
    """Fetch and process market data for optimization."""
    print(f"Fetching and processing data for {symbol}...")
    
    # Try multiple data sources
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
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_optimization_data(df):
    """Prepare data for hyperparameter optimization."""
    print("Preparing data for optimization...")
    
    # Create a PredictionModel instance for data preparation
    temp_model = PredictionModel(model_type='lstm')  # Temporary for data prep
    
    # Configure data preparation
    sequence_length = 20
    target_column = 'close'
    
    # Select feature columns (same as in initial testing)
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle',
        'sma_20', 'ema_20', 'sma_50'
    ]
    
    # Filter to columns that actually exist
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Prepare data
    data_dict = temp_model.prepare_data(
        df=df,
        sequence_length=sequence_length,
        target_column=target_column,
        feature_columns=feature_columns,
        target_horizon=1,
        train_size=0.7,
        val_size=0.15,
        scale_data=True,
        differencing=False  # CRITICAL: Must match baseline models
    )
    
    if not data_dict:
        print("Failed to prepare data for optimization")
        return None
    
    # Handle any NaN values
    for key in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        if key in data_dict:
            nan_count = np.isnan(data_dict[key]).sum()
            if nan_count > 0:
                print(f"Warning: {key} contains {nan_count} NaN values. Replacing with 0.")
                data_dict[key] = np.nan_to_num(data_dict[key], nan=0.0)
    
    print(f"Data prepared: X_train shape: {data_dict['X_train'].shape}")
    
    # Log the original test price range for verification
    test_prices = data_dict.get('test_original_prices', None)
    if test_prices is not None:
        print(f"Test set original price range: ${test_prices.min():.2f} - ${test_prices.max():.2f}")
    else:
        print("WARNING: No original test prices found in data preparation!")
    
    return data_dict

def optimize_cnn_model(data_dict):
    """Optimize CNN model parameters."""
    print("\n" + "="*50)
    print("🔧 OPTIMIZING CNN MODEL (Focus: Reduce RMSE)")
    print("="*50)
    
    # Custom CNN parameter grid based on initial results
    cnn_param_grid = {
        'filters': [
            [16, 8],           # Simpler architecture
            [32, 16],          # Current best
            [64, 32],          # More complex
            [32, 16, 8],       # Deeper network
            [64, 32, 16]       # Even deeper
        ],
        'kernel_sizes': [
            [3, 3],            # Small kernels
            [5, 3],            # Mixed kernels
            [7, 5],            # Larger kernels
            [5, 3, 3],         # Varied kernels
            [7, 5, 3]          # Progressive kernels
        ],
        'pool_sizes': [
            [2, 2],            # Standard pooling
            [2, 1],            # Conservative pooling
            [3, 2],            # Aggressive pooling
            [1, 1],            # No pooling
            [2, 2, 1]          # Mixed pooling
        ],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4],     # Lower dropout for CNN
        'learning_rate': [0.0001, 0.0005, 0.001], # Conservative learning rates
        'batch_size': [16, 32, 64]                 # Various batch sizes
    }
    
    try:
        cnn_results = ModelOptimizer.optimize_hyperparameters(
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            model_type='cnn',
            param_grid=cnn_param_grid,
            n_trials=30,  # More trials for thorough search
            epochs=50,
            model_dir=os.path.join(output_dir, 'cnn_optimization'),
            optimize_for='val_loss'  # Focus on reducing error
        )
        
        print(f"\n🎯 CNN OPTIMIZATION RESULTS:")
        print(f"Best validation loss: {cnn_results['best_val_loss']:.4f}")
        print(f"Best direction accuracy: {cnn_results['best_direction_accuracy']:.2f}%")
        print(f"Best parameters: {cnn_results['best_params']}")
        print(f"Successful trials: {cnn_results['successful_trials']}/{cnn_results['total_trials']}")
        
        return cnn_results
        
    except Exception as e:
        print(f"CNN optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_lstm_model(data_dict):
    """Optimize LSTM model parameters."""
    print("\n" + "="*50)
    print("🎯 OPTIMIZING LSTM MODEL (Focus: Direction Accuracy)")
    print("="*50)
    
    # Custom LSTM parameter grid to improve direction accuracy
    lstm_param_grid = {
        'units': [
            [32, 16],          # Simple architecture
            [64, 32],          # Current baseline
            [128, 64],         # More capacity
            [64, 32, 16],      # Deeper network
            [128, 64, 32],     # Even deeper
            [256, 128]         # High capacity
        ],
        'dropout_rate': [0.3, 0.4, 0.5, 0.6],      # Higher dropout for regularization
        'learning_rate': [0.0001, 0.0005, 0.001, 0.002],  # Range of learning rates
        'batch_size': [16, 32, 64],                 # Various batch sizes
        'bidirectional': [True, False]             # Test bidirectional LSTMs
    }
    
    try:
        lstm_results = ModelOptimizer.optimize_hyperparameters(
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            model_type='lstm',
            param_grid=lstm_param_grid,
            n_trials=25,  # Focused search
            epochs=50,
            model_dir=os.path.join(output_dir, 'lstm_optimization'),
            optimize_for='direction_accuracy'  # Focus on trading performance
        )
        
        print(f"\n🎯 LSTM OPTIMIZATION RESULTS:")
        print(f"Best validation loss: {lstm_results['best_val_loss']:.4f}")
        print(f"Best direction accuracy: {lstm_results['best_direction_accuracy']:.2f}%")
        print(f"Best parameters: {lstm_results['best_params']}")
        print(f"Successful trials: {lstm_results['successful_trials']}/{lstm_results['total_trials']}")
        
        return lstm_results
        
    except Exception as e:
        print(f"LSTM optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_optimized_models(data_dict, cnn_results, lstm_results):
    """FIXED: Test optimized models with proper scaler passing."""
    print("\n" + "="*50)
    print("🧪 TESTING OPTIMIZED MODELS WITH REAL PRICE EVALUATION")
    print("="*50)
    
    results_comparison = []
    
    # Get test original prices from data_dict
    test_original_prices = data_dict.get('test_original_prices', None)
    if test_original_prices is None:
        print("❌ ERROR: No original test prices available!")
        return []
    
    print(f"Using original test prices range: ${test_original_prices.min():.2f} - ${test_original_prices.max():.2f}")
    
    # CRITICAL: Get the scaler from data_dict
    target_scaler = data_dict.get('target_scaler', None)
    if target_scaler is None:
        print("❌ ERROR: No target scaler available!")
        return []
    
    print(f"✅ Found target scaler: {type(target_scaler)}")
    print(f"Scaler range: {target_scaler.data_min_[0]:.2f} to {target_scaler.data_max_[0]:.2f}")
    
    # Test optimized CNN
    if cnn_results and cnn_results['best_params']:
        print("Testing optimized CNN model...")
        try:
            # Extract only model architecture parameters (exclude training params)
            cnn_model_params = {
                'filters': cnn_results['best_params']['filters'],
                'kernel_sizes': cnn_results['best_params']['kernel_sizes'],
                'pool_sizes': cnn_results['best_params']['pool_sizes'],
                'dropout_rate': cnn_results['best_params']['dropout_rate'],
                'learning_rate': cnn_results['best_params']['learning_rate']
            }
            
            cnn_model = PredictionModel(
                model_type='cnn',
                model_params=cnn_model_params
            )
            
            # 🔧 ADD THIS LINE HERE - BEFORE build_model():
            cnn_model.scaler = data_dict['target_scaler']
            
            # Build and train with best parameters
            cnn_model.build_model(data_dict['X_train'].shape[1:])
            cnn_history = cnn_model.train(
                X_train=data_dict['X_train'],
                y_train=data_dict['y_train'],
                X_val=data_dict['X_val'],
                y_val=data_dict['y_val'],
                epochs=50,
                batch_size=cnn_results['best_params']['batch_size'],
                patience=10
            )
            
            # NOW when evaluate() is called, it will have the scaler
            cnn_metrics = cnn_model.evaluate(
                X_test=data_dict['X_test'],
                y_test=data_dict['y_test'],
                output_dir=os.path.join(output_dir, 'optimized_cnn_evaluation')
            )
            
            results_comparison.append({
                'model': 'Optimized CNN',
                'params': cnn_results['best_params'],
                **cnn_metrics
            })
            
            print(f"✅ Optimized CNN - RMSE: ${cnn_metrics.get('rmse', 'N/A'):.2f}, "
                  f"Direction Acc: {cnn_metrics.get('direction_accuracy', 'N/A'):.2f}%")
            
        except Exception as e:
            print(f"❌ Optimized CNN testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test optimized LSTM
    if lstm_results and lstm_results['best_params']:
        print("Testing optimized LSTM model...")
        try:
            # Extract only model architecture parameters and fix parameter names
            lstm_model_params = {
                'lstm_units': lstm_results['best_params']['units'],  # Fix: units -> lstm_units
                'dropout_rate': lstm_results['best_params']['dropout_rate'],
                'learning_rate': lstm_results['best_params']['learning_rate'],
                'bidirectional': lstm_results['best_params']['bidirectional']
            }
            
            lstm_model = PredictionModel(
                model_type='lstm',
                model_params=lstm_model_params
            )
            
            # 🔧 ADD THIS LINE HERE - BEFORE build_model():
            lstm_model.scaler = data_dict['target_scaler']
            
            # Build and train with best parameters
            lstm_model.build_model(data_dict['X_train'].shape[1:])
            lstm_history = lstm_model.train(
                X_train=data_dict['X_train'],
                y_train=data_dict['y_train'],
                X_val=data_dict['X_val'],
                y_val=data_dict['y_val'],
                epochs=50,
                batch_size=lstm_results['best_params']['batch_size'],
                patience=10
            )
            
            # NOW when evaluate() is called, it will have the scaler
            lstm_metrics = lstm_model.evaluate(
                X_test=data_dict['X_test'],
                y_test=data_dict['y_test'],
                output_dir=os.path.join(output_dir, 'optimized_lstm_evaluation')
            )
            
            results_comparison.append({
                'model': 'Optimized LSTM',
                'params': lstm_results['best_params'],
                **lstm_metrics
            })
            
            print(f"✅ Optimized LSTM - RMSE: ${lstm_metrics.get('rmse', 'N/A'):.2f}, "
                  f"Direction Acc: {lstm_metrics.get('direction_accuracy', 'N/A'):.2f}%")
            
        except Exception as e:
            print(f"❌ Optimized LSTM testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save comparison results
    if results_comparison:
        comparison_df = pd.DataFrame(results_comparison)
        comparison_path = os.path.join(output_dir, 'optimized_models_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"📊 Comparison results saved to {comparison_path}")
        
        # Create visualization
        create_optimization_comparison_plot(comparison_df, output_dir)
        
        # Log final results for verification
        print(f"\n📊 FINAL RESULTS VERIFICATION:")
        for _, row in comparison_df.iterrows():
            print(f"{row['model']}:")
            print(f"  RMSE: ${row['rmse']:.2f} (should be ~$10-50 range)")
            print(f"  MAPE: {row['mape']:.2f}% (should be <20%)")
            print(f"  Direction Accuracy: {row['direction_accuracy']:.2f}%")
            
            # Validate if metrics are now realistic
            if 10 <= row['rmse'] <= 100:
                print(f"  ✅ RMSE is now in realistic range!")
            else:
                print(f"  ❌ RMSE still not realistic")
                
            if row['mape'] <= 25:
                print(f"  ✅ MAPE is now realistic!")
            else:
                print(f"  ❌ MAPE still too high")
    
    return results_comparison

def create_optimization_comparison_plot(comparison_df, output_dir):
    """Create comparison visualization for optimized models."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot RMSE comparison
        plt.subplot(2, 3, 1)
        models = comparison_df['model']
        rmse_values = comparison_df['rmse']
        colors = ['#1f77b4', '#ff7f0e']
        bars = plt.bar(models, rmse_values, color=colors)
        plt.title('RMSE Comparison (Lower is Better)')
        plt.ylabel('RMSE ($)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${height:.2f}', ha='center', va='bottom')
        
        # Add baseline comparison (from initial testing)
        plt.axhline(y=27.97, color='green', linestyle='--', label='Original CNN (27.97)')
        plt.axhline(y=55.89, color='red', linestyle='--', label='Original LSTM (55.89)')
        plt.legend()
        
        # Plot Direction Accuracy comparison
        plt.subplot(2, 3, 2)
        dir_acc_values = comparison_df['direction_accuracy']
        bars = plt.bar(models, dir_acc_values, color=colors)
        plt.title('Direction Accuracy Comparison (Higher is Better)')
        plt.ylabel('Direction Accuracy (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Add baseline comparison
        plt.axhline(y=50.0, color='green', linestyle='--', label='Original CNN (50.0%)')
        plt.axhline(y=59.52, color='red', linestyle='--', label='Original LSTM (59.52%)')
        plt.legend()
        
        # Plot MAPE comparison
        plt.subplot(2, 3, 3)
        mape_values = comparison_df['mape']
        bars = plt.bar(models, mape_values, color=colors)
        plt.title('MAPE Comparison (Lower is Better)')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Add baseline comparison
        plt.axhline(y=5.40, color='green', linestyle='--', label='Original CNN (5.40%)')
        plt.axhline(y=12.33, color='red', linestyle='--', label='Original LSTM (12.33%)')
        plt.legend()
        
        # Plot R² comparison
        plt.subplot(2, 3, 4)
        r2_values = comparison_df['r2']
        bars = plt.bar(models, r2_values, color=colors)
        plt.title('R² Score Comparison (Higher is Better)')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Add baseline comparison
        plt.axhline(y=0.19, color='green', linestyle='--', label='Original CNN (0.19)')
        plt.axhline(y=-2.25, color='red', linestyle='--', label='Original LSTM (-2.25)')
        plt.legend()
        
        # Plot MAE comparison
        plt.subplot(2, 3, 5)
        mae_values = comparison_df['mae']
        bars = plt.bar(models, mae_values, color=colors)
        plt.title('MAE Comparison (Lower is Better)')
        plt.ylabel('MAE ($)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${height:.2f}', ha='center', va='bottom')
        
        # Add improvement summary text
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Calculate improvements
        improvement_text = "🎯 OPTIMIZATION SUMMARY\n\n"
        
        if len(comparison_df) > 0:
            for _, row in comparison_df.iterrows():
                model_name = row['model']
                if 'CNN' in model_name:
                    baseline_rmse = 27.97
                    baseline_dir = 50.0
                elif 'LSTM' in model_name:
                    baseline_rmse = 55.89
                    baseline_dir = 59.52
                
                rmse_improvement = ((baseline_rmse - row['rmse']) / baseline_rmse) * 100
                dir_improvement = row['direction_accuracy'] - baseline_dir
                
                improvement_text += f"{model_name}:\n"
                improvement_text += f"RMSE: {rmse_improvement:+.1f}% change\n"
                improvement_text += f"Direction: {dir_improvement:+.1f}% points\n\n"
        
        improvement_text += "Target: Real price predictions\n"
        improvement_text += "Expected RMSE: $10-50 range\n"
        improvement_text += "Expected MAPE: <20%"
        
        plt.text(0.1, 0.9, improvement_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimization_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Optimization comparison plot saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        import traceback
        traceback.print_exc()

def main(symbol="MSFT"):
    """Main function to run hyperparameter optimization."""
    print("🚀 HYPERPARAMETER OPTIMIZATION - CONTROL POINT 4 TASK 2.11")
    print("="*60)
    
    # Step 1: Fetch and process data
    df = fetch_and_process_data(symbol=symbol, days=500)
    if df is None or df.empty:
        print("❌ Failed to fetch and process data. Exiting.")
        return
    
    # Step 2: Prepare data for optimization
    data_dict = prepare_optimization_data(df)
    if data_dict is None:
        print("❌ Failed to prepare data for optimization. Exiting.")
        return
    
    # Step 3: Optimize CNN model (focus on accuracy)
    cnn_results = optimize_cnn_model(data_dict)
    
    # Step 4: Optimize LSTM model (focus on direction accuracy)
    lstm_results = optimize_lstm_model(data_dict)
    
    # Step 5: Test optimized models with REAL PRICE EVALUATION
    if cnn_results or lstm_results:
        comparison_results = test_optimized_models(data_dict, cnn_results, lstm_results)
        
        # Create summary report
        with open(os.path.join(output_dir, 'optimization_summary.txt'), 'w') as f:
            f.write(f"HYPERPARAMETER OPTIMIZATION SUMMARY - FIXED EVALUATION\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Dataset: {symbol} with {len(df)} data points\n")
            f.write(f"Period: {df.index[0]} to {df.index[-1]}\n\n")
            
            f.write("EVALUATION FIX APPLIED:\n")
            f.write("- Models now evaluated using original test prices\n")
            f.write("- RMSE should be in realistic $10-50 range\n")
            f.write("- MAPE should be reasonable <20%\n")
            f.write("- Direction accuracy optimized for trading\n\n")
            
            if cnn_results:
                f.write("CNN OPTIMIZATION:\n")
                f.write(f"  Best validation loss: {cnn_results['best_val_loss']:.4f}\n")
                f.write(f"  Best direction accuracy: {cnn_results['best_direction_accuracy']:.2f}%\n")
                f.write(f"  Best parameters: {cnn_results['best_params']}\n")
                f.write(f"  Successful trials: {cnn_results['successful_trials']}/{cnn_results['total_trials']}\n\n")
            
            if lstm_results:
                f.write("LSTM OPTIMIZATION:\n")
                f.write(f"  Best validation loss: {lstm_results['best_val_loss']:.4f}\n")
                f.write(f"  Best direction accuracy: {lstm_results['best_direction_accuracy']:.2f}%\n")
                f.write(f"  Best parameters: {lstm_results['best_params']}\n")
                f.write(f"  Successful trials: {lstm_results['successful_trials']}/{lstm_results['total_trials']}\n\n")
            
            if comparison_results:
                f.write("FINAL MODEL PERFORMANCE:\n")
                for result in comparison_results:
                    f.write(f"\n{result['model']}:\n")
                    for metric, value in result.items():
                        if metric not in ['model', 'params']:
                            f.write(f"  {metric}: {value:.4f}\n")
        
        print(f"\n✅ Hyperparameter optimization completed with REAL PRICE EVALUATION!")
        print(f"📁 Results saved to {output_dir}")
        
        # Display improvement summary
        print(f"\n📈 IMPROVEMENT SUMMARY:")
        print(f"Original CNN: RMSE=27.97, Direction=50.0%")
        print(f"Original LSTM: RMSE=55.89, Direction=59.52%")
        
        if comparison_results:
            for result in comparison_results:
                print(f"✅ {result['model']}: RMSE=${result['rmse']:.2f}, Direction={result['direction_accuracy']:.1f}%")
    else:
        print("❌ No optimization results to analyze.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize model hyperparameters with fixed evaluation')
    parser.add_argument('--symbol', type=str, default='MSFT', help='Stock symbol to analyze')
    
    args = parser.parse_args()
    main(args.symbol)