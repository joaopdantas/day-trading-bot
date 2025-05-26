"""
DIAGNOSTIC TEST: Let's figure out exactly what's happening with the scaling

This will help us understand where the issue is occurring and fix it properly.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.models import PredictionModel
from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.indicators.technical import TechnicalIndicators

def diagnostic_test():
    """Run diagnostic test to understand the scaling issue."""
    
    print("🔍 DIAGNOSTIC TEST - Understanding the Scaling Issue")
    print("="*60)
    
    # Step 1: Get some simple data
    print("Step 1: Fetching data...")
    try:
        api = get_data_api("alpha_vantage")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        df = api.fetch_historical_data(
            symbol="MSFT",
            interval="1d",
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            print("❌ Failed to get data")
            return
            
        print(f"✅ Got {len(df)} data points")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
    except Exception as e:
        print(f"❌ Error getting data: {e}")
        return
    
    # Step 2: Prepare data and examine what happens
    print("\nStep 2: Preparing data...")
    
    model = PredictionModel(model_type='lstm')
    
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    
    data_dict = model.prepare_data(
        df=df,
        sequence_length=10,
        target_column='close',
        feature_columns=feature_columns,
        target_horizon=1,
        train_size=0.7,
        val_size=0.15,
        scale_data=True,
        differencing=False
    )
    
    if not data_dict:
        print("❌ Data preparation failed")
        return
    
    print("✅ Data preparation successful")
    print(f"X_test shape: {data_dict['X_test'].shape}")
    print(f"y_test shape: {data_dict['y_test'].shape}")
    
    # Step 3: Examine the scaler
    print("\nStep 3: Examining the scaler...")
    scaler = data_dict['target_scaler']
    
    if scaler is not None:
        print(f"Scaler type: {type(scaler)}")
        print(f"Scaler feature range: {scaler.feature_range}")
        print(f"Scaler data_min_: {scaler.data_min_}")
        print(f"Scaler data_max_: {scaler.data_max_}")
        print(f"Scaler data_range_: {scaler.data_range_}")
        
        # Test the scaler manually
        print("\nStep 4: Testing scaler manually...")
        
        # Get a few test values
        y_test_sample = data_dict['y_test'][:5]
        print(f"Sample y_test (scaled): {y_test_sample.flatten()}")
        
        # Try inverse transform
        try:
            y_test_inverse = scaler.inverse_transform(y_test_sample)
            print(f"Sample y_test (inverse): {y_test_inverse.flatten()}")
            print(f"Inverse transform range: ${y_test_inverse.min():.2f} - ${y_test_inverse.max():.2f}")
            
            # Compare with original close prices
            original_close_sample = df['close'].tail(10).head(5).values
            print(f"Original close prices: ${original_close_sample.min():.2f} - ${original_close_sample.max():.2f}")
            
            # Check if they're in the same ballpark
            if abs(y_test_inverse.mean() - original_close_sample.mean()) < 50:
                print("✅ Inverse transform looks correct!")
            else:
                print("❌ Inverse transform looks wrong!")
                print(f"Expected around: ${original_close_sample.mean():.2f}")
                print(f"Got: ${y_test_inverse.mean():.2f}")
                
        except Exception as e:
            print(f"❌ Inverse transform failed: {e}")
            print(f"y_test_sample shape: {y_test_sample.shape}")
            print(f"Scaler was fitted on data with shape: probably (n_samples, 1)")
    
    else:
        print("❌ No scaler found!")
    
    # Step 5: Train a simple model and test evaluation
    print("\nStep 5: Training simple model and testing evaluation...")
    
    try:
        # Build and train a very simple model
        model.build_model(data_dict['X_test'].shape[1:])
        
        # Train for just 1 epoch to get something
        history = model.train(
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            epochs=1,
            batch_size=16
            # Note: removed verbose=0 as it's not supported
        )
        
        print("✅ Model trained (1 epoch)")
        
        # Test prediction
        y_pred_scaled = model.predict(data_dict['X_test'][:5], inverse_transform=False)
        print(f"Predictions (scaled): {y_pred_scaled.flatten()}")
        
        y_pred_unscaled = model.predict(data_dict['X_test'][:5], inverse_transform=True)
        print(f"Predictions (unscaled): {y_pred_unscaled.flatten()}")
        
        # Check if unscaled predictions are in realistic range
        if abs(y_pred_unscaled.max()) > 50:  # Should be in hundreds for stock prices
            print("✅ Unscaled predictions look realistic!")
        else:
            print("❌ Unscaled predictions still look scaled!")
        
        # Test the evaluation method
        print("\nStep 6: Testing evaluation...")
        
        # Use just a few samples for testing
        X_test_sample = data_dict['X_test'][:5]
        y_test_sample = data_dict['y_test'][:5]
        
        print(f"Testing with {len(X_test_sample)} samples")
        print(f"y_test_sample shape: {y_test_sample.shape}")
        print(f"y_test_sample range: {y_test_sample.min():.4f} to {y_test_sample.max():.4f}")
        
        # Test the predict method first
        print("\nTesting prediction method...")
        try:
            y_pred_sample = model.predict(X_test_sample, inverse_transform=False)
            print(f"✅ Prediction successful!")
            print(f"y_pred shape: {y_pred_sample.shape}")
            print(f"y_pred range: {y_pred_sample.min():.4f} to {y_pred_sample.max():.4f}")
            
            # Test inverse transform manually
            if model.scaler is not None:
                print("\nTesting manual inverse transform...")
                try:
                    y_pred_unscaled = model.scaler.inverse_transform(y_pred_sample)
                    y_test_unscaled = model.scaler.inverse_transform(y_test_sample)
                    print(f"✅ Manual inverse transform successful!")
                    print(f"y_pred_unscaled range: ${y_pred_unscaled.min():.2f} to ${y_pred_unscaled.max():.2f}")
                    print(f"y_test_unscaled range: ${y_test_unscaled.min():.2f} to ${y_test_unscaled.max():.2f}")
                except Exception as e:
                    print(f"❌ Manual inverse transform failed: {e}")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            print("\nCalling model.evaluate()...")
            metrics = model.evaluate(
                X_test=X_test_sample,
                y_test=y_test_sample,
                output_dir='diagnostic_test_output'
            )
            
            print("✅ Evaluation completed!")
            print("Metrics returned:")
            print(f"  Type: {type(metrics)}")
            print(f"  Content: {metrics}")
            
            if isinstance(metrics, dict) and metrics:
                print("Metrics breakdown:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                    
                # Check if metrics are realistic
                if 'rmse' in metrics:
                    if metrics['rmse'] > 1 and metrics['rmse'] < 200:
                        print("✅ RMSE looks realistic for stock prices!")
                    else:
                        print(f"❌ RMSE looks wrong: {metrics['rmse']:.2f}")
                else:
                    print("❌ No RMSE found in metrics!")
                    
                if 'mape' in metrics:
                    if 0 <= metrics['mape'] <= 50:
                        print("✅ MAPE looks realistic!")
                    else:
                        print(f"❌ MAPE looks wrong: {metrics['mape']:.2f}%")
                else:
                    print("❌ No MAPE found in metrics!")
            else:
                print("❌ Metrics is empty or not a dictionary!")
                
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Model training/evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    diagnostic_test()