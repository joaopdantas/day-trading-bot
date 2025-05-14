"""
Utility functions for machine learning models.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def handle_nan_values(data: np.ndarray, strategy: str = 'zeros') -> np.ndarray:
    """
    Handle NaN values in data.
    
    Args:
        data: Input data
        strategy: Strategy for handling NaN values ('zeros', 'mean', 'median')
        
    Returns:
        Cleaned data
    """
    if not np.isnan(data).any():
        return data
        
    logger.info(f"Handling NaN values using {strategy} strategy")
    
    if strategy == 'zeros':
        return np.nan_to_num(data, nan=0.0)
    elif strategy == 'mean':
        # Calculate mean per feature
        means = np.nanmean(data, axis=0)
        # Replace NaNs with means
        return np.where(np.isnan(data), means, data)
    elif strategy == 'median':
        # Calculate median per feature
        medians = np.nanmedian(data, axis=0)
        # Replace NaNs with medians
        return np.where(np.isnan(data), medians, data)
    else:
        logger.warning(f"Unknown NaN handling strategy: {strategy}. Using zeros.")
        return np.nan_to_num(data, nan=0.0)


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str],
    target_column: str,
    target_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        df: DataFrame with features and target
        sequence_length: Length of input sequences
        feature_columns: Columns to use as features
        target_column: Column to predict
        target_horizon: How many steps ahead to predict
        
    Returns:
        Tuple of (X, y) where X is sequences and y is targets
    """
    X, y = [], []
    
    # Create future target values if needed
    if target_horizon > 1:
        target_col = f"{target_column}_future"
        df[target_col] = df[target_column].shift(-target_horizon)
        # Remove last rows where future data is not available
        df = df.iloc[:-target_horizon]
    else:
        target_col = target_column
        
    # Create sequences
    for i in range(len(df) - sequence_length):
        # Use feature data for input sequence
        features_seq = df[feature_columns].iloc[i:(i + sequence_length)].values
        # Target value
        target_val = df[target_col].iloc[i + sequence_length]
        
        X.append(features_seq)
        y.append(target_val)
    
    return np.array(X), np.array(y).reshape(-1, 1)


# In src/models/utils.py - fix the calculate_direction_accuracy function
def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate direction accuracy with improved error handling."""
    if len(y_true) <= 1:
        return 50.0  # Default to random chance
    
    try:
        # Calculate direction changes
        actual_diff = np.diff(y_true.flatten())
        pred_diff = np.diff(y_pred.flatten())
        
        # Filter out very small changes to avoid noise
        threshold = 1e-6
        valid_indices = np.where(np.abs(actual_diff) > threshold)[0]
        
        if len(valid_indices) > 0:
            actual_direction = actual_diff[valid_indices] > 0
            pred_direction = pred_diff[valid_indices] > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            return float(direction_accuracy)
        else:
            logger.warning("No significant direction changes found in data")
            return 50.0
    except Exception as e:
        logger.warning(f"Error calculating direction accuracy: {e}")
        return 50.0  # Return default value instead of NaN


def plot_training_history(history: Dict, output_dir: str, model_name: str = "model") -> None:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
        model_name: Name for plot files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create MAE plot if available
    if 'mae' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE', color='blue')
        if 'val_mae' in history:
            plt.plot(history['val_mae'], label='Validation MAE', color='orange')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_history.png'), dpi=300)
    plt.close()
    
    logger.info(f"Training history plot saved to {output_dir}")


def is_sequence_data(X: np.ndarray) -> bool:
    """
    Check if input data is sequence data (3D).
    
    Args:
        X: Input features
        
    Returns:
        True if sequence data, False otherwise
    """
    return len(X.shape) == 3


def is_valid_timestamp_index(df: pd.DataFrame) -> bool:
    """
    Check if DataFrame has a valid timestamp index.
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if valid timestamp index, False otherwise
    """
    return isinstance(df.index, pd.DatetimeIndex)


def rolling_window_evaluation(
    prediction_model: Any,
    df: pd.DataFrame,
    window_size: int,
    sequence_length: int,
    feature_columns: List[str],
    target_column: str,
    step_size: int = 1,
    output_dir: str = None
) -> Dict[str, float]:
    """
    Evaluate model using rolling window approach.
    
    Args:
        prediction_model: Model to evaluate
        df: DataFrame with features and target
        window_size: Size of evaluation window
        sequence_length: Length of input sequences
        feature_columns: Columns to use as features
        target_column: Column to predict
        step_size: Steps between evaluation windows
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(df) < window_size + sequence_length:
        logger.error("Data too short for rolling window evaluation")
        return {}
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_y_true = []
    all_y_pred = []
    
    # Loop through windows
    for i in range(0, len(df) - window_size - sequence_length, step_size):
        # Split train and test
        train_df = df.iloc[i:i+sequence_length]
        test_df = df.iloc[i+sequence_length:i+sequence_length+window_size]
        
        # Create sequences
        X_test, y_test = create_sequences(
            pd.concat([train_df.iloc[-sequence_length+1:], test_df]),
            sequence_length,
            feature_columns,
            target_column,
            target_horizon=1
        )
        
        # Make predictions
        y_pred = prediction_model.predict(X_test)
        
        # Store results
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
    
    # Concatenate results
    y_true = np.vstack(all_y_true)
    y_pred = np.vstack(all_y_pred)
    
    # Calculate metrics
    metrics = calculate_prediction_metrics(y_true, y_pred)
    
    # Create plots if output directory specified
    if output_dir:
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
        plt.title('Rolling Window Evaluation')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics annotation
        metrics_text = "\n".join([f"{m.upper()}: {v:.4f}" for m, v in metrics.items()])
        plt.figtext(0.02, 0.02, metrics_text, fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, 'rolling_window_evaluation.png'), dpi=300)
        plt.close()
    
    return metrics


def calculate_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate prediction metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score
    )
    
    try:
        # Handle NaN values
        y_true = handle_nan_values(y_true)
        y_pred = handle_nan_values(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE with safeguards
        try:
            # Avoid division by zero and extremely small values
            valid_indices = abs(y_true) > 1e-8
            if sum(valid_indices) > 0:
                mape = np.mean(np.abs((y_true[valid_indices] - y_pred[valid_indices]) / 
                            y_true[valid_indices])) * 100
                # Cap at a reasonable maximum
                mape = min(float(mape), 1000.0)
            else:
                mape = 1000.0  # Default high value
        except Exception as e:
            logger.warning(f"Error calculating MAPE: {e}")
            mape = 1000.0
        
        # Calculate direction accuracy
        direction_accuracy = calculate_direction_accuracy(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            'mse': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'mape': float('nan'),
            'direction_accuracy': 50.0
        }