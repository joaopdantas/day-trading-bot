"""
Model trainer module for training and evaluating ML models.
"""

import os
import json
import logging
import datetime
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training and evaluating models."""

    def __init__(self, model_dir: str = 'trained_models'):
        """
        Initialize ModelTrainer.

        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train_model(
        self,
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        reduce_lr_factor: float = 0.2,
        reduce_lr_patience: int = 5,
        model_name: str = 'model',
        use_tensorboard: bool = False
    ) -> Tuple[Sequential, Dict]:
        """
        Train a model with early stopping, learning rate reduction, and checkpointing.

        Args:
            model: Keras Sequential model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (if None, uses validation_split)
            y_val: Validation targets (if None, uses validation_split)
            validation_split: Fraction of data to use for validation if X_val/y_val not provided
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            reduce_lr_factor: Factor to reduce learning rate by
            reduce_lr_patience: Number of epochs to wait before reducing learning rate
            model_name: Name for saving the model
            use_tensorboard: Whether to use TensorBoard for logging

        Returns:
            Tuple of (trained model, training history)
        """
        # Create model save path
        model_path = os.path.join(self.model_dir, f'{model_name}.h5')
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Add TensorBoard if requested
        if use_tensorboard:
            log_dir = os.path.join("logs", model_name + "_" + 
                                  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks.append(
                TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True
                )
            )
        
        # Train the model
        if X_val is not None and y_val is not None:
            # Use provided validation data
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Use validation split
            history = model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

        # Save training history
        history_path = os.path.join(self.model_dir, f'{model_name}_history.json')
        with open(history_path, 'w') as f:
            history_dict = {key: [float(x) for x in value]
                            for key, value in history.history.items()}
            json.dump(history_dict, f)
            
        logger.info(f"Model trained and saved to {model_path}")
        logger.info(f"Training history saved to {history_path}")

        return model, history.history

    def evaluate_model(
    self,
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Any = None,
    output_dir: str = None,
    y_pred_override: np.ndarray = None,
    y_test_override: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data and generate visualization.

        Args:
            model: Trained Keras Sequential model
            X_test: Test features
            y_test: Test targets
            scaler: Scaler used to transform the target variable (for inverse transform)
            output_dir: Directory to save evaluation plots (if None, uses model_dir)
            y_pred_override: Override for predictions (e.g., for differenced data)
            y_test_override: Override for test values (e.g., for differenced data)

        Returns:
            Dictionary of evaluation metrics
        """
        if output_dir is None:
            output_dir = self.model_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Use overrides if provided (for differenced data or custom processing)
        if y_pred_override is not None and y_test_override is not None:
            y_pred_orig = y_pred_override
            y_test_orig = y_test_override
            logger.info("Using provided override values for evaluation")
        else:
            # Standard inverse transform if scaler provided
            if scaler is not None:
                # Reshape for inverse_transform
                if len(y_test.shape) == 1:
                    y_test_reshaped = y_test.reshape(-1, 1)
                    y_pred_reshaped = y_pred.reshape(-1, 1)
                else:
                    y_test_reshaped = y_test
                    y_pred_reshaped = y_pred
                    
                # Try to inverse_transform
                try:
                    y_test_orig = scaler.inverse_transform(y_test_reshaped)
                    y_pred_orig = scaler.inverse_transform(y_pred_reshaped)
                    
                    # Flatten if needed
                    if len(y_test.shape) == 1:
                        y_test_orig = y_test_orig.flatten()
                        y_pred_orig = y_pred_orig.flatten()
                except Exception as e:
                    logger.warning(f"Could not inverse transform with scaler: {e}")
                    y_test_orig = y_test
                    y_pred_orig = y_pred
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred

        # Check for NaN or inf values
        if np.isnan(y_test_orig).any() or np.isinf(y_test_orig).any():
            logger.warning("Test data contains NaN or inf values. Cleaning...")
            y_test_orig = np.nan_to_num(y_test_orig, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if np.isnan(y_pred_orig).any() or np.isinf(y_pred_orig).any():
            logger.warning("Prediction data contains NaN or inf values. Cleaning...")
            y_pred_orig = np.nan_to_num(y_pred_orig, nan=0.0, posinf=1e10, neginf=-1e10)

        # Calculate metrics on original scale
        try:
            mse = mean_squared_error(y_test_orig, y_pred_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            
            # Calculate R2 with safeguards
            try:
                r2 = r2_score(y_test_orig, y_pred_orig)
                # Cap extremely negative R2 values to a reasonable minimum
                if r2 < -10:
                    logger.warning(f"Extremely negative R2 score: {r2}. Capping at -10.")
                    r2 = -10.0
            except Exception as e:
                logger.warning(f"Error calculating R2 score: {e}")
                r2 = -1.0  # Default poor value
            
            # Calculate MAPE with robust handling
            try:
                # Filter out values where the true value is close to zero
                valid_indices = np.abs(y_test_orig) > 1e-6
                if np.any(valid_indices):
                    # Calculate MAPE only on valid indices
                    mape = np.mean(np.abs((y_test_orig[valid_indices] - y_pred_orig[valid_indices]) / 
                                    y_test_orig[valid_indices])) * 100
                    # Cap at a reasonable maximum
                    mape = min(float(mape), 1000.0)
                else:
                    logger.warning("Cannot calculate MAPE: All true values are near zero")
                    mape = 1000.0  # Use a high but reasonable default
            except Exception as e:
                logger.warning(f"Error calculating MAPE: {e}")
                mape = 1000.0
            
            # Direction accuracy with robust handling
            if len(y_test_orig) > 1:
                try:
                    # Calculate directions
                    actual_diff = np.diff(y_test_orig.flatten())
                    pred_diff = np.diff(y_pred_orig.flatten())
                    
                    # Check if we have meaningful differences (avoid tiny fluctuations)
                    meaningful_indices = np.abs(actual_diff) > 1e-8
                    
                    if np.any(meaningful_indices):  # Check if any meaningful differences exist
                        actual_direction = (actual_diff[meaningful_indices] > 0).astype(int)
                        pred_direction = (pred_diff[meaningful_indices] > 0).astype(int)
                        
                        # Now use these filtered arrays for the calculation
                        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                    else:
                        # No meaningful price movements found
                        logger.warning("No meaningful price movements found for direction accuracy")
                        direction_accuracy = 50.0  # Default to random chance
                except Exception as e:
                    logger.warning(f"Error calculating direction accuracy: {e}")
                    direction_accuracy = 50.0  # Default to random chance
            else:
                direction_accuracy = 50.0

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Provide default values for metrics in case of error
            mse = 1000.0
            rmse = 31.6  # sqrt(1000)
            mae = 25.0
            r2 = -1.0
            mape = 1000.0
            direction_accuracy = 50.0

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }

        # Log evaluation results
        logger.info("Model Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")

        # Create prediction vs actual plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_orig, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred_orig, label='Predicted', color='red', alpha=0.7)
        plt.title('Prediction vs Actual')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics annotation
        metrics_text = "\n".join([f"{m.upper()}: {v:.4f}" for m, v in metrics.items()])
        plt.figtext(0.02, 0.02, metrics_text, fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'prediction_vs_actual.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create prediction error plot
        plt.figure(figsize=(12, 6))
        error = y_test_orig - y_pred_orig
        plt.scatter(y_test_orig, error, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.xlabel('Actual Value')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.grid(True, alpha=0.3)
        
        # Save the error plot
        error_plot_path = os.path.join(output_dir, 'prediction_error.png')
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {output_dir}")

        return metrics
    
    def _create_evaluation_plots(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, float],
        output_dir: str
    ) -> None:
        """
        Create evaluation plots.
        
        Args:
            y_test: Test targets (original scale)
            y_pred: Predictions (original scale)
            metrics: Dictionary of evaluation metrics
            output_dir: Directory to save plots
        """
        # Create prediction vs actual plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
        plt.title('Prediction vs Actual')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics annotation
        metrics_text = "\n".join([f"{m.upper()}: {v:.4f}" for m, v in metrics.items()])
        plt.figtext(0.02, 0.02, metrics_text, fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'prediction_vs_actual.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create prediction error plot
        plt.figure(figsize=(12, 6))
        error = y_test - y_pred
        plt.scatter(y_test, error, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.xlabel('Actual Value')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.grid(True, alpha=0.3)
        
        # Save the error plot
        error_plot_path = os.path.join(output_dir, 'prediction_error.png')
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {output_dir}")

    def save_model(
        self,
        model: Sequential,
        model_name: str,
        metadata: Dict = None,
        scaler: Any = None
    ) -> None:
        """
        Save model, metadata and scaler if provided.

        Args:
            model: Trained Keras Sequential model
            model_name: Name for the saved model
            metadata: Additional metadata to save with the model
            scaler: Scaler used for data preprocessing
        """
        # Save the model
        model_path = os.path.join(self.model_dir, f'{model_name}.h5')
        model.save(model_path)

        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(
                self.model_dir, f'{model_name}_metadata.json')
            with open(metadata_path, 'w') as f:
                # Convert any numpy types to native Python types for JSON serialization
                clean_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (np.int32, np.int64)):
                        clean_metadata[k] = int(v)
                    elif isinstance(v, (np.float32, np.float64)):
                        clean_metadata[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean_metadata[k] = v.tolist()
                    else:
                        clean_metadata[k] = v
                        
                json.dump(clean_metadata, f)
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = os.path.join(self.model_dir, f'{model_name}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

        logger.info(f"Model saved to {model_path}")
        if metadata:
            logger.info(f"Metadata saved to {metadata_path}")

    def load_model(self, model_name: str) -> Tuple[Sequential, Dict, Any]:
        """
        Load a saved model, its metadata and scaler if exists.

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (loaded model, metadata if exists, scaler if exists)
        """
        model_path = os.path.join(self.model_dir, f'{model_name}.h5')
        metadata_path = os.path.join(
            self.model_dir, f'{model_name}_metadata.json')
        scaler_path = os.path.join(self.model_dir, f'{model_name}_scaler.pkl')

        try:
            model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Metadata loaded from {metadata_path}")
            
            scaler = None
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
                
            return model, metadata, scaler
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None, None