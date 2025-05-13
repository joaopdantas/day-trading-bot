"""
Machine Learning Models Module for Day Trading Bot.

This module implements various ML models for price prediction and pattern recognition.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Models
from keras.models import Sequential
from keras.models import load_model

# Layers
from keras.layers import (
    LSTM,
    Dense, 
    Dropout,
    GRU,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Bidirectional
)

# Callbacks
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

# Optimizers
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBuilder:
    """Class for building and training different types of neural networks."""

    @staticmethod
    def build_lstm_model(
        input_shape: Tuple[int, int],
        output_units: int = 1,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        bidirectional: bool = False
    ) -> Sequential:
        """
        Build an LSTM model for time series prediction.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            output_units: Number of output units (1 for regression, >1 for classification)
            lstm_units: List of units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            bidirectional: Whether to use bidirectional LSTM layers

        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()

        # First LSTM layer
        if bidirectional:
            model.add(Bidirectional(LSTM(lstm_units[0], return_sequences=len(lstm_units) > 1), 
                                    input_shape=input_shape))
        else:
            model.add(LSTM(lstm_units[0], input_shape=input_shape,
                      return_sequences=len(lstm_units) > 1))
        model.add(Dropout(dropout_rate))

        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:]):
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=i < len(lstm_units)-2)))
            else:
                model.add(LSTM(units, return_sequences=i < len(lstm_units)-2))
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(output_units))

        # Compile with specified learning rate
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def build_gru_model(
        input_shape: Tuple[int, int],
        output_units: int = 1,
        gru_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ) -> Sequential:
        """
        Build a GRU model for time series prediction.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            output_units: Number of output units
            gru_units: List of units in each GRU layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer

        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()

        # First GRU layer
        model.add(GRU(gru_units[0], input_shape=input_shape,
                  return_sequences=len(gru_units) > 1))
        model.add(Dropout(dropout_rate))

        # Additional GRU layers
        for i, units in enumerate(gru_units[1:]):
            model.add(GRU(units, return_sequences=i < len(gru_units)-2))
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(output_units))

        # Compile with specified learning rate
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def build_cnn_model(
        input_shape: Tuple[int, int],
        output_units: int = 1,
        filters: List[int] = [64, 32],
        kernel_sizes: List[int] = [3, 3],
        pool_sizes: List[int] = [2, 2],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ) -> Sequential:
        """
        Build a CNN model for time series prediction.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            output_units: Number of output units
            filters: List of filter numbers for each Conv1D layer
            kernel_sizes: List of kernel sizes for each Conv1D layer
            pool_sizes: List of pooling sizes for each MaxPooling1D layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer

        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()

        # Add Conv1D layers
        for i, (f, k, p) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
            if i == 0:
                model.add(Conv1D(filters=f, kernel_size=k,
                          activation='relu', input_shape=input_shape))
            else:
                model.add(Conv1D(filters=f, kernel_size=k, activation='relu'))
            model.add(MaxPooling1D(pool_size=p))
            model.add(Dropout(dropout_rate))

        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_units))

        # Compile with specified learning rate
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model


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
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
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
        scaler: Optional[Any] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data and generate visualization.

        Args:
            model: Trained Keras Sequential model
            X_test: Test features
            y_test: Test targets
            scaler: Scaler used to transform the target variable (for inverse transform)
            output_dir: Directory to save evaluation plots (if None, uses model_dir)

        Returns:
            Dictionary of evaluation metrics
        """
        if output_dir is None:
            output_dir = self.model_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform if scaler provided
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

        # Calculate metrics on original scale
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        # Calculate MAPE and handle division by zero
        try:
            mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100
        except:
            # Avoid division by zero
            mape = np.mean(np.abs((y_test_orig - y_pred_orig) / 
                                 (y_test_orig + 1e-10))) * 100
        
        # Direction accuracy (for regression tasks)
        if len(y_test_orig) > 1:
            actual_direction = np.diff(y_test_orig) > 0
            pred_direction = np.diff(y_pred_orig) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 0.0

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

    def load_model(self, model_name: str) -> Tuple[Sequential, Optional[Dict], Optional[Any]]:
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
            model = load_model(model_path)
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


class ModelOptimizer:
    """Class for hyperparameter optimization."""

    @staticmethod
    def optimize_hyperparameters(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = 'lstm',
        param_grid: Dict = None,
        n_trials: int = 10,
        epochs: int = 50,
        model_dir: str = 'model_optimization'
    ) -> Dict:
        """
        Optimize model hyperparameters using random search with validation data.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Type of model ('lstm', 'gru', or 'cnn')
            param_grid: Dictionary of parameter ranges to search
            n_trials: Number of random trials
            epochs: Number of epochs for each trial
            model_dir: Directory to save optimization results

        Returns:
            Dictionary with best parameters
        """
        os.makedirs(model_dir, exist_ok=True)
        
        if param_grid is None:
            param_grid = {
                'units': [[32, 16], [64, 32], [128, 64], [128, 64, 32]],
                'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.001, 0.005, 0.01, 0.0005],
                'batch_size': [16, 32, 64, 128],
                'bidirectional': [True, False]  # Only for LSTM
            }

        best_val_loss = float('inf')
        best_params = None
        best_epoch = 0
        best_direction_acc = 0
        
        # Create results log
        results = []

        for trial in range(n_trials):
            # Randomly sample parameters
            current_params = {}
            for key, value in param_grid.items():
                # Skip bidirectional parameter if not LSTM
                if key == 'bidirectional' and model_type != 'lstm':
                    continue
                current_params[key] = np.random.choice(value)
            
            logger.info(f"Trial {trial+1}/{n_trials} with params: {current_params}")

            # Build model with current parameters
            if model_type == 'lstm':
                model = ModelBuilder.build_lstm_model(
                    input_shape=X_train.shape[1:],
                    lstm_units=current_params['units'],
                    dropout_rate=current_params['dropout_rate'],
                    learning_rate=current_params['learning_rate'],
                    bidirectional=current_params.get('bidirectional', False)
                )
            elif model_type == 'gru':
                model = ModelBuilder.build_gru_model(
                    input_shape=X_train.shape[1:],
                    gru_units=current_params['units'],
                    dropout_rate=current_params['dropout_rate'],
                    learning_rate=current_params['learning_rate']
                )
            else:  # CNN
                model = ModelBuilder.build_cnn_model(
                    input_shape=X_train.shape[1:],
                    dropout_rate=current_params['dropout_rate'],
                    learning_rate=current_params['learning_rate']
                )

            # Create early stopping to avoid wasting time on poor models
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=current_params['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )

            # Check if this is the best model so far based on validation loss
            val_loss = min(history.history['val_loss'])
            best_epoch_idx = np.argmin(history.history['val_loss'])
            
            # Calculate direction accuracy on validation set
            y_pred = model.predict(X_val)
            actual_direction = np.diff(y_val.flatten()) > 0
            pred_direction = np.diff(y_pred.flatten()) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Log results
            logger.info(f"Trial {trial+1} val_loss: {val_loss:.4f}, direction acc: {direction_accuracy:.2f}%, "
                       f"best epoch: {best_epoch_idx+1}/{len(history.history['val_loss'])}")
            
            # Store result
            trial_result = {
                'trial': trial,
                'params': current_params,
                'val_loss': float(val_loss),
                'direction_accuracy': float(direction_accuracy),
                'best_epoch': int(best_epoch_idx + 1),
                'model_type': model_type
            }
            results.append(trial_result)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = current_params
                best_epoch = best_epoch_idx + 1
                best_direction_acc = direction_accuracy
                
                # Save best model so far
                model.save(os.path.join(model_dir, f'best_{model_type}_optimization.h5'))
                
                logger.info(f"New best model found! Val Loss: {best_val_loss:.4f}, "
                           f"Direction Acc: {best_direction_acc:.2f}%, Best Epoch: {best_epoch}")

        # Save optimization results
        results_path = os.path.join(model_dir, f'{model_type}_optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot optimization results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        trial_nums = [r['trial'] for r in results]
        val_losses = [r['val_loss'] for r in results]
        plt.plot(trial_nums, val_losses, '-o')
        plt.title('Validation Loss by Trial')
        plt.xlabel('Trial')
        plt.ylabel('Validation Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        dir_accs = [r['direction_accuracy'] for r in results]
        plt.plot(trial_nums, dir_accs, '-o')
        plt.title('Direction Accuracy by Trial')
        plt.xlabel('Trial')
        plt.ylabel('Direction Accuracy (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'{model_type}_optimization_results.png'))
        plt.close()
        
        logger.info(f"Optimization completed. Best params: {best_params}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best direction accuracy: {best_direction_acc:.2f}%")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Results saved to {results_path}")

        return {
            'best_params': best_params,
            'best_val_loss': float(best_val_loss),
            'best_direction_accuracy': float(best_direction_acc),
            'best_epoch': int(best_epoch),
            'model_type': model_type
        }


class PredictionModel:
    """Main class for managing the prediction model lifecycle."""

    def __init__(
        self,
        model_type: str = 'lstm',
        model_params: Dict = None,
        model_dir: str = 'trained_models'
    ):
        """
        Initialize PredictionModel.

        Args:
            model_type: Type of model to use ('lstm', 'gru', or 'cnn')
            model_params: Parameters for model construction
            model_dir: Directory for saving models
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.trainer = ModelTrainer(model_dir)
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.metadata = {
            'model_type': model_type,
            'creation_date': datetime.datetime.now().isoformat(),
            'model_params': model_params or {}
        }

    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        target_column: str = 'close',
        feature_columns: List[str] = None,
        target_horizon: int = 1,
        train_size: float = 0.7,
        val_size: float = 0.15,
        scale_data: bool = True,
        differencing: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for model training with time series validation.

        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences
            target_column: Column to predict
            feature_columns: Columns to use as features
            target_horizon: How many steps ahead to predict
            train_size: Fraction of data to use for training
            val_size: Fraction of data to use for validation
            scale_data: Whether to scale the data
            differencing: Whether to use differencing for stationarity

        Returns:
            Dictionary with train, validation, test datasets and scalers
        """
        try:
            # Save these parameters in metadata
            self.metadata.update({
                'sequence_length': sequence_length,
                'target_column': target_column,
                'feature_columns': feature_columns,
                'target_horizon': target_horizon,
                'differencing': differencing
            })
            
            if df.empty:
                logger.error("Empty DataFrame provided for data preparation")
                return {}
            
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Default to all columns if not specified
            if feature_columns is None:
                feature_columns = [col for col in data.columns 
                                  if col != target_column and not col.startswith('target_')]
            
            # Ensure all feature columns exist
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}")
                feature_columns = [col for col in feature_columns if col in data.columns]
            
            # Handle differencing if requested
            if differencing:
                logger.info("Applying differencing for stationarity")
                # Save original values for later inverse transformation
                original_values = data[target_column].values
                # Apply differencing to target column
                data[f'{target_column}_diff'] = data[target_column].diff()
                # Replace target with differenced version
                target_for_model = f'{target_column}_diff'
                # Drop the first row which is NaN after differencing
                data = data.iloc[1:].reset_index(drop=True)
            else:
                target_for_model = target_column
                original_values = None
            
            # Create future target values
            if target_horizon > 1:
                data[f'{target_for_model}_future'] = data[target_for_model].shift(-target_horizon)
                # Remove last rows where future data is not available
                data = data.iloc[:-target_horizon].reset_index(drop=True)
                target_for_prediction = f'{target_for_model}_future'
            else:
                target_for_prediction = target_for_model
            
            # Scale the data if requested
            if scale_data:
                logger.info("Scaling features and target")
                # Create and fit scalers
                self.feature_scaler = StandardScaler()
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
                
                # Scale features
                scaled_features = self.feature_scaler.fit_transform(data[feature_columns])
                
                # Scale target
                scaled_target = self.scaler.fit_transform(data[[target_for_prediction]])
                
                # Create a DataFrame of scaled features with column names preserved
                scaled_data = pd.DataFrame(scaled_features, columns=feature_columns)
                
                # Add the scaled target to the DataFrame
                scaled_data[target_for_prediction] = scaled_target
            else:
                scaled_data = data.copy()
            
            # Create sequences
            X, y = [], []
            
            for i in range(len(scaled_data) - sequence_length):
                # Use scaled feature data for input sequence
                features_seq = scaled_data[feature_columns].iloc[i:(i + sequence_length)].values
                # Next value of target variable as prediction target
                target_val = scaled_data[target_for_prediction].iloc[i + sequence_length]
                
                X.append(features_seq)
                y.append(target_val)
            
            X = np.array(X)
            y = np.array(y).reshape(-1, 1)
            
            # REPLACE THIS SECTION WITH THE TIME SERIES SPLIT CODE
            # Calculate sizes for the time series split
            total_samples = len(X)
            test_size = int(total_samples * (1 - train_size - val_size))
            val_size_samples = int(total_samples * val_size)
            
            # Use TimeSeriesSplit to create test set
            tscv = TimeSeriesSplit(n_splits=2, test_size=test_size)
            # Get the indices from the last fold
            for train_indices, test_indices in tscv.split(X):
                pass  # We just want the last fold
                
            # Further split train indices into train and validation
            train_indices_final = train_indices[:-val_size_samples]
            val_indices = train_indices[-val_size_samples:]
            
            # Create train, validation, and test sets
            X_train, y_train = X[train_indices_final], y[train_indices_final]
            X_val, y_val = X[val_indices], y[val_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            # END REPLACEMENT SECTION
            
            logger.info(f"Data preparation complete: X_train shape: {X_train.shape}, "
                       f"X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
            
            # Store in metadata
            self.metadata.update({
                'scaled_data': scale_data,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': 1 - train_size - val_size,
                'n_features': len(feature_columns),
                'feature_columns': feature_columns,
                'x_shape': X.shape,
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'n_test_samples': len(X_test)
            })
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.scaler,
                'original_values': original_values,
                'target_for_prediction': target_for_prediction,
                'feature_columns': feature_columns
            }
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the model based on specified type and parameters.

        Args:
            input_shape: Shape of input data
        """
        if self.model_type == 'lstm':
            self.model = ModelBuilder.build_lstm_model(input_shape, **self.model_params)
        elif self.model_type == 'gru':
            self.model = ModelBuilder.build_gru_model(input_shape, **self.model_params)
        else:  # CNN
            self.model = ModelBuilder.build_cnn_model(input_shape, **self.model_params)
        
        # Store model summary as string
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        self.metadata['model_summary'] = '\n'.join(summary_list)
        
        logger.info(f"Built {self.model_type.upper()} model with shape {input_shape}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict:
        """
        Train the model with enhanced options.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(X_train.shape[1:])

        # Create model name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_model_{timestamp}"
        
        # Call trainer
        self.model, history = self.trainer.train_model(
            self.model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
            **kwargs
        )
        
        # Update metadata with training info
        self.metadata.update({
            'training_date': timestamp,
            'training_params': kwargs,
            'train_samples': len(X_train),
            'final_loss': float(history['loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]) if 'val_loss' in history else None,
        })

        return history

    def predict(
        self, 
        X: np.ndarray, 
        inverse_transform: bool = True
    ) -> np.ndarray:
        """
        Make predictions with the trained model and optionally inverse-transform.

        Args:
            X: Input features
            inverse_transform: Whether to inverse-transform the predictions

        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Make predictions
        predictions = self.model.predict(X)
        
        # Inverse transform if requested
        if inverse_transform and self.scaler is not None:
            try:
                predictions = self.scaler.inverse_transform(predictions)
            except Exception as e:
                logger.warning(f"Could not inverse transform predictions: {e}")
        
        return predictions

    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        output_dir: str = None
    ) -> Dict[str, float]:
        """
        Evaluate the model with enhanced visualization.

        Args:
            X_test: Test features
            y_test: Test targets
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        return self.trainer.evaluate_model(
            self.model, 
            X_test, 
            y_test, 
            scaler=self.scaler,
            output_dir=output_dir
        )

    def save(self, model_name: str = None, metadata: Dict = None) -> str:
        """
        Save the model with metadata and scalers.

        Args:
            model_name: Name of the model to save (default: auto-generated)
            metadata: Additional metadata to save

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_model_{timestamp}"
        
        # Combine with existing metadata
        if metadata:
            self.metadata.update(metadata)

        # Save model, metadata, and scalers
        self.trainer.save_model(
            self.model,
            model_name,
            self.metadata,
            self.scaler
        )
        
        # Save feature scaler if exists
        if self.feature_scaler is not None:
            feature_scaler_path = os.path.join(self.trainer.model_dir, f'{model_name}_feature_scaler.pkl')
            joblib.dump(self.feature_scaler, feature_scaler_path)
            logger.info(f"Feature scaler saved to {feature_scaler_path}")
        
        return os.path.join(self.trainer.model_dir, f'{model_name}.h5')

    def load(self, model_name: str) -> bool:
        """
        Load a saved model with metadata and scalers.

        Args:
            model_name: Name of the model to load

        Returns:
            True if loading was successful
        """
        self.model, metadata, self.scaler = self.trainer.load_model(model_name)
        
        if self.model is not None:
            if metadata:
                self.metadata = metadata
                self.model_type = metadata.get('model_type', self.model_type)
                self.model_params = metadata.get('model_params', {})
            
            # Try to load feature scaler
            feature_scaler_path = os.path.join(self.trainer.model_dir, f'{model_name}_feature_scaler.pkl')
            if os.path.exists(feature_scaler_path):
                self.feature_scaler = joblib.load(feature_scaler_path)
                logger.info(f"Feature scaler loaded from {feature_scaler_path}")
            
            return True
        
        return False

    def predict_next_n_days(
        self, 
        current_data: pd.DataFrame, 
        n_days: int = 5,
        feature_columns: List[str] = None,
        sequence_length: int = None,
        target_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Make predictions for the next n days using recursive forecasting.

        Args:
            current_data: Current market data DataFrame
            n_days: Number of days to predict
            feature_columns: Feature columns to use (uses metadata if None)
            sequence_length: Sequence length (uses metadata if None)
            target_column: Target column to predict

        Returns:
            DataFrame with predictions for the next n days
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Use metadata if parameters not provided
        if feature_columns is None:
            feature_columns = self.metadata.get('feature_columns', None)
            if feature_columns is None:
                raise ValueError("Feature columns not specified and not found in metadata")
        
        if sequence_length is None:
            sequence_length = self.metadata.get('sequence_length', None)
            if sequence_length is None:
                raise ValueError("Sequence length not specified and not found in metadata")
        
        # Make a copy to avoid modifying the original
        data = current_data.copy()
        
        # Ensure all required columns exist
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Prepare results DataFrame
        last_date = data.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_days)]
        predictions_df = pd.DataFrame(index=future_dates, columns=[target_column])
        
        # Scale the feature data
        if self.feature_scaler is not None:
            scaled_features = self.feature_scaler.transform(data[feature_columns])
            scaled_data = pd.DataFrame(scaled_features, index=data.index, columns=feature_columns)
        else:
            scaled_data = data[feature_columns].copy()
        
        # For each future day
        for i in range(n_days):
            # Get the last sequence_length data points
            last_sequence = scaled_data.iloc[-sequence_length:].values
            
            # Reshape for model input
            X_pred = np.array([last_sequence])
            
            # Make prediction
            pred = self.model.predict(X_pred)
            
            # Inverse transform if scaler exists
            if self.scaler is not None:
                pred = self.scaler.inverse_transform(pred)
            
            # Add prediction to results DataFrame
            predictions_df.iloc[i, 0] = pred[0][0]
            
            # Update data for next prediction (recursive forecasting)
            # Create a new row with prediction
            new_row = pd.DataFrame([data.iloc[-1].copy()], index=[future_dates[i]])
            new_row[target_column] = pred[0][0]
            
            # Add engineered features based on the prediction if needed
            # This depends on your feature engineering approach
            
            # Append to data
            data = pd.concat([data, new_row])
            
            # Scale new data point if scaler exists
            if self.feature_scaler is not None:
                scaled_new_features = self.feature_scaler.transform(data[feature_columns].iloc[-1:])
                new_scaled_row = pd.DataFrame(scaled_new_features, index=[future_dates[i]], 
                                            columns=feature_columns)
                scaled_data = pd.concat([scaled_data, new_scaled_row])
            else:
                scaled_data = pd.concat([scaled_data, data[feature_columns].iloc[-1:]])
        
        return predictions_df


class EnsembleModel:
    """
    Class for creating ensemble models from multiple base models.
    This improves prediction stability and accuracy.
    """
    
    def __init__(self, models: List[PredictionModel], ensemble_type: str = 'average'):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of trained PredictionModel instances
            ensemble_type: Type of ensemble ('average', 'weighted')
        """
        self.models = models
        self.ensemble_type = ensemble_type
        self.weights = None
        self.target_scaler = None
        
        # Validate that all models are trained
        for i, model in enumerate(models):
            if model.model is None:
                raise ValueError(f"Model {i} is not trained")
                
        # Use the first model's scaler for inverse transformation if needed
        if models and models[0].scaler is not None:
            self.target_scaler = models[0].scaler
            
        # Default to equal weights
        self.weights = np.ones(len(models)) / len(models)
    
    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set weights for weighted ensemble.
        
        Args:
            weights: Array of weights for each model (must sum to 1)
        """
        if len(weights) != len(self.models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(self.models)})")
            
        if abs(np.sum(weights) - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {np.sum(weights)}, normalizing to 1.0")
            weights = weights / np.sum(weights)
            
        self.weights = weights
        logger.info(f"Ensemble weights set: {self.weights}")
    
    def predict(self, X: np.ndarray, inverse_transform: bool = True) -> np.ndarray:
        """
        Make predictions using the ensemble of models.
        
        Args:
            X: Input features
            inverse_transform: Whether to inverse-transform the predictions
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            pred = model.predict(X, inverse_transform=False)  # Keep scaled predictions for now
            predictions.append(pred)
            
        # Stack predictions horizontally for aggregation
        stacked_preds = np.hstack(predictions)
        
        # Apply ensemble method
        if self.ensemble_type == 'weighted':
            # Apply weights for weighted average
            ensemble_preds = np.sum(stacked_preds * self.weights, axis=1, keepdims=True)
        else:
            # Default to simple average
            ensemble_preds = np.mean(stacked_preds, axis=1, keepdims=True)
            
        # Inverse transform if requested
        if inverse_transform and self.target_scaler is not None:
            try:
                ensemble_preds = self.target_scaler.inverse_transform(ensemble_preds)
            except Exception as e:
                logger.warning(f"Could not inverse transform ensemble predictions: {e}")
                
        return ensemble_preds
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, output_dir: str = None) -> Dict[str, float]:
        """
        Evaluate the ensemble model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get ensemble predictions
        y_pred = self.predict(X_test)
        
        # Check if y_test needs inverse transform
        if self.target_scaler is not None and not np.array_equal(y_test, self.target_scaler.inverse_transform(y_test)):
            y_test_orig = self.target_scaler.inverse_transform(y_test)
        else:
            y_test_orig = y_test
        
        # Calculate metrics
        mse = mean_squared_error(y_test_orig, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        
        # Calculate MAPE and handle division by zero
        try:
            mape = mean_absolute_percentage_error(y_test_orig, y_pred) * 100
        except:
            # Avoid division by zero
            mape = np.mean(np.abs((y_test_orig - y_pred) / 
                             (y_test_orig + 1e-10))) * 100
        
        # Direction accuracy
        if len(y_test_orig) > 1:
            actual_direction = np.diff(y_test_orig.flatten()) > 0
            pred_direction = np.diff(y_pred.flatten()) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 0.0
            
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }
        
        # Log results
        logger.info(f"Ensemble Model Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
            
        # Create visualization if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create prediction vs actual plot
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_orig, label='Actual', color='blue', alpha=0.7)
            plt.plot(y_pred, label='Ensemble Prediction', color='red', alpha=0.7)
            plt.title('Ensemble Prediction vs Actual')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add metrics annotation
            metrics_text = "\n".join([f"{m.upper()}: {v:.4f}" for m, v in metrics.items()])
            plt.figtext(0.02, 0.02, metrics_text, fontsize=9, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, 'ensemble_prediction_vs_actual.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create individual model predictions for comparison
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_orig, label='Actual', color='blue', alpha=0.7)
            
            for i, model in enumerate(self.models):
                y_pred_single = model.predict(X_test)
                plt.plot(y_pred_single, label=f'Model {i+1}', alpha=0.4)
                
            plt.plot(y_pred, label='Ensemble', color='red', linewidth=2)
            plt.title('Individual Models vs Ensemble Prediction')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'ensemble_vs_individual_models.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
        return metrics