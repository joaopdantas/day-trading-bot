"""
Machine Learning Models Module for Day Trading Bot.

This module implements various ML models for price prediction and pattern recognition.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        dropout_rate: float = 0.2
    ) -> Sequential:
        """
        Build an LSTM model for time series prediction.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            output_units: Number of output units (1 for regression, >1 for classification)
            lstm_units: List of units in each LSTM layer
            dropout_rate: Dropout rate for regularization

        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(lstm_units[0], input_shape=input_shape,
                  return_sequences=len(lstm_units) > 1))
        model.add(Dropout(dropout_rate))

        # Additional LSTM layers
        for units in lstm_units[1:]:
            model.add(LSTM(units, return_sequences=False))
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(output_units))

        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def build_gru_model(
        input_shape: Tuple[int, int],
        output_units: int = 1,
        gru_units: List[int] = [64, 32],
        dropout_rate: float = 0.2
    ) -> Sequential:
        """
        Build a GRU model for time series prediction.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            output_units: Number of output units
            gru_units: List of units in each GRU layer
            dropout_rate: Dropout rate for regularization

        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()

        # First GRU layer
        model.add(GRU(gru_units[0], input_shape=input_shape,
                  return_sequences=len(gru_units) > 1))
        model.add(Dropout(dropout_rate))

        # Additional GRU layers
        for units in gru_units[1:]:
            model.add(GRU(units, return_sequences=False))
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(output_units))

        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def build_cnn_model(
        input_shape: Tuple[int, int],
        output_units: int = 1,
        filters: List[int] = [64, 32],
        kernel_sizes: List[int] = [3, 3],
        pool_sizes: List[int] = [2, 2],
        dropout_rate: float = 0.2
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

        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
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
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        model_name: str = 'model'
    ) -> Tuple[Sequential, Dict]:
        """
        Train a model with early stopping and checkpointing.

        Args:
            model: Keras Sequential model to train
            X_train: Training features
            y_train: Training targets
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            model_name: Name for saving the model

        Returns:
            Tuple of (trained model, training history)
        """
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience,
                          restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'{model_name}.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save training history
        with open(os.path.join(self.model_dir, f'{model_name}_history.json'), 'w') as f:
            history_dict = {key: [float(x) for x in value]
                            for key, value in history.history.items()}
            json.dump(history_dict, f)

        return model, history.history

    def evaluate_model(
        self,
        model: Sequential,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            model: Trained Keras Sequential model
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred))
        }

        # Log evaluation results
        logger.info("Model Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")

        return metrics

    def save_model(
        self,
        model: Sequential,
        model_name: str,
        metadata: Dict = None
    ) -> None:
        """
        Save model and its metadata.

        Args:
            model: Trained Keras Sequential model
            model_name: Name for the saved model
            metadata: Additional metadata to save with the model
        """
        # Save the model
        model_path = os.path.join(self.model_dir, f'{model_name}.h5')
        model.save(model_path)

        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(
                self.model_dir, f'{model_name}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_name: str) -> Tuple[Sequential, Optional[Dict]]:
        """
        Load a saved model and its metadata.

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (loaded model, metadata if exists)
        """
        model_path = os.path.join(self.model_dir, f'{model_name}.h5')
        metadata_path = os.path.join(
            self.model_dir, f'{model_name}_metadata.json')

        try:
            model = load_model(model_path)
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None


class ModelOptimizer:
    """Class for hyperparameter optimization."""

    @staticmethod
    def optimize_hyperparameters(
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'lstm',
        param_grid: Dict = None,
        n_trials: int = 10,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Optimize model hyperparameters using random search.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model ('lstm', 'gru', or 'cnn')
            param_grid: Dictionary of parameter ranges to search
            n_trials: Number of random trials
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with best parameters
        """
        if param_grid is None:
            param_grid = {
                'lstm_units': [[32, 16], [64, 32], [128, 64]],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01, 0.1],
                'batch_size': [16, 32, 64]
            }

        best_val_loss = float('inf')
        best_params = None

        for _ in range(n_trials):
            # Randomly sample parameters
            current_params = {
                key: np.random.choice(value) for key, value in param_grid.items()
            }

            # Build model with current parameters
            if model_type == 'lstm':
                model = ModelBuilder.build_lstm_model(
                    input_shape=X_train.shape[1:],
                    lstm_units=current_params['lstm_units'],
                    dropout_rate=current_params['dropout_rate']
                )
            elif model_type == 'gru':
                model = ModelBuilder.build_gru_model(
                    input_shape=X_train.shape[1:],
                    gru_units=current_params['lstm_units'],
                    dropout_rate=current_params['dropout_rate']
                )
            else:  # CNN
                model = ModelBuilder.build_cnn_model(
                    input_shape=X_train.shape[1:],
                    dropout_rate=current_params['dropout_rate']
                )

            # Train model
            history = model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=50,
                batch_size=current_params['batch_size'],
                verbose=0
            )

            # Check if this is the best model so far
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = current_params

        return best_params


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

    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        target_column: str = 'close',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.

        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences
            target_column: Column to predict
            test_size: Fraction of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from ..data.preprocessor import DataPreprocessor

        # Create sequences
        preprocessor = DataPreprocessor()
        X, y = preprocessor.create_sequences(
            df,
            sequence_length=sequence_length,
            target_column=target_column
        )

        # Split data
        return train_test_split(X, y, test_size=test_size, shuffle=False)

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the model based on specified type and parameters.

        Args:
            input_shape: Shape of input data
        """
        if self.model_type == 'lstm':
            self.model = ModelBuilder.build_lstm_model(
                input_shape, **self.model_params)
        elif self.model_type == 'gru':
            self.model = ModelBuilder.build_gru_model(
                input_shape, **self.model_params)
        else:  # CNN
            self.model = ModelBuilder.build_cnn_model(
                input_shape, **self.model_params)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional training parameters

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(X_train.shape[1:])

        self.model, history = self.trainer.train_model(
            self.model,
            X_train,
            y_train,
            model_name=f"{self.model_type}_model",
            **kwargs
        )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        return self.trainer.evaluate_model(self.model, X_test, y_test)

    def save(self, metadata: Dict = None) -> None:
        """
        Save the model.

        Args:
            metadata: Additional metadata to save
        """
        if self.model is None:
            raise ValueError("No model to save")

        self.trainer.save_model(
            self.model,
            f"{self.model_type}_model",
            metadata
        )

    def load(self, model_name: str) -> None:
        """
        Load a saved model.

        Args:
            model_name: Name of the model to load
        """
        self.model, metadata = self.trainer.load_model(model_name)
        if metadata:
            self.model_params = metadata.get('model_params', {})
