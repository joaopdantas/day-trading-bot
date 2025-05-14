"""
Model builder module for creating neural network architectures.
"""

import logging
from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    LSTM, Dense, Dropout, GRU,
    Conv1D, MaxPooling1D, Flatten, Bidirectional
)
from keras.optimizers import Adam

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

    # In src/models/builder.py - update the build_cnn_model method
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
        """Build a CNN model with dimension checking to prevent errors."""
        model = Sequential()
        
        # Calculate sequence length after each pooling
        curr_seq_len = input_shape[0]
        
        # Add Conv1D layers with dimension checks
        for i, (f, k, p) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
            if i == 0:
                model.add(Conv1D(filters=f, kernel_size=k, activation='relu', 
                                padding='same', input_shape=input_shape))
            else:
                model.add(Conv1D(filters=f, kernel_size=k, activation='relu', 
                                padding='same'))
                
            # Check if pooling would reduce dimension to zero
            if curr_seq_len // p > 0:
                model.add(MaxPooling1D(pool_size=p))
                curr_seq_len = curr_seq_len // p
            else:
                # Skip pooling if dimension would become too small
                logger.warning(f"Skipping pooling layer {i} to prevent dimension error")
            
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