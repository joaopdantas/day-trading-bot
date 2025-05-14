"""
PredictionModel module providing the main user interface for model training and prediction.
"""

import os
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

from .builder import ModelBuilder
from .trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            
            # Initialize original_values for all cases
            original_values = data[target_column].values
            
            # Handle differencing if requested
            if differencing:
                logger.info("Applying differencing for stationarity")
                # Store original values for later inverse transformation
                self.metadata['original_target'] = original_values
                # Apply differencing to target column
                data[f'{target_column}_diff'] = data[target_column].diff()
                # Replace target with differenced version
                target_for_model = f'{target_column}_diff'
                # Drop the first row which is NaN after differencing
                data = data.iloc[1:].reset_index(drop=True)
                # Update original_values to match the shifted data
                original_values = original_values[1:]
            else:
                target_for_model = target_column
                self.metadata['differencing'] = False
            
            # Create future target values
            if target_horizon > 1:
                data[f'{target_for_model}_future'] = data[target_for_model].shift(-target_horizon)
                # Remove last rows where future data is not available
                data = data.iloc[:-target_horizon].reset_index(drop=True)
                target_for_prediction = f'{target_for_model}_future'
                # Update original_values again if needed
                original_values = original_values[:len(data)]
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
            
            # Use TimeSeriesSplit for proper time series validation
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

        # Check for NaN values
        if np.isnan(X).any():
            logger.warning("Input data contains NaN values. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0)

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
            X_test, 
            y_test, 
            output_dir=None
            ):
        """Evaluate with proper inverse transform for differenced data."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test, inverse_transform=False)
        
        # Special handling for differenced data
        if self.metadata.get('differencing', False) and 'original_target' in self.metadata:
            # We need to convert differenced predictions back to original scale
            # This requires cumulative sum starting from the last known value
            logger.info("Applying inverse differencing transform")
            original_target = self.metadata['original_target']
            last_known_value = original_target[-len(y_test)-1]  # Value before test set
            
            # Inverse transform predictions and test data if scaled
            if self.scaler is not None:
                y_pred_diff = self.scaler.inverse_transform(y_pred)
                y_test_diff = self.scaler.inverse_transform(y_test)
            else:
                y_pred_diff = y_pred
                y_test_diff = y_test
                
            # Convert from differences back to levels
            y_pred_levels = np.zeros(y_pred_diff.shape)
            y_test_levels = np.zeros(y_test_diff.shape)
            
            # First value uses the last known value from training
            y_pred_levels[0] = last_known_value + y_pred_diff[0]
            y_test_levels[0] = last_known_value + y_test_diff[0]
            
            # Cumulative sum for the rest
            for i in range(1, len(y_pred_diff)):
                y_pred_levels[i] = y_pred_levels[i-1] + y_pred_diff[i]
                y_test_levels[i] = y_test_levels[i-1] + y_test_diff[i]
                
            # Use these levels for evaluation
            return self.trainer.evaluate_model(
                self.model,
                X_test,
                y_test,
                scaler=None,  # Already inverse transformed
                output_dir=output_dir,
                y_pred_override=y_pred_levels,
                y_test_override=y_test_levels
            )
        else:
            # Standard evaluation
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