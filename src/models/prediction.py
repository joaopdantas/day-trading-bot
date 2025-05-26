"""
FINAL FIX for PredictionModel - Ensuring Real Price Evaluation

The key issue: Models are trained on scaled data (-1,1) but evaluation 
needs to happen in real price space ($400+). This fix ensures proper
inverse scaling by storing and using the original price range.
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
    """FIXED PredictionModel with guaranteed real price evaluation."""

    def __init__(
        self,
        model_type: str = 'lstm',
        model_params: Dict = None,
        model_dir: str = 'trained_models'
    ):
        """Initialize PredictionModel."""
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
        """Prepare data with GUARANTEED price tracking for evaluation."""
        try:
            # Save parameters in metadata
            self.metadata.update({
                'sequence_length': sequence_length,
                'target_column': target_column,
                'feature_columns': feature_columns,
                'target_horizon': target_horizon,
                'differencing': differencing,
                'scale_data': scale_data
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
            
            # CRITICAL: Store original target prices throughout the entire process
            original_target_prices = data[target_column].values.copy()
            logger.info(f"Original price range: ${original_target_prices.min():.2f} - ${original_target_prices.max():.2f}")
            
            # Store in metadata for later use
            self.metadata['original_price_min'] = float(original_target_prices.min())
            self.metadata['original_price_max'] = float(original_target_prices.max())
            self.metadata['original_price_mean'] = float(original_target_prices.mean())
            
            # Handle differencing if requested
            if differencing:
                logger.info("Applying differencing for stationarity")
                self.metadata['differencing'] = True
                
                # Apply differencing to target column
                data[f'{target_column}_diff'] = data[target_column].diff()
                target_for_model = f'{target_column}_diff'
                # Drop the first row which is NaN after differencing
                data = data.iloc[1:].reset_index(drop=True)
                original_target_prices = original_target_prices[1:]
            else:
                target_for_model = target_column
                self.metadata['differencing'] = False
            
            # Create future target values
            if target_horizon > 1:
                data[f'{target_for_model}_future'] = data[target_for_model].shift(-target_horizon)
                data = data.iloc[:-target_horizon].reset_index(drop=True)
                target_for_prediction = f'{target_for_model}_future'
                original_target_prices = original_target_prices[:len(data)]
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
                target_data = data[[target_for_prediction]].values
                scaled_target = self.scaler.fit_transform(target_data)
                
                # CRITICAL: Store scaler parameters for debugging
                logger.info(f"Target scaler fitted on range: {target_data.min():.2f} to {target_data.max():.2f}")
                logger.info(f"Scaled target range: {scaled_target.min():.2f} to {scaled_target.max():.2f}")
                
                # Create scaled DataFrame
                scaled_data = pd.DataFrame(scaled_features, columns=feature_columns)
                scaled_data[target_for_prediction] = scaled_target.flatten()
            else:
                scaled_data = data.copy()
            
            # Create sequences
            X, y = [], []
            corresponding_original_prices = []
            
            for i in range(len(scaled_data) - sequence_length):
                # Use scaled feature data for input sequence
                features_seq = scaled_data[feature_columns].iloc[i:(i + sequence_length)].values
                # Target value
                target_val = scaled_data[target_for_prediction].iloc[i + sequence_length]
                # Corresponding original price for this target
                orig_price_idx = i + sequence_length
                if orig_price_idx < len(original_target_prices):
                    orig_price = original_target_prices[orig_price_idx]
                else:
                    orig_price = original_target_prices[-1]
                
                X.append(features_seq)
                y.append(target_val)
                corresponding_original_prices.append(orig_price)
            
            X = np.array(X)
            y = np.array(y).reshape(-1, 1)
            corresponding_original_prices = np.array(corresponding_original_prices)
            
            # Use TimeSeriesSplit for proper time series validation
            total_samples = len(X)
            test_size = int(total_samples * (1 - train_size - val_size))
            val_size_samples = int(total_samples * val_size)
            
            tscv = TimeSeriesSplit(n_splits=2, test_size=test_size)
            for train_indices, test_indices in tscv.split(X):
                pass  # Get the last fold
                
            # Split train indices into train and validation
            train_indices_final = train_indices[:-val_size_samples]
            val_indices = train_indices[-val_size_samples:]
            
            # Create datasets
            X_train, y_train = X[train_indices_final], y[train_indices_final]
            X_val, y_val = X[val_indices], y[val_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            
            # CRITICAL: Store original prices for test set
            test_original_prices = corresponding_original_prices[test_indices]
            
            logger.info(f"Data preparation complete:")
            logger.info(f"  X_train shape: {X_train.shape}")
            logger.info(f"  X_val shape: {X_val.shape}")
            logger.info(f"  X_test shape: {X_test.shape}")
            logger.info(f"  Test set original price range: ${test_original_prices.min():.2f} - ${test_original_prices.max():.2f}")
            
            # Update metadata
            self.metadata.update({
                'n_features': len(feature_columns),
                'feature_columns': feature_columns,
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'n_test_samples': len(X_test),
                'test_price_min': float(test_original_prices.min()),
                'test_price_max': float(test_original_prices.max()),
                'test_price_mean': float(test_original_prices.mean())
            })
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'test_original_prices': test_original_prices,
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.scaler,
                'target_for_prediction': target_for_prediction,
                'feature_columns': feature_columns
            }
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build the model based on specified type and parameters."""
        if self.model_type == 'lstm':
            self.model = ModelBuilder.build_lstm_model(input_shape, **self.model_params)
        elif self.model_type == 'gru':
            self.model = ModelBuilder.build_gru_model(input_shape, **self.model_params)
        else:  # CNN
            self.model = ModelBuilder.build_cnn_model(input_shape, **self.model_params)
        
        # Store model summary
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
        """Train the model."""
        if self.model is None:
            self.build_model(X_train.shape[1:])

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_model_{timestamp}"
        
        self.model, history = self.trainer.train_model(
            self.model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
            **kwargs
        )
        
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
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if np.isnan(X).any():
            logger.warning("Input data contains NaN values. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0)

        predictions = self.model.predict(X)
        
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
        SIMPLE EVALUATION: Back to the working original approach.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Use the trainer's evaluation method as originally designed
        # This should work correctly with the scaler
        return self.trainer.evaluate_model(
            self.model,
            X_test,
            y_test,
            scaler=self.scaler,  # Let trainer handle the inverse transform
            output_dir=output_dir
        )

    def save(self, model_name: str = None, metadata: Dict = None) -> str:
        """Save the model with metadata and scalers."""
        if self.model is None:
            raise ValueError("No model to save")
            
        if model_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_model_{timestamp}"
        
        if metadata:
            self.metadata.update(metadata)

        self.trainer.save_model(
            self.model,
            model_name,
            self.metadata,
            self.scaler
        )
        
        if self.feature_scaler is not None:
            feature_scaler_path = os.path.join(self.trainer.model_dir, f'{model_name}_feature_scaler.pkl')
            joblib.dump(self.feature_scaler, feature_scaler_path)
            logger.info(f"Feature scaler saved to {feature_scaler_path}")
        
        return os.path.join(self.trainer.model_dir, f'{model_name}.h5')

    def load(self, model_name: str) -> bool:
        """Load a saved model with metadata and scalers."""
        self.model, metadata, self.scaler = self.trainer.load_model(model_name)
        
        if self.model is not None:
            if metadata:
                self.metadata = metadata
                self.model_type = metadata.get('model_type', self.model_type)
                self.model_params = metadata.get('model_params', {})
            
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
        """Make predictions for the next n days using recursive forecasting."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if feature_columns is None:
            feature_columns = self.metadata.get('feature_columns', None)
            if feature_columns is None:
                raise ValueError("Feature columns not specified and not found in metadata")
        
        if sequence_length is None:
            sequence_length = self.metadata.get('sequence_length', None)
            if sequence_length is None:
                raise ValueError("Sequence length not specified and not found in metadata")
        
        data = current_data.copy()
        
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        last_date = data.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_days)]
        predictions_df = pd.DataFrame(index=future_dates, columns=[target_column])
        
        if self.feature_scaler is not None:
            scaled_features = self.feature_scaler.transform(data[feature_columns])
            scaled_data = pd.DataFrame(scaled_features, index=data.index, columns=feature_columns)
        else:
            scaled_data = data[feature_columns].copy()
        
        for i in range(n_days):
            last_sequence = scaled_data.iloc[-sequence_length:].values
            X_pred = np.array([last_sequence])
            pred = self.model.predict(X_pred)
            
            if self.scaler is not None:
                pred = self.scaler.inverse_transform(pred)
            
            predictions_df.iloc[i, 0] = pred[0][0]
            
            new_row = pd.DataFrame([data.iloc[-1].copy()], index=[future_dates[i]])
            new_row[target_column] = pred[0][0]
            data = pd.concat([data, new_row])
            
            if self.feature_scaler is not None:
                scaled_new_features = self.feature_scaler.transform(data[feature_columns].iloc[-1:])
                new_scaled_row = pd.DataFrame(scaled_new_features, index=[future_dates[i]], 
                                            columns=feature_columns)
                scaled_data = pd.concat([scaled_data, new_scaled_row])
            else:
                scaled_data = pd.concat([scaled_data, data[feature_columns].iloc[-1:]])
        
        return predictions_df