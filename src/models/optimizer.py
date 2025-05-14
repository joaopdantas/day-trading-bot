"""
Model optimizer module for hyperparameter tuning.
"""

import os
import json
import logging
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from .builder import ModelBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            early_stop = keras.callbacks.EarlyStopping(
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
        
        # Create visualization of results
        ModelOptimizer._plot_optimization_results(results, model_type, model_dir)
        
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
    
    @staticmethod
    def _plot_optimization_results(results: List[Dict], model_type: str, output_dir: str) -> None:
        """
        Plot optimization results.
        
        Args:
            results: List of trial results
            model_type: Type of model
            output_dir: Directory to save plots
        """
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
        plt.savefig(os.path.join(output_dir, f'{model_type}_optimization_results.png'))
        plt.close()