"""
Enhanced Model optimizer module with model-specific hyperparameter grids.
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
    """Class for hyperparameter optimization with model-specific parameter grids."""

    @staticmethod
    def _convert_numpy_types(obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object that might contain numpy types
            
        Returns:
            Object with numpy types converted to Python types
        """
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [ModelOptimizer._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(ModelOptimizer._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: ModelOptimizer._convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj

    @staticmethod
    def get_default_param_grid(model_type: str) -> Dict:
        """
        Get default parameter grid for specific model type.
        
        Args:
            model_type: Type of model ('lstm', 'gru', or 'cnn')
            
        Returns:
            Dictionary with model-specific parameter ranges
        """
        if model_type == 'lstm':
            return {
                'units': [[32, 16], [64, 32], [128, 64], [64, 32, 16]],
                'dropout_rate': [0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
                'batch_size': [16, 32, 64],
                'bidirectional': [True, False]
            }
        elif model_type == 'gru':
            return {
                'units': [[32, 16], [64, 32], [128, 64], [64, 32, 16]],
                'dropout_rate': [0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
                'batch_size': [16, 32, 64]
                # No bidirectional for GRU in current implementation
            }
        elif model_type == 'cnn':
            return {
                'filters': [[16, 8], [32, 16], [64, 32], [32, 16, 8]],
                'kernel_sizes': [[3, 3], [5, 3], [7, 5], [5, 3, 3]],
                'pool_sizes': [[2, 2], [2, 1], [3, 2]],  # More conservative pooling
                'dropout_rate': [0.2, 0.3, 0.4],
                'learning_rate': [0.0001, 0.0005, 0.001],
                'batch_size': [16, 32, 64]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def optimize_hyperparameters(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = 'lstm',
        param_grid: Dict = None,
        n_trials: int = 20,  # Increased default trials
        epochs: int = 50,
        model_dir: str = 'model_optimization',
        optimize_for: str = 'val_loss'  # New parameter for optimization target
    ) -> Dict:
        """
        Optimize model hyperparameters using random search with model-specific grids.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Type of model ('lstm', 'gru', or 'cnn')
            param_grid: Custom parameter grid (uses default if None)
            n_trials: Number of random trials
            epochs: Number of epochs for each trial
            model_dir: Directory to save optimization results
            optimize_for: Metric to optimize ('val_loss' or 'direction_accuracy')

        Returns:
            Dictionary with best parameters and results
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Get model-specific parameter grid
        if param_grid is None:
            param_grid = ModelOptimizer.get_default_param_grid(model_type)
            logger.info(f"Using default parameter grid for {model_type}")
        
        # Validate model type and parameters
        if model_type not in ['lstm', 'gru', 'cnn']:
            raise ValueError(f"Unsupported model type: {model_type}")

        best_val_loss = float('inf')
        best_direction_acc = 0
        best_params = None
        best_epoch = 0
        
        # Create results log
        results = []

        logger.info(f"Starting optimization for {model_type.upper()} model with {n_trials} trials")
        logger.info(f"Parameter space: {param_grid}")

        for trial in range(n_trials):
            # Randomly sample parameters based on model type
            current_params = ModelOptimizer._sample_params(param_grid, model_type)
            
            logger.info(f"Trial {trial+1}/{n_trials} with params: {current_params}")

            try:
                # Build model with current parameters
                model = ModelOptimizer._build_model_with_params(
                    model_type, X_train.shape[1:], current_params
                )

                # Create callbacks for training
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=7,  # Slightly more patience
                        restore_best_weights=True,
                        verbose=0
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6,
                        verbose=0
                    )
                ]

                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=current_params['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )

                # Calculate metrics
                val_loss = min(history.history['val_loss'])
                best_epoch_idx = np.argmin(history.history['val_loss'])
                
                # Calculate direction accuracy on validation set
                y_pred = model.predict(X_val, verbose=0)
                direction_accuracy = ModelOptimizer._calculate_direction_accuracy(y_val, y_pred)
                
                # Calculate additional metrics
                mae = min(history.history.get('val_mae', [float('inf')]))
                
                # Log results
                logger.info(f"Trial {trial+1}: val_loss={val_loss:.4f}, "
                           f"direction_acc={direction_accuracy:.2f}%, "
                           f"val_mae={mae:.4f}, epoch={best_epoch_idx+1}")
                
                # Store result
                trial_result = {
                    'trial': trial,
                    'params': current_params,
                    'val_loss': float(val_loss),
                    'val_mae': float(mae),
                    'direction_accuracy': float(direction_accuracy),
                    'best_epoch': int(best_epoch_idx + 1),
                    'model_type': model_type,
                    'training_time': len(history.history['loss'])
                }
                results.append(trial_result)

                # Determine if this is the best model based on optimization target
                is_best = False
                if optimize_for == 'val_loss' and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best = True
                elif optimize_for == 'direction_accuracy' and direction_accuracy > best_direction_acc:
                    best_direction_acc = direction_accuracy
                    is_best = True
                
                if is_best:
                    best_params = current_params
                    best_epoch = best_epoch_idx + 1
                    
                    # Save best model
                    model_path = os.path.join(model_dir, f'best_{model_type}_optimization.h5')
                    model.save(model_path)
                    
                    logger.info(f"🎯 New best model! Optimizing for {optimize_for}: "
                               f"val_loss={val_loss:.4f}, direction_acc={direction_accuracy:.2f}%")

            except Exception as e:
                logger.error(f"Trial {trial+1} failed: {e}")
                # Add failed trial to results
                trial_result = {
                    'trial': trial,
                    'params': current_params,
                    'val_loss': float('inf'),
                    'val_mae': float('inf'),
                    'direction_accuracy': 0.0,
                    'best_epoch': 0,
                    'model_type': model_type,
                    'error': str(e)
                }
                results.append(trial_result)
                continue

        # Save optimization results with JSON serialization fix
        results_path = os.path.join(model_dir, f'{model_type}_optimization_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = []
            for result in results:
                clean_result = {}
                for key, value in result.items():
                    if key == 'params':
                        # Clean the params dictionary
                        clean_params = {}
                        for param_key, param_value in value.items():
                            clean_params[param_key] = ModelOptimizer._convert_numpy_types(param_value)
                        clean_result[key] = clean_params
                    else:
                        clean_result[key] = ModelOptimizer._convert_numpy_types(value)
                serializable_results.append(clean_result)
            
            json.dump(serializable_results, f, indent=2)
        
        # Create visualization of results
        ModelOptimizer._plot_optimization_results(results, model_type, model_dir)
        
        # Calculate final metrics
        successful_results = [r for r in results if 'error' not in r]
        
        logger.info(f"Optimization completed for {model_type.upper()}")
        logger.info(f"Successful trials: {len(successful_results)}/{n_trials}")
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best direction accuracy: {best_direction_acc:.2f}%")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Results saved to {results_path}")

        return {
            'best_params': best_params,
            'best_val_loss': float(best_val_loss),
            'best_direction_accuracy': float(best_direction_acc),
            'best_epoch': int(best_epoch),
            'model_type': model_type,
            'successful_trials': len(successful_results),
            'total_trials': n_trials,
            'optimization_target': optimize_for,
            'all_results': results
        }
    
    @staticmethod
    def _sample_params(param_grid: Dict, model_type: str) -> Dict:
        """Sample parameters from the grid based on model type."""
        current_params = {}
        
        for key, values in param_grid.items():
            # Skip parameters not relevant to the model type
            if key == 'bidirectional' and model_type != 'lstm':
                continue
            if key in ['filters', 'kernel_sizes', 'pool_sizes'] and model_type != 'cnn':
                continue
            if key == 'units' and model_type == 'cnn':
                continue
                
            current_params[key] = np.random.choice(values) if isinstance(values[0], (int, float, bool)) else np.random.choice(values, replace=False)
        
        return current_params
    
    @staticmethod
    def _build_model_with_params(model_type: str, input_shape: tuple, params: Dict):
        """Build model with given parameters, filtering out training-only params."""
        # Filter out training-only parameters
        model_params = {k: v for k, v in params.items() if k != 'batch_size'}
        
        if model_type == 'lstm':
            return ModelBuilder.build_lstm_model(
                input_shape=input_shape,
                lstm_units=model_params['units'],  # Map 'units' to 'lstm_units'
                dropout_rate=model_params['dropout_rate'],
                learning_rate=model_params['learning_rate'],
                bidirectional=model_params.get('bidirectional', False)
            )
        elif model_type == 'gru':
            return ModelBuilder.build_gru_model(
                input_shape=input_shape,
                gru_units=model_params['units'],  # Map 'units' to 'gru_units'
                dropout_rate=model_params['dropout_rate'],
                learning_rate=model_params['learning_rate']
            )
        elif model_type == 'cnn':
            return ModelBuilder.build_cnn_model(
                input_shape=input_shape,
                filters=model_params['filters'],
                kernel_sizes=model_params['kernel_sizes'],
                pool_sizes=model_params['pool_sizes'],
                dropout_rate=model_params['dropout_rate'],
                learning_rate=model_params['learning_rate']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate direction accuracy with robust error handling."""
        if len(y_true) <= 1:
            return 50.0
        
        try:
            actual_diff = np.diff(y_true.flatten())
            pred_diff = np.diff(y_pred.flatten())
            
            # Filter out very small changes
            threshold = 1e-6
            valid_indices = np.where(np.abs(actual_diff) > threshold)[0]
            
            if len(valid_indices) > 0:
                actual_direction = actual_diff[valid_indices] > 0
                pred_direction = pred_diff[valid_indices] > 0
                return np.mean(actual_direction == pred_direction) * 100
            else:
                return 50.0
        except:
            return 50.0
    
    @staticmethod
    def _plot_optimization_results(results: List[Dict], model_type: str, output_dir: str) -> None:
        """Create enhanced optimization result plots."""
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            logger.warning("No successful trials to plot")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Validation Loss vs Trial
        plt.subplot(2, 3, 1)
        trial_nums = [r['trial'] for r in successful_results]
        val_losses = [r['val_loss'] for r in successful_results]
        plt.scatter(trial_nums, val_losses, alpha=0.6)
        plt.plot(trial_nums, val_losses, alpha=0.3)
        plt.title('Validation Loss by Trial')
        plt.xlabel('Trial')
        plt.ylabel('Validation Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Direction Accuracy vs Trial
        plt.subplot(2, 3, 2)
        dir_accs = [r['direction_accuracy'] for r in successful_results]
        plt.scatter(trial_nums, dir_accs, alpha=0.6, color='green')
        plt.plot(trial_nums, dir_accs, alpha=0.3, color='green')
        plt.title('Direction Accuracy by Trial')
        plt.xlabel('Trial')
        plt.ylabel('Direction Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Validation Loss vs Direction Accuracy
        plt.subplot(2, 3, 3)
        plt.scatter(val_losses, dir_accs, alpha=0.6, color='purple')
        plt.xlabel('Validation Loss')
        plt.ylabel('Direction Accuracy (%)')
        plt.title('Loss vs Direction Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training Epochs Distribution
        plt.subplot(2, 3, 4)
        epochs = [r['training_time'] for r in successful_results]
        plt.hist(epochs, bins=10, alpha=0.7, color='orange')
        plt.title('Training Epochs Distribution')
        plt.xlabel('Epochs to Convergence')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Best Epoch Distribution
        plt.subplot(2, 3, 5)
        best_epochs = [r['best_epoch'] for r in successful_results]
        plt.hist(best_epochs, bins=10, alpha=0.7, color='red')
        plt.title('Best Epoch Distribution')
        plt.xlabel('Best Epoch')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Parameter Analysis (example with dropout_rate)
        plt.subplot(2, 3, 6)
        if 'dropout_rate' in successful_results[0]['params']:
            dropout_rates = [r['params']['dropout_rate'] for r in successful_results]
            val_losses_for_dropout = [r['val_loss'] for r in successful_results]
            plt.scatter(dropout_rates, val_losses_for_dropout, alpha=0.6)
            plt.xlabel('Dropout Rate')
            plt.ylabel('Validation Loss')
            plt.title('Dropout Rate vs Performance')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_type}_optimization_analysis.png'), dpi=300)
        plt.close()
        
        logger.info(f"Optimization analysis plots saved to {output_dir}")